"""Plan assisted chunks from one observation; execute without intermediate input.

Only block boundaries consume observations. Within a plan, ``cursor`` is the
last planned command, NOT a measured state or evidence of contact. Existing FK,
IK and trajectory builders are reused; feedback-gated next/commit methods are
not used. Planned state is published only when the entire block is acknowledged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from .config import PIPER_JOINT_MIN, PIPER_JOINT_MAX, compensate_gripper_action
from .grasp_dip_runtime import GraspDipRuntime
from .handover import HandoverAssist
from .kinematics import fingertip_xyz
from .relative_handover import RelativeHandover
from .runtime import AssistRuntime, clip_actions
from .timing import ActionSchedule
from .bell_press import DualBellPressRuntime


HZ = 10
CLOSE_FRAMES = 3  # Fixed 0.3 s command dwell, never a feedback confirmation.
MODEL_FALLBACK_TASKS = frozenset({
    "click_bell", "sample_loading", "items_handover", "manipulate_pipette",
    "table_rearrangement", "water_pouring",
})
LOGGER = logging.getLogger(__name__)


@dataclass
class GripIntent:
    closed: bool = False
    baseline: float = 0.0
    release_count: int = 0
    saw_open: bool = False
    peak_open: float = 0.0


class ChunkPlanner:
    """Persistent planning state; cloned transactionally for each new block."""

    def __init__(self, args):
        self.args = args
        self.schedule = ActionSchedule(
            mode=args.schedule, legacy_steps=args.legacy_steps,
            source_fps=args.model_action_fps, control_hz=args.control_hz,
            sample_start_step=args.sample_start_step,
        )
        # These tasks are model-only, even if legacy options enable assists.
        self.helpers = (None if args.task in ("mixer_operating", "item_assembly")
                        else AssistRuntime(args))
        self.grips = [GripIntent(), GripIntent()]
        self.lifted = [False, False]  # Scheduled lift complete, not object grasped.
        self.lift_targets = [None, None]
        self.handover_done = False
        self.terminal = None
        self.cycle = -1

    def model_state(self, state):
        if self.helpers is None:
            return np.asarray(state, dtype=np.float32).copy()
        return self.helpers.model_state(state)

    def _emit(self, action, kind):
        if len(self.actions) >= self.budget:
            return False
        self.cursor = clip_actions(np.asarray(action)[None, :])[0]
        self.actions.append(self.cursor.copy())
        self.records.append({"kind": kind, "source_row": self.source_row})
        return True

    def _hold(self, frames, kind):
        for _ in range(min(frames, self.budget - len(self.actions))):
            self._emit(self.cursor, kind)

    def _path(self, path, kind, side=None):
        for row in path:
            action = self.cursor.copy()
            if side is None:
                action = np.asarray(row).copy()
            else:
                action[side * 7:side * 7 + 6] = row
            if not self._emit(action, kind):
                break

    def model_actions(self, raw, state):
        """Prepare model commands without any geometric or grip assistance."""
        raw = np.asarray(raw, dtype=np.float32)
        if self.helpers is not None:
            raw = compensate_gripper_action(
                raw, self.helpers.offsets,
                enabled=self.args.gripper_offset_compensation,
            )
        control = clip_actions(self.schedule.make_plan(raw, state=state))
        return raw, control

    def plan(self, raw, state, *, budget):
        self.cycle += 1
        self.budget = budget
        self.actions, self.records = [], []
        self.source_row = 0
        self.cursor = np.asarray(state, dtype=np.float32).copy()
        raw, control = self.model_actions(raw, state)
        h = self.helpers
        if h is None:
            self._path(control, "model")
        elif self.terminal is not None:
            if self.args.task in ("water_pouring", "sample_loading"):
                # Keep completion latched so assistance cannot restart, but
                # let fresh model blocks finish the task after scripted release.
                self._path(control, "model")
            else:
                self.cursor = self.terminal.copy()
                self._hold(len(control), "completed_hold")
        else:
            if h.floor is not None:
                h.floor.begin_chunk()
                control = h.floor.apply(control)
                raw = h.floor.apply(clip_actions(raw))
            if h.bell is not None:
                bell = DualBellPressRuntime(h.bell, control, lookahead=raw[self.args.legacy_steps:])
                while bell.row < len(control) and len(self.actions) < budget:
                    self.source_row = bell.row
                    step = bell.next(self.cursor, self.cursor)
                    self._emit(step.action, step.kind)
                    bell.commit(step)
            else:
                self._plan_rows(raw, control)
        if not self.actions:
            raise ValueError("Chunk planner produced no actions")
        actions = np.asarray(self.actions, dtype=np.float32)
        records = self.records
        # Do not retain completed block arrays inside the next planner snapshot.
        del self.actions, self.records, self.cursor
        return actions, records

    def _plan_rows(self, raw, control):
        h = self.helpers
        dip_runtime = None
        if h.dip is not None:
            plan = h.dip.plan(control, previous_action=self.cursor, physical_state=self.cursor)
            dip_runtime = GraspDipRuntime(
                h.dip, control, plan, lookahead=raw[self.args.legacy_steps:],
                preclose_config=h.grip_config
                if self.args.task in ("sample_loading", "manipulate_pipette") else None,
            )
        # The press trigger examines model stability, not measured contact.
        if h.pipette is not None and self.grips[1].closed and self._pipette_ready(raw):
            if h.pipette.plan(self.cursor, cycle=self.cycle):
                self._path(h.pipette.path, "pipette_press")
                h.pipette.phase = "completed"
                if h.pipette.config.ready_only:
                    self.terminal = self.cursor.copy()
                return
            raise ValueError(f"Pipette press planning failed: {h.pipette.reason}")
        for index, model in enumerate(control):
            if len(self.actions) >= self.budget:
                break
            self.source_row = index
            target = model.copy()
            if dip_runtime is not None:
                target = self._dip_row(dip_runtime, index)
            # Once lifted, keep that arm at its planned endpoint while the other
            # arm finishes pickup; stale model rows must not undo the lift.
            for side, goal in enumerate(self.lift_targets):
                if goal is not None:
                    start = side * 7
                    target[start:start + 6] = goal
            # Latch/release only model rows. Inserted dwell rows aren't evidence.
            closed, released = self._grips(target, model)
            if h.pour is not None:
                self._pour_pickups(target, model)
                target = h.pour.apply_right_grip_cap(target, model[13])
            # A release trajectory must keep the gripper shut until its end.
            if any(released) and self._release_paths(model, released):
                return  # Fresh inference after a changed release pose.
            if h.handover_observation is not None and self.handover_done:
                target = h.handover_observation.apply(self.cursor, target, completed=True)
                h.handover_observation.commit(target)
            if self.args.task == "sample_loading" and self._handover_requested(model):
                # Hold the receiver while sending the donor's final command.
                target[:7] = self.cursor[:7]
                self._emit(target, "handover_position_right")
                if closed[1]:
                    self._hold(math.ceil(.4 * HZ), "grasp_close")
                self._handover(control[index:])
                return
            self._emit(target, "model")
            if self.args.task == "items_handover" and closed[1]:
                # Fixed 0.4 s after the close command, before motion/transfer.
                self._hold(math.ceil(.4 * HZ), "grasp_dwell")
            if self.args.task == "sample_loading" and closed[1]:
                self._hold(math.ceil(.4 * HZ), "grasp_close")
            if self._handover_requested(model):
                self._handover(control[index:])
                return
            if h.assembly_pickup is not None:
                self._assembly_lift(closed)
                if all(self.lifted):
                    if h.assembly.plan(self.cursor, cycle=self.cycle):
                        self._path(h.assembly.path, "assembly_insert")
                        h.assembly.phase = "done"
                        self.terminal = self.cursor.copy()
                    return
            if h.pour is not None and self._pour_requested(control[index:]):
                self._pour(control[index:])
                return

    def _dip_row(self, runtime, index):
        """Reuse preparation geometry, skipping all arrival/contact wait logic."""
        h = self.helpers
        runtime.row = index
        preparation = runtime._prepare(self.cursor, self.cursor)
        if preparation is not None:
            for j, action in enumerate(preparation.actions):
                self._emit(action, "grasp_prepare")
                h.dip.commit(preparation, j)
            if 1 in runtime.preparing_sides and runtime.preclose_config is not None:
                self._hold(2, "grasp_dwell")
                action = self.cursor.copy()
                action[13] = runtime.close_target
                self._emit(action, "grasp_close")
                close_dwell = (math.ceil(.4 * HZ) if self.args.task == "sample_loading"
                               else CLOSE_FRAMES - 1)
                self._hold(close_dwell, "grasp_close")
                if self.args.task == "sample_loading":
                    # Preparation already validated the predicted pickup pose.
                    # Preserve this close through resumed rows and lifting.
                    self.grips[1] = GripIntent(True, float(runtime.close_target))
                closes = np.flatnonzero(
                    runtime.actions[index:, 13] <= h.dip.config.close_threshold
                )
                end = index + int(closes[0]) + 1 if len(closes) else len(runtime.actions)
                runtime.actions[index:end, 13] = np.minimum(
                    runtime.actions[index:end, 13], runtime.close_target
                )
            runtime.plan = h.dip.plan(
                runtime.actions[index:], previous_action=self.cursor,
                physical_state=self.cursor,
            )
            runtime.plan_start = index
        offset = index - runtime.plan_start
        target = runtime.plan.actions[offset].copy()
        h.dip.commit(runtime.plan, offset)
        return target

    def _grips(self, target, model):
        closed, released = [False, False], [False, False]
        h = self.helpers
        for side, helper in enumerate((h.left, h.right)):
            if helper is None:
                continue
            cfg, intent, i = helper.config, self.grips[side], side * 7 + 6
            command = float(getattr(cfg, "command", getattr(cfg, "command_cap", 0.0)))
            threshold = getattr(cfg, "closed_threshold", getattr(cfg, "close_threshold", None))
            if self.args.task == "sample_loading" and side == 1 and h.dip is not None:
                # Do not latch before the dip's own closing transition; doing
                # so would make preparation skip an already closed gripper.
                threshold = min(threshold, h.dip.config.close_threshold)
            value = float(model[i])
            closing_allowed = True
            if side == 0 and self.args.task == "handle_basket" and not intent.closed:
                # Initial closed commands followed by opening are setup, not
                # release. Require an open -> close model transition first.
                if value >= cfg.open_threshold:
                    intent.saw_open = True
                    intent.peak_open = max(intent.peak_open, value)
                closing_allowed = (intent.saw_open
                                   and intent.peak_open - value >= cfg.close_delta)
            if not intent.closed and threshold is not None and value <= threshold:
                # A handover donor stays released after the scripted transfer.
                if (closing_allowed and not (side == 1 and self.handover_done)
                        and self._grip_pose_allowed(model, side)):
                    intent.closed, intent.baseline = True, value
                    intent.release_count = 0
                    closed[side] = True
            elif intent.closed:
                opening = value >= cfg.open_threshold and value - intent.baseline >= cfg.release_delta
                allowed = getattr(cfg, "model_release_enabled", True)
                if h.handover is not None and side == 1 and not self.handover_done:
                    allowed = False
                intent.release_count = intent.release_count + 1 if opening and allowed else 0
                frames = 1 if side == 0 and self.args.task == "handle_basket" else cfg.release_frames
                if intent.release_count >= frames:
                    intent.closed = False
                    released[side] = True
            if intent.closed:
                if self.args.task == "sample_loading" and side == 1:
                    command = min(command, intent.baseline)
                target[i] = min(float(target[i]), command)
        return closed, released

    def _grip_pose_allowed(self, model, side):
        """Gate grip latching using the dip's geometry and predicted pose."""
        if (self.args.task not in ("table_rearrangement", "manipulate_pipette")
                and not (self.args.task == "sample_loading" and side == 1)):
            return True
        dip = self.helpers.dip
        if dip is None or dip.config.target_z_mm[side] is None:
            return False
        limits = dip.config.z_ranges[side]
        joints = model[side * 7:side * 7 + 6]
        return (limits[0] <= dip._xyz(joints)[2] <= limits[1]
                and dip._points_down(dip._tilt(joints)))

    def _release_paths(self, model, released):
        h = self.helpers
        paths, holds = {}, {}
        for side, name in enumerate(("left", "right")):
            if not released[side]:
                continue
            if name in h.table_releases:
                helper = h.table_releases[name]
                paths[side], _, _ = helper._path(self.cursor)
                holds[side] = max(0, math.ceil(helper.config.hold_seconds * HZ) - 1)
        if not paths:
            return False
        origin = self.cursor.copy()
        duration = max(len(path) + holds[side] + 1 for side, path in paths.items())
        for frame in range(duration):
            action = origin.copy()
            for side, path in paths.items():
                start = side * 7
                action[start:start + 6] = path[min(frame, len(path) - 1)]
                if frame >= len(path) + holds[side]:
                    action[start + 6] = model[start + 6]
            if not self._emit(action, "release_path"):
                break
        return True

    def _handover_requested(self, model):
        h = self.helpers.handover
        if h is None or self.handover_done or not self.grips[1].closed:
            return False
        cfg = h.config
        # Evaluate predicted geometry, never wait for measured arrival.
        if isinstance(h, RelativeHandover):
            # The relative planner supplies precise alignment; use the broader
            # original region only to avoid triggering immediately at pickup.
            region = HandoverAssist.poses(h, model)["in_handover_region"]
        else:
            region = h.poses(model)["in_handover_region"]
        intent = model[13] >= cfg.right_open_threshold or model[6] <= cfg.left_closed_threshold
        return bool(region and intent)

    def _handover(self, remaining):
        h = self.helpers.handover
        cfg = h.config
        if isinstance(h, RelativeHandover):
            relative = h.relative_config
            # Fixed command dwell only; no observation or arrival polling.
            self._hold(relative.stable_frames, "handover_right_dwell")
            action = self.cursor.copy()
            action[6] = relative.left_opening_mm / 100
            self._emit(action, "handover_open_left")
            _, right = h.tips(self.cursor)
            path, _ = h.plan_approach(self.cursor, right)
            self._path(path, "handover_approach", side=0)
            opening = relative.right_opening_mm / 100
        else:
            opening = max(float(np.max(remaining[:, 13])), cfg.right_open_threshold)
        action = self.cursor.copy()
        action[6] = cfg.gripper_cap if cfg.gripper_cap is not None else cfg.left_close_action
        self._emit(action, "handover_close_left")
        self._hold(CLOSE_FRAMES - 1, "handover_close_left")
        start = float(self.cursor[13])
        count = max(1, math.ceil(max(0.0, opening - start) / cfg.contact_release_step))
        for value in np.linspace(start, opening, count + 1)[1:]:
            action = self.cursor.copy()
            action[13] = value
            if not self._emit(action, "handover_open_right"):
                break
        self.handover_done = True
        h.state.phase = "completed"
        self.grips[0] = GripIntent(True, float(self.cursor[6]))
        self.grips[1] = GripIntent()
        if self.helpers.placement is not None:
            self._placement()

    def _placement(self):
        helper = self.helpers.placement
        helper.start(self.cursor, self.cursor)
        if helper.phase == "planning_failed":
            raise ValueError(f"Placement planning failed: {helper.reason}")
        self.cursor = helper.hold.copy()
        for kind, path in zip(("placement_approach", "placement_descend"), helper.paths):
            self._path(path, kind, side=0)
        self._hold(math.ceil(helper.config.release_delay_s * HZ), "placement_dwell")
        action = self.cursor.copy()
        action[6] = helper.config.opening_mm / 100
        self._emit(action, "placement_release")
        self._hold(9, "placement_release")
        helper.phase = "done"
        self.terminal = self.cursor.copy()

    def _assembly_lift(self, closed):
        helper = self.helpers.assembly_pickup
        paths = {}
        for side in (0, 1):
            if not closed[side] or self.lifted[side]:
                continue
            arm = helper.arms[side]
            arm["hold"] = self.cursor[side * 7:side * 7 + 6].copy()
            helper._plan(arm, self.cycle)
            if arm["reason"]:
                raise ValueError(f"Assembly lift planning failed: {arm['reason']}")
            paths[side] = arm["path"]
        if not paths:
            return
        self._hold(CLOSE_FRAMES - 1, "assembly_close")
        for frame in range(max(map(len, paths.values()))):
            action = self.cursor.copy()
            for side, path in paths.items():
                action[side * 7:side * 7 + 6] = path[min(frame, len(path) - 1)]
            if not self._emit(action, "assembly_lift"):
                break
        for side in paths:
            self.lifted[side] = True
            self.lift_targets[side] = self.cursor[side * 7:side * 7 + 6].copy()
            helper.arms[side]["phase"] = "raised"

    def _pipette_ready(self, raw):
        helper = self.helpers.pipette
        if helper.phase != "idle" or len(raw) < 2:
            return False
        poses = np.asarray([helper.fk.pose(q) for q in np.unique(raw[:, 7:13], axis=0)])
        tips = np.asarray([fingertip_xyz(p) for p in poses])
        rotations = Rotation.from_euler("xyz", poses[:, 3:], degrees=True)
        distance = np.linalg.norm(tips[:, None] - tips[None, :], axis=-1).max()
        angle = max(np.rad2deg((r.inv() * rotations).magnitude()).max() for r in rotations)
        return distance <= helper.config.stable_mm and angle <= helper.config.stable_deg

    def _pour_pickups(self, target, model):
        helper = self.helpers.pour
        cfg = helper.config
        for side, start in enumerate((0, 7)):
            if model[start + 6] >= cfg.open_threshold:
                helper.pickup[side] = None
            elif model[start + 6] <= cfg.close_threshold and helper.pickup[side] is None:
                helper.pickup[side] = target[start:start + 6].copy()

    def _pour_requested(self, remaining):
        h = self.helpers.pour
        cfg = h.config
        if h.active or any(q is None for q in h.pickup):
            return False
        if np.any(remaining[0, [6, 13]] >= cfg.open_threshold):
            return False
        geometry = h.geometry(self.cursor)
        ranges = (cfg.align_distance_mm, cfg.align_horizontal_mm) if cfg.relative_align else (cfg.distance_mm, cfg.horizontal_mm)
        if not all(lo <= geometry[key] <= hi for key, (lo, hi) in zip(
            ("distance_mm", "horizontal_mm", "height_mm"), (*ranges, cfg.height_mm)
        )) or geometry["angle_deg"] > cfg.angle_deg:
            return False
        if cfg.relative_align:
            return True
        q = np.rad2deg(remaining[:, 12])
        return len(q) > 1 and q[-1] - q[0] >= cfg.trend_deg

    def _pour(self, remaining):
        h = self.helpers.pour
        cfg = h.config
        h.latch(self.cursor, self.cursor)
        if cfg.relative_align:
            h.align_left = self.cursor[:6].copy()
            h.phase = "align_pending"
            h.plan_alignment(self.cursor)
            if h.phase != "align":
                raise ValueError(f"Pour alignment failed: {h.reason}")
            self._path(h.paths["align"], "pour_align")
            h.align_j6 = float(self.cursor[12])
            if cfg.align_only:
                self.terminal = self.cursor.copy()
                return
        desired = (float(h.full_joints[5]) if cfg.relative_align and cfg.full_pose else
                   float(np.clip(np.max(remaining[:, 12]) + np.deg2rad(cfg.extra_deg),
                                 PIPER_JOINT_MIN[5], PIPER_JOINT_MAX[5])))
        start = float(self.cursor[12])
        count = max(1, math.ceil(abs(desired - start) / np.deg2rad(cfg.tilt_speed_deg_s) * HZ))
        for q in np.linspace(start, desired, count + 1)[1:]:
            action = self.cursor.copy()
            action[12] = q
            if not self._emit(action, "pour_tilt"):
                break
        self._hold(math.ceil(cfg.hold_seconds * HZ), "pour_hold")
        h.phase = "return_pending"
        h.plan_return(self.cursor)
        if h.phase == "planning_failed":
            raise ValueError(f"Pour return failed: {h.reason}")
        phases = ("joint_return",) if "joint_return" in h.paths else ("untilt", "approach", "descend")
        for phase in phases:
            self._path(h.paths[phase], "pour_" + phase)
        self._hold(math.ceil(cfg.release_delay_s * HZ), "pour_release_dwell")
        action = self.cursor.copy()
        action[[6, 13]] = np.asarray([cfg.left_opening_mm, cfg.right_opening_mm]) / 100
        self._emit(action, "pour_release")
        h.phase = "done"
        self.terminal = self.cursor.copy()


class ChunkRuntime:
    """Execution cursor plus a private speculative planner for the next block."""

    def __init__(self, args):
        self.planner = ChunkPlanner(args)
        self._planned = None
        self.steps = 0
        self.cycle = -1
        self.row = 0
        self.control = np.empty((0, 14), dtype=np.float32)
        self.records = []
        self.pending = self.previous = None
        self.last_record = {}

    @property
    def needs_chunk(self):
        return self.row >= len(self.control)

    def clock(self):
        return self.steps / HZ

    def model_state(self, state):
        return self.planner.model_state(state)

    def begin_chunk(self, raw, state):
        if self.pending is not None or not self.needs_chunk:
            raise RuntimeError("Execute and acknowledge the current block before planning another")
        if self.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        planner = deepcopy(self.planner)
        try:
            actions, records = planner.plan(np.array(raw, copy=True), state,
                                            budget=1000 - self.steps)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if self.planner.args.task not in MODEL_FALLBACK_TASKS:
                if isinstance(exc, RuntimeError):
                    raise ValueError(f"Chunk planning failed: {exc}") from exc
                raise
            # Nothing from the failed speculative plan has been executed.
            # Retain only committed helper state and execute the current model
            # prefix. Invalid model commands still raise rather than being sent.
            planner = deepcopy(self.planner)
            _, actions = planner.model_actions(raw, state)
            actions = actions[:1000 - self.steps].copy()
            records = [{"kind": "model_fallback", "source_row": i,
                        "assist_error": str(exc)} for i in range(len(actions))]
            planner.cycle += 1
            LOGGER.warning("%s assist planning failed: %s; executing %d model frames",
                           planner.args.task, exc, len(actions))
        self._planned = planner
        self.control, self.records, self.row = actions, records, 0
        self.cycle += 1

    def propose(self, state=None, *, feedback=None):
        # Deliberately don't inspect state/feedback, including on cached frames.
        if self.pending is not None:
            raise RuntimeError("Call commit_action before requesting another action")
        if self.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        if self.needs_chunk:
            return None
        action = self.control[self.row].copy()
        self.pending = {"target": action.copy(), "record": self.records[self.row]}
        return action

    def commit(self):
        if self.pending is None:
            raise RuntimeError("No proposed action to acknowledge")
        self.previous = self.pending["target"].copy()
        self.last_record = dict(self.pending["record"], frame=self.steps, cycle=self.cycle)
        self.steps += 1
        if not self.pending.get("recovery_hold"):
            self.row += 1
            if self.needs_chunk:
                self.planner = self._planned
                self._planned = None
        self.pending = None
