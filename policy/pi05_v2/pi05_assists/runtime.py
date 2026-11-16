"""Feedback-driven action postprocessing. No robot, camera or transport access.

begin_chunk -> propose(state) -> official executor -> commit(). A successful
commit acknowledges one submitted control frame; measured state always comes
from the next observation, never from a commanded target.
"""

from __future__ import annotations

from copy import deepcopy
from .assembly_pickup import AssemblyPickupLift
from .handover_observation import HandoverObservation
import numpy as np

from .options import (
    config_from_args,
    assembly_config_from_args,
    bell_config_from_args,
    handover_config_from_args,
    relative_config_from_args,
    grip_hold_config_from_args,
    left_grip_hold_config_from_args,
    basket_dip_config_from_args,
    basket_release_config_from_args,
    table_release_config_from_args,
    placement_config_from_args,
    assembly_insert_config_from_args,
    pipette_press_config_from_args,
    pour_config_from_args,
    DualBellPress,
    DualBellPressRuntime,
    RelativeHandover,
    HandoverAssist,
    AssemblyGripConfig,
    BasketLeftGripConfig,
    AssemblyRightGripHold,
    BasketRightGripHold,
    RightGripHold,
    AssemblyLeftGripHold,
    BasketLeftGripHold,
    TableLeftGripHold,
    LeftGripHold,
    BasketDipHold,
    BasketReleaseAdvance,
    TableReleaseLower,
    GraspDip,
    GraspDipRuntime,
    DipPlan,
    ModelZFloor,
    SamplePlacement,
    ItemAssemblyInsert,
    PipettePress,
    WaterPouring,
    ActionSchedule,
    make_piper_fk,
)
from .config import (
    PIPER_JOINT_MIN,
    PIPER_JOINT_MAX,
    compensate_gripper_action,
    compensate_gripper_state,
)


def checked_state(state):
    """Copy measured state without validating or altering observation values."""
    return np.asarray(state, dtype=np.float32).copy()


def clip_actions(actions):
    values = np.asarray(actions, dtype=np.float32).copy()
    if (
        values.ndim != 2
        or values.shape[1] != 14
        or len(values) == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected nonempty finite actions [T,14]")
    for start in (0, 7):
        values[:, start : start + 6] = np.clip(
            values[:, start : start + 6], PIPER_JOINT_MIN, PIPER_JOINT_MAX
        )
    values[:, [6, 13]] = np.clip(values[:, [6, 13]], 0, 1)
    return values


class AssistRuntime:
    def __init__(self, args):
        if args.control_hz != 10:
            raise ValueError("Final-round execution is fixed at 10 Hz")
        self.args = args
        self.offsets = (args.left_gripper_offset, args.right_gripper_offset)
        self.schedule = ActionSchedule(
            mode=args.schedule,
            legacy_steps=args.legacy_steps,
            sample_start_step=args.sample_start_step,
            source_fps=args.model_action_fps,
            control_hz=10,
        )
        self.dip_config = config_from_args(args)
        assembly_config = assembly_config_from_args(args)
        bell_config = bell_config_from_args(args)
        self.bell = (
            DualBellPress(bell_config, make_piper_fk())
            if bell_config is not None
            else None
        )
        handover_config = handover_config_from_args(args)
        relative_config = relative_config_from_args(args)
        self.handover = (
            RelativeHandover(handover_config, make_piper_fk(), relative_config)
            if relative_config is not None
            else HandoverAssist(handover_config, make_piper_fk())
            if handover_config is not None
            else None
        )
        self.handover_observation = (
            HandoverObservation(
                args.handover_right_observe_reference,
                args.handover_right_observe_speed_deg_s,
                10,
                args.handover_right_observe_model_reference,
            )
            if args.handover_right_observe
            else None
        )
        self.grip_config = assembly_config or grip_hold_config_from_args(args)
        cfg = self.grip_config
        self.right = (
            (
                AssemblyRightGripHold(cfg)
                if isinstance(cfg, AssemblyGripConfig)
                else BasketRightGripHold(cfg)
                if isinstance(cfg, BasketLeftGripConfig)
                else RightGripHold(cfg, make_piper_fk())
            )
            if cfg is not None
            else None
        )
        cfg = assembly_config or left_grip_hold_config_from_args(args)
        self.left = (
            (
                AssemblyLeftGripHold(cfg)
                if isinstance(cfg, AssemblyGripConfig)
                else BasketLeftGripHold(cfg)
                if isinstance(cfg, BasketLeftGripConfig)
                else TableLeftGripHold(cfg, make_piper_fk())
                if args.task == "table_rearrangement"
                else LeftGripHold(cfg)
            )
            if cfg is not None
            else None
        )
        basket_config = basket_dip_config_from_args(args)
        if basket_config is not None:
            self.left = BasketDipHold(
                cfg, basket_config, make_piper_fk(), clock=self.clock
            )
        cfg = basket_release_config_from_args(args)
        self.basket_release = (
            BasketReleaseAdvance(cfg, make_piper_fk()) if cfg is not None else None
        )
        cfg = table_release_config_from_args(args)
        self.table_releases = (
            {side: TableReleaseLower(cfg, make_piper_fk(), side) for side in ("left", "right")}
            if cfg is not None else {}
        )
        self.dip = (
            GraspDip(self.dip_config, make_piper_fk())
            if self.dip_config is not None
            else None
        )
        self.floor = (
            ModelZFloor(make_piper_fk(), args.model_min_z_mm)
            if args.model_z_floor
            else None
        )
        cfg = placement_config_from_args(args)
        self.placement = (
            SamplePlacement(cfg, make_piper_fk(), clock=self.clock)
            if cfg is not None
            else None
        )
        cfg = assembly_insert_config_from_args(args)
        self.assembly = (
            ItemAssemblyInsert(
                cfg, make_piper_fk(), grip_command=assembly_config.command
            )
            if cfg is not None
            else None
        )
        self.assembly_pickup = (
            AssemblyPickupLift(cfg, self.assembly.fk, assembly_config, 10)
            if self.assembly is not None
            else None
        )
        cfg = pipette_press_config_from_args(args)
        self.pipette = PipettePress(cfg, make_piper_fk(), 10) if cfg is not None else None
        cfg = pour_config_from_args(args)
        self.pour = (
            WaterPouring(cfg, make_piper_fk(), 10, clock=self.clock)
            if cfg is not None
            else None
        )
        self.previous = None
        self.steps = 0
        self.cycle = -1
        self.row = 0
        self.control = np.empty((0, 14), dtype=np.float32)
        self.dip_runtime = self.bell_runtime = self.pending = None
        self.last_record = {}
        self.planning_check = None

    def clock(self):
        # Inference/planning latency must not count toward held-action durations.
        return self.steps / 10.0

    @property
    def needs_chunk(self):
        return self.row >= len(self.control)

    def model_state(self, physical):
        physical = checked_state(physical)
        if self.handover_observation is not None:
            physical = self.handover_observation.model_state(physical)
        return compensate_gripper_state(
            physical,
            self.offsets,
            enabled=self.args.gripper_offset_compensation,
        )

    def _reset_dip(self):
        if self.dip is not None:
            self.dip = GraspDip(
                self.dip_config, self.dip.fk, down_tilt=self.dip.down_tilt
            )
        self.dip_runtime = None

    def begin_chunk(self, raw, physical):
        if self.pending is not None:
            raise RuntimeError("Commit the outstanding action before starting a chunk")
        physical = checked_state(physical)
        # The latest pipette trigger checks the complete raw prediction,
        # including the unexecuted tail, before clipping or height correction.
        self.pipette_model_chunk = np.asarray(raw, dtype=np.float32).copy()
        raw = compensate_gripper_action(
            raw, self.offsets, enabled=self.args.gripper_offset_compensation
        )
        control = clip_actions(
            self.schedule.make_plan(raw, state=physical, action_semantics="absolute")
        )
        if self.floor is not None:
            self.floor.begin_chunk()
            control = self.floor.apply(control)
            raw = self.floor.apply(clip_actions(raw))
        self.model_control = control.copy()
        self.control = control
        self.cycle += 1
        self.row = 0
        self.dip_runtime = self.bell_runtime = None
        if self.pour is not None:
            self.pour.plan_pending(physical)
        if self.handover is not None:
            self.handover.begin_inference_result()
        exclusive = any(
            x is not None and x.active
            for x in (self.pour, self.placement, self.assembly, self.pipette)
        )
        exclusive |= (
            isinstance(self.handover, RelativeHandover) and self.handover.owns_motion
        )
        suspended = self.handover is not None and self.handover.state.phase in (
            "waiting",
            "ready",
            "releasing",
        )
        if self.dip is not None and not exclusive and not suspended:
            try:
                plan = self.dip.plan(
                    control,
                    previous_action=physical
                    if self.previous is None
                    else self.previous,
                    physical_state=physical,
                )
            except ValueError as exc:
                self._reset_dip()
                plan = DipPlan(
                    control.copy(),
                    [deepcopy(self.dip._states) for _ in control],
                    [
                        {
                            "side": side,
                            "status": "fallback",
                            "start_row": 0,
                            "reason": str(exc),
                        }
                        for side, z in zip(
                            ("left", "right"), self.dip_config.target_z_mm
                        )
                        if z is not None
                    ],
                )
            start = min(self.args.legacy_steps, len(raw))
            count = int(
                np.ceil(self.dip_config.pre_steps * self.schedule.source_fps / 10)
            )
            tail = (
                raw[start : start + count]
                if self.args.schedule != "resample"
                else raw[:0]
            )
            self.dip_runtime = GraspDipRuntime(
                self.dip,
                control,
                plan,
                lookahead=tail,
                preclose_config=self.grip_config
                if self.args.task in ("sample_loading", "manipulate_pipette")
                else None,
                clock=self.clock,
                control_hz=10,
            )
            self.control = plan.actions
        if self.bell is not None:
            tail = (
                raw[min(self.args.legacy_steps, len(raw)) :]
                if self.args.schedule != "resample"
                else raw[:0]
            )
            self.bell_runtime = DualBellPressRuntime(
                self.bell, self.control, lookahead=tail
            )

    def propose(self, physical, *, feedback=None):
        if self.pending is not None:
            raise RuntimeError(
                "Previous action has not been acknowledged; call commit() or reset the episode"
            )
        if self.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        if self.needs_chunk:
            return None
        current = checked_state(physical)
        previous = current if self.previous is None else self.previous
        if self.planning_check is not None:
            kind, anchor = self.planning_check
            self.planning_check = None
            joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            if np.max(np.abs(current[joints] - anchor[joints])) > 0.03:
                if kind == "pipette":
                    self.pipette.defer("feedback_changed_during_planning", self.cycle)
                else:
                    self.assembly.phase = "idle"
                    self.assembly.reason = "feedback_changed_during_planning"
                self.row = len(self.control)
                return None
        # Each call represents a fresh official observation. Optional metadata can
        # supply real observation sequence stamps; no device status is fabricated.
        if feedback is None:
            feedback = {
                side: {
                    "valid": True,
                    "stamp": self.steps,
                    "opening": float(current[index]),
                }
                for side, index in (("left", 6), ("right", 13))
            }
        remaining = self.model_control[self.row :]
        target = self.control[self.row].copy()
        dip_step = bell_step = handover_step = right_step = left_step = None
        if self.pour is not None:
            self.pour.observe(current, previous, remaining, (self.cycle, self.row))
            if self.pour.active:
                self.dip_runtime = None
        if self.pipette is not None and self.pipette.phase == "idle" and self.row == 0:
            if self.pipette.observe(
                current,
                right_holding=self.right.state.phase == "holding",
                feedback=feedback,
                cycle=self.cycle,
                model_chunk=self.pipette_model_chunk,
            ):
                if self.pipette.plan(current, cycle=self.cycle):
                    self._reset_dip()
                    # Hold one frame after potentially slow planning; next command
                    # is checked against a new observation from the official runner.
                    target = previous.copy()
                    self.pending = dict(
                        current=current,
                        target=clip_actions(target[None, :])[0],
                        planning_hold=True,
                        planning_kind="pipette",
                    )
                    return self.pending["target"].copy()
        if self.assembly is not None and not self.assembly.active:
            if self.assembly.observe(
                current,
                left_holding=self.left.state.phase == "holding",
                right_holding=self.right.state.phase == "holding",
                cycle=self.cycle,
                feedback=feedback,
                pickup_ready=self.assembly_pickup.ready,
            ):
                if self.assembly.plan(current, cycle=self.cycle):
                    self.pending = dict(
                        current=current,
                        target=clip_actions(previous[None, :])[0],
                        planning_hold=True,
                        planning_kind="assembly",
                    )
                    return self.pending["target"].copy()
        if (
            self.placement is not None
            and not self.placement.active
            and self.handover.state.phase == "completed"
        ):
            self.placement.start(current, previous)
            self.pending = dict(
                current=current,
                target=clip_actions(previous[None, :])[0],
                planning_hold=True,
            )
            return self.pending["target"].copy()
        placement_active = self.placement is not None and self.placement.active
        assembly_active = self.assembly is not None and self.assembly.active
        pipette_active = self.pipette is not None and self.pipette.active
        if self.handover is not None and not placement_active and not assembly_active:
            if (
                self.handover_observation is not None
                and self.handover_observation.phase != "idle"
            ):
                self.handover.state.phase = "completed"
                self.handover.keep_completed = True
            if isinstance(self.handover, RelativeHandover):
                self.handover.donor_holding = (
                    self.right is not None and self.right.state.phase == "holding"
                )
            handover_step = self.handover.next(
                current,
                previous,
                remaining
                if self.right is not None
                or self.handover.config.gripper_cap is not None
                else self.control[self.row :],
                feedback=feedback,
            )
        override = handover_step is not None and handover_step.action is not None
        if override:
            target = handover_step.action
        if (
            self.dip_runtime is not None
            and not override
            and not placement_active
            and not assembly_active
            and not pipette_active
        ):
            n = len(self.dip_runtime.events)
            dip_step = self.dip_runtime.next(
                current,
                previous,
                feedback=feedback["right"]
                if self.handover is not None or self.args.task == "manipulate_pipette"
                else None,
            )
            if dip_step is None:
                self.row = len(self.control)
                return None
            if self.args.task == "manipulate_pipette" and any(
                e["status"] in ("close_empty", "close_timeout")
                for e in self.dip_runtime.events[n:]
            ):
                self.right.state.phase = "released"
                self._reset_dip()
                self.row = len(self.control)
                return None
            target = dip_step.action
        if self.bell_runtime is not None:
            bell_step = self.bell_runtime.next(current, previous)
            if bell_step is None:
                self.row = len(self.control)
                return None
            target = bell_step.action
        model_chunk = (
            handover_step is not None
            and handover_step.kind == "relative_handover_model_chunk"
        )
        if (
            self.right is not None
            and not placement_active
            and not assembly_active
            and not pipette_active
            and not model_chunk
        ):
            kwargs = (
                {"allow_close": dip_step is None or dip_step.kind == "model"}
                if isinstance(self.right, AssemblyRightGripHold)
                else {}
            )
            if self.assembly_pickup is not None and self.assembly_pickup.owns(1):
                kwargs["allow_close"] = False
            if (
                isinstance(self.right, RightGripHold)
                and self.dip_config is not None
                and self.dip_config.target_z_mm[1] is not None
            ):
                preparing = dip_step is not None and dip_step.kind in (
                    "grasp_dip_prepare", "grasp_dip_wait", "grasp_dip_dwell"
                )
                kwargs["allow_close"] = not preparing and (
                    (dip_step is not None and dip_step.kind == "grasp_dip_close")
                    or self.dip._states[1].closed
                    or float(remaining[0, 13]) <= self.dip_config.close_threshold
                )
            right_step = self.right.next(
                current,
                previous,
                float(remaining[0, 13]),
                token=(self.cycle, self.row),
                handover=None if handover_step is None else handover_step.record,
                **kwargs,
            )
            if "right" in self.table_releases:
                right_step = self.table_releases["right"].next(
                    right_step, self.right.state, current, previous, cycle=self.cycle
                )
            if getattr(right_step, "arm_action", None) is not None:
                target = target.copy()
                target[7:14] = right_step.arm_action
            if (
                right_step.force_close
                and not (isinstance(self.handover, RelativeHandover) and override)
                and not (
                    dip_step is not None
                    and (
                        dip_step.kind == "grasp_dip_dwell"
                        or (
                            self.args.task == "manipulate_pipette"
                            and dip_step.kind in ("grasp_dip_prepare", "grasp_dip_wait")
                        )
                    )
                )
            ):
                target = target.copy()
                target[13] = (
                    min(float(target[13]), right_step.record["hold_command"])
                    if dip_step is not None and dip_step.kind == "grasp_dip_close"
                    else right_step.record["hold_command"]
                )
        if (
            self.left is not None
            and not placement_active
            and not assembly_active
            and not model_chunk
        ):
            kwargs = (
                {"previous": previous, "feedback": feedback["left"]}
                if isinstance(self.left, BasketDipHold)
                else {}
            )
            if isinstance(self.left, AssemblyLeftGripHold):
                kwargs["allow_close"] = dip_step is None or dip_step.kind == "model"
            if self.assembly_pickup is not None and self.assembly_pickup.owns(0):
                kwargs["allow_close"] = False
            left_step = self.left.next(
                current,
                float(remaining[0, 6]),
                token=(self.cycle, self.row),
                handover=None if handover_step is None else handover_step.record,
                **kwargs,
            )
            if self.basket_release is not None:
                left_step = self.basket_release.next(
                    left_step, self.left.state, current, previous, cycle=self.cycle
                )
            if "left" in self.table_releases:
                left_step = self.table_releases["left"].next(
                    left_step, self.left.state, current, previous, cycle=self.cycle
                )
            if left_step.force_close and not (
                isinstance(self.handover, RelativeHandover) and override
            ):
                target = target.copy()
                target[6] = left_step.record["hold_command"]
            if getattr(left_step, "arm_action", None) is not None:
                target = target.copy()
                target[:7] = left_step.arm_action
        record = {}
        if self.assembly_pickup is not None and not assembly_active:
            holding = (
                left_step is not None and left_step.force_close,
                right_step is not None and right_step.force_close,
            )
            target = self.assembly_pickup.next(
                current, target, holding, feedback, self.cycle
            )
            record["assembly_pickup"] = self.assembly_pickup.record()
        for name, helper, active in (
            ("placement", self.placement, placement_active),
            ("assembly", self.assembly, assembly_active),
            ("pipette", self.pipette, pipette_active),
        ):
            if active:
                target, record[name] = helper.next(current)
                self.dip_runtime = None
        if self.pour is not None:
            pour_target, record["pour"] = self.pour.next(current, remaining)
            if pour_target is not None:
                target = pour_target
            target = self.pour.apply_right_grip_cap(target, float(remaining[0, 13]))
            record["pour"] = self.pour.record()
        if self.handover_observation is not None:
            target = self.handover_observation.apply(
                current, target, completed=self.handover.state.phase == "completed"
            )
            record["handover_observation"] = self.handover_observation.record()
        target = clip_actions(target[None, :])[0]
        if self.floor is not None:
            target = self.floor.apply(target[None, :])[0]
        if not np.isfinite(target).all():
            raise ValueError("Assistance produced a nonfinite action")
        self.pending = dict(
            current=current,
            target=target.copy(),
            dip=dip_step,
            bell=bell_step,
            handover=handover_step,
            right=right_step,
            left=left_step,
            override=override,
            placement=placement_active,
            assembly=assembly_active,
            pipette=pipette_active,
            record=record,
        )
        return target.copy()

    def commit(self):
        if self.pending is None:
            raise RuntimeError("No proposed action to acknowledge")
        p = self.pending
        self.previous = p["target"].copy()
        if p.get("recovery_hold"):
            self.last_record = dict(p["record"], frame=self.steps, cycle=self.cycle)
            self.steps += 1
            self.pending = None
            return
        if p.get("planning_hold"):
            if p.get("planning_kind"):
                self.planning_check = (p["planning_kind"], p["current"].copy())
            self.steps += 1
            self.pending = None
            return
        current, target = p["current"], p["target"]
        if self.handover_observation is not None:
            self.handover_observation.commit(target)
        if self.assembly_pickup is not None and not p["assembly"]:
            self.assembly_pickup.commit(current, target)
        if self.pour is not None:
            self.pour.commit(current, target)
        if p["pipette"]:
            self.pipette.commit(current, target, cycle=self.cycle)
        if self.dip_runtime is not None and p["dip"] is not None:
            self.dip_runtime.commit(p["dip"])
        if p["assembly"]:
            self.assembly.commit(current, target)
        for name in ("handover", "right", "left"):
            if p[name] is not None:
                getattr(self, name).commit(p[name])
                if name in self.table_releases:
                    self.table_releases[name].commit(p[name])
                if name == "left" and self.basket_release is not None:
                    self.basket_release.commit(p[name])
        if self.bell_runtime is not None and p["bell"] is not None:
            self.bell_runtime.commit(p["bell"])
        if p["placement"]:
            self.placement.commit(current, target)
        if p["override"]:
            self._reset_dip()
        self.row = (
            self.bell_runtime.row
            if self.bell_runtime is not None
            else self.dip_runtime.row
            if self.dip_runtime is not None
            else self.row + 1
        )
        if (p["pipette"] and not self.pipette.active) or (
            isinstance(self.handover, RelativeHandover)
            and self.handover.state.phase == "model_chunk_pending"
        ):
            self.row = len(self.control)
        if self.assembly is not None and self.assembly.phase == "failed":
            self.assembly.last_attempt_cycle = self.cycle + 1
            self._reset_dip()
            self.row = len(self.control)
        self.last_record = dict(p["record"], frame=self.steps, cycle=self.cycle)
        self.steps += 1
        self.pending = None
