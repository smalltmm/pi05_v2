"""Relative left approach and one to three offset base-Z pipette presses."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .config import JOINT_INDICES, PIPER_JOINT_MAX, PIPER_JOINT_MIN
from .handover import transform
from .kinematics import FINGERTIP_OFFSET_MM, fingertip_xyz, solve_ik
from .relative_handover import checked_transform, pose_error


DEFAULT_REFERENCE = Path(__file__).parent / "calibration/pipette_press_20261107_063626_utc.json"
# Leave slow/unreachable auxiliary plans to the next fresh model block.
MAX_PLANNING_SECONDS = 2.0


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Pipette planning time budget exceeded")


@dataclass(frozen=True)
class PipettePressConfig:
    reference: Path = DEFAULT_REFERENCE
    ready_only: bool = False
    offset_xyz_mm: tuple = (0., 0., 0.)
    offset_rpy_deg: tuple = (0., 0., 0.)
    depth_mm: float = 5.
    step_mm: float = 1.
    count: int = 1
    retry_offset_xyz_mm: tuple = (0., 0., 3.)
    pattern: str = 'line'
    spread_mm: float = 6.
    approach_clearance_mm: float = 40.
    stable_frames: int = 5
    stable_mm: float = 3.
    stable_deg: float = 3.
    position_tolerance_mm: float = 1.
    arrival_position_tolerance_mm: float = 2.
    orientation_tolerance_deg: float = 2.
    move_speed_mm_s: float | None = None
    joint_speed_deg_s: float | None = None
    approach_speed_mm_s: float | None = None
    approach_joint_speed_deg_s: float | None = None

    def __post_init__(self):
        if not isinstance(self.count, int) or not 1 <= self.count <= 3:
            raise ValueError("pipette press count must be 1..3")
        if self.pattern not in ('line', 'triangle'):
            raise ValueError('pipette press pattern must be line or triangle')
        if not np.isfinite(self.spread_mm) or self.spread_mm <= 0:
            raise ValueError('pipette spread must be finite and positive')
        if self.pattern == 'triangle' and self.count != 3:
            raise ValueError('triangle pressing requires count=3')
        if np.asarray(self.retry_offset_xyz_mm).shape != (3,) or not np.isfinite(self.retry_offset_xyz_mm).all():
            raise ValueError("pipette retry offset must contain three finite values")
        for value in (self.move_speed_mm_s, self.joint_speed_deg_s,
                      self.approach_speed_mm_s, self.approach_joint_speed_deg_s):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("pipette press speeds must be finite and positive")
        positive = (self.step_mm, self.approach_clearance_mm, self.stable_mm, self.stable_deg,
                    self.position_tolerance_mm, self.arrival_position_tolerance_mm,
                    self.orientation_tolerance_deg)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("pipette press steps and tolerances must be finite and positive")
        if not np.isfinite(self.depth_mm) or self.depth_mm < 0 or self.step_mm > 5:
            raise ValueError("pipette press requires finite depth >= 0 mm and step <= 5 mm")
        if self.arrival_position_tolerance_mm < self.position_tolerance_mm:
            raise ValueError("pipette arrival tolerance must be at least the IK position tolerance")
        if self.stable_frames < 2:
            raise ValueError("pipette press needs >=2 stable frames")
        for v in (self.offset_xyz_mm, self.offset_rpy_deg):
            if np.asarray(v).shape != (3,) or not np.isfinite(v).all():
                raise ValueError("pipette press offsets must contain three finite values")


class PipettePress:
    def __init__(self, config, fk, control_hz=10.):
        self.config, self.fk = config, fk
        self.hz = control_hz
        self.timed_motion = any(value is not None for value in (
            config.move_speed_mm_s, config.joint_speed_deg_s,
            config.approach_speed_mm_s, config.approach_joint_speed_deg_s))
        data = json.loads(config.reference.read_text())
        if data.get("task") != "manipulate_pipette":
            raise ValueError("pipette press requires a manipulate_pipette reference")
        if not np.allclose(data["fingertip_offset_mm"], FINGERTIP_OFFSET_MM):
            raise ValueError("pipette press reference tool length mismatch")
        self.base = transform([*data["right_base_xyz_mm"], *data["right_base_rpy_deg"]])
        self.relative = checked_transform(data["left_tip_in_right_tip"]).copy()
        self.relative[:3, 3] += config.offset_xyz_mm
        # Both offsets are expressed in the right fingertip frame.
        self.relative[:3, :3] = (Rotation.from_euler("xyz", config.offset_rpy_deg, degrees=True).as_matrix()
                                  @ self.relative[:3, :3])
        self.left_grip = float(data["left_gripper_command"])
        self.reference_left_q = np.asarray(data["state_14"][:6], dtype=np.float32)
        if self.reference_left_q.shape != (6,) or not np.isfinite(self.reference_left_q).all():
            raise ValueError("invalid pipette reference left joints")
        if not np.isfinite(self.left_grip) or not 0 <= self.left_grip <= 1:
            raise ValueError("invalid pipette reference left gripper command")
        self.phase = "idle"
        self.reason = None
        self.last_model_cycle = None
        self.trigger_metrics = {}
        self.retry_after_cycle = -1
        self.path, self.labels, self.tip_targets = [], [], []
        self.index = self.wait_frames = 0
        self.anchor = self.target = None
        self.approach_method = "cartesian"
        self.approach_fallback_reason = None

    @property
    def active(self):
        return self.phase in ("reconfigure", "approach", "down", "up", "retract", "ready_hold")

    def settings(self):
        return {**asdict(self.config), "reference": str(self.config.reference),
                "left_tip_in_right_tip": self.relative.tolist(), "left_gripper_command": self.left_grip,
                "right_gripper_command": 0., "vertical_frame": "left_base",
                "trigger_mode": "stable_model_full_chunk"}

    def tips(self, physical):
        result = []
        for start in (0, 7):
            pose = np.asarray(self.fk.pose(physical[start:start + 6]))
            tip = transform(pose)
            tip[:3, 3] = fingertip_xyz(pose)
            result.append(tip)
        return result[0], self.base @ result[1]

    def observe(self, physical, *, right_holding, feedback, cycle, model_chunk=None):
        if self.phase != "idle" or cycle <= self.retry_after_cycle:
            return False
        if cycle == self.last_model_cycle:
            return False
        self.last_model_cycle = cycle
        self.trigger_metrics = {"model_stability_cycle": cycle}
        if not right_holding:
            self.reason = "waiting_for_right_grip_hold"
            return False
        if not all(feedback.get(s, {}).get("valid", False) for s in ("left", "right")):
            self.reason = "waiting_for_valid_feedback"
            return False
        model = np.asarray(model_chunk, dtype=float)
        if model.ndim != 2 or model.shape[1] != 14 or len(model) < 2 or not np.isfinite(model).all():
            self.reason = "waiting_for_model_chunk"
            return False
        # Inspect the entire raw model chunk, not just endpoints or resampled
        # control rows. Ignore left-arm motion and gripper predictions.
        poses = [self.fk.pose(q) for q in np.unique(model[:, 7:13], axis=0)]
        tips = np.asarray([fingertip_xyz(p) for p in poses])
        rotations = Rotation.from_euler("xyz", np.asarray(poses)[:, 3:], degrees=True)
        distance = float(np.max(np.linalg.norm(tips[:, None] - tips[None, :], axis=-1)))
        angle = max(float(np.rad2deg((r.inv() * rotations).magnitude()).max()) for r in rotations)
        self.trigger_metrics.update(model_chunk_rows=len(model), model_right_span_mm=distance,
                                    model_right_span_deg=angle)
        ready = distance <= self.config.stable_mm and angle <= self.config.stable_deg
        self.reason = "model_chunk_stable" if ready else "waiting_for_stable_model_chunk"
        return ready

    def defer(self, reason, cycle):
        self.phase, self.reason = "idle", reason
        # Finish this chunk and one complete fresh chunk before another attempt.
        self.retry_after_cycle = cycle + 1

    def _joint_approach(self, state, target_q, *, deadline=None):
        """Continuous wrist reconfiguration above the work, with sampled FK checks."""
        cfg = self.config
        first, right = self.tips(state)
        end = state.copy()
        end[:6] = target_q
        last, _ = self.tips(end)
        floor = min(first[2, 3], last[2, 3]) - cfg.position_tolerance_mm
        # Keep the two fingertips at least 50 mm apart during the joint-clearance
        # transition.  The previous endpoint-derived limit could be about 120 mm
        # and reject otherwise usable approach paths.
        min_distance = 50.
        delta = target_q - state[:6]
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / .03)))
        for _ in range(8):
            rows, tips = [], []
            previous = first
            ratio = 1.
            for alpha in np.linspace(0, 1, count + 1)[1:]:
                _check_deadline(deadline)
                row = state.copy()
                row[:6] = state[:6] + alpha * delta
                if np.any(row[:6] < PIPER_JOINT_MIN) or np.any(row[:6] > PIPER_JOINT_MAX):
                    raise ValueError("pipette reconfiguration exceeds joint limits")
                tip, _ = self.tips(row)
                if (tip[2, 3] < max(0., floor)
                        or np.linalg.norm(tip[:3, 3] - right[:3, 3]) < min_distance):
                    raise ValueError("pipette reconfiguration violates height/separation envelope")
                distance, angle = pose_error(previous, tip)
                ratio = max(ratio, distance / cfg.step_mm, angle / 2.)
                rows.append(row)
                tips.append(tip)
                previous = tip
            if ratio <= 1.00001:
                self.path.extend(rows)
                self.tip_targets.extend(tips)
                self.labels.extend(["reconfigure"] * len(rows))
                return last
            count = int(np.ceil(count * ratio * 1.05))
        raise ValueError("pipette reconfiguration could not meet per-step limits")

    def press_offsets(self, right):
        if self.config.pattern == 'line':
            lateral = right[:3, :3] @ np.asarray(self.config.retry_offset_xyz_mm)
            lateral[2] = 0.
            return [(0., 1., -1.)[i] * lateral for i in range(self.config.count)]
        # Build an orthonormal horizontal basis from the right gripper, so the
        # radius and equilateral spacing remain exact even with a tilted wrist.
        axis = right[:3, 2].copy()
        axis[2] = 0.
        if np.linalg.norm(axis) < 1e-6:
            axis = right[:3, 1].copy()
            axis[2] = 0.
        axis /= np.linalg.norm(axis)
        across = np.cross([0., 0., 1.], axis)
        return [self.config.spread_mm * (np.cos(a)*axis + np.sin(a)*across)
                for a in np.deg2rad([180., 60., -60.])]


    def plan(self, physical, *, cycle):
        cfg = self.config
        deadline = time.perf_counter() + MAX_PLANNING_SECONDS
        self.path, self.labels, self.tip_targets = [], [], []
        self.reason = None
        self.approach_method = "cartesian"
        self.approach_fallback_reason = None
        state = np.asarray(physical, dtype=np.float32).copy()
        try:
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("pipette press needs finite physical state")
            left, right = self.tips(state)
            self.anchor = right.copy()
            self.target = right @ self.relative
            if np.linalg.norm(self.target[:3, 3] - left[:3, 3]) > 400:
                raise ValueError("pipette relative target is more than 400 mm from left tip")
            down = self.target.copy()
            down[2, 3] -= 0 if cfg.ready_only else cfg.depth_mm
            if min(self.target[2, 3], down[2, 3]) < 0:
                raise ValueError("pipette target tip is below base Z=0")
            state[6], state[13] = self.left_grip, 0.
            q = state[:6].copy()

            def segment(start, end, label):
                nonlocal q
                distance, angle = pose_error(start, end)
                n = max(1, int(np.ceil(distance / cfg.step_mm)), int(np.ceil(angle / 2)))
                r0 = Rotation.from_matrix(start[:3, :3])
                turn = (r0.inv() * Rotation.from_matrix(end[:3, :3])).as_rotvec()

                def waypoint(goal, depth=0):
                    nonlocal q
                    new, _, _ = solve_ik(self.fk, q, goal[:3, 3],
                                         Rotation.from_matrix(goal[:3, :3]).as_euler("xyz", degrees=True),
                                         position_tolerance_mm=cfg.position_tolerance_mm,
                                         orientation_tolerance_deg=cfg.orientation_tolerance_deg,
                                         prefer_local_solution=True,
                                         local_only=True, deadline=deadline)
                    if not np.isfinite(new).all() or np.any(new < PIPER_JOINT_MIN) or np.any(new > PIPER_JOINT_MAX):
                        raise ValueError("pipette IK joint limits exceeded")
                    if np.max(np.abs(new - q)) > .03:
                        if np.max(np.abs(new - q)) > .5:
                            raise ValueError("pipette IK branch discontinuity")
                        if depth >= 5:
                            raise ValueError("pipette IK branch discontinuity")
                        old = state.copy()
                        old[:6] = q
                        actual, _ = self.tips(old)
                        midpoint = goal.copy()
                        midpoint[:3, 3] = (actual[:3, 3] + goal[:3, 3]) / 2
                        ra = Rotation.from_matrix(actual[:3, :3])
                        midpoint[:3, :3] = (ra * Rotation.from_rotvec(
                            .5 * (ra.inv() * Rotation.from_matrix(goal[:3, :3])).as_rotvec())).as_matrix()
                        waypoint(midpoint, depth + 1)
                        waypoint(goal, depth + 1)
                        return
                    row = state.copy()
                    row[:6] = q = new
                    self.path.append(row)
                    self.labels.append(label)
                    self.tip_targets.append(goal.copy())

                for alpha in np.linspace(0, 1, n + 1)[1:]:
                    goal = start.copy()
                    goal[:3, 3] = start[:3, 3] + alpha * (end[:3, 3] - start[:3, 3])
                    goal[:3, :3] = (r0 * Rotation.from_rotvec(alpha * turn)).as_matrix()
                    waypoint(goal)

            try:
                segment(left, self.target, "approach")
            except (ValueError, RuntimeError) as exc:
                # Do not execute any prefix of a failed Cartesian path. Select
                # the demonstrated wrist branch at the final goal, then move
                # continuously to a point above it before Cartesian approach.
                self.approach_fallback_reason = str(exc)
                self.approach_method = "joint_clearance"
                self.path, self.labels, self.tip_targets = [], [], []
                rpy = Rotation.from_matrix(self.target[:3, :3]).as_euler("xyz", degrees=True)
                kwargs = dict(position_tolerance_mm=cfg.position_tolerance_mm,
                              orientation_tolerance_deg=cfg.orientation_tolerance_deg,
                              prefer_local_solution=True, deadline=deadline)
                final_q, _, _ = solve_ik(self.fk, self.reference_left_q, self.target[:3, 3], rpy, **kwargs)
                pre = self.target.copy()
                pre[2, 3] += cfg.approach_clearance_mm
                pre_q, _, _ = solve_ik(self.fk, final_q, pre[:3, 3], rpy, **kwargs)
                actual_pre = self._joint_approach(state, pre_q, deadline=deadline)
                q = pre_q.copy()
                segment(actual_pre, self.target, "approach")
            if not cfg.ready_only:
                # Offset in right-tip coordinates, projected onto the horizontal
                # plane of the left base. Always retract before shifting sideways.
                ready = self.target.copy()
                for attempt, offset in enumerate(self.press_offsets(right)):
                    next_ready = self.target.copy()
                    next_ready[:3, 3] += offset
                    if attempt or np.linalg.norm(offset) > 0:
                        segment(ready, next_ready, "approach")
                    down = next_ready.copy()
                    down[2, 3] -= cfg.depth_mm
                    segment(next_ready, down, "down")
                    segment(down, next_ready, "up")
                    ready = next_ready
                if cfg.count > 1:
                    segment(ready, self.target, "approach")
            if self.timed_motion:
                self._time_path(state, deadline=deadline)
            _check_deadline(deadline)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self.path, self.labels, self.tip_targets = [], [], []
            self.defer(str(exc), cycle)
            return False
        self.index = self.wait_frames = 0
        self.phase = self.labels[0]
        return True

    def _time_path(self, start, *, deadline=None):
        """Sample each phase at control_hz, retaining its final arrival target."""
        path, labels, targets = [], [], []
        begin = 0
        while begin < len(self.path):
            _check_deadline(deadline)
            end = begin + 1
            label = self.labels[begin]
            speed = self.config.move_speed_mm_s or 40.
            joint_speed = self.config.joint_speed_deg_s or 20.
            if label in ('approach', 'reconfigure'):
                speed = self.config.approach_speed_mm_s or speed
                joint_speed = self.config.approach_joint_speed_deg_s or joint_speed
            joint_speed = np.deg2rad(joint_speed)
            while end < len(self.path) and self.labels[end] == label:
                end += 1
            rows = np.asarray([start, *self.path[begin:end]])
            tips = np.asarray([self.tips(row)[0][:3, 3] for row in rows])
            durations = np.maximum(np.linalg.norm(np.diff(tips, axis=0), axis=1) / speed,
                                   np.max(np.abs(np.diff(rows[:, :6], axis=0)), axis=1) / joint_speed)
            times = np.r_[0., np.cumsum(np.maximum(durations, 1e-9))]
            count = max(1, int(np.ceil(times[-1] * self.hz)))
            samples = np.linspace(0., times[-1], count + 1)[1:]
            commands = np.column_stack([np.interp(samples, times, rows[:, j]) for j in range(14)]).astype(np.float32)
            for row in commands:
                _check_deadline(deadline)
                path.append(row)
                labels.append(label)
                targets.append(self.tips(row)[0])
            targets[-1] = self.tip_targets[end - 1].copy()
            start = rows[-1]
            begin = end
        self.path, self.labels, self.tip_targets = path, labels, targets

    def next(self, physical):
        left, right = self.tips(physical)
        goal = self.tip_targets[self.index]
        error, angle = pose_error(left, goal)
        return self.path[self.index].copy(), {
            "phase": self.phase, "path_index": self.index, "path_length": len(self.path),
            "left_tip_in_left_base": left.tolist(), "right_tip_in_left_base": right.tolist(),
            "left_tip_in_right_tip": (np.linalg.inv(right) @ left).tolist(),
            "ready_target_in_left_base": self.target.tolist(), "waypoint_in_left_base": goal.tolist(),
            "position_error_mm": error, "orientation_error_deg": angle,
            "depth_mm": 0. if self.config.ready_only else self.config.depth_mm,
            "press_count": self.config.count,
            "press_pattern": self.config.pattern,
            "press_offsets_in_left_base_mm": [v.tolist() for v in self.press_offsets(self.anchor)],
            "press_attempt": sum(label == 'down' and (i == 0 or self.labels[i-1] != 'down')
                                 for i, label in enumerate(self.labels[:self.index+1])),
            "ready_only": self.config.ready_only, "reason": self.reason,
            "wait_frames": self.wait_frames,
            "approach_method": self.approach_method,
            "approach_fallback_reason": self.approach_fallback_reason,
            **self.trigger_metrics,
        }

    def commit(self, physical, sent, *, cycle):
        if self.phase == "ready_hold":
            return
        if self.timed_motion:
            # Called after a successful send (including final Z-floor changes).
            # Advance at endpoints too, without feedback arrival/drift gates.
            self.index += 1
            self.wait_frames = 0
            self.phase = self.labels[self.index] if self.index < len(self.path) else "completed"
            if self.phase == "completed" and self.config.ready_only:
                self.index = len(self.path) - 1
                self.phase = "ready_hold"
            return
        left, right = self.tips(physical)
        cfg = self.config
        drift, turn = pose_error(right, self.anchor)
        if drift > max(10., cfg.stable_mm * 2) or turn > max(5., cfg.stable_deg * 2):
            self.defer("right_pose_changed", cycle)
            return
        error, angle = pose_error(left, self.tip_targets[self.index])
        commanded_tip, _ = self.tips(self.path[self.index])
        tracking_error, _ = pose_error(left, commanded_tip)
        arrived = (error <= cfg.arrival_position_tolerance_mm and angle <= cfg.orientation_tolerance_deg
                   and tracking_error <= min(cfg.arrival_position_tolerance_mm, max(1., cfg.step_mm / 2))
                   and (self.phase != "reconfigure"
                        or np.max(np.abs(np.asarray(physical)[:6] - self.path[self.index][:6])) <= .02)
                   and np.max(np.abs(np.asarray(physical)[7:13] - self.path[self.index][7:13])) <= .02
                   and np.allclose(np.asarray(sent)[list(JOINT_INDICES)],
                                   self.path[self.index][list(JOINT_INDICES)], atol=1e-5))
        if arrived:
            self.index += 1
            self.wait_frames = 0
            self.phase = self.labels[self.index] if self.index < len(self.path) else "completed"
            if self.phase == "completed" and cfg.ready_only:
                self.index = len(self.path) - 1
                self.phase = "ready_hold"
        else:
            self.wait_frames += 1
            # Keep sending the current waypoint until fresh feedback satisfies
            # the arrival test. Ctrl+C and max-control-steps are the runtime
            # stop paths; waiting does not fall back to model control.


def add_arguments(parser):
    parser.add_argument("--pipette-press", action=argparse.BooleanOptionalAction, default=False,
                        help="manipulate_pipette: relative left approach, vertical press, return; once per run")
    for name, value in asdict(PipettePressConfig()).items():
        if name == 'pattern':
            parser.add_argument('--pipette-press-pattern', choices=('line','triangle'), default=value,
                                help='line: center/+offset/-offset; triangle: three vertices around center')
            continue
        if name == 'spread_mm':
            parser.add_argument('--pipette-press-spread-mm', type=float, default=value,
                                help='triangle radius in the horizontal left-base plane; side = sqrt(3)*radius')
            continue
        if name == 'count':
            parser.add_argument('--pipette-press-count', type=int, default=value,
                                help='number of down/up strokes, 1..3; triangle requires 3')
            continue
        if name == "stable_frames":
            parser.add_argument("--pipette-press-stable-frames", type=int, default=value,
                                help="legacy option; model-chunk stability no longer counts feedback frames")
            continue
        if name == "ready_only":
            parser.add_argument("--pipette-press-ready-only", action=argparse.BooleanOptionalAction,
                                default=value, help="reach and hold press-ready pose without pressing or resuming model motion")
            continue
        kwargs = {"nargs": 3, "type": float} if name.startswith("offset_") or name == "retry_offset_xyz_mm" else {
            "type": Path if name == "reference" else int if name.endswith("frames") or name == "count" else float}
        parser.add_argument("--pipette-press-" + name.replace("_", "-"), default=value, **kwargs)


def config_from_args(args):
    if not args.pipette_press:
        return None
    if args.task != "manipulate_pipette" or not args.right_grip_hold:
        raise ValueError("--pipette-press requires --task manipulate_pipette --right-grip-hold")
    if args.bell_press:
        raise ValueError("--pipette-press cannot be combined with bell press")
    return PipettePressConfig(**{name: getattr(args, "pipette_press_" + name)
                                for name in asdict(PipettePressConfig())})
