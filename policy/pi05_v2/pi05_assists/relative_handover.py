"""Demonstrated fingertip-relative handover with feedback-gated arm motion."""

from __future__ import annotations

import copy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .handover import HandoverAssist
from .handover import HandoverState
from .handover import HandoverStep
from .handover import transform
from .kinematics import FINGERTIP_OFFSET_MM
from .kinematics import fingertip_xyz
from .kinematics import plan_tip_line
from .kinematics import solve_ik

DEFAULT_REFERENCE = Path(__file__).parent / "calibration/sample_loading_handover_20261105.json"


def checked_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ValueError("relative handover reference must contain rigid 4x4 transforms")
    return matrix


def pose_error(actual, target):
    return (
        float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        float(np.rad2deg(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())),
    )


@dataclass(frozen=True)
class RelativeConfig:
    reference: Path = DEFAULT_REFERENCE
    stable_frames: int = 5
    stable_mm: float = 3.0
    stable_deg: float = 3.0
    position_tolerance_mm: float = 5.0
    orientation_tolerance_deg: float = 3.0
    approach_mm: float = 30.0
    left_opening_mm: float = 60.0
    right_opening_mm: float = 80.0
    right_release_threshold: float = 0.5

    def __post_init__(self):
        for name in (
            "stable_mm",
            "stable_deg",
            "position_tolerance_mm",
            "orientation_tolerance_deg",
            "approach_mm",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"relative handover {name} must be finite and positive")
        if self.stable_frames < 2:
            raise ValueError("relative handover stable frames must be at least 2")
        for name in ("left_opening_mm", "right_opening_mm"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 100:
                raise ValueError(f"relative handover {name} must be in (0, 100]")
        if (
            not np.isfinite(self.right_release_threshold)
            or not 0 < self.right_release_threshold <= self.right_opening_mm / 100
        ):
            raise ValueError(
                "relative handover right release threshold must be positive and no greater than right opening / 100"
            )


@dataclass
class RelativeState(HandoverState):
    right_samples: list = field(default_factory=list)
    path: np.ndarray | None = None
    path_index: int = 0
    target_tip: np.ndarray | None = None
    right_tip: np.ndarray | None = None
    arrival_count: int = 0
    approach_done: bool = False
    reason: str | None = None
    last_feedback_stamps: tuple | None = None
    model_rows_remaining: int = 0
    planning_attempts: int = 0


class RelativeHandover(HandoverAssist):
    def __init__(self, config, fk, relative_config):
        super().__init__(config, fk)
        self.relative_config = relative_config
        self.reference_data = json.loads(Path(relative_config.reference).read_text())
        self.relative = checked_transform(self.reference_data["left_tip_in_right_tip"])
        self.demo_right = checked_transform(self.reference_data["right_tip_in_left_base"])
        reference_base = transform(
            [*self.reference_data["right_base_xyz_mm"], *self.reference_data["right_base_rpy_deg"]]
        )
        if not np.allclose(self.base, reference_base, atol=1e-6):
            raise ValueError("relative handover base calibration differs from captured reference")
        if not np.allclose(self.reference_data["fingertip_offset_mm"], FINGERTIP_OFFSET_MM):
            raise ValueError("relative handover fingertip offset differs from captured reference")
        if min(relative_config.left_opening_mm, relative_config.right_opening_mm) / 100 <= config.right_open_threshold:
            raise ValueError("relative handover open commands must exceed the handover open threshold")
        self.state = RelativeState()
        self.donor_holding = False

    @property
    def owns_motion(self):
        return self.state.phase != "idle"

    def begin_inference_result(self):
        super().begin_inference_result()
        if self.state.phase in ("model_chunk_pending", "model_chunk"):
            self.state.phase = "model_chunk"
            # Interrupted/failed cycles must execute a whole new chunk before
            # retrying. Only committed sent rows consume this count.
            self.state.model_rows_remaining = 0

    def settings(self):
        result = super().settings()
        result["relative_handover"] = {
            **asdict(self.relative_config),
            "reference": str(self.relative_config.reference),
            "left_tip_in_right_tip": self.relative.tolist(),
            "captured_at": self.reference_data.get("captured_at"),
        }
        return result

    def tips(self, physical):
        tips = []
        for start in (0, 7):
            pose = self.fk.pose(physical[start : start + 6])
            tip = transform(pose)
            tip[:3, 3] = fingertip_xyz(pose)
            tips.append(tip)
        return tips[0], self.base @ tips[1]

    def poses(self, physical):
        record = super().poses(physical)
        left, right = self.tips(physical)
        target = right @ self.relative
        error, angle = pose_error(left, target)
        record.update(
            left_fingertip_in_left_base=left.tolist(),
            right_fingertip_in_left_base=right.tolist(),
            left_fingertip_in_right_fingertip=(np.linalg.inv(right) @ left).tolist(),
            relative_position_error_mm=error,
            relative_rotation_error_deg=angle,
        )
        # Region for the confirmation state machine is the demonstrated full
        # relative pose, not the old items_handover rotation template.
        record["in_handover_region"] = (
            error <= self.relative_config.position_tolerance_mm
            and angle <= self.relative_config.orientation_tolerance_deg
        )
        return record

    def plan_approach(self, physical, right_tip):
        cfg = self.relative_config
        target = right_tip @ self.relative
        pre = target.copy()
        pre[:3, 3] -= cfg.approach_mm * target[:3, 2]
        current = np.asarray(physical[:6]).copy()
        rpy = Rotation.from_matrix(target[:3, :3]).as_euler("xyz", degrees=True)

        def solve(seed, goal_tip):
            goal, _, _ = solve_ik(
                self.fk,
                seed,
                goal_tip[:3, 3],
                rpy,
                position_tolerance_mm=cfg.position_tolerance_mm,
                orientation_tolerance_deg=cfg.orientation_tolerance_deg,
                prefer_local_solution=True,
            )
            return goal

        def path(start, goal, goal_tip):
            return plan_tip_line(self.fk, start, goal, goal_tip[:3, 3], position_tolerance_mm=cfg.position_tolerance_mm)

        near = solve(current, pre)
        final = solve(near, target)
        paths = [path(current, near, pre), path(near, final, target)]
        return np.concatenate(paths), target

    def _plan_transfer(self, state, physical, right, record, events):
        started = time.perf_counter()
        state.planning_attempts += 1
        try:
            state.path, state.target_tip = self.plan_approach(physical, right)
            state.right_tip = right.copy()
            state.path_index = state.arrival_count = 0
            state.phase, state.reason = "approaching", None
            events.append("relative_approach_planned")
        except (ValueError, RuntimeError) as exc:
            state.phase, state.reason = "model_chunk_pending", str(exc)
            state.model_rows_remaining = 0
            events.append("relative_plan_failed")
        record["planning_seconds"] = time.perf_counter() - started

    @staticmethod
    def fresh_stamps(feedback):
        if feedback is None or not all(feedback.get(s, {}).get("valid", False) for s in ("left", "right")):
            return None
        stamps = tuple(feedback[s].get("stamp") for s in ("left", "right"))
        if any(x is None or not np.isfinite(x) for x in stamps):
            return None
        return stamps

    def next(self, physical, previous, remaining, *, feedback=None):
        cfg = self.relative_config
        physical, previous = np.asarray(physical), np.asarray(previous)
        state = copy.deepcopy(self.state)
        record = self.poses(physical)
        left, right = self.tips(physical)
        stamps = self.fresh_stamps(feedback)
        fresh = stamps is not None and (
            state.last_feedback_stamps is None
            or all(x > y for x, y in zip(stamps, state.last_feedback_stamps, strict=True))
        )
        events = []
        if state.phase == "idle":
            distance, angle = pose_error(right, self.demo_right)
            # The demonstrated translation defines the relative left target,
            # not an absolute region the donor must revisit.
            eligible = self.donor_holding
            if fresh:
                state.last_feedback_stamps = stamps
            if eligible and fresh:
                state.right_samples = ([*state.right_samples, right.copy()])[-cfg.stable_frames :]
            else:
                state.right_samples = []
            stable = len(state.right_samples) >= cfg.stable_frames and all(
                pose_error(p, right)[0] <= cfg.stable_mm and pose_error(p, right)[1] <= cfg.stable_deg
                for p in state.right_samples
            )
            record.update(
                right_reference_distance_mm=distance,
                right_reference_angle_deg=angle,
                right_stable_frames=len(state.right_samples),
                donor_holding=self.donor_holding,
            )
            if stable:
                state.hold = physical.copy()
                cap = self.config.gripper_cap if self.config.gripper_cap is not None else self.config.left_close_action
                state.hold[13] = min(float(previous[13]), cap)
                state.hold[6] = max(float(physical[6]), cfg.left_opening_mm / 100)
                state.right_tip = right.copy()
                self._plan_transfer(state, physical, right, record, events)
                # This frame holds pose and opens receiver; never jumps into
                # motion after the potentially slow planning computation.
                action = state.hold.copy()
            else:
                action = None
        elif state.phase == "model_chunk_pending":
            action = state.hold.copy()
        elif state.phase == "model_chunk":
            if state.model_rows_remaining == 0:
                state.model_rows_remaining = len(remaining)
                events.append("model_chunk_started")
            action = np.asarray(remaining[0], dtype=np.float32).copy()
            state.model_rows_remaining -= 1
            if state.model_rows_remaining == 0:
                state.phase = "replan_pending"
                events.append("model_chunk_completed")
        elif state.phase == "replan_pending":
            # Relatch both poses from the NEW feedback after the model chunk.
            # Do not reuse the old failed target or wait for a timer.
            state.hold = physical.copy()
            cap = self.config.gripper_cap if self.config.gripper_cap is not None else self.config.left_close_action
            state.hold[13] = min(float(previous[13]), cap)
            state.hold[6] = max(float(physical[6]), cfg.left_opening_mm / 100)
            action = state.hold.copy()
            if fresh:
                state.last_feedback_stamps = stamps
                events.append("relative_replanning")
                self._plan_transfer(state, physical, right, record, events)
        elif state.phase == "approaching":
            action = state.hold.copy()
            drift, rotation_drift = pose_error(right, state.right_tip)
            right_stable = drift <= cfg.stable_mm and rotation_drift <= cfg.stable_deg
            opened = physical[6] >= state.hold[6] - 0.05
            if right_stable and opened and fresh:
                state.last_feedback_stamps = stamps
                waypoint = state.path[min(state.path_index, len(state.path) - 1)]
                if np.max(np.abs(physical[:6] - waypoint)) <= 0.02 and np.max(np.abs(previous[:6] - waypoint)) <= 1e-5:
                    state.path_index = min(state.path_index + 1, len(state.path))
                action[:6] = state.path[min(state.path_index, len(state.path) - 1)]
                error, angle = pose_error(left, state.target_tip)
                arrived = (
                    state.path_index >= len(state.path)
                    and error <= cfg.position_tolerance_mm
                    and angle <= cfg.orientation_tolerance_deg
                    and record["in_handover_region"]
                )
                state.arrival_count = state.arrival_count + 1 if arrived else 0
                if state.arrival_count >= 3:
                    state.phase = "waiting"
                    state.hold[:6] = state.path[-1]
                    state.hold[6] = (
                        self.config.gripper_cap
                        if self.config.gripper_cap is not None
                        else self.config.left_close_action
                    )
                    state.samples = ()
                    state.contact_samples = ()
                    state.left_peak, state.left_shrunk = float(physical[6]), False
                    state.approach_done = True
                    action = state.hold.copy()
                    events.append("relative_arrived_closing_left")
            else:
                # Keep the last sent left joint pose until feedback is valid,
                # the receiver is open and the donor is stable again.
                action[:6] = previous[:6]
                state.arrival_count = 0
                events.append("relative_waiting_for_open_and_stable_feedback")
            record.update(
                path_index=state.path_index,
                path_length=len(state.path),
                donor_drift_mm=drift,
                donor_drift_deg=rotation_drift,
            )
        elif state.phase in ("waiting", "ready", "releasing"):
            drift, rotation_drift = pose_error(right, state.right_tip)
            if not record["in_handover_region"] or drift > cfg.stable_mm or rotation_drift > cfg.stable_deg:
                state.phase = "waiting"
                state.samples = state.contact_samples = ()
                action = state.hold.copy()
                events.append("relative_pose_lost_hold_donor")
            else:
                # Reuse fresh CAN contact/threshold confirmation, but control
                # release ourselves instead of following model gripper intent.
                synthetic = np.repeat(state.hold[None, :], len(remaining), axis=0)
                synthetic[:, 13] = cfg.right_opening_mm / 100
                step = super().next(physical, previous, synthetic, feedback=feedback)
                state, record = step.state, step.record
                events = list(record["events"])
                action = step.action
                if action is None:
                    action = state.hold.copy()
                if step.kind == "handover_release":
                    action[:6], action[7:13] = state.hold[:6], state.hold[7:13]
                    if not record["left_closed_stable"]:
                        state.phase = "waiting"
                        action = state.hold.copy()
                        events = ["left_closure_lost"]
                    else:
                        action[13] = min(
                            cfg.right_opening_mm / 100,
                            min(float(previous[13]), float(physical[13]))
                            + (self.config.contact_release_step if record["right_feedback_fresh"] else 0.0),
                        )
                        opened = (
                            record["right_feedback_fresh"]
                            and physical[13] >= cfg.right_release_threshold
                            and previous[13] >= cfg.right_release_threshold
                        )
                        state.phase = "completed" if opened else "releasing"
                        events = [e for e in events if e != "handover_completed"]
                        if opened:
                            # Placement latches the last sent donor command.
                            # Keep opening to the full target while the left
                            # arm moves; do not latch the partial release value.
                            action[13] = cfg.right_opening_mm / 100
                            events.append("handover_completed")
        else:
            # One demonstrated transfer per episode. Placement takes ownership
            # next; if disabled, hold the completed transfer pose.
            action = state.hold.copy()
            if state.phase == "completed":
                action[13] = cfg.right_opening_mm / 100
        record.update(
            phase=state.phase,
            events=events,
            feedback=feedback,
            relative_control=True,
            right_release_threshold=cfg.right_release_threshold,
            planning_attempts=state.planning_attempts,
            model_rows_remaining=state.model_rows_remaining,
            reason=state.reason,
            target_left_tip=None if state.target_tip is None else state.target_tip.tolist(),
        )
        # Normal monitor status expects these during the confirmation phases.
        for key, value in {
            "left_closed_stable": False,
            "left_previous_action": float(previous[6]),
            "left_action_state_gap": float(physical[6] - previous[6]),
            "left_contact_frames": 0,
            "left_closing_frames": 0,
            "left_observed_closing": False,
            "left_feedback_fresh": fresh,
            "confirmation": None,
        }.items():
            record.setdefault(key, value)
        kind = "relative_handover_" + state.phase if action is not None else "model"
        if self.state.phase == "model_chunk":
            kind = "relative_handover_model_chunk"
        return HandoverStep(action, state, kind, record)


def add_arguments(parser):
    parser.add_argument(
        "--relative-handover",
        action="store_true",
        help="use captured relative fingertip pose for sample_loading handover",
    )
    for name, value in vars(RelativeConfig()).items():
        parser.add_argument(
            "--relative-handover-" + name.replace("_", "-"),
            type=Path if name == "reference" else int if name == "stable_frames" else float,
            default=value,
        )


def config_from_args(args):
    if not args.relative_handover:
        return None
    if args.task != "sample_loading" or not args.handover_assist or args.right_grip_hold is False:
        raise ValueError("--relative-handover requires sample_loading, --handover-assist and right grip hold")
    if args.bell_press:
        raise ValueError("--relative-handover cannot be combined with --bell-press")
    return RelativeConfig(**{name: getattr(args, "relative_handover_" + name) for name in vars(RelativeConfig())})
