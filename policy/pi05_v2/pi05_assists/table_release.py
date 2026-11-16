"""Optional independent left/right lowering before a model-requested release."""

import argparse
import copy
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import config as timing
from .config import PIPER_JOINT_MIN, PIPER_JOINT_MAX


@dataclass(frozen=True)
class TableReleaseConfig:
    z_mm: float = 155.
    speed_mm_s: float = 50.
    hold_seconds: float = .2

    def __post_init__(self):
        if not np.isfinite([self.z_mm, self.speed_mm_s]).all() or self.z_mm < 0 or self.speed_mm_s <= 0:
            raise ValueError("table release needs finite Z >= 0 and speed > 0")
        if not np.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError("table release hold seconds must be finite and non-negative")


@dataclass
class ReleaseState:
    phase: str = "idle"
    path: list = field(default_factory=list)
    index: int = 0
    close_grip: float = 0.
    open_grip: float = 0.
    release_cycle: int | None = None
    start_z_mm: float | None = None
    end_z_mm: float | None = None
    hold_remaining: int = 0


class TableReleaseLower:
    def __init__(self, config, fk, side):
        self.config, self.fk, self.side = config, fk, side
        self.start = {"left": 0, "right": 7}[side]
        self.state = ReleaseState()

    def _solve(self, q, xyz, rotation):
        # Z uses the same SDK endpoint as --print-ee-z-only and model Z floor,
        # rather than the separate fingertip offset used by relative handover.
        def residual(candidate):
            pose = np.asarray(self.fk.pose(candidate))
            angle = (rotation.inv() * Rotation.from_euler("xyz", pose[3:], degrees=True)).as_rotvec()
            return np.r_[(pose[:3] - xyz) / 100., angle]

        solved = least_squares(residual, np.clip(q, PIPER_JOINT_MIN, PIPER_JOINT_MAX),
                               bounds=(PIPER_JOINT_MIN.astype(float), PIPER_JOINT_MAX.astype(float)),
                               max_nfev=160)
        result = solved.x.astype(np.float32)
        pose = np.asarray(self.fk.pose(result))
        angle = np.rad2deg((rotation.inv() * Rotation.from_euler("xyz", pose[3:], degrees=True)).magnitude())
        if not np.isfinite(pose).all() or np.linalg.norm(pose[:3] - xyz) > 1. or angle > 2.:
            raise ValueError(f"{self.side} table release lowering IK failed; gripper not released")
        return result

    def _path(self, physical):
        q = np.asarray(physical[self.start:self.start + 6], dtype=np.float32).copy()
        pose = np.asarray(self.fk.pose(q), dtype=float)
        end_z = min(pose[2], self.config.z_mm)  # Already lower: never raise before releasing.
        distance = pose[2] - end_z
        rotation = Rotation.from_euler("xyz", pose[3:], degrees=True)
        release_rpy = pose[3:].copy()
        release_rpy[0] = -180. if self.side == "left" else 180.
        release_rotation = Rotation.from_euler("xyz", release_rpy, degrees=True)
        path = []
        previous_xyz = pose[:3].copy()

        def waypoint(xyz, orientation):
            nonlocal q, previous_xyz
            goal = self._solve(q, xyz, orientation)
            steps = max(1, int(np.ceil(np.linalg.norm(xyz - previous_xyz) /
                                       (self.config.speed_mm_s * timing.CONTROL_DT))),
                        int(np.ceil(np.max(np.abs(goal - q)) /
                                    (np.deg2rad(60.) * timing.CONTROL_DT))))
            path.extend((q + t * (goal - q)).astype(np.float32)
                        for t in np.linspace(0., 1., steps + 1)[1:])
            q, previous_xyz = goal, xyz.copy()

        # Correct endpoint Euler roll while still above the release height.
        # Pitch/yaw stay at their trigger values; all six joints remain free
        # to solve this Cartesian pose (there is no fixed J6 target).
        turn = (rotation.inv() * release_rotation).as_rotvec()
        angle = float(np.rad2deg(np.linalg.norm(turn)))
        if angle > 1e-6:
            count = max(1, int(np.ceil(angle / 2.)))
            for alpha in np.linspace(0., 1., count + 1)[1:]:
                waypoint(pose[:3], rotation * Rotation.from_rotvec(alpha * turn))
        # Solve a few Cartesian waypoints, then sample their joint segments at
        # the configured control rate. No per-row IK or arrival feedback wait.
        count = max(1, int(np.ceil(distance / 3.)))
        for alpha in np.linspace(0., 1., count + 1)[1:]:
            xyz = pose[:3].copy()
            xyz[2] = pose[2] - alpha * distance
            if distance > 0:
                waypoint(xyz, release_rotation)
            else:
                path.append(q.copy())
        return path, float(pose[2]), float(end_z)

    def next(self, step, holding_state, physical, previous, *, cycle):
        state = copy.copy(self.state)
        result = copy.deepcopy(step)
        events = result.record["events"]
        if state.phase == "released" and state.release_cycle != cycle:
            state = ReleaseState()
        if state.phase == "idle" and "released_by_model" in events:
            path, start_z, end_z = self._path(physical)
            state = ReleaseState(phase="lowering", path=path,
                                 close_grip=float(previous[self.start + 6]),
                                 open_grip=float(result.record["model_grip"]),
                                 start_z_mm=start_z, end_z_mm=end_z,
                                 # The final lowering row already holds for one
                                 # control interval before the next command.
                                 hold_remaining=max(0, int(np.ceil(self.config.hold_seconds /
                                                                  timing.CONTROL_DT)) - 1))
            events.append("release_lower_started")
        if state.phase == "lowering":
            events[:] = [e for e in events if e != "released_by_model"]
            if state.index < len(state.path):
                result.state = copy.deepcopy(holding_state)
                result.force_close = True
                result.arm_action = np.r_[state.path[state.index], state.close_grip]
                result.record.update(phase="holding", force_close=True, hold_command=state.close_grip)
                state.index += 1
            else:
                state.phase = "holding_before_release"
                events.append("release_lower_hold_started")
        if state.phase == "holding_before_release":
            events[:] = [e for e in events if e != "released_by_model"]
            if state.hold_remaining:
                result.state = copy.deepcopy(holding_state)
                result.force_close = True
                result.arm_action = np.r_[state.path[-1], state.close_grip]
                result.record.update(phase="holding", force_close=True, hold_command=state.close_grip)
                state.hold_remaining -= 1
            else:
                state.phase, state.release_cycle = "released", cycle
                events.append("released_after_lowering")
        if state.phase == "released":
            result.state.phase = "released"
            result.force_close = False
            result.arm_action = np.r_[state.path[-1], state.open_grip]
            result.record.update(phase="released", force_close=False, hold_command=None)
        # Retain the release pose through this chunk; the next inference sees
        # the lowered state rather than resuming stale pre-release arm targets.
        result.record["release_lower"] = dict(
            **asdict(self.config), phase=state.phase, path_index=state.index, path_length=len(state.path),
            hold_remaining=state.hold_remaining,
            target_roll_deg=-180. if self.side == "left" else 180.,
            pitch_yaw="preserve_at_release_trigger",
            start_z_mm=state.start_z_mm, end_z_mm=state.end_z_mm, frame=f"{self.side}_base_sdk_endpoint")
        result.release_lower_state = state
        return result

    def commit(self, step):
        self.state = step.release_lower_state


def add_arguments(parser):
    parser.add_argument("--table-release-lower", action=argparse.BooleanOptionalAction, default=False,
                        help="table_rearrangement: lower each held arm before releasing its gripper")
    parser.add_argument("--table-release-z-mm", type=float, default=155.)
    parser.add_argument("--table-release-speed-mm-s", type=float, default=50.)
    parser.add_argument("--table-release-hold-seconds", type=float, default=.2)


def config_from_args(args):
    if not args.table_release_lower:
        return None
    if args.task != "table_rearrangement" or not args.right_grip_hold or not args.left_grip_hold:
        raise ValueError("--table-release-lower requires --task table_rearrangement --left-grip-hold --right-grip-hold")
    config = TableReleaseConfig(args.table_release_z_mm, args.table_release_speed_mm_s,
                                args.table_release_hold_seconds)
    if args.model_z_floor and args.model_min_z_mm > config.z_mm:
        raise ValueError("--table-release-z-mm must be at least --model-min-z-mm")
    return config
