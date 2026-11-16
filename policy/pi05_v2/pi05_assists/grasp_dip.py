"""Optional joint-2 or Cartesian correction around an open-to-close grasp transition.

Operates on absolute, scheduled control rows. Planning is speculative; only
commit() advances state after a row has actually been sent/recorded.
Heights are SDK link-6 reference positions in each arm's base frame, in mm.
They are NOT fingertip clearances above a table.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .config import PIPER_JOINT_MAX, PIPER_JOINT_MIN


def down_tilt_from_rpy_deg(rpy) -> float:
    """Angle between SDK tool +Z and base -Z, for R = Rz(yaw) Ry(pitch) Rx(roll).

    SDK Euler angles are degrees. R[2,2] = cos(roll)*cos(pitch), so yaw and
    the +180/-180 roll wrap do not affect this direction test.
    """
    rpy = np.asarray(rpy, dtype=float)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("grasp dip orientation must be finite SDK roll/pitch/yaw in degrees")
    roll, pitch, _ = np.deg2rad(rpy)
    return float(np.rad2deg(np.arccos(np.clip(-np.cos(roll) * np.cos(pitch), -1.0, 1.0))))


class _PiperFK:
    def __init__(self, sdk_fk):
        self.sdk_fk = sdk_fk

    def pose(self, joints):
        return np.asarray(self.sdk_fk.CalFK(np.asarray(joints, dtype=float).tolist())[-1], dtype=float)

    def __call__(self, joints):
        return self.pose(joints)[:3]

    def down_tilt_deg(self, joints):
        return down_tilt_from_rpy_deg(self.pose(joints)[3:])


def make_piper_fk() -> _PiperFK:
    # Pure numeric FK; no SDK device imports or connections.
    from .piper_fk import C_PiperForwardKinematics

    return _PiperFK(C_PiperForwardKinematics(dh_is_offset=1))


@dataclass(frozen=True)
class GraspDipConfig:
    target_z_mm: tuple[float | None, float | None] = (None, None)
    z_ranges: tuple[tuple[float, float] | None, tuple[float, float] | None] = (None, None)
    close_threshold: float = 0.1
    open_threshold: float = 0.2
    pre_steps: int = 3
    release_steps: int = 3
    lift_mm: float = 10.0
    max_drop_mm: float = 3.0
    max_tilt_deg: float = 30.0
    forward_mm: tuple[float, float] = (0.0, 0.0)
    vertical_down: bool = False
    staged_vertical: bool = False

    def __post_init__(self):
        if len(self.forward_mm) != 2 or not np.isfinite(self.forward_mm).all():
            raise ValueError("grasp dip forward offsets must contain two finite distances")
        if len(self.target_z_mm) != 2 or len(self.z_ranges) != 2:
            raise ValueError("grasp dip requires left and right settings")
        if all(z is None for z in self.target_z_mm):
            raise ValueError("--grasp-dip requires --left-grasp-dip-target-z-mm or --right-grasp-dip-target-z-mm")
        for side, target, limits in zip(("left", "right"), self.target_z_mm, self.z_ranges, strict=True):
            if target is not None and not np.isfinite(target):
                raise ValueError(f"{side} grasp dip target Z must be finite")
            if limits is not None and (len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
                raise ValueError(f"{side} grasp dip Z range must contain finite MIN < MAX in base-frame mm")
            if target is not None and limits is None:
                raise ValueError(f"--{side}-grasp-dip-z-range-mm MIN MAX is required with a target Z")
        if not 0 <= self.close_threshold < self.open_threshold <= 1:
            raise ValueError("grasp dip thresholds require 0 <= close < open <= 1")
        if self.pre_steps < 1 or self.release_steps < 1:
            raise ValueError("grasp dip pre/release steps must be positive")
        for value in (self.lift_mm, self.max_drop_mm):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("grasp dip lift/drop distances must be finite and positive")
        if not np.isfinite(self.max_tilt_deg) or not 0 <= self.max_tilt_deg < 90:
            raise ValueError("grasp dip maximum tilt from downward must be finite and in [0, 90) degrees")


@dataclass
class _ArmState:
    armed: bool = False
    phase: str = "idle"
    offset: float | np.ndarray = 0.0
    grasp_offset: float | np.ndarray = 0.0
    grasp_z: float = 0.0
    closed: bool = False
    prepared: bool = False
    prepare_grip: float = 1.0
    target_x: float | None = None
    target_y: float | None = None
    target_heading_deg: float | None = None


@dataclass
class DipPlan:
    actions: np.ndarray
    states: list[list[_ArmState]]
    events: list[dict]


class GraspDip:
    def __init__(self, config: GraspDipConfig, fk: Callable[[np.ndarray], np.ndarray], *,
                 down_tilt: Callable[[np.ndarray], float] | None = None):
        self.config = config
        self.fk = fk
        self.down_tilt = down_tilt if down_tilt is not None else getattr(fk, "down_tilt_deg", None)
        if not callable(self.down_tilt):
            raise ValueError("grasp dip requires an orientation FK callback; XYZ alone cannot check downward posture")
        self._states = [_ArmState(), _ArmState()]

    def _xyz(self, joints):
        xyz = np.asarray(self.fk(joints), dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("grasp dip FK must return finite XYZ in mm")
        return xyz

    def _tilt(self, joints):
        tilt = float(self.down_tilt(joints))
        if not np.isfinite(tilt) or not 0 <= tilt <= 180:
            raise ValueError("grasp dip orientation FK must return a finite downward angle in [0, 180] degrees")
        return tilt

    def _corrected_tilt(self, joints, offset):
        candidate = self.apply_offset(joints, offset)
        return self._tilt(candidate)

    def _points_down(self, tilt):
        return tilt <= self.config.max_tilt_deg + 1e-6

    def _check(self, joints, offset, side):
        candidate = self.apply_offset(joints, offset)
        before, after = self._xyz(joints), self._xyz(candidate)
        drop = float(before[2] - after[2])
        reason = None
        if np.any(candidate < PIPER_JOINT_MIN - 1e-6) or np.any(candidate > PIPER_JOINT_MAX + 1e-6):
            reason = "joint2_limit"
        elif drop <= (-0.01 if self.config.forward_mm[side] or self.config.vertical_down else 0):
            reason = "offset_does_not_lower_z"
        elif drop > self.config.max_drop_mm:
            reason = "max_drop_exceeded"
        return reason, before, after

    @staticmethod
    def apply_offset(joints, offset):
        result = np.asarray(joints, dtype=float).copy()
        if np.ndim(offset):
            result += offset
        else:
            result[1] += offset
        return result

    @staticmethod
    def offset_event(offset):
        return None if offset is None else float(offset[1] if np.ndim(offset) else offset)

    def solve_correction(self, joints, target_z, side, *, target_x=None, target_y=None,
                         target_heading_deg=None, fraction=1.0):
        if not self.config.forward_mm[side] and not self.config.vertical_down:
            return self.solve_joint2(joints, target_z, side)
        from .grasp_dip_cartesian import solve_translation
        xyz = self._xyz(joints)
        target = xyz.copy()
        target[0] = xyz[0] + fraction * self.config.forward_mm[side] if target_x is None else target_x
        if target_y is not None:
            target[1] = target_y
        # Never move up merely because the policy has already reached target Z.
        target[2] = min(target_z, xyz[2])
        return solve_translation(self, joints, target, side, orientation_fraction=fraction,
                                 heading_deg=target_heading_deg)

    def solve_joint2(self, joints: np.ndarray, target_z: float, side: int):
        """Find the nearest local J2 solution to FK(q).z == target_z.

        All five other joints stay fixed. Search both signs within hardware
        joint limits, then bisect the first bracket in each direction.
        The angle is a computed result, never a prescribed dip amount.
        """
        q = np.asarray(joints, dtype=np.float64).copy()
        original_z = float(self._xyz(q)[2])
        if target_z >= original_z - 0.001:
            return None, "target_not_below_model_z"
        if original_z - target_z > self.config.max_drop_mm + 0.001:
            return None, "max_drop_exceeded"
        roots = []
        for direction in (1, -1):
            extent = float(PIPER_JOINT_MAX[1] - q[1]) if direction > 0 else float(q[1] - PIPER_JOINT_MIN[1])
            if extent <= 0:
                continue
            previous_delta = 0.0
            for distance in np.linspace(extent / 256, extent, 256):
                delta = direction * float(distance)
                candidate = q.copy()
                candidate[1] += delta
                if self._xyz(candidate)[2] <= target_z:
                    above, below = previous_delta, delta
                    for _ in range(35):
                        midpoint = (above + below) / 2
                        candidate[1] = q[1] + midpoint
                        if self._xyz(candidate)[2] > target_z:
                            above = midpoint
                        else:
                            below = midpoint
                    roots.append((above + below) / 2)
                    break
                previous_delta = delta
        if not roots:
            return None, "target_z_unreachable_within_j2_limit"
        reason = None
        for delta in sorted(roots, key=abs):
            reason, _, _ = self._check(np.asarray(joints, dtype=np.float32), delta, side)
            if reason is None:
                return delta, None
        return None, reason

    def plan(self, actions: np.ndarray, *, previous_action: np.ndarray, physical_state: np.ndarray) -> DipPlan:
        base = np.asarray(actions, dtype=np.float32)
        previous = np.asarray(previous_action, dtype=np.float32)
        physical = np.asarray(physical_state, dtype=np.float32)
        if base.ndim != 2 or base.shape[1] != 14 or len(base) == 0:
            raise ValueError("grasp dip requires nonempty [T,14] absolute control actions")
        if previous.shape != (14,) or physical.shape != (14,):
            raise ValueError("grasp dip previous action and physical state must be [14]")
        if not all(np.all(np.isfinite(x)) for x in (base, previous, physical)):
            raise ValueError("grasp dip inputs must be finite")
        output = base.copy()
        snapshots = [copy.deepcopy(self._states) for _ in base]
        events = []
        cfg = self.config
        close_limit = np.float32(cfg.close_threshold)
        open_limit = np.float32(cfg.open_threshold)
        for side, start in enumerate((0, 7)):
            target_z = cfg.target_z_mm[side]
            if target_z is None:
                continue
            name = ("left", "right")[side]
            limits = cfg.z_ranges[side]
            physical_z = float(self._xyz(physical[start:start + 6])[2])
            physical_tilt = self._tilt(physical[start:start + 6])
            arm = copy.deepcopy(self._states[side])
            if arm.phase == "idle" and previous[start + 6] >= open_limit:
                arm.armed = True
            i = 0
            while i < len(base):
                grip = base[i, start + 6]
                if arm.phase == "idle":
                    if grip >= open_limit:
                        arm.armed = True
                    if cfg.staged_vertical:
                        # Runtime inserts a timed rotate-then-lower path from
                        # the current pose. Never also send the legacy short
                        # ramp that blends rotation and lowering in three rows.
                        snapshots[i][side] = copy.deepcopy(arm)
                        i += 1
                        continue
                    # Find the next closed row in THIS executable chunk, not
                    # in the model tail that the scheduler will discard.
                    closes = np.flatnonzero(base[i:, start + 6] <= close_limit)
                    if arm.armed and len(closes):
                        close = i + int(closes[0])
                        begin = max(i, close - cfg.pre_steps)
                        if i == begin:
                            close_xyz = self._xyz(base[close, start:start + 6])
                            close_tilt = self._tilt(base[close, start:start + 6])
                            reason = None
                            bias = None
                            ramp_offsets = {}
                            if close == begin:
                                reason = "no_open_control_row_before_close"
                            elif not limits[0] <= physical_z <= limits[1]:
                                reason = "physical_height_outside_range"
                            elif not limits[0] <= close_xyz[2] <= limits[1]:
                                reason = "close_height_outside_range"
                            elif not self._points_down(physical_tilt):
                                reason = "physical_orientation_not_downward"
                            elif not self._points_down(close_tilt):
                                reason = "close_orientation_not_downward"
                            else:
                                bias, reason = self.solve_correction(base[close, start:start + 6], target_z, side)
                            if reason is None:
                                # Check full correction through the upcoming
                                # low grasp segment before starting any dip.
                                for j in range(begin, len(base)):
                                    q = base[j, start:start + 6]
                                    xyz = self._xyz(q)
                                    if j > close and (xyz[2] >= close_xyz[2] + cfg.lift_mm
                                                      or base[j, start + 6] >= open_limit
                                                      or not self._points_down(self._tilt(q))
                                                      or not self._points_down(self._corrected_tilt(q, bias))):
                                        break
                                    if j < close and not limits[0] <= xyz[2] <= limits[1]:
                                        reason = "approach_height_outside_range"
                                        break
                                    if not self._points_down(self._tilt(q)):
                                        reason = "approach_orientation_not_downward"
                                        break
                                    if j < close:
                                        fraction = (j - begin + 1) / (close - begin)
                                        ramp_z = float(xyz[2] + fraction * (target_z - xyz[2]))
                                        if abs(ramp_z - xyz[2]) <= 0.001 and not cfg.forward_mm[side] and not cfg.vertical_down:
                                            ramp_offsets[j] = 0.0
                                        else:
                                            ramp_offsets[j], reason = self.solve_correction(q, ramp_z, side, fraction=fraction)
                                    else:
                                        reason, _, _ = self._check(q, bias, side)
                                    if reason is None and not self._points_down(
                                        self._corrected_tilt(q, ramp_offsets[j] if j < close else bias)
                                    ):
                                        reason = "corrected_orientation_not_downward"
                                    if reason:
                                        break
                            event = {
                                "side": name, "start_row": begin, "close_row": close,
                                "physical_z_mm": physical_z,
                                "trigger_z_range_mm": list(limits),
                                "close_z_mm": float(close_xyz[2]), "j2_offset_rad": self.offset_event(bias),
                                "forward_mm": cfg.forward_mm[side],
                                "target_z_mm": target_z,
                                "physical_down_tilt_deg": physical_tilt,
                                "close_down_tilt_deg": close_tilt,
                                "max_tilt_deg": cfg.max_tilt_deg,
                                "status": "skipped" if reason else "applied", "reason": reason,
                            }
                            events.append(event)
                            if reason is None:
                                _, _, corrected_xyz = self._check(base[close, start:start + 6], bias, side)
                                event["target_x_mm"] = float(close_xyz[0] + cfg.forward_mm[side])
                                event["corrected_close_x_mm"] = float(corrected_xyz[0])
                                event["corrected_close_z_mm"] = float(corrected_xyz[2])
                                event["close_drop_mm"] = float(close_xyz[2] - corrected_xyz[2])
                                event["corrected_close_down_tilt_deg"] = self._corrected_tilt(base[close, start:start + 6], bias)
                                arm.phase = "hold"
                                arm.grasp_offset = bias
                                arm.target_x = float(close_xyz[0] + cfg.forward_mm[side])
                                arm.grasp_z = float(close_xyz[2])
                                arm.closed = False
                                # Reach full offset at the last still-open row.
                                for j in range(begin, close):
                                    arm.offset = ramp_offsets[j]
                                    output[j, start:start + 6] = self.apply_offset(base[j, start:start + 6], arm.offset)
                                    snapshots[j][side] = copy.deepcopy(arm)
                                arm.armed = False
                                i = close
                                continue
                            # Consume this transition even if skipped. A later
                            # opening may arm a new transition.
                            arm.armed = False
                            for j in range(i, close + 1):
                                snapshots[j][side] = copy.deepcopy(arm)
                            i = close + 1
                            continue
                else:
                    xyz = self._xyz(base[i, start:start + 6])
                    if arm.phase == "hold":
                        # A runtime preparation may precede the next inference.
                        # Solve its still-open approach and actual closing pose
                        # again; the predicted tail was only a closure hint.
                        cancelled_preparation = (arm.prepared and not arm.closed and grip >= open_limit
                                                 and grip > arm.prepare_grip + 0.05)
                        if (arm.prepared and not arm.closed and not cancelled_preparation
                                and xyz[2] < arm.grasp_z + cfg.lift_mm):
                            bias, reason = self.solve_correction(
                                base[i, start:start + 6], target_z, side, target_x=arm.target_x,
                                target_y=arm.target_y, target_heading_deg=arm.target_heading_deg)
                            if reason == "target_not_below_model_z":
                                # The new policy already incorporated the dip
                                # through feedback. Do not lower it a second time
                                # or treat an already-satisfied target as a fault.
                                bias = 0.0
                                reason = None
                                events.append({"side": name, "status": "satisfied", "start_row": i,
                                               "reason": "model_at_or_below_target_z",
                                               "target_z_mm": target_z, "z_mm": float(xyz[2])})
                            if reason is not None:
                                raise ValueError(f"{name} prepared grasp dip rejected: {reason}")
                            arm.grasp_offset = bias
                            if grip <= close_limit:
                                arm.grasp_z = float(xyz[2])
                        # Retain the angle solved at closure while allowing the
                        # model's subsequent trajectory (including lift) to run.
                        arm.offset = arm.grasp_offset
                        arm.closed = arm.closed or grip <= close_limit
                        tilt = self._tilt(base[i, start:start + 6])
                        corrected_tilt = self._corrected_tilt(base[i, start:start + 6], arm.offset)
                        orientation_changed = not all(self._points_down(value) for value in (physical_tilt, tilt, corrected_tilt))
                        if (xyz[2] >= arm.grasp_z + cfg.lift_mm
                                or (arm.closed and grip >= open_limit) or orientation_changed or cancelled_preparation):
                            arm.phase = "release"
                            events.append({"side": name, "status": "release", "start_row": i,
                                           "reason": "orientation_not_downward" if orientation_changed else "lift_or_reopen",
                                           "z_mm": float(xyz[2]), "physical_down_tilt_deg": physical_tilt,
                                           "down_tilt_deg": tilt, "corrected_down_tilt_deg": corrected_tilt})
                    if arm.phase == "release":
                        magnitude = np.maximum(0.0, np.abs(arm.offset) - np.abs(arm.grasp_offset) / cfg.release_steps)
                        arm.offset = np.copysign(magnitude, arm.grasp_offset)
                        if not np.any(np.abs(arm.offset) > 1e-8):
                            arm = _ArmState(armed=bool(grip >= open_limit))
                    if np.any(np.abs(arm.offset) > 1e-8):
                        reason, before, after = self._check(base[i, start:start + 6], arm.offset, side)
                        if reason:
                            # Do not abruptly undo an already executed dip or
                            # silently send an invalid correction on replanning.
                            raise ValueError(f"{name} active grasp dip rejected: {reason}; Z {before[2]:.2f} -> {after[2]:.2f} mm")
                        output[i, start:start + 6] = self.apply_offset(base[i, start:start + 6], arm.offset)
                snapshots[i][side] = copy.deepcopy(arm)
                i += 1
        return DipPlan(output, snapshots, events)

    def commit(self, plan: DipPlan, row_index: int) -> None:
        self._states = copy.deepcopy(plan.states[row_index])

    def settings(self) -> dict:
        return asdict(self.config)


def add_arguments(parser):
    import argparse

    parser.add_argument("--grasp-dip", action=argparse.BooleanOptionalAction, default=False,
                        help="lower joint-2 targets before closing, hold through the grasp, then fade on lift/reopen")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-grasp-dip-target-z-mm", type=float, default=None,
                            help="desired SDK end-reference Z in base-frame mm; omit to disable this arm; J2 is solved by FK")
        parser.add_argument(f"--{side}-grasp-dip-forward-mm", type=float, default=0.0,
                            help="base +X translation during dip (mm); nonzero uses Cartesian IK; default 0 keeps J2-only dip")
        parser.add_argument(f"--{side}-grasp-dip-z-range-mm", type=float, nargs=2, metavar=("MIN", "MAX"),
                            help="required trigger-only Z window for current feedback and predicted grasp; target may be below MIN")
    parser.add_argument("--grasp-dip-close-threshold", type=float, default=0.1)
    parser.add_argument("--grasp-dip-open-threshold", type=float, default=0.2)
    parser.add_argument("--grasp-dip-pre-steps", type=int, default=3, help="10 Hz open rows used to ramp into the offset")
    parser.add_argument("--grasp-dip-release-steps", type=int, default=3)
    parser.add_argument("--grasp-dip-lift-mm", type=float, default=10.0)
    parser.add_argument("--grasp-dip-max-drop-mm", type=float, default=3.0)
    parser.add_argument("--grasp-dip-max-tilt-deg", type=float, default=30.0,
                        help="maximum angle between tool +Z and base -Z for downward grasps (default: 30 degrees)")
    parser.add_argument("--grasp-dip-vertical-down", action=argparse.BooleanOptionalAction, default=False,
                        help="during grasp dip, use full-arm IK to point tool +Z down while preserving its heading")
    parser.add_argument("--assembly-grasp-assist", action=argparse.BooleanOptionalAction, default=False,
                        help="item_assembly only: Cartesian dip ending vertically downward, then hold both grippers")


def config_from_args(args):
    if args.grasp_dip_vertical_down and not args.grasp_dip:
        raise ValueError("--grasp-dip-vertical-down requires --grasp-dip")
    if args.assembly_grasp_assist and (args.task != "item_assembly" or not args.grasp_dip
                                      or args.left_grasp_dip_target_z_mm is None
                                      or args.right_grasp_dip_target_z_mm is None):
        raise ValueError("--assembly-grasp-assist requires item_assembly and --grasp-dip with both target Z values")
    if not args.grasp_dip:
        return None
    return GraspDipConfig(
        target_z_mm=(args.left_grasp_dip_target_z_mm, args.right_grasp_dip_target_z_mm),
        z_ranges=(args.left_grasp_dip_z_range_mm, args.right_grasp_dip_z_range_mm),
        close_threshold=args.grasp_dip_close_threshold, open_threshold=args.grasp_dip_open_threshold,
        pre_steps=args.grasp_dip_pre_steps, release_steps=args.grasp_dip_release_steps,
        lift_mm=args.grasp_dip_lift_mm, max_drop_mm=args.grasp_dip_max_drop_mm,
        max_tilt_deg=args.grasp_dip_max_tilt_deg,
        forward_mm=(args.left_grasp_dip_forward_mm, args.right_grasp_dip_forward_mm),
        vertical_down=args.assembly_grasp_assist or args.grasp_dip_vertical_down,
        staged_vertical=args.task == "table_rearrangement" and args.grasp_dip_vertical_down,
    )
