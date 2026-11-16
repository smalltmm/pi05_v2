"""Pure fingertip geometry and bounded IK; returns joint targets only."""
from __future__ import annotations
from itertools import pairwise


import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import PIPER_JOINT_MAX
from .config import PIPER_JOINT_MIN

FINGERTIP_OFFSET_MM = np.asarray((0.0, 0.0, 135.8))
# Average left link6 rotation over sample_loading01 frames 491-494,
# immediately before the first clear gripper-opening command at frame 495.
SAMPLE_LOADING_RELEASE_RPY_DEG = np.asarray((-33.45469554, 86.55296625, -67.18979470))
DEFAULT_POSITION_TOLERANCE_MM = 5.0
DEFAULT_ORIENTATION_TOLERANCE_DEG = 3.0


def fingertip_xyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return pose[:3] + Rotation.from_euler("xyz", pose[3:], degrees=True).apply(FINGERTIP_OFFSET_MM)


def solve_ik(
    fk,
    current_joints: np.ndarray,
    target_xyz: np.ndarray,
    target_rpy_deg: np.ndarray,
    *,
    position_tolerance_mm: float = DEFAULT_POSITION_TOLERANCE_MM,
    orientation_tolerance_deg: float = DEFAULT_ORIENTATION_TOLERANCE_DEG,
    prefer_local_solution: bool = False,
    joint_min: np.ndarray | None = None,
    joint_max: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Find a bounded fingertip IK solution with the requested tool rotation.

    Online callers may accept a valid solution seeded from the current joints
    immediately. If that local solve fails, retain the full nearest-solution
    search instead of accepting the first arbitrary random-seed branch.
    """
    if not np.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0:
        raise ValueError("position tolerance must be finite and positive")
    if not np.isfinite(orientation_tolerance_deg) or orientation_tolerance_deg <= 0:
        raise ValueError("orientation tolerance must be finite and positive")
    current_joints = np.asarray(current_joints, dtype=float).reshape(6)
    lower = np.asarray(PIPER_JOINT_MIN if joint_min is None else joint_min, dtype=float)
    upper = np.asarray(PIPER_JOINT_MAX if joint_max is None else joint_max, dtype=float)
    if (lower.shape != (6,) or upper.shape != (6,) or not np.isfinite([lower, upper]).all()
            or np.any(lower >= upper) or np.any(lower < PIPER_JOINT_MIN)
            or np.any(upper > PIPER_JOINT_MAX)):
        raise ValueError("IK joint bounds must be finite ordered six-vectors within Piper limits")
    target_rotation = Rotation.from_euler("xyz", target_rpy_deg, degrees=True)

    def residual(joints: np.ndarray) -> np.ndarray:
        pose = np.asarray(fk.pose(joints), dtype=float)
        rotation = Rotation.from_euler("xyz", pose[3:], degrees=True)
        return np.r_[(fingertip_xyz(pose) - target_xyz) / 100.0, (target_rotation.inv() * rotation).as_rotvec()]

    rng = np.random.default_rng(9)
    starts = [np.clip(current_joints, lower, upper)]
    starts.extend(lower + (upper - lower) * rng.random(6) for _ in range(100))
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for seed_index, start in enumerate(starts):
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=200,
        )
        pose = np.asarray(fk.pose(result.x), dtype=float)
        position_error_mm = float(np.linalg.norm(fingertip_xyz(pose) - target_xyz))
        rotation = Rotation.from_euler("xyz", pose[3:], degrees=True)
        orientation_error_deg = float(np.rad2deg((target_rotation.inv() * rotation).magnitude()))
        if position_error_mm <= position_tolerance_mm and orientation_error_deg <= orientation_tolerance_deg:
            if prefer_local_solution and seed_index == 0:
                return result.x.astype(np.float32), position_error_mm, orientation_error_deg
            joint_distance = float(np.linalg.norm(result.x - current_joints))
            candidates.append((joint_distance, result.x, position_error_mm, orientation_error_deg))
    if not candidates:
        raise RuntimeError(
            f"no IK solution within {position_tolerance_mm:g} mm, {orientation_tolerance_deg:g} degrees, "
            "and Piper joint limits; no motion sent"
        )
    _, joints, position_error_mm, orientation_error_deg = min(candidates, key=lambda item: item[0])
    return joints.astype(np.float32), position_error_mm, orientation_error_deg


def plan_tip_line(
    fk,
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    target_xyz: np.ndarray,
    *,
    position_tolerance_mm: float = DEFAULT_POSITION_TOLERANCE_MM,
) -> np.ndarray:
    """Track a straight fingertip path and check all interpolated commands."""
    current_joints = np.asarray(current_joints, dtype=float)
    target_joints = np.asarray(target_joints, dtype=float)
    start_xyz = fingertip_xyz(fk.pose(current_joints))
    segments = max(2, int(np.ceil(np.linalg.norm(target_xyz - start_xyz) / 5.0)))
    previous = np.clip(current_joints, PIPER_JOINT_MIN, PIPER_JOINT_MAX)
    path = [current_joints]
    for alpha in np.linspace(0.0, 1.0, segments + 1)[1:]:
        desired = start_xyz + alpha * (target_xyz - start_xyz)
        nominal = current_joints + alpha * (target_joints - current_joints)
        result = least_squares(
            lambda joints, desired=desired, nominal=nominal: np.r_[
                (fingertip_xyz(fk.pose(joints)) - desired) / 100.0, 0.02 * (joints - nominal)
            ],
            previous,
            bounds=(PIPER_JOINT_MIN.astype(float), PIPER_JOINT_MAX.astype(float)),
            max_nfev=150,
        )
        position_error = np.linalg.norm(fingertip_xyz(fk.pose(result.x)) - desired)
        if position_error > 1.0:
            raise RuntimeError(f"Cartesian path IK missed waypoint by {position_error:.2f} mm")
        path.append(result.x)
        previous = result.x
    path.append(target_joints)

    commands = [path[0]]
    for start, end in pairwise(path):
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / 0.03)))
        commands.extend(start + fraction * (end - start) for fraction in np.linspace(0.0, 1.0, steps + 1)[1:])
    command_array = np.asarray(commands, dtype=np.float32)
    direction = target_xyz - start_xyz
    length_squared = float(np.dot(direction, direction))
    for index, joints in enumerate(command_array):
        point = fingertip_xyz(fk.pose(joints))
        fraction = np.clip(np.dot(point - start_xyz, direction) / length_squared, 0, 1) if length_squared else 0.0
        line = start_xyz + fraction * direction
        if (
            np.linalg.norm(point - line) > max(3.0, position_tolerance_mm)
            or point[2] < min(start_xyz[2], target_xyz[2]) - position_tolerance_mm
        ):
            raise RuntimeError(f"Cartesian path check failed at command {index}: tip={np.round(point, 2).tolist()}")
    return command_array[1:]

