"""Bounded local XYZ correction with optional downward tool rotation."""

# Internal companion to GraspDip; shares its geometry checks.
# ruff: noqa: SLF001
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import PIPER_JOINT_MAX
from .config import PIPER_JOINT_MIN

DEFAULT_POSITION_TOLERANCE_MM = 3.0
DEFAULT_ORIENTATION_TOLERANCE_DEG = 5.0


def solve_translation(helper, joints, target, side, *, orientation_fraction=1.0, heading_deg=None):
    q = np.asarray(joints, dtype=float)
    if not callable(getattr(helper.fk, "pose", None)):
        raise ValueError("forward grasp dip requires full pose FK")
    original = np.asarray(helper.fk.pose(q), dtype=float)
    if original.shape != (6,) or not np.isfinite(original).all():
        raise ValueError("forward grasp dip requires finite XYZ/RPY FK")
    if original[2] - target[2] > helper.config.max_drop_mm + 0.001:
        return None, "max_drop_exceeded"
    reference = Rotation.from_euler("xyz", original[3:], degrees=True)
    if helper.config.vertical_down:
        # Preserve the tool's heading while rotating its +Z axis toward base -Z.
        tool_x = reference.as_matrix()[:2, 0]
        heading = np.rad2deg(np.arctan2(tool_x[1], tool_x[0])) if heading_deg is None else heading_deg
        downward = Rotation.from_euler("xyz", [180.0, 0.0, heading], degrees=True)
        turn = (reference.inv() * downward).as_rotvec()
        reference = reference * Rotation.from_rotvec(orientation_fraction * turn)

    def residual(candidate):
        pose = helper.fk.pose(candidate)
        angle = (reference.inv() * Rotation.from_euler("xyz", pose[3:], degrees=True)).as_rotvec()
        return np.r_[(pose[:3] - target) / 100.0, angle, 0.001 * (candidate - q)]

    result = least_squares(
        residual,
        np.clip(q, PIPER_JOINT_MIN, PIPER_JOINT_MAX),
        bounds=(PIPER_JOINT_MIN.astype(float), PIPER_JOINT_MAX.astype(float)),
        max_nfev=120,
    )
    pose = helper.fk.pose(result.x)
    error = np.linalg.norm(pose[:3] - target)
    angle = np.rad2deg((reference.inv() * Rotation.from_euler("xyz", pose[3:], degrees=True)).magnitude())
    if error > DEFAULT_POSITION_TOLERANCE_MM:
        return None, "cartesian_target_unreachable"
    if angle > DEFAULT_ORIENTATION_TOLERANCE_DEG or not helper._points_down(helper._tilt(result.x)):
        return None, "cartesian_orientation_unreachable"
    return result.x - q, None
