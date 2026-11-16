"""Clamp model endpoint height in each Piper base frame (SDK FK reference)."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .config import PIPER_JOINT_MIN, PIPER_JOINT_MAX


class ModelZFloor:
    def __init__(self, fk, minimum_mm):
        self.fk = fk
        self.minimum_mm = float(minimum_mm)
        self.cache = {}

    def begin_chunk(self):
        self.cache.clear()

    def apply(self, actions):
        result = np.asarray(actions, dtype=np.float32).copy()
        for row in result:
            for start in (0, 7):
                q = row[start:start + 6].copy()
                key = q.tobytes()
                if key not in self.cache:
                    pose = np.asarray(self.fk.pose(q), dtype=float)
                    if pose[2] >= self.minimum_mm:
                        self.cache[key] = q
                    else:
                        self.cache[key] = self._solve(q, pose)
                row[start:start + 6] = self.cache[key]
        return result

    def _solve(self, q, pose):
        target = pose[:3].copy()
        # Sub-millimetre margin absorbs float32/FK rounding at the boundary.
        target[2] = self.minimum_mm + .01
        rotation = Rotation.from_euler('xyz', pose[3:], degrees=True)

        def residual(candidate):
            actual = np.asarray(self.fk.pose(candidate))
            angle = (rotation.inv() * Rotation.from_euler('xyz', actual[3:], degrees=True)).as_rotvec()
            return np.r_[(actual[:3] - target) / 100., angle]

        def objective(candidate):
            error = residual(candidate)
            return .5 * float(error @ error)

        # Height is a hard constraint, not a term the optimizer may trade
        # against rotation near a joint limit. Keep the same final XY/angle
        # tolerances and float32 height verification below.
        solved = minimize(
            objective, np.clip(q, PIPER_JOINT_MIN, PIPER_JOINT_MAX).astype(float),
            method='SLSQP',
            bounds=list(zip(PIPER_JOINT_MIN.astype(float), PIPER_JOINT_MAX.astype(float))),
            constraints=[{'type': 'ineq', 'fun': lambda candidate:
                          (self.fk.pose(candidate)[2] - target[2]) / 100.}],
            options={'ftol': 1e-12, 'maxiter': 160},
        )
        joints = solved.x.astype(np.float32)
        actual = np.asarray(self.fk.pose(joints))
        angle = np.rad2deg((rotation.inv() * Rotation.from_euler('xyz', actual[3:], degrees=True)).magnitude())
        if (not np.isfinite(actual).all() or actual[2] < self.minimum_mm
                or np.linalg.norm(actual[:3] - target) > 1. or angle > 2.):
            raise ValueError(f'model Z floor IK failed: target_z_mm={self.minimum_mm:g}, '
                             f'solved_z_mm={actual[2]:.3f}; model chunk not sent')
        return joints
