"""Hold each assembly grasp still, then lift vertically before pose changes."""

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import fingertip_xyz, solve_ik


class AssemblyPickupLift:
    def __init__(self, config, fk, grip_config, hz):
        self.config, self.fk, self.grip_config, self.hz = config, fk, grip_config, hz
        self.arms = [dict(phase="idle", hold=None, samples=[], stamp=None, path=[], index=0,
                          attempt=-1, reason=None, initial_opening=None) for _ in range(2)]

    @property
    def ready(self):
        return all(a["phase"] == "raised" for a in self.arms)

    def owns(self, side):
        return self.arms[side]["phase"] != "idle"

    def _plan(self, arm, cycle):
        arm["attempt"] = cycle
        q = arm["hold"].copy()
        pose = self.fk.pose(q)
        xyz = fingertip_xyz(pose)
        path = []
        distance = self.config.trigger_lift_mm
        try:
            for height in np.linspace(0., distance, max(1, int(np.ceil(distance))) + 1)[1:]:
                target = xyz + [0., 0., height]
                new, _, _ = solve_ik(self.fk, q, target, pose[3:], position_tolerance_mm=.5,
                                     orientation_tolerance_deg=1., prefer_local_solution=True)
                # Dense vertical IK waypoints, then rate-limited joint commands.
                dz = abs(fingertip_xyz(self.fk.pose(new))[2] - fingertip_xyz(self.fk.pose(q))[2])
                duration = max(dz / self.config.pickup_lift_speed_mm_s,
                               float(np.max(np.abs(new - q))) / np.deg2rad(20.))
                count = max(1, int(np.ceil(duration * self.hz)))
                path.extend(q + alpha * (new - q) for alpha in np.linspace(0, 1, count + 1)[1:])
                q = new
            arm.update(path=path, index=0, phase="lifting", goal_xyz=target, goal_rpy=np.asarray(pose[3:]), reason=None)
        except (ValueError, RuntimeError) as exc:
            arm["reason"] = str(exc)

    def next(self, physical, target, holding, feedback, cycle):
        result = np.asarray(target).copy()
        for side, start in enumerate((0, 7)):
            arm = self.arms[side]
            if arm["phase"] == "idle" and holding[side]:
                arm.update(phase="closing", hold=np.asarray(physical[start:start + 6]).copy(),
                           initial_opening=float(physical[start + 6]))
            if arm["phase"] == "idle":
                continue
            info = feedback.get(("left", "right")[side], {})
            stamp = info.get("stamp")
            fresh = (info.get("valid", False) and stamp is not None and np.isfinite(stamp)
                     and (arm["stamp"] is None or stamp > arm["stamp"]))
            if fresh:
                arm["stamp"] = stamp
            if arm["phase"] == "closing" and fresh:
                opening = float(info["opening"])
                contact = (self.grip_config.command + self.config.contact_gap <= opening
                           <= self.grip_config.close_threshold
                           and arm["initial_opening"] - opening >= .03)
                arm["samples"] = (arm["samples"] + [opening])[-self.config.stable_frames:] if contact else []
                stable = len(arm["samples"]) == self.config.stable_frames and np.ptp(arm["samples"]) <= .01
                if stable and arm["attempt"] != cycle:
                    self._plan(arm, cycle)
            q = arm["path"][min(arm["index"], len(arm["path"]) - 1)] if arm["path"] else arm["hold"]
            result[start:start + 6] = q
            result[start + 6] = self.grip_config.command
        return result

    def commit(self, physical, sent):
        for side, start in enumerate((0, 7)):
            arm = self.arms[side]
            if arm["phase"] != "lifting":
                continue
            expected = arm["path"][min(arm["index"], len(arm["path"]) - 1)]
            if np.allclose(sent[start:start + 6], expected, atol=1e-5):
                arm["index"] = min(arm["index"] + 1, len(arm["path"]))
            if arm["index"] == len(arm["path"]):
                pose = self.fk.pose(physical[start:start + 6])
                angle = (Rotation.from_euler("xyz", arm["goal_rpy"], degrees=True).inv()
                         * Rotation.from_euler("xyz", pose[3:], degrees=True)).magnitude()
                if np.linalg.norm(fingertip_xyz(pose) - arm["goal_xyz"]) <= 2. and np.rad2deg(angle) <= 2.:
                    arm["phase"] = "raised"

    def record(self):
        return {name: {"phase": arm["phase"], "path_index": arm["index"], "path_length": len(arm["path"]),
                       "reason": arm["reason"]} for name, arm in zip(("left", "right"), self.arms, strict=True)}
