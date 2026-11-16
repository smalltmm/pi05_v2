"""Move along the left tool's forward axis before releasing a basket grasp."""
import argparse
import copy
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from . import config as control_timing
from .kinematics import solve_ik, fingertip_xyz


@dataclass(frozen=True)
class BasketReleaseConfig:
    forward_mm: float = 15.
    speed_mm_s: float = 30.

    def __post_init__(self):
        if (not np.isfinite([self.forward_mm, self.speed_mm_s]).all()
                or min(self.forward_mm, self.speed_mm_s) <= 0):
            raise ValueError("basket release distance and speed must be finite and positive")


@dataclass
class ReleaseState:
    phase: str = "idle"
    path: list = field(default_factory=list)
    index: int = 0
    close_grip: float = 0.
    open_grip: float = 0.
    release_cycle: int | None = None
    direction: list = field(default_factory=list)


class BasketReleaseAdvance:
    def __init__(self, config, fk):
        self.config = config
        self.state = ReleaseState()
        self.fk = fk

    def _path(self, physical):
        q = np.asarray(physical[:6], dtype=float)
        pose = np.asarray(self.fk.pose(q))
        start = fingertip_xyz(pose)
        # Tool +Z points from the flange toward the fingertips. Freeze this
        # direction at release intent; preserve the original tool rotation.
        direction = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()[:, 2]
        count = max(1, int(np.ceil(self.config.forward_mm /
                                  (self.config.speed_mm_s * control_timing.CONTROL_DT))))
        path = []
        for alpha in np.linspace(0., 1., count + 1)[1:]:
            target = start + alpha * self.config.forward_mm * direction
            q, _, _ = solve_ik(self.fk, q, target, pose[3:],
                               position_tolerance_mm=5., orientation_tolerance_deg=20.,
                               prefer_local_solution=True)
            path.append(q.copy())
        return path, direction.tolist()

    def next(self, step, holding_state, physical, previous, *, cycle):
        d = copy.copy(self.state)
        result = copy.deepcopy(step)
        events = result.record["events"]
        if d.phase == "released" and cycle != d.release_cycle:
            d = ReleaseState()
        if d.phase == "idle" and "released_by_model" in events:
            path, direction = self._path(physical)
            d = ReleaseState(phase="advancing", path=path, direction=direction,
                             close_grip=float(previous[6]),
                             open_grip=float(result.record["model_grip"]))
            events.append("release_advance_started")
        if d.phase == "advancing":
            events[:] = [e for e in events if e != "released_by_model"]
            if d.index < len(d.path):
                result.state = copy.deepcopy(holding_state)
                result.force_close = True
                result.arm_action = np.r_[d.path[d.index], d.close_grip]
                result.record.update(phase="holding", force_close=True, hold_command=d.close_grip)
                d.index += 1
            else:
                d.phase = "released"
                d.release_cycle = cycle
                events.append("released_after_advance")
        if d.phase == "released":
            # Finish this model chunk at the release pose, then use a fresh
            # inference so stale pre-release arm targets cannot pull it back.
            result.state.phase = "released"
            result.force_close = False
            result.arm_action = np.r_[d.path[-1], d.open_grip]
            result.record.update(phase="released", force_close=False, hold_command=None)
        result.record["release_advance"] = dict(
            **asdict(self.config), phase=d.phase, path_index=d.index, path_length=len(d.path),
            direction_in_left_base=d.direction, axis="left_gripper_+Z")
        result.release_state = d
        return result

    def commit(self, step):
        self.state = step.release_state


def add_arguments(parser):
    parser.add_argument("--basket-left-release-advance", action=argparse.BooleanOptionalAction, default=False,
                        help="advance along left gripper +Z before opening")
    parser.add_argument("--basket-left-release-forward-mm", type=float, default=15.)
    parser.add_argument("--basket-left-release-speed-mm-s", type=float, default=30.)


def config_from_args(args):
    if not args.basket_left_release_advance:
        return None
    if args.task != "handle_basket" or not args.basket_left_grip_hold:
        raise ValueError("--basket-left-release-advance requires --task handle_basket --basket-left-grip-hold")
    return BasketReleaseConfig(args.basket_left_release_forward_mm, args.basket_left_release_speed_mm_s)
