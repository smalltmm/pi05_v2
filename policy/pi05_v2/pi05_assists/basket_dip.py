"""Optional feedback-gated descent before a handle_basket left grasp."""
import copy
from dataclasses import asdict, dataclass, field
import time

import numpy as np

from . import config as control_timing
from .grasp_dip import GraspDip, GraspDipConfig
from .grasp_dip_cartesian import solve_translation
from .grip_hold import BasketLeftGripHold, GripHoldStep


@dataclass(frozen=True)
class BasketDipConfig:
    target_z_mm: float = 155.
    z_range_mm: tuple = (155., 180.)
    max_drop_mm: float = 40.
    speed_mm_s: float = 50.
    backward_mm: float = 10.

    def __post_init__(self):
        lo, hi = self.z_range_mm
        if not np.isfinite([self.target_z_mm, lo, hi, self.max_drop_mm, self.speed_mm_s, self.backward_mm]).all():
            raise ValueError("basket dip settings must be finite")
        if (not lo <= self.target_z_mm <= hi or lo >= hi
                or min(self.max_drop_mm, self.speed_mm_s) <= 0 or self.backward_mm < 0):
            raise ValueError("basket dip needs MIN <= target <= MAX, MIN < MAX, positive drop/speed and non-negative backward distance")


@dataclass
class DipState:
    phase: str = "idle"
    path: list = field(default_factory=list)
    index: int = 0
    open_grip: float = 0.
    close_grip: float = 0.
    started: float = 0.
    dwell_started: float | None = None
    samples: tuple = ()
    stamp: float | None = None
    finish_cycle: int | None = None
    retreat_steps: int = 0
    retreat_ready: bool = False


@dataclass
class BasketDipStep(GripHoldStep):
    dip_state: DipState
    arm_action: np.ndarray | None = None


class BasketDipHold(BasketLeftGripHold):
    """Model inference continues; only the left arm is overridden while dipping."""

    def __init__(self, config, dip_config, fk, *, clock=time.monotonic):
        super().__init__(config)
        self.dip_config = dip_config
        self.clock = clock
        self.fk = fk
        self.dip_state = DipState()
        # Use the shared Cartesian solver while retaining the starting rotation.
        # Basket grasps need not point downward, unlike flat-object grasp dip.
        self.solver = GraspDip(GraspDipConfig(target_z_mm=(dip_config.target_z_mm, None),
                              z_ranges=(dip_config.z_range_mm, None), max_drop_mm=dip_config.max_drop_mm),
                              fk, down_tilt=lambda q: 0.)

    def settings(self):
        return {**super().settings(), "dip": asdict(self.dip_config)}

    def _path(self, physical):
        cfg = self.dip_config
        q = np.asarray(physical[:6], dtype=float)
        xyz = self.solver._xyz(q)
        if not cfg.z_range_mm[0] <= xyz[2] <= cfg.z_range_mm[1]:
            raise ValueError("physical_height_outside_range")
        distance = max(0., xyz[2] - cfg.target_z_mm)
        if distance > cfg.max_drop_mm:
            raise ValueError("max_drop_exceeded")
        path = []
        # First translate backwards at the current height, then descend from
        # that shifted XY position. "Backward" is -X in the Piper base frame.
        segments = []
        if cfg.backward_mm > 0:
            retreat = xyz.copy()
            retreat[0] -= cfg.backward_mm
            segments.append((xyz.copy(), retreat))
            xyz = retreat
        segments.append((xyz.copy(), np.array([xyz[0], xyz[1], cfg.target_z_mm])))
        for start, end in segments:
            distance_segment = float(np.linalg.norm(end - start))
            count = max(1, int(np.ceil(distance_segment / (cfg.speed_mm_s * control_timing.CONTROL_DT))))
            for alpha in np.linspace(0., 1., count + 1)[1:]:
                target = start + alpha * (end - start)
                # Only Z contributes to the total descent limit.
                if start[2] - target[2] > cfg.max_drop_mm + 1e-6:
                    raise ValueError("max_drop_exceeded")
                bias, reason = solve_translation(self.solver, q, target, 0)
                if reason:
                    raise ValueError(reason)
                path.append(q + bias)
        return path

    def next(self, physical, model_grip, *, token, handover=None, previous=None, feedback=None):
        d = copy.deepcopy(self.dip_state)
        now = self.clock()
        active = d.phase in ("descending", "dwell", "closing", "ready")
        # Do not accumulate policy release votes while inserted motion owns the arm.
        base = (GripHoldStep(True, copy.deepcopy(self.state), {}) if active
                else super().next(physical, model_grip, token=token, handover=handover))
        events = list(base.record.get("events", []))
        if not active and "closing_trend_holding" in events:
            try:
                d = DipState(phase="descending", path=self._path(physical), started=now,
                             open_grip=max(float(physical[6]), float(physical[6] if previous is None else previous[6])),
                             close_grip=min(float(model_grip), self.config.command_cap),
                             retreat_steps=int(np.ceil(self.dip_config.backward_mm /
                                                       (self.dip_config.speed_mm_s * control_timing.CONTROL_DT))))
                events.append("dip_started")
            except ValueError as exc:
                d.phase = "skipped"
                events.append("dip_skipped:" + str(exc))
        if d.phase == "ready" and token[0] != d.finish_cycle:
            d.phase = "done"
            base = super().next(physical, model_grip, token=token, handover=handover)
            events.extend(base.record["events"])
            events.append("dip_resume_fresh_model")
        arm = None
        z = float(self.solver._xyz(physical[:6])[2])
        if d.phase in ("descending", "dwell", "closing", "ready"):
            arm = np.r_[d.path[min(d.index, len(d.path) - 1)], d.open_grip]
            at_target = np.linalg.norm(self.solver._xyz(physical[:6]) - self.solver._xyz(d.path[-1])) <= 2.
            if d.phase == "descending":
                waiting_retreat = False
                if d.retreat_steps and d.index == d.retreat_steps and not d.retreat_ready:
                    retreat_q = d.path[d.retreat_steps - 1]
                    d.retreat_ready = bool(np.linalg.norm(self.solver._xyz(physical[:6]) -
                                                         self.solver._xyz(retreat_q)) <= 2.)
                    if d.retreat_ready:
                        events.append("dip_retreat_ready")
                    else:
                        arm[:6] = retreat_q
                        waiting_retreat = True
                if now - d.started > len(d.path) * control_timing.CONTROL_DT + 3. and not at_target:
                    d.phase = "skipped"
                    arm = None
                    events.append("dip_arrival_timeout_resume_model")
                elif waiting_retreat:
                    pass
                elif d.index < len(d.path):
                    d.index += 1
                elif at_target:
                    d.phase = "dwell"
                    d.dwell_started = now
                    events.append("dip_at_target")
            if d.phase == "dwell":
                if not at_target:
                    d.phase = "descending"
                    d.dwell_started = None
                elif now - d.dwell_started >= .2:
                    d.phase = "closing"
                    d.started = now
                    events.append("dip_closing_in_place")
            if d.phase in ("closing", "ready"):
                arm[6] = d.close_grip
            if d.phase == "closing":
                stamp = None if feedback is None else feedback.get("stamp")
                fresh = feedback is None or (feedback.get("valid", False) and stamp is not None
                          and np.isfinite(stamp) and (d.stamp is None or stamp > d.stamp))
                if fresh:
                    d.stamp = stamp
                    d.samples = (*d.samples, float(physical[6]))[-3:]
                else:
                    d.samples = ()
                stable = len(d.samples) == 3 and np.ptp(d.samples) <= .01
                closed = (physical[6] <= d.close_grip + .01 or
                          (physical[6] <= self.config.close_threshold and d.open_grip - physical[6] >= .05))
                if (stable and closed and at_target) or now - d.started >= 2.:
                    events.append("dip_close_ready" if stable and closed and at_target else "dip_close_timeout")
                    d.phase = "ready"
                    d.finish_cycle = token[0]
        force = base.state.phase == "holding"
        command = float(arm[6]) if arm is not None else min(float(model_grip), self.config.command_cap) if force else None
        record = dict(phase=base.state.phase, force_close=force, feedback=float(physical[6]),
                      hold_command=command, model_grip=float(model_grip), baseline=base.state.baseline,
                      release_count=base.state.release_count, events=events, dip_phase=d.phase,
                      target_z_mm=self.dip_config.target_z_mm, physical_z_mm=z)
        return BasketDipStep(force, base.state, record, d, arm)

    def commit(self, step):
        super().commit(step)
        self.dip_state = copy.deepcopy(step.dip_state)


def add_arguments(parser):
    import argparse
    parser.add_argument("--basket-left-grasp-dip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--basket-left-grasp-dip-target-z-mm", type=float, default=155.)
    parser.add_argument("--basket-left-grasp-dip-z-range-mm", type=float, nargs=2, default=(155., 180.))
    parser.add_argument("--basket-left-grasp-dip-max-drop-mm", type=float, default=40.)
    parser.add_argument("--basket-left-grasp-dip-speed-mm-s", type=float, default=50.)
    parser.add_argument("--basket-left-grasp-dip-backward-mm", type=float, default=10.,
                        help="move left end-effector -X this distance before descending")


def config_from_args(args):
    if not args.basket_left_grasp_dip:
        return None
    if args.task != "handle_basket" or not args.basket_left_grip_hold:
        raise ValueError("--basket-left-grasp-dip requires --task handle_basket --basket-left-grip-hold")
    if args.grasp_dip:
        raise ValueError("basket left grasp dip cannot be combined with --grasp-dip")
    return BasketDipConfig(args.basket_left_grasp_dip_target_z_mm, tuple(args.basket_left_grasp_dip_z_range_mm),
                           args.basket_left_grasp_dip_max_drop_mm, args.basket_left_grasp_dip_speed_mm_s,
                           args.basket_left_grasp_dip_backward_mm)
