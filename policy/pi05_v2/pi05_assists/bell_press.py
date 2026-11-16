"""One press per downward approach for click_bell, with a closed gripper.

No XY/contact detector or feedback arrival wait. Commands are ramped, held
for a fixed duration, then follow the policy again. Only committed commands
advance the persistent press/rearm state.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np

from . import config as control_timing
from .grasp_dip import GraspDip, GraspDipConfig


@dataclass(frozen=True)
class BellPressConfig:
    z_range_mm: tuple[float, float]
    target_z_mm: float
    max_tilt_deg: float = 20.0
    hold_seconds: float = 0.2
    rearm_rise_mm: float = 20.0
    max_drop_mm: float = 3.0
    closed_threshold: float = 0.1
    ramp_steps: int = 3
    release_steps: int = 3
    down_speed_mm_s: float = 100.0

    def __post_init__(self):
        GraspDipConfig(target_z_mm=(None, self.target_z_mm), z_ranges=(None, self.z_range_mm),
                       max_drop_mm=self.max_drop_mm, max_tilt_deg=self.max_tilt_deg)
        if not np.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError("--bell-press-hold-seconds must be finite and non-negative")
        if not np.isfinite(self.rearm_rise_mm) or self.rearm_rise_mm <= 0:
            raise ValueError("--bell-press-rearm-rise-mm must be finite and positive")
        if not np.isfinite(self.closed_threshold) or not 0 <= self.closed_threshold < 1:
            raise ValueError("--bell-press-closed-threshold must be in [0, 1)")
        if self.ramp_steps < 1 or self.release_steps < 1:
            raise ValueError("bell press ramp/release steps must be positive")
        if not np.isfinite(self.down_speed_mm_s) or self.down_speed_mm_s <= 0:
            raise ValueError("--bell-press-down-speed-mm-s must be finite and positive")


@dataclass
class _State:
    armed: bool = True
    low_z: float = math.inf
    previous_z: float | None = None
    descending_frames: int = 0
    offset: float = 0.0
    anchor_z: float = 0.0
    releasing: bool = False
    release_left: int = 0


@dataclass
class BellStep:
    action: np.ndarray
    state: _State
    source_row: int
    kind: str = "model"


class BellPress:
    def __init__(self, config, fk, *, side=1):
        self.config = config
        self.side = side
        self.name = ("left", "right")[side]
        self.solver = GraspDip(GraspDipConfig(
            target_z_mm=tuple(config.target_z_mm if index == side else None for index in (0, 1)),
            z_ranges=tuple(config.z_range_mm if index == side else None for index in (0, 1)),
            max_drop_mm=config.max_drop_mm, max_tilt_deg=config.max_tilt_deg,
        ), fk)
        self.state = _State()

    def settings(self):
        return asdict(self.config)


class BellPressRuntime:
    def __init__(self, helper, actions, *, lookahead=None):
        self.helper = helper
        self.joints = slice(helper.side * 7, helper.side * 7 + 6)
        self.j2 = helper.side * 7 + 1
        self.grip = helper.side * 7 + 6
        self.actions = np.asarray(actions, dtype=np.float32).copy()
        self.lookahead = np.empty((0, 14), dtype=np.float32) if lookahead is None else np.asarray(lookahead, dtype=np.float32)
        self.row = 0
        self.inserted = []
        self.inserted_index = 0
        self.events = []
        self._reported = set()

    def _event(self, status, **details):
        key = (self.row, status, details.get("reason"))
        if key not in self._reported:
            self.events.append(dict(side=self.helper.name, status=status, source_row=self.row,
                                    target_z_mm=self.helper.config.target_z_mm, **details))
            self._reported.add(key)

    def _limit_descent_speed(self, prepared, physical):
        """Place a Z waypoint every speed * dt mm; round the last interval up.

        Start at measured joints, not the model pose, so the initial approach
        is included. This limits command spacing, not physical contact force.
        """
        solver = self.helper.solver
        max_step = self.helper.config.down_speed_mm_s * control_timing.CONTROL_DT
        start = np.asarray(physical[self.joints], dtype=float)
        last = prepared[-1]
        end = last.action[self.joints].astype(float)
        zs = np.array([solver._xyz(start + alpha * (end - start))[2]
                       for alpha in np.linspace(0., 1., 65)])
        if np.any(np.diff(zs) > 1e-4):
            raise ValueError("bell press approach is not monotonically downward")
        distance = max(0., float(zs[0] - zs[-1]))
        count = max(1, int(math.ceil(max(0., distance - 1e-4) / max_step)))
        if count > 10000:
            raise ValueError("bell press speed requires too many trajectory steps")
        result = []
        for index in range(1, count + 1):
            if index == count:
                q = end
            else:
                desired_z = zs[0] - index * max_step
                lo, hi = 0., 1.
                for _ in range(40):
                    alpha = (lo + hi) / 2
                    if solver._xyz(start + alpha * (end - start))[2] > desired_z:
                        lo = alpha
                    else:
                        hi = alpha
                q = start + (lo + hi) / 2 * (end - start)
            if not solver._points_down(solver._tilt(q)):
                raise ValueError("bell press ramp orientation not downward")
            row = copy.deepcopy(last)
            row.action[self.joints] = q
            result.append(row)
        return result

    def next(self, physical, previous):
        try:
            return self._next(physical, previous)
        except ValueError as exc:
            # Reject only this assist, not the policy's original command.
            # Monitor still validates that command before sending it.
            state = copy.deepcopy(self.helper.state)
            state.armed = False
            state.offset = 0.0
            state.releasing = False
            state.release_left = 0
            state.low_z = min(state.low_z, float(self.helper.solver._xyz(np.asarray(physical)[self.joints])[2]))
            self.inserted = []
            self.inserted_index = 0
            self._event("fallback_to_model", reason=str(exc))
            return BellStep(self.actions[self.row].copy(), state, self.row)

    def _next(self, physical, previous):
        if self.row >= len(self.actions):
            return None
        if self.inserted_index < len(self.inserted):
            step = copy.deepcopy(self.inserted[self.inserted_index])
            z = float(self.helper.solver._xyz(np.asarray(physical)[self.joints])[2])
            step.state.low_z = min(step.state.low_z, self.helper.state.low_z, z)
            step.state.previous_z = z
            return step
        cfg = self.helper.config
        solver = self.helper.solver
        state = copy.deepcopy(self.helper.state)
        physical = np.asarray(physical, dtype=np.float32)
        target = self.actions[self.row].copy()
        z = float(solver._xyz(physical[self.joints])[2])
        model_z = float(solver._xyz(target[self.joints])[2])
        if state.previous_z is not None:
            if z < state.previous_z - 0.2:
                state.descending_frames += 1
            elif z > state.previous_z + 0.2:
                state.descending_frames = 0
        state.previous_z = z
        if not state.armed:
            state.low_z = min(state.low_z, z)
            if z >= state.low_z + cfg.rearm_rise_mm and state.offset == 0:
                state.armed = True
                state.descending_frames = 0
                self._event("rearmed", physical_z_mm=z)
            else:
                # Follow the model after pressing; fade correction only when
                # it lifts/reopens or rotates away, so there is no bounce at
                # the end of the fixed hold period.
                if state.offset:
                    tilt = solver._tilt(target[self.joints])
                    corrected_tilt = solver._corrected_tilt(target[self.joints], state.offset)
                    drop_limit_reached = model_z - cfg.target_z_mm > cfg.max_drop_mm
                    if (model_z > state.anchor_z + 1.0 or drop_limit_reached or target[self.grip] > cfg.closed_threshold
                            or not solver._points_down(tilt) or not solver._points_down(corrected_tilt)):
                        if not state.releasing:
                            state.releasing = True
                            state.release_left = cfg.release_steps
                            self._event("releasing", physical_z_mm=z,
                                        reason="target_drop_exceeds_limit" if drop_limit_reached else "lift_reopen_or_orientation",
                                        model_z_mm=model_z)
                    if state.releasing:
                        state.offset *= (state.release_left - 1) / state.release_left
                        state.release_left -= 1
                    elif model_z <= cfg.target_z_mm + 0.001:
                        state.offset = 0.0
                    else:
                        # A fresh inference can incorporate the press in its
                        # baseline pose. Recompute the residual, never add the
                        # old correction on top of that new low target.
                        state.offset, reason = solver.solve_joint2(target[self.joints], cfg.target_z_mm, self.helper.side)
                        if reason:
                            raise ValueError(f"{self.helper.name} active bell press rejected: {reason}")
                        if not solver._points_down(solver._corrected_tilt(target[self.joints], state.offset)):
                            raise ValueError(f"{self.helper.name} active bell press rejected: corrected_orientation_not_downward")
                    if state.offset:
                        reason, _, _ = solver._check(target[self.joints], state.offset, self.helper.side)
                        if reason:
                            raise ValueError(f"{self.helper.name} active bell press rejected: {reason}")
                        target[self.j2] += state.offset
                return BellStep(target, state, self.row)

        lo, hi = cfg.z_range_mm
        if not (lo <= z <= hi and lo <= model_z <= hi):
            return BellStep(target, state, self.row)
        if physical[self.grip] > cfg.closed_threshold or target[self.grip] > cfg.closed_threshold:
            return BellStep(target, state, self.row)

        # Look ahead to the first predicted low point. Tail rows are hints
        # only; no discarded model row can become a hardware command.
        forecast = np.concatenate((self.actions[self.row:], self.lookahead))
        zs = []
        for action in forecast:
            if action[self.grip] > cfg.closed_threshold:
                break
            value = float(solver._xyz(action[self.joints])[2])
            if zs and value > min(zs) + 1.0:
                break
            zs.append(value)
        predicted_descent = len(zs) >= 2 and (z - zs[0] > 0.5 or zs[0] - min(zs) > 0.5)
        if not (state.descending_frames >= 2 or predicted_descent):
            return BellStep(target, state, self.row)
        # Keep following the policy until close to its predicted bottom pose.
        if zs and model_z > min(zs) + 2.0:
            return BellStep(target, state, self.row)

        physical_tilt = solver._tilt(physical[self.joints])
        model_tilt = solver._tilt(target[self.joints])
        details = dict(physical_z_mm=z, model_z_mm=model_z,
                       physical_down_tilt_deg=physical_tilt, model_down_tilt_deg=model_tilt)
        if not solver._points_down(physical_tilt) or not solver._points_down(model_tilt):
            self._event("not_triggered", reason="orientation_not_downward", **details)
            return BellStep(target, state, self.row)
        if model_z <= cfg.target_z_mm + 0.001:
            state.armed = False
            state.low_z = z
            state.offset = 0.0
            self._event("target_already_satisfied", **details)
            return BellStep(target, state, self.row)
        bias, reason = solver.solve_joint2(target[self.joints], cfg.target_z_mm, self.helper.side)
        if reason is None and not solver._points_down(solver._corrected_tilt(target[self.joints], bias)):
            reason = "corrected_orientation_not_downward"
        if reason:
            self._event("not_triggered", reason=reason, **details)
            return BellStep(target, state, self.row)

        # Prepare around the selected executable policy pose. Other arm
        # joints follow that normal pose; only J2 gets an extra correction.
        state.armed = False
        state.low_z = z
        state.anchor_z = model_z
        state.releasing = False
        closed_grip = min(float(previous[self.grip]), float(physical[self.grip]), float(target[self.grip]))
        prepared = []
        for index in range(cfg.ramp_steps):
            ramp_z = model_z + (index + 1) / cfg.ramp_steps * (cfg.target_z_mm - model_z)
            delta, reason = solver.solve_joint2(target[self.joints], ramp_z, self.helper.side)
            if reason or not solver._points_down(solver._corrected_tilt(target[self.joints], delta)):
                self._event("not_triggered", reason=reason or "corrected_orientation_not_downward", **details)
                return BellStep(self.actions[self.row].copy(), copy.deepcopy(self.helper.state), self.row)
            action = np.asarray(previous, dtype=np.float32).copy()
            action[self.joints] = target[self.joints]
            action[self.j2] += delta
            action[self.grip] = closed_grip
            snapshot = copy.deepcopy(state)
            snapshot.offset = delta
            prepared.append(BellStep(action, snapshot, self.row, "bell_press_down"))
        prepared = self._limit_descent_speed(prepared, physical)
        down_steps = len(prepared)
        for _ in range(int(math.ceil(cfg.hold_seconds / control_timing.CONTROL_DT))):
            step = copy.deepcopy(prepared[-1])
            step.kind = "bell_press_hold"
            prepared.append(step)
        self.inserted = prepared
        self.inserted_index = 0
        self._event("planned_press", j2_offset_rad=bias, down_steps=down_steps,
                    down_seconds=down_steps * control_timing.CONTROL_DT, down_speed_mm_s=cfg.down_speed_mm_s, **details)
        return prepared[0]

    def commit(self, step):
        # Advance only after sending and retain the state across inferences.
        self.helper.state = copy.deepcopy(step.state)
        if step.kind == "model":
            self.row += 1
        else:
            if self.inserted_index == 0:
                self._event("press_command_sent")
            self.inserted_index += 1
            if self.inserted_index == len(self.inserted):
                self._event("hold_complete")


@dataclass
class DualBellStep:
    action: np.ndarray
    source_row: int
    kind: str
    steps: dict


class DualBellPress:
    def __init__(self, configs, fk):
        self.arms = {side: BellPress(config, fk, side=side) for side, config in configs.items()}

    def settings(self):
        # Preserve the existing right-only metadata layout for old consumers.
        if set(self.arms) == {1}:
            return self.arms[1].settings()
        return {arm.name: arm.settings() for arm in self.arms.values()}


class DualBellPressRuntime:
    """Both arms share model time; each owns its press and rearm state.

    During extra press frames, advance only arms sending inserted commands.
    All other arms hold their last command. Never consume a model row until
    both arms have completed any inserted ramp/hold frames at that row.
    """

    def __init__(self, helper, actions, *, lookahead=None):
        self.actions = np.asarray(actions, dtype=np.float32).copy()
        self.arms = {side: BellPressRuntime(arm, actions, lookahead=lookahead)
                     for side, arm in helper.arms.items()}
        self.row = 0
        self.events = []
        self.event_counts = {side: 0 for side in self.arms}

    def _collect_events(self):
        for side, runtime in self.arms.items():
            self.events.extend(runtime.events[self.event_counts[side]:])
            self.event_counts[side] = len(runtime.events)

    def next(self, physical, previous):
        if self.row >= len(self.actions):
            return None
        proposals = {side: runtime.next(physical, previous) for side, runtime in self.arms.items()}
        inserted = {side: step for side, step in proposals.items() if step.kind != "model"}
        selected = inserted or proposals
        action = np.asarray(previous, dtype=np.float32).copy() if inserted else self.actions[self.row].copy()
        for side, step in selected.items():
            action[side * 7:side * 7 + 7] = step.action[side * 7:side * 7 + 7]
        kind = "model"
        if inserted:
            kind = "bell_press_down" if any(s.kind == "bell_press_down" for s in inserted.values()) else "bell_press_hold"
        self._collect_events()
        return DualBellStep(action, self.row, kind, selected)

    def commit(self, step):
        for side, arm_step in step.steps.items():
            self.arms[side].commit(arm_step)
        if step.kind == "model":
            self.row += 1
        self._collect_events()


def add_arguments(parser):
    import argparse

    parser.add_argument("--bell-press", action=argparse.BooleanOptionalAction, default=False,
                        help="closed-gripper downward press assistance for click_bell only")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-bell-press-z-range-mm", type=float, nargs=2, metavar=("MIN", "MAX"))
        parser.add_argument(f"--{side}-bell-press-target-z-mm", type=float)
    parser.add_argument("--bell-press-max-tilt-deg", type=float, default=20.0)
    parser.add_argument("--bell-press-hold-seconds", type=float, default=0.2)
    parser.add_argument("--bell-press-rearm-rise-mm", type=float, default=20.0)
    parser.add_argument("--bell-press-max-drop-mm", type=float, default=3.0)
    parser.add_argument("--bell-press-down-speed-mm-s", type=float, default=100.0,
                        help="maximum commanded Z speed during inserted press descent (mm/s)")
    parser.add_argument("--bell-press-closed-threshold", type=float, default=0.1)


def config_from_args(args):
    if not args.bell_press:
        return None
    if args.task != "click_bell":
        raise ValueError("--bell-press requires --task click_bell")
    if args.grasp_dip:
        raise ValueError("--bell-press and --grasp-dip cannot be enabled together")
    configs = {}
    for side, name in enumerate(("left", "right")):
        limits = getattr(args, f"{name}_bell_press_z_range_mm")
        target = getattr(args, f"{name}_bell_press_target_z_mm")
        if limits is None and target is None:
            continue
        if limits is None or target is None:
            raise ValueError(f"--{name}-bell-press-z-range-mm and --{name}-bell-press-target-z-mm must be provided together")
        configs[side] = BellPressConfig(
            z_range_mm=tuple(limits), target_z_mm=target,
            max_tilt_deg=args.bell_press_max_tilt_deg, hold_seconds=args.bell_press_hold_seconds,
            rearm_rise_mm=args.bell_press_rearm_rise_mm, max_drop_mm=args.bell_press_max_drop_mm,
            closed_threshold=args.bell_press_closed_threshold,
            down_speed_mm_s=args.bell_press_down_speed_mm_s)
    if not configs:
        raise ValueError("--bell-press requires a target Z and Z range for at least one arm")
    return configs
