"""Keep donor and receiver grippers closed until deliberate release."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass

import numpy as np

from .handover import gripper_cap_from_args


@dataclass(frozen=True)
class GripHoldConfig:
    closed_threshold: float = .25
    open_threshold: float = .35
    close_delta: float = .05
    lift_mm: float = 15.
    stable_frames: int = 3
    stable_tolerance: float = .01
    release_delta: float = .15
    release_frames: int = 3
    command_cap: float = 0.
    model_release_enabled: bool = True
    prelift_compensation: bool = False
    min_contact_gap: float = 0.

    def __post_init__(self):
        values = [self.closed_threshold, self.open_threshold, self.close_delta,
                  self.stable_tolerance, self.release_delta]
        if not np.isfinite(values).all() or not all(0 <= x <= 1 for x in values):
            raise ValueError("right grip hold gripper values must be finite and in [0, 1]")
        if not self.closed_threshold < self.open_threshold or self.close_delta <= 0 or self.release_delta <= 0:
            raise ValueError("right grip hold requires closed < open and positive close/release deltas")
        if not np.isfinite(self.lift_mm) or self.lift_mm <= 0 or min(self.stable_frames, self.release_frames) < 1:
            raise ValueError("right grip hold lift and frame counts must be positive")
        if not np.isfinite(self.command_cap) or not 0 <= self.command_cap <= self.closed_threshold:
            raise ValueError("right grip hold command cap must be within closed threshold")
        if not np.isfinite(self.min_contact_gap) or not 0 <= self.min_contact_gap < self.closed_threshold:
            raise ValueError("right grip hold minimum contact gap must be below closed threshold")


@dataclass
class GripHoldState:
    phase: str = "idle"
    peak_open: float = 0.
    saw_open: bool = False
    close_z: float | None = None
    baseline: float | None = None
    samples: tuple[float, ...] = ()
    release_count: int = 0
    last_token: tuple[int, int] | None = None


@dataclass
class GripHoldStep:
    force_close: bool
    state: GripHoldState
    record: dict


class RightGripHold:
    def __init__(self, config, fk):
        self.config = config
        self.fk = fk
        self.state = GripHoldState()

    def settings(self):
        return asdict(self.config)

    def next(self, physical, previous, model_grip, *, token, handover=None, allow_close=True):
        """Speculate once per send; model_grip is the original scheduled intent.

        Repeated preparation rows may update physical evidence but cannot count
        the same model row repeatedly toward deliberate release.
        """
        cfg = self.config
        state = copy.deepcopy(self.state)
        grip = float(physical[13])
        z = float(self.fk(physical[7:13])[2])
        if not np.isfinite([grip, z, model_grip, previous[13]]).all():
            raise ValueError("right grip hold needs finite feedback and model intent")
        events = []
        phase = "idle" if handover is None else handover["phase"]
        in_region = False if handover is None else handover["in_handover_region"]

        # Handover owns release, including the current donor-opening command.
        if phase in ("releasing", "completed"):
            if state.phase not in ("released", "idle"):
                events.append("released_by_handover")
            state = GripHoldState(phase="released")
        elif state.phase == "released":
            # Do not relatch on delayed closed feedback after releasing.
            if grip >= cfg.open_threshold:
                state = GripHoldState(saw_open=True, peak_open=grip)
                events.append("rearmed_after_open")
        elif state.phase != "holding":
            if grip >= cfg.open_threshold:
                if state.phase == "closing":
                    events.append("closure_cancelled")
                state = GripHoldState(saw_open=True, peak_open=max(state.peak_open, grip))
            elif (allow_close and state.saw_open and state.peak_open - grip >= cfg.close_delta
                  and grip <= cfg.closed_threshold and previous[13] <= cfg.closed_threshold):
                if state.phase != "closing":
                    state.phase = "closing"
                    state.close_z = z
                    events.append("closed_waiting_for_lift")
                state.close_z = min(state.close_z, z)
                state.samples = (*state.samples, grip)[-cfg.stable_frames:]
                stable = len(state.samples) == cfg.stable_frames and np.ptp(state.samples) <= cfg.stable_tolerance
                contact = grip - cfg.command_cap >= cfg.min_contact_gap
                if stable and not contact and previous[13] <= cfg.command_cap + cfg.stable_tolerance:
                    state.phase = "released"
                    events.append("empty_grasp")
                elif stable and contact and z - state.close_z >= cfg.lift_mm:
                    state.phase = "holding"
                    state.baseline = grip
                    events.append("lift_confirmed_holding")
            else:
                state.samples = ()
                if state.phase == "closing":
                    state.phase = "idle"
                    state.close_z = None

        if state.phase == "holding" and cfg.min_contact_gap > 0:
            state.samples = (*state.samples, grip)[-cfg.stable_frames:]
            if (len(state.samples) == cfg.stable_frames
                    and max(state.samples) - cfg.command_cap < cfg.min_contact_gap
                    and previous[13] <= cfg.command_cap + cfg.stable_tolerance):
                state.phase = "released"
                events.append("lost_contact")

        if state.phase == "holding" and token != state.last_token:
            state.last_token = token
            opening = model_grip >= cfg.open_threshold and model_grip - state.baseline >= cfg.release_delta
            # Within handover, only receiver confirmation may unlock the donor.
            if not cfg.model_release_enabled or in_region or phase in ("waiting", "ready"):
                state.release_count = 0
            else:
                state.release_count = state.release_count + 1 if opening else 0
                if state.release_count >= cfg.release_frames:
                    state.phase = "released"
                    events.append("released_by_model")

        force_close = state.phase == "holding" or (cfg.prelift_compensation and state.phase == "closing")
        record = dict(phase=state.phase, force_close=force_close, feedback=grip,
                      hold_command=min(float(model_grip), cfg.command_cap) if force_close else None,
                      model_grip=float(model_grip), z_mm=z, close_z_mm=state.close_z,
                      baseline=state.baseline, release_count=state.release_count,
                      events=events)
        return GripHoldStep(force_close, state, record)

    def commit(self, step):
        self.state = copy.deepcopy(step.state)


class TableLeftGripHold(RightGripHold):
    """Use the same close/lift latch for the left arm in table tasks."""

    def next(self, physical, model_grip, *, token, handover=None, previous=None):
        mapped = np.asarray(physical, dtype=np.float32).copy()
        mapped[7:13] = mapped[:6]
        mapped[13] = mapped[6]
        prior = mapped.copy() if previous is None else np.asarray(previous, dtype=np.float32).copy()
        prior[7:13] = prior[:6]
        prior[13] = prior[6]
        return super().next(mapped, prior, model_grip, token=token, handover=handover)


@dataclass(frozen=True)
class LeftGripHoldConfig:
    open_threshold: float = .35
    release_delta: float = .15
    release_frames: int = 3
    command_cap: float = 0.

    def __post_init__(self):
        values = [self.open_threshold, self.release_delta]
        if not np.isfinite(values).all() or not all(0 < x <= 1 for x in values):
            raise ValueError("left grip hold open threshold and release delta must be finite and in (0, 1]")
        if self.release_frames < 1:
            raise ValueError("left grip hold release frames must be positive")
        if not np.isfinite(self.command_cap) or not 0 <= self.command_cap < self.open_threshold:
            raise ValueError("left grip hold command cap must be below open threshold")


class LeftGripHold:
    """Latch on an actual handover completion event, not proximity or phase alone."""

    def __init__(self, config):
        self.config = config
        self.state = GripHoldState()

    def settings(self):
        return asdict(self.config)

    def next(self, physical, model_grip, *, token, handover):
        state = copy.deepcopy(self.state)
        cfg = self.config
        grip = float(physical[6])
        if not np.isfinite([grip, model_grip]).all():
            raise ValueError("left grip hold needs finite feedback and model intent")
        events = []
        if "handover_completed" in handover["events"]:
            # Start with a closed completion frame. Old pre-handover predictions
            # cannot accumulate release evidence during donor release.
            state = GripHoldState(phase="holding", baseline=grip, last_token=token)
            events.append("handover_completed_holding")
        elif state.phase == "holding":
            if handover["phase"] in ("waiting", "ready", "releasing"):
                state.release_count = 0
                state.last_token = token
            elif token != state.last_token:
                state.last_token = token
                opening = model_grip >= cfg.open_threshold and model_grip - state.baseline >= cfg.release_delta
                state.release_count = state.release_count + 1 if opening else 0
                if state.release_count >= cfg.release_frames:
                    state.phase = "released"
                    events.append("released_by_model")
        # After releasing, only a new completed handover can relatch the left.
        force_close = state.phase == "holding"
        record = dict(phase=state.phase, force_close=force_close, feedback=grip,
                      hold_command=min(float(model_grip), cfg.command_cap) if force_close else None,
                      model_grip=float(model_grip), baseline=state.baseline,
                      release_count=state.release_count, events=events)
        return GripHoldStep(force_close, state, record)

    def commit(self, step):
        self.state = copy.deepcopy(step.state)


@dataclass(frozen=True)
class BasketLeftGripConfig:
    close_threshold: float = .55
    close_delta: float = .15
    open_threshold: float = .70
    release_delta: float = .15
    release_frames: int = 3
    command_cap: float = .35

    def __post_init__(self):
        values = (self.close_threshold, self.close_delta, self.open_threshold,
                  self.release_delta, self.command_cap)
        if not np.isfinite(values).all() or not (0 <= self.command_cap < self.close_threshold < self.open_threshold <= 1):
            raise ValueError("basket grip requires finite cap < close threshold < open threshold <= 1")
        if not (0 < self.close_delta <= 1 and 0 < self.release_delta <= 1) or self.release_frames < 1:
            raise ValueError("basket grip deltas and release frames must be positive")


class BasketLeftGripHold:
    """Latch a clear model closing transition, then require deliberate reopening."""

    grip_index = 6

    def __init__(self, config):
        self.config = config
        self.state = GripHoldState()

    def settings(self):
        return asdict(self.config)

    def next(self, physical, model_grip, *, token, handover=None):
        state = copy.deepcopy(self.state)
        cfg = self.config
        grip = float(physical[self.grip_index])
        model_grip = float(model_grip)
        if not np.isfinite([grip, model_grip]).all():
            raise ValueError("basket grip needs finite feedback and model intent")
        events = []
        if state.phase == "holding":
            if token != state.last_token:
                state.last_token = token
                opening = (model_grip >= cfg.open_threshold
                           and model_grip - state.baseline >= cfg.release_delta)
                state.release_count = state.release_count + 1 if opening else 0
                if state.release_count >= cfg.release_frames:
                    state.phase = "released"
                    events.append("released_by_model")
        else:
            if grip >= cfg.open_threshold:
                state.saw_open = True
                state.peak_open = max(state.peak_open, grip)
            if model_grip >= cfg.open_threshold:
                if state.phase == "released" and grip >= cfg.open_threshold:
                    state = GripHoldState(saw_open=True, peak_open=grip)
                    events.append("rearmed_after_open")
                elif state.phase == "idle":
                    state.saw_open = True
                    state.peak_open = max(state.peak_open, model_grip)
            if (state.phase == "idle" and state.saw_open and token != state.last_token
                    and model_grip <= cfg.close_threshold
                    and state.peak_open - model_grip >= cfg.close_delta):
                state.phase = "holding"
                state.baseline = model_grip
                state.last_token = token
                events.append("closing_trend_holding")
        force_close = state.phase == "holding"
        record = dict(phase=state.phase, force_close=force_close, feedback=grip,
                      hold_command=min(model_grip, cfg.command_cap) if force_close else None,
                      model_grip=model_grip, baseline=state.baseline,
                      release_count=state.release_count, events=events)
        return GripHoldStep(force_close, state, record)

    def commit(self, step):
        self.state = copy.deepcopy(step.state)


class BasketRightGripHold(BasketLeftGripHold):
    """Same trend latch with the right-arm monitor call signature."""

    grip_index = 13

    def next(self, physical, previous, model_grip, *, token, handover=None):
        return super().next(physical, model_grip, token=token, handover=handover)


@dataclass(frozen=True)
class AssemblyGripConfig:
    close_threshold: float = .4
    open_threshold: float = .6
    command: float = .2
    release_delta: float = .15
    release_frames: int = 3

    def __post_init__(self):
        values = (self.close_threshold, self.open_threshold, self.command, self.release_delta)
        if not np.isfinite(values).all() or not (0 <= self.command < self.close_threshold < self.open_threshold <= 1):
            raise ValueError("assembly grip requires command < close threshold < open threshold <= 1")
        if not 0 < self.release_delta <= 1 or self.release_frames < 1:
            raise ValueError("assembly grip release delta and frame count must be positive")


class AssemblyLeftGripHold:
    grip_index = 6

    def __init__(self, config):
        self.config = config
        self.state = GripHoldState()

    def settings(self):
        return asdict(self.config)

    def next(self, physical, model_grip, *, token, handover=None, allow_close=True):
        state = copy.deepcopy(self.state)
        cfg = self.config
        grip = float(physical[self.grip_index])
        model_grip = float(model_grip)
        if not np.isfinite([grip, model_grip]).all():
            raise ValueError("assembly grip needs finite feedback and model intent")
        events = []
        if state.phase == "holding":
            if allow_close and token != state.last_token:
                state.last_token = token
                opening = model_grip >= cfg.open_threshold and model_grip - state.baseline >= cfg.release_delta
                state.release_count = state.release_count + 1 if opening else 0
                if state.release_count >= cfg.release_frames:
                    state.phase = "released"
                    events.append("released_by_model")
        else:
            if grip >= cfg.open_threshold:
                state.saw_open = True
                state.peak_open = max(state.peak_open, grip)
            if allow_close and model_grip >= cfg.open_threshold:
                if state.phase == "released" and grip >= cfg.open_threshold:
                    state = GripHoldState(saw_open=True, peak_open=grip)
                    events.append("rearmed_after_open")
                elif state.phase == "idle":
                    state.saw_open = True
                    state.peak_open = max(state.peak_open, model_grip)
            if (allow_close and state.phase == "idle" and state.saw_open and token != state.last_token
                    and model_grip <= cfg.close_threshold and state.peak_open - model_grip >= cfg.release_delta):
                state.phase = "holding"
                state.baseline = model_grip
                state.last_token = token
                events.append("closing_trend_holding")
        force_close = state.phase == "holding"
        record = dict(phase=state.phase, force_close=force_close, feedback=grip,
                      hold_command=cfg.command if force_close else None,
                      model_grip=model_grip, baseline=state.baseline,
                      release_count=state.release_count, events=events)
        return GripHoldStep(force_close, state, record)

    def commit(self, step):
        self.state = copy.deepcopy(step.state)


class AssemblyRightGripHold(AssemblyLeftGripHold):
    grip_index = 13

    def next(self, physical, previous, model_grip, *, token, handover=None, allow_close=True):
        return super().next(physical, model_grip, token=token, allow_close=allow_close)


def add_arguments(parser):
    parser.add_argument("--right-grip-hold", action=argparse.BooleanOptionalAction, default=None,
                        help="hold right gripper after closure and lift; defaults on with handover assist, explicit for manipulate_pipette")
    defaults = GripHoldConfig()
    for name in ("closed_threshold", "open_threshold", "close_delta", "lift_mm", "stable_frames",
                 "stable_tolerance", "release_delta", "release_frames"):
        parser.add_argument("--right-grip-hold-" + name.replace("_", "-"),
                            type=int if name.endswith("frames") else float,
                            default=None if name in ("closed_threshold", "open_threshold") else getattr(defaults, name))
    parser.add_argument("--left-grip-hold", action=argparse.BooleanOptionalAction, default=None,
                        help="hold left gripper after handover; defaults on with handover assist")
    for name, value in asdict(LeftGripHoldConfig()).items():
        if name == "command_cap":
            continue
        parser.add_argument("--left-grip-hold-" + name.replace("_", "-"),
                            type=int if name.endswith("frames") else float, default=value)
    for side in ("left", "right"):
        parser.add_argument(f"--basket-{side}-grip-hold", action=argparse.BooleanOptionalAction, default=False,
                            help=f"explicitly enable {side} gripper trend hold for handle_basket")
        for name, value in asdict(BasketLeftGripConfig()).items():
            if side == "right" and name == "command_cap":
                value = 0.0
            parser.add_argument(f"--basket-{side}-grip-" + name.replace("_", "-"),
                                type=int if name.endswith("frames") else float, default=value)
    parser.add_argument("--assembly-grip-command", type=float, default=.2)
    parser.add_argument("--assembly-grip-release-delta", type=float, default=.15)
    parser.add_argument("--assembly-grip-release-frames", type=int, default=3)


def assembly_config_from_args(args):
    if not args.assembly_grasp_assist:
        return None
    return AssemblyGripConfig(close_threshold=args.grasp_dip_close_threshold,
                              open_threshold=args.grasp_dip_open_threshold,
                              command=args.assembly_grip_command,
                              release_delta=args.assembly_grip_release_delta,
                              release_frames=args.assembly_grip_release_frames)


def config_from_args(args):
    if args.basket_right_grip_hold:
        if args.task != "handle_basket":
            raise ValueError("--basket-right-grip-hold requires --task handle_basket")
        if args.right_grip_hold:
            raise ValueError("--basket-right-grip-hold and --right-grip-hold cannot both be enabled")
        return BasketLeftGripConfig(**{name: getattr(args, "basket_right_grip_" + name)
                                       for name in asdict(BasketLeftGripConfig())})
    enabled = args.handover_assist if args.right_grip_hold is None else args.right_grip_hold
    if not enabled:
        return None
    if args.task not in ("items_handover", "sample_loading", "manipulate_pipette", "table_rearrangement", "mixer_operating") or (args.task not in ("manipulate_pipette", "table_rearrangement", "mixer_operating") and not args.handover_assist):
        raise ValueError("--right-grip-hold requires manipulate_pipette/table_rearrangement or handover assist for items_handover/sample_loading")
    values = {name: getattr(args, "right_grip_hold_" + name) for name in asdict(GripHoldConfig())
              if name not in ("command_cap", "model_release_enabled", "prelift_compensation", "min_contact_gap")}
    if values["closed_threshold"] is None:
        values["closed_threshold"] = .45 if args.task == "manipulate_pipette" else .30 if args.task == "sample_loading" else .25
    if values["open_threshold"] is None:
        values["open_threshold"] = .55 if args.task == "manipulate_pipette" else .35
    values["command_cap"] = 0. if args.task == "manipulate_pipette" else gripper_cap_from_args(args) or 0.
    values["model_release_enabled"] = args.task != "manipulate_pipette" and not getattr(args, "relative_handover", False)
    values["prelift_compensation"] = args.task in ("sample_loading", "manipulate_pipette")
    values["min_contact_gap"] = .02 if args.task == "manipulate_pipette" else 0.
    return GripHoldConfig(**values)


def left_config_from_args(args):
    if args.basket_left_grip_hold:
        if args.task != "handle_basket":
            raise ValueError("--basket-left-grip-hold requires --task handle_basket")
        if args.left_grip_hold:
            raise ValueError("--basket-left-grip-hold and --left-grip-hold cannot both be enabled")
        return BasketLeftGripConfig(**{name: getattr(args, "basket_left_grip_" + name)
                                       for name in asdict(BasketLeftGripConfig())})
    enabled = args.handover_assist if args.left_grip_hold is None else args.left_grip_hold
    if not enabled:
        return None
    if args.task not in ("items_handover", "sample_loading", "table_rearrangement") or (args.task in ("items_handover", "sample_loading") and not args.handover_assist):
        raise ValueError("--left-grip-hold requires table_rearrangement, or items_handover/sample_loading with --handover-assist")
    if args.task == "table_rearrangement":
        values = {name: getattr(args, "right_grip_hold_" + name)
                  for name in asdict(GripHoldConfig())
                  if name not in ("command_cap", "model_release_enabled", "prelift_compensation", "min_contact_gap")}
        if values["closed_threshold"] is None:
            values["closed_threshold"] = .25
        if values["open_threshold"] is None:
            values["open_threshold"] = .35
        values.update(command_cap=0., model_release_enabled=True, prelift_compensation=False, min_contact_gap=0.)
        return GripHoldConfig(**values)
    values = {name: getattr(args, "left_grip_hold_" + name) for name in asdict(LeftGripHoldConfig()) if name != "command_cap"}
    values["command_cap"] = gripper_cap_from_args(args) or 0.
    return LeftGripHoldConfig(**values)
