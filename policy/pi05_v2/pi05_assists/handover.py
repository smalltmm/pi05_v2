"""Right-to-left handover guard with continued inference while holding pose."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np


def rotation(rpy_deg):
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def transform(pose):
    result = np.eye(4)
    result[:3, :3] = rotation(pose[3:])
    result[:3, 3] = pose[:3]
    return result


@dataclass(frozen=True)
class HandoverConfig:
    right_base_xyz_mm: tuple[float, float, float] = (0., -600., 0.)
    right_base_rpy_deg: tuple[float, float, float] = (0., 0., 0.)
    relative_rpy_deg: tuple[float, float, float] = (-147., -9., 5.)
    distance_mm: float = 220.0
    angle_deg: float = 30.0
    rearm_distance_mm: float = 300.0
    left_closed_threshold: float = 0.1
    right_closed_threshold: float = 0.1
    right_open_threshold: float = 0.2
    left_close_action: float = 0.0
    gripper_cap: float | None = None
    stable_frames: int = 3
    stable_tolerance: float = 0.01
    contact_confirmation: bool = True
    contact_gap: float = .05
    contact_frames: int = 5
    # Retained for CLI/log compatibility; neither gates contact confirmation.
    contact_close_delta: float = .03
    contact_settle_frames: int = 3
    contact_release_step: float = .05

    def __post_init__(self):
        for name in ("right_base_xyz_mm", "right_base_rpy_deg", "relative_rpy_deg"):
            value = np.asarray(getattr(self, name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"handover {name} must contain three finite values")
        if not np.isclose(np.linalg.norm(self.right_base_xyz_mm), 600., atol=1.):
            raise ValueError("handover right-base translation must have length 600 mm (tolerance 1 mm)")
        if not np.isfinite([self.distance_mm, self.angle_deg, self.rearm_distance_mm]).all():
            raise ValueError("handover distance/angle limits must be finite")
        if not 0 < self.distance_mm < self.rearm_distance_mm or not 0 <= self.angle_deg < 180:
            raise ValueError("handover requires 0 < distance < rearm-distance and angle in [0, 180)")
        values = [self.left_closed_threshold, self.right_closed_threshold, self.right_open_threshold,
                  self.left_close_action, self.stable_tolerance]
        if not np.isfinite(values).all() or not all(0 <= x <= 1 for x in values):
            raise ValueError("handover gripper parameters must be in [0, 1]")
        if self.right_closed_threshold >= self.right_open_threshold or self.left_close_action > self.left_closed_threshold:
            raise ValueError("handover gripper thresholds/close action are inconsistent")
        if self.gripper_cap is not None:
            if (not np.isfinite(self.gripper_cap) or not 0 <= self.gripper_cap <= 1
                    or self.gripper_cap > min(self.left_closed_threshold, self.right_closed_threshold)
                    or self.gripper_cap >= self.right_open_threshold):
                raise ValueError("handover gripper cap must be within closed thresholds and below open threshold")
        if self.stable_frames < 1:
            raise ValueError("--handover-stable-frames must be positive")
        if min(self.contact_frames, self.contact_settle_frames) < 1:
            raise ValueError("handover contact frame counts must be positive")
        if not all(np.isfinite(x) and 0 < x <= 1 for x in
                   (self.contact_gap, self.contact_close_delta, self.contact_release_step)):
            raise ValueError("handover contact gap, closing delta and release step must be in (0, 1]")


@dataclass
class HandoverState:
    phase: str = "idle"
    hold: np.ndarray | None = None
    samples: tuple[float, ...] = ()
    left_peak: float = 0.
    left_shrunk: bool = False
    left_stamp: float | None = None
    right_stamp: float | None = None
    last_left_command: float | None = None
    closing_frames: int = 0
    contact_samples: tuple[float, ...] = ()
    confirmation: str | None = None
    contact_reference: float | None = None


@dataclass
class HandoverStep:
    action: np.ndarray | None
    state: HandoverState
    kind: str
    record: dict


class HandoverAssist:
    def __init__(self, config, fk):
        self.config = config
        self.fk = fk
        self.base = transform([*config.right_base_xyz_mm, *config.right_base_rpy_deg])
        self.reference = rotation(config.relative_rpy_deg)
        self.state = HandoverState()

    def settings(self):
        return asdict(self.config)

    def poses(self, physical):
        physical = np.asarray(physical, dtype=float)
        if physical.shape != (14,) or not np.isfinite(physical).all():
            raise ValueError("handover requires finite physical state[14]")
        left_pose = np.asarray(self.fk.pose(physical[:6]), dtype=float)
        right_pose = np.asarray(self.fk.pose(physical[7:13]), dtype=float)
        if left_pose.shape != (6,) or right_pose.shape != (6,) or not np.isfinite([left_pose, right_pose]).all():
            raise ValueError("handover FK must return finite XYZ(mm), RPY(deg)")
        left = transform(left_pose)
        right = self.base @ transform(right_pose)
        relative = np.eye(4)
        relative[:3, :3] = left[:3, :3].T @ right[:3, :3]
        relative[:3, 3] = left[:3, :3].T @ (right[:3, 3] - left[:3, 3])
        error = self.reference.T @ relative[:3, :3]
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(error)-1)/2, -1, 1))))
        distance = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
        return dict(left_base_pose_xyz_mm_rpy_deg=left_pose.tolist(),
                    right_base_pose_xyz_mm_rpy_deg=right_pose.tolist(),
                    left_pose_in_left_base=left.tolist(), right_pose_in_left_base=right.tolist(),
                    right_pose_in_left_tool=relative.tolist(), distance_mm=distance,
                    relative_orientation_error_deg=angle,
                    in_handover_region=distance <= self.config.distance_mm and angle <= self.config.angle_deg)

    def begin_inference_result(self):
        # Ready is committed by a real feedback/control frame. Only a model
        # result from a subsequent inference can release the held pose.
        if self.state.phase == "ready":
            self.state.phase = "releasing"

    def next(self, physical, previous, remaining, *, feedback=None):
        cfg = self.config
        physical = np.asarray(physical, dtype=np.float32)
        previous = np.asarray(previous, dtype=np.float32)
        remaining = np.asarray(remaining, dtype=np.float32)
        if remaining.ndim != 2 or remaining.shape[1] != 14 or not len(remaining) or not np.isfinite(remaining).all():
            raise ValueError("handover needs a finite executable action chunk")
        if previous.shape != (14,) or not np.isfinite(previous).all():
            raise ValueError("handover needs the previous finite sent action")
        record = self.poses(physical)
        state = copy.deepcopy(self.state)
        events = []
        value = float(physical[6])
        if (state.phase == "completed" and record["distance_mm"] >= cfg.rearm_distance_mm
                and not getattr(self, "keep_completed", False)):
            state = HandoverState()
            events.append("rearmed")
        fresh, valid = {}, {}
        for side in ("left", "right"):
            info = (feedback or {}).get(side, {})
            stamp = info.get("stamp")
            last_stamp = getattr(state, side + "_stamp")
            valid[side] = bool(info.get("valid", False) and stamp is not None and np.isfinite(stamp)
                               and (last_stamp is None or stamp >= last_stamp))
            fresh[side] = valid[side] and (last_stamp is None or stamp > last_stamp)
            if fresh[side]:
                setattr(state, side + "_stamp", stamp)
        # Offline/legacy callers may still use the original threshold path.
        # The contact path always requires verified, advancing CAN feedback.
        threshold_fresh = feedback is None or fresh["left"]
        if value > cfg.left_closed_threshold or (feedback is not None and not valid["left"]):
            state.samples = ()
        elif threshold_fresh:
            state.samples = (*state.samples, value)[-cfg.stable_frames:]
        threshold_stable = len(state.samples) == cfg.stable_frames and np.ptp(state.samples) <= cfg.stable_tolerance
        command = float(previous[6])
        gap = value - command
        if fresh["left"]:
            state.left_peak = max(state.left_peak, value)
            if gap >= cfg.contact_gap and state.left_peak - value >= cfg.contact_close_delta:
                state.left_shrunk = True
        closing = valid["left"] and command <= cfg.left_closed_threshold and gap >= cfg.contact_gap
        if not closing:
            state.closing_frames = 0
        elif fresh["left"]:
            unchanged = state.last_left_command is not None and abs(command - state.last_left_command) <= cfg.stable_tolerance
            state.closing_frames = state.closing_frames + 1 if unchanged else 1
        state.last_left_command = command
        opening = bool(np.any(remaining[:, 13] >= cfg.right_open_threshold))
        if (state.phase == "idle" and record["in_handover_region"] and opening
                and physical[13] < cfg.right_open_threshold and previous[13] <= cfg.right_closed_threshold):
            state.phase = "waiting"
            state.hold = physical.copy()
            state.hold[6] = cfg.left_close_action
            state.hold[13] = min(float(previous[13]), float(physical[13]))
            events.append("waiting_for_left")
        # A closed command blocked at a stable opening is enough. Requiring
        # another shrink after a small rebound deadlocks an already held object.
        contact_ok = (cfg.contact_confirmation and state.phase in ("waiting", "ready", "releasing")
                      and closing)
        if not contact_ok:
            state.contact_samples = ()
        elif fresh["left"]:
            state.contact_samples = (*state.contact_samples, value)[-cfg.contact_frames:]
        # Re-reading a still-valid CAN sample neither counts nor erases evidence.
        contact_stable = (len(state.contact_samples) == cfg.contact_frames
                          and np.ptp(state.contact_samples) <= cfg.stable_tolerance)
        if state.contact_reference is not None and value > state.contact_reference + cfg.stable_tolerance:
            contact_stable = False
        stable = threshold_stable or contact_stable
        if state.phase == "waiting" and stable:
            state.phase = "ready"
            state.confirmation = "threshold" if threshold_stable else "contact"
            state.contact_reference = value if state.confirmation == "contact" else None
            events.append("left_closed_stable_waiting_for_fresh_inference" if threshold_stable
                          else "left_contact_stable_waiting_for_fresh_inference")
        action = None
        kind = "model"
        if state.phase == "ready" and state.confirmation == "contact" and not stable:
            state.phase = "waiting"
            state.contact_samples = ()
            if fresh["left"] and state.contact_reference is not None and value > state.contact_reference + cfg.stable_tolerance:
                state.left_peak, state.left_shrunk = value, False
            state.contact_reference = None
            events.append("left_contact_lost")
        if state.phase in ("waiting", "ready"):
            action = state.hold.copy()
            kind = "handover_hold"
        elif state.phase == "releasing":
            if not stable:
                # Receiver reopened before donor released: return to the
                # established handover pose and block release immediately.
                if physical[13] < cfg.right_open_threshold or state.confirmation == "contact":
                    state.phase = "waiting"
                    action = state.hold.copy()
                    kind = "handover_hold"
                    events.append("left_closure_lost")
                    state.contact_samples = ()
                    if fresh["left"] and state.contact_reference is not None and value > state.contact_reference + cfg.stable_tolerance:
                        state.left_peak, state.left_shrunk = value, False
                    state.contact_reference = None
            if kind != "handover_hold":
                action = remaining[0].copy()
                action[6] = cfg.left_close_action
                kind = "handover_release"
                if state.confirmation == "contact":
                    # Transfer load while keeping both arm poses fixed. Do not
                    # jump to the full model opening at the old open threshold.
                    desired = float(remaining[0, 13])
                    action = state.hold.copy()
                    base = min(float(previous[13]), float(physical[13]))
                    action[13] = min(desired, base + (cfg.contact_release_step if fresh["right"] else 0.))
                    if fresh["right"]:
                        events.append("right_contact_release_step")
                    if (fresh["right"] and desired >= cfg.right_open_threshold
                            and physical[13] >= desired - cfg.stable_tolerance
                            and previous[13] >= desired - cfg.stable_tolerance):
                        state.phase = "completed"
                        events.append("handover_completed")
                elif physical[13] >= cfg.right_open_threshold:
                    state.phase = "completed"
                    events.append("handover_completed")
        if action is not None and cfg.gripper_cap is not None:
            # Keep joint pose latched, but evaluate each new model gripper row.
            action[6] = min(float(remaining[0, 6]), cfg.gripper_cap)
            if kind == "handover_hold":
                action[13] = min(float(remaining[0, 13]), cfg.gripper_cap)
        record.update(phase=state.phase, left_closed_stable=bool(stable),
                      left_threshold_stable=bool(threshold_stable), left_contact_stable=bool(contact_stable),
                      left_previous_action=command, left_action_state_gap=gap,
                      left_closing_frames=state.closing_frames, left_contact_frames=len(state.contact_samples),
                      left_observed_closing=state.left_shrunk, confirmation=state.confirmation,
                      left_feedback_fresh=fresh["left"], right_feedback_fresh=fresh["right"],
                      feedback=feedback,
                      left_gripper_feedback=value, right_gripper_feedback=float(physical[13]),
                      right_open_predicted=opening, gripper_cap=cfg.gripper_cap, events=events)
        return HandoverStep(action, state, kind, record)

    def commit(self, step):
        self.state = copy.deepcopy(step.state)


def add_arguments(parser):
    import argparse
    parser.add_argument("--handover-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--handover-right-base-xyz-mm", type=float, nargs=3, default=(0., -600., 0.), metavar=("X", "Y", "Z"),
                        help="right base origin expressed in left base coordinates; length 600 mm")
    parser.add_argument("--handover-right-base-rpy-deg", type=float, nargs=3, default=(0., 0., 0.), metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--handover-relative-rpy-deg", type=float, nargs=3, default=HandoverConfig().relative_rpy_deg, metavar=("ROLL", "PITCH", "YAW"),
                        help="reference right tool orientation expressed in left tool coordinates")
    parser.add_argument("--handover-distance-mm", type=float, default=None,
                        help="default: sample_loading 280 mm, items_handover 220 mm")
    parser.add_argument("--handover-angle-deg", type=float, default=None,
                        help="default: sample_loading 25 degrees, items_handover 30 degrees")
    parser.add_argument("--handover-rearm-distance-mm", type=float, default=300.)
    parser.add_argument("--handover-left-closed-threshold", type=float, default=None)
    parser.add_argument("--handover-right-closed-threshold", type=float, default=None)
    parser.add_argument("--handover-right-open-threshold", type=float, default=None)
    parser.add_argument("--handover-left-close-action", type=float, default=0.)
    parser.add_argument("--handover-gripper-cap", type=float, default=None,
                        help="use min(compensated model, cap) for both grip holds and handover; sample_loading default .25")
    parser.add_argument("--handover-stable-frames", type=int, default=3)
    parser.add_argument("--handover-stable-tolerance", type=float, default=.01)
    parser.add_argument("--handover-contact-confirmation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--handover-contact-gap", type=float, default=.05)
    parser.add_argument("--handover-contact-frames", type=int, default=5)
    parser.add_argument("--handover-contact-close-delta", type=float, default=.03,
                        help="legacy closing diagnostic threshold; no longer required for contact confirmation")
    parser.add_argument("--handover-contact-settle-frames", type=int, default=3,
                        help="legacy compatibility option; contact confirmation uses one contact-frames window")
    parser.add_argument("--handover-contact-release-step", type=float, default=.05)


def gripper_cap_from_args(args):
    return .25 if args.task == "sample_loading" and args.handover_gripper_cap is None else args.handover_gripper_cap


def config_from_args(args):
    if not args.handover_assist:
        return None
    if args.task not in ("items_handover", "sample_loading"):
        raise ValueError("--handover-assist requires --task items_handover or sample_loading")
    for name in ("right_base_xyz_mm", "right_base_rpy_deg", "relative_rpy_deg"):
        if getattr(args, "handover_" + name) is None:
            raise ValueError("--handover-assist requires --handover-" + name.replace("_", "-"))
    defaults = vars(HandoverConfig()).copy()
    if args.task == "sample_loading":
        defaults.update(distance_mm=280., angle_deg=25., left_closed_threshold=.30,
                        right_closed_threshold=.30, right_open_threshold=.35)
    values = {name: getattr(args, "handover_" + name) for name in defaults}
    values = {name: defaults[name] if value is None else value for name, value in values.items()}
    values["gripper_cap"] = gripper_cap_from_args(args)
    return HandoverConfig(**values)
