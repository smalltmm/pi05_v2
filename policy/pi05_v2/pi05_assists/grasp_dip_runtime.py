"""Feedback-gated preparation for closures missed at an inference boundary."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from .grasp_dip import DipPlan, GraspDip


@dataclass
class DipStep:
    plan: DipPlan
    index: int
    source_row: int
    kind: str = "model"

    @property
    def action(self):
        return self.plan.actions[self.index]


class GraspDipRuntime:
    """Insert recorded 10 Hz preparation rows without consuming model rows.

    Only commit() advances a step. A prediction beyond the executed prefix is
    a hint, never an instruction to execute that discarded trajectory.
    """

    SETTLE_TOLERANCE_MM = 2.0
    MAX_SETTLE_STEPS = 20

    def __init__(self, helper: GraspDip, actions, initial_plan: DipPlan, *, lookahead=None,
                 preclose_config=None, clock=time.monotonic, control_hz=10.0):
        self.helper = helper
        self.actions = np.asarray(actions, dtype=np.float32).copy()
        self.lookahead = np.empty((0, 14), dtype=np.float32) if lookahead is None else np.asarray(lookahead, dtype=np.float32)
        self.plan = initial_plan
        self.plan_start = 0
        self.row = 0
        self.preparation = None
        self.prep_row = 0
        self.settle_steps = 0
        self.preparing_sides = []
        self.events = []
        self._reported = set()
        self.fallback = False
        self.preclose_config = preclose_config
        self.clock = clock
        self.control_hz = float(control_hz)
        self.close_plan = None
        self.close_started = None
        self.close_samples = ()
        self.close_stamp = None
        self.close_pending_sample = None
        self.close_peak = None
        self.close_target = None
        self.dwell_started = None

    def _event(self, event):
        key = (self.row, event["side"], event["status"], event.get("reason"))
        if key not in self._reported:
            self.events.append(dict(event, source_row=self.row))
            self._reported.add(key)

    def _prepare(self, physical, previous):
        if self.helper.config.staged_vertical:
            from .grasp_dip_vertical import prepare
            return prepare(self, physical, previous)
        h = self.helper
        cfg = h.config
        remaining = self.actions[self.row:]
        # Hold the other arm at its last commanded position during added rows.
        rows = np.repeat(previous[None, :], cfg.pre_steps, axis=0)
        states = [copy.deepcopy(h._states) for _ in rows]
        selected = []
        events = []
        for side, start in enumerate((0, 7)):
            target_z = cfg.target_z_mm[side]
            arm = h._states[side]
            if target_z is None or arm.phase != "idle":
                continue
            armed = arm.armed or previous[start + 6] > cfg.close_threshold
            if not armed or physical[start + 6] <= cfg.close_threshold:
                continue  # Do not reopen a gripper that is already closed.
            closes = np.flatnonzero(remaining[:, start + 6] <= np.float32(cfg.close_threshold))
            if len(closes):
                close = int(closes[0])
                if close > cfg.pre_steps:
                    continue
                candidate = remaining[close]
                # Existing normal pre-close ramp already covers this transition.
                if not (side == 1 and self.preclose_config is not None) and not cfg.forward_mm[side] and any(e["side"] == ("left", "right")[side] and e["status"] == "applied"
                       and e.get("close_row") == self.row - self.plan_start + close
                       and e["start_row"] >= self.row - self.plan_start for e in self.plan.events):
                    continue
                source = "executed_prefix"
            else:
                closes = np.flatnonzero(self.lookahead[:, start + 6] <= np.float32(cfg.close_threshold))
                if not len(closes) or len(remaining) > cfg.pre_steps:
                    continue
                # Require an actual closing trend, not a distant static prediction.
                if remaining[0, start + 6] >= previous[start + 6] - 1e-4:
                    continue
                candidate = self.lookahead[int(closes[0])]
                source = "prediction_tail"
            limits = cfg.z_ranges[side]
            xyz = h._xyz(physical[start:start + 6])
            close_xyz = h._xyz(candidate[start:start + 6])
            if not limits[0] <= xyz[2] <= limits[1]:
                continue  # Recheck using fresh feedback before the next row.
            event = {"side": ("left", "right")[side], "status": "skipped", "start_row": 0,
                     "reason": None, "target_z_mm": target_z, "physical_z_mm": float(xyz[2]),
                     "close_z_mm": float(close_xyz[2]), "trigger_source": source}
            reason = None
            if not limits[0] <= close_xyz[2] <= limits[1]:
                reason = "close_height_outside_range"
            elif not h._points_down(h._tilt(physical[start:start + 6])):
                reason = "physical_orientation_not_downward"
            elif not h._points_down(h._tilt(candidate[start:start + 6])):
                reason = "close_orientation_not_downward"
            else:
                close_bias, reason = h.solve_correction(candidate[start:start + 6], target_z, side)
                if reason is None and not h._points_down(h._corrected_tilt(candidate[start:start + 6], close_bias)):
                    reason = "corrected_orientation_not_downward"
            side_rows = []
            if reason is None:
                for j in range(cfg.pre_steps):
                    q = physical[start:start + 6].copy()
                    ramp_z = float(xyz[2] + (j + 1) / cfg.pre_steps * (target_z - xyz[2]))
                    bias, reason = h.solve_correction(q, ramp_z, side,
                                                      target_x=float(xyz[0] + (j + 1) / cfg.pre_steps *
                                                                     (close_xyz[0] + cfg.forward_mm[side] - xyz[0]))
                                                      if cfg.forward_mm[side] else None,
                                                      fraction=(j + 1) / cfg.pre_steps)
                    if reason is not None:
                        break
                    q = h.apply_offset(q, bias)
                    if not h._points_down(h._tilt(q)):
                        reason = "corrected_orientation_not_downward"
                        break
                    side_rows.append((q, bias))
            if reason is not None:
                event["reason"] = reason
                self._event(event)
                continue
            event.update(status="applied", corrected_close_z_mm=target_z, j2_offset_rad=h.offset_event(close_bias),
                         forward_mm=cfg.forward_mm[side],
                         target_x_mm=float(close_xyz[0] + cfg.forward_mm[side]))
            for j, (q, bias) in enumerate(side_rows):
                rows[j, start:start + 6] = q
                # Preserve the more-open of feedback and the last command; do
                # not copy a predicted close into the preparation frames.
                rows[j, start + 6] = max(float(physical[start + 6]), float(previous[start + 6]))
                state = states[j][side]
                state.phase = "hold"
                state.armed = False
                state.closed = False
                state.prepared = True
                state.prepare_grip = float(rows[j, start + 6])
                state.offset = bias
                state.grasp_offset = close_bias
                state.grasp_z = float(close_xyz[2])
                state.target_x = float(close_xyz[0] + cfg.forward_mm[side])
            selected.append(side)
            if side == 1 and self.preclose_config is not None:
                self.close_target = min(float(candidate[13]), self.preclose_config.command_cap)
            events.append(event)
        if not selected:
            return None
        # Validate the resumed model trajectory before sending any inserted row.
        preview = GraspDip(cfg, h.fk, down_tilt=h.down_tilt)
        preview._states = copy.deepcopy(states[-1])
        preview.plan(remaining, previous_action=rows[-1], physical_state=rows[-1])
        self.preparing_sides = selected
        for event in events:
            self._event(event)
        return DipPlan(rows, states, events)

    def next(self, physical, previous, *, feedback=None):
        try:
            return self._next(physical, previous, feedback=feedback)
        except (ValueError, TimeoutError) as exc:
            # Keep the original scheduled rows; cancel only the failed assist.
            fresh = GraspDip(self.helper.config, self.helper.fk, down_tilt=self.helper.down_tilt)
            self.helper._states = fresh._states
            self.preparation = None
            self.fallback = True
            self.plan_start = 0
            self.plan = DipPlan(self.actions.copy(), [copy.deepcopy(fresh._states) for _ in self.actions], [])
            for side, target in enumerate(self.helper.config.target_z_mm):
                if target is not None:
                    self._event({"side": ("left", "right")[side], "status": "fallback", "start_row": self.row,
                                 "target_z_mm": target, "reason": str(exc)})
            return DipStep(self.plan, self.row, self.row)

    def _close_in_place(self, physical, previous, feedback):
        cfg = self.preclose_config
        if self.close_plan is None:
            target = previous.copy()
            target[7:13] = self.preparation.actions[-1, 7:13]
            target[13] = self.close_target
            self.close_plan = DipPlan(target[None, :], [copy.deepcopy(self.helper._states)], [])
            self.close_peak = float(physical[13])
            self._event({"side": "right", "status": "closing_in_place", "start_row": 0,
                         "gripper_command": self.close_target})
        elapsed = 0. if self.close_started is None else self.clock() - self.close_started
        value = float(physical[13])
        stamp = None if feedback is None else feedback.get("stamp")
        fresh = feedback is None or bool(feedback.get("valid", False) and stamp is not None
                                        and np.isfinite(stamp) and (self.close_stamp is None or stamp > self.close_stamp))
        self.close_pending_sample = (value, stamp) if fresh else None
        samples = (*self.close_samples, value)[-cfg.stable_frames:] if fresh else ()
        stable = len(samples) == cfg.stable_frames and np.ptp(samples) <= cfg.stable_tolerance
        reached = value <= self.close_target + cfg.stable_tolerance
        # A tube can prevent reaching the commanded aperture. Accept a stable
        # residual gap only after measurable closure under this command.
        contact = (value <= cfg.closed_threshold and self.close_peak - value >= cfg.close_delta
                   and value - self.close_target >= max(.015, cfg.min_contact_gap))
        at_pose = np.linalg.norm(self.helper._xyz(physical[7:13]) -
                                 self.helper._xyz(self.close_plan.actions[0, 7:13])) <= self.SETTLE_TOLERANCE_MM
        ready = self.close_started is not None and stable and at_pose and (contact or (reached and cfg.min_contact_gap == 0))
        empty = (cfg.min_contact_gap > 0 and self.close_started is not None and stable and at_pose
                 and value - self.close_target < cfg.min_contact_gap)
        if ready or empty or elapsed >= 2.:
            self._event({"side": "right", "status": "close_empty" if empty else "close_ready" if ready else "close_timeout",
                         "start_row": 0, "gripper_command": self.close_target,
                         "gripper_feedback": value,
                         "reason": "no_object_gap" if empty else ("aperture_reached" if reached else "stable_contact") if ready
                                   else "closure_not_confirmed_resume_model"})
            # Preparation can precede the policy's close by several rows. Do
            # not replay those larger apertures after closing in place.
            closes = np.flatnonzero(self.actions[self.row:, 13] <= self.helper.config.close_threshold)
            end = self.row + int(closes[0]) + 1 if len(closes) else len(self.actions)
            self.actions[self.row:end, 13] = np.minimum(self.actions[self.row:end, 13], self.close_target)
            return None
        return DipStep(self.close_plan, 0, self.row, "grasp_dip_close")

    def _next(self, physical, previous, *, feedback=None):
        if self.row >= len(self.actions):
            return None
        if self.fallback:
            return DipStep(self.plan, self.row, self.row)
        physical = np.asarray(physical, dtype=np.float32)
        previous = np.asarray(previous, dtype=np.float32)
        if self.preparation is not None:
            if self.helper.config.staged_vertical:
                if self.prep_row < len(self.preparation.actions):
                    return DipStep(self.preparation, self.prep_row, self.row, "grasp_dip_prepare")
                # The timed rotation/descent and endpoint dwell are open loop.
                # Do not reuse the old 20-row arrival gate (only .2 s at 100 Hz).
                self.preparation = None
                self.plan = self.helper.plan(self.actions[self.row:], previous_action=previous,
                                             physical_state=physical)
                self.plan_start = self.row
                return DipStep(self.plan, 0, self.row)
            for side in self.preparing_sides:
                if not self.helper._points_down(self.helper._tilt(physical[side * 7:side * 7 + 6])):
                    raise ValueError("grasp dip preparation: physical orientation no longer downward")
            if self.prep_row < len(self.preparation.actions):
                return DipStep(self.preparation, self.prep_row, self.row, "grasp_dip_prepare")
            settled = True
            for side in self.preparing_sides:
                q = physical[side * 7:side * 7 + 6]
                if not self.helper._points_down(self.helper._tilt(q)):
                    raise ValueError("grasp dip preparation: physical orientation no longer downward")
                if self.helper.config.forward_mm[side] or self.helper.config.vertical_down:
                    expected = self.helper._xyz(self.preparation.actions[-1, side * 7:side * 7 + 6])
                    settled &= np.linalg.norm(self.helper._xyz(q) - expected) <= self.SETTLE_TOLERANCE_MM
                    if self.helper.config.vertical_down:
                        settled &= self.helper._tilt(q) <= 5.0
                else:
                    settled &= abs(self.helper._xyz(q)[2] - self.helper.config.target_z_mm[side]) <= self.SETTLE_TOLERANCE_MM
            if not settled and self.close_plan is None:
                self.dwell_started = None
                if self.settle_steps >= self.MAX_SETTLE_STEPS:
                    raise TimeoutError("grasp dip preparation did not reach target position; reverting to model actions")
                return DipStep(self.preparation, len(self.preparation.actions) - 1, self.row, "grasp_dip_wait")
            for side in self.preparing_sides:
                self._event({"side": ("left", "right")[side], "status": "ready", "start_row": 0,
                             "target_z_mm": self.helper.config.target_z_mm[side]})
            if 1 in self.preparing_sides and self.preclose_config is not None:
                # Keep the preparation aperture for 0.2 s at the reached pose,
                # then close. Closing itself has no fixed minimum dwell.
                if self.close_plan is None and (self.dwell_started is None or
                                               self.clock() - self.dwell_started < .2):
                    self._event({"side": "right", "status": "dwell_before_close", "start_row": 0})
                    return DipStep(self.preparation, len(self.preparation.actions) - 1,
                                   self.row, "grasp_dip_dwell")
                closing = self._close_in_place(physical, previous, feedback)
                if closing is not None:
                    return closing
            self.preparation = None
            self.plan = self.helper.plan(self.actions[self.row:], previous_action=previous, physical_state=physical)
            self.plan_start = self.row
        else:
            self.preparation = self._prepare(physical, previous)
            if self.preparation is not None:
                self.prep_row = 0
                self.settle_steps = 0
                self.close_plan = None
                self.close_started = None
                self.close_samples = ()
                self.close_stamp = None
                self.dwell_started = None
                return DipStep(self.preparation, 0, self.row, "grasp_dip_prepare")
        if self.helper.config.vertical_down and not self.helper.config.staged_vertical:
            waiting = []
            plan_row = self.row - self.plan_start
            for event in self.plan.events:
                if event.get("status") != "applied" or event.get("close_row") != plan_row:
                    continue
                side = ("left", "right").index(event["side"])
                start = side * 7
                expected = self.plan.actions[plan_row - 1, start:start + 6]
                actual = physical[start:start + 6]
                if (np.linalg.norm(self.helper._xyz(actual) - self.helper._xyz(expected)) > self.SETTLE_TOLERANCE_MM
                        or self.helper._tilt(actual) > 5.0):
                    waiting.append((side, start, expected))
            if waiting:
                if self.settle_steps >= self.MAX_SETTLE_STEPS:
                    raise TimeoutError("assembly grasp pose did not settle before closing")
                hold = previous.copy()
                for _side, start, expected in waiting:
                    hold[start:start + 6] = expected
                    hold[start + 6] = max(float(previous[start + 6]), float(physical[start + 6]))
                pause = DipPlan(hold[None, :], [copy.deepcopy(self.helper._states)], [])
                return DipStep(pause, 0, self.row, "grasp_dip_wait")
            self.settle_steps = 0
        return DipStep(self.plan, self.row - self.plan_start, self.row)

    def commit(self, step):
        self.helper.commit(step.plan, step.index)
        if step.kind == "model":
            self.row += 1
        elif step.kind == "grasp_dip_prepare":
            self.prep_row += 1
        elif step.kind == "grasp_dip_dwell":
            if self.dwell_started is None:
                self.dwell_started = self.clock()
        elif step.kind == "grasp_dip_close":
            if self.close_started is None:
                self.close_started = self.clock()
            if self.close_pending_sample is not None:
                value, self.close_stamp = self.close_pending_sample
                self.close_samples = (*self.close_samples, value)[-self.preclose_config.stable_frames:]
            else:
                self.close_samples = ()
        else:
            self.settle_steps += 1
