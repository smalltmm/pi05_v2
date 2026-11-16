"""Opt-in pouring assistance. Produces targets; never connects to hardware."""

import argparse
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .config import PIPER_JOINT_MAX, PIPER_JOINT_MIN
from .kinematics import fingertip_xyz, solve_ik

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
RIGHT_BASE_XYZ_MM = np.array([0., -600., 0.])


@dataclass(frozen=True)
class PourConfig:
    relative_align: bool = False
    align_only: bool = False
    # User-designated fully poured pose captured from fresh CAN feedback.
    full_pose: bool = True
    prepare_j6_backoff_deg: float = 20.
    right_j4_min_deg: float = -95.
    align_xyz_mm: tuple = (-221.59293302, 104.50285813, 26.81573730)
    align_rpy_deg: tuple = (111.34068199, -32.80485649, 165.87208893)
    align_stable_seconds: float = .3
    align_stable_mm: float = 3.
    align_stable_deg: float = 5.
    align_distance_mm: tuple = (120., 300.)
    align_horizontal_mm: tuple = (60., 250.)
    distance_mm: tuple = (120., 210.)
    horizontal_mm: tuple = (60., 145.)
    height_mm: tuple = (60., 150.)
    relative_rpy_deg: tuple = (-109.2, -36.2, -13.)
    angle_deg: float = 45.
    region_seconds: float = .25
    trend_deg: float = 3.
    trend_seconds: float = .5
    extra_deg: float = 10.
    tilt_speed_deg_s: float = 10.
    hold_seconds: float = 1.5
    return_trend_deg: float = 5.
    return_seconds: float = .3
    close_threshold: float = .6
    open_threshold: float = .75
    contact_delta: float = .08
    right_grip_command_cap: float | None = None
    lift_mm: float = 10.
    clearance_mm: float = 80.
    move_speed_mm_s: float = 40.
    joint_speed_deg_s: float = 20.
    arrival_mm: float = 10.
    arrival_deg: float = 10.
    return_orientation_tolerance_deg: float = 15.
    return_position_tolerance_mm: float = 10.
    settle_seconds: float = .2
    release_delay_s: float = .1
    left_place_xyz_mm: tuple | None = None
    right_place_xyz_mm: tuple | None = None
    left_place_z_mm: float | None = None
    right_place_z_mm: float | None = None
    left_opening_mm: float = 100.
    right_opening_mm: float = 100.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and not np.isfinite(value).all():
                raise ValueError(f'pour {name} must be finite')
        for name in ('distance_mm', 'horizontal_mm', 'height_mm', 'align_distance_mm', 'align_horizontal_mm'):
            lo, hi = getattr(self, name)
            if lo < 0 or lo >= hi:
                raise ValueError(f'pour {name} requires 0 <= MIN < MAX')
        for name in ('align_xyz_mm', 'align_rpy_deg', 'relative_rpy_deg', 'left_place_xyz_mm', 'right_place_xyz_mm'):
            if getattr(self, name) is not None and np.asarray(getattr(self, name)).shape != (3,):
                raise ValueError(f'pour {name} requires three values')
        if self.align_only and not self.relative_align:
            raise ValueError('--pour-align-only requires --pour-relative-align')
        if not 0 < abs(self.prepare_j6_backoff_deg) <= 90:
            raise ValueError('pour preparation J6 backoff magnitude must be within (0, 90] degrees')
        if not PIPER_JOINT_MIN[3] <= np.deg2rad(self.right_j4_min_deg) < PIPER_JOINT_MAX[3]:
            raise ValueError('pour right J4 planning minimum must be within Piper joint limits')
        if not 0 < self.close_threshold < self.open_threshold <= 1:
            raise ValueError('pour requires 0 < close threshold < open threshold <= 1')
        if self.right_grip_command_cap is not None and not 0 <= self.right_grip_command_cap <= self.close_threshold:
            raise ValueError('pour right grip command cap must be within 0..close threshold')
        for name in ('angle_deg', 'region_seconds', 'trend_deg', 'trend_seconds', 'extra_deg',
                     'tilt_speed_deg_s', 'hold_seconds', 'return_trend_deg', 'return_seconds',
                     'contact_delta', 'lift_mm', 'clearance_mm', 'move_speed_mm_s',
                     'joint_speed_deg_s', 'arrival_mm', 'arrival_deg', 'return_orientation_tolerance_deg',
                     'return_position_tolerance_mm', 'settle_seconds'):
            if getattr(self, name) <= 0:
                raise ValueError(f'pour {name} must be positive')
        if min(self.align_stable_seconds, self.align_stable_mm, self.align_stable_deg) <= 0:
            raise ValueError('pour alignment stability tolerances and duration must be positive')
        if (self.angle_deg >= 180 or self.arrival_deg >= 180
                or self.return_orientation_tolerance_deg >= 180 or self.extra_deg > 30):
            raise ValueError('pour angles must be below 180 degrees; extra angle must be <= 30 degrees')
        if self.release_delay_s < 0 or not all(0 <= x <= 100 for x in (self.left_opening_mm, self.right_opening_mm)):
            raise ValueError('pour release delay must be nonnegative; openings must be within 0..100 mm')


class WaterPouring:
    def __init__(self, config, fk, control_hz, clock=time.monotonic):
        self.config, self.fk, self.hz, self.clock = config, fk, control_hz, clock
        self.phase = 'idle'
        self.reason = 'waiting_for_grasp'
        self.pickup = [None, None]
        self.open_seen = [False, False]
        self.close_opening = [None, None]
        self.lifted = [False, False]
        self.region_since = None
        self.reversing_since = None
        self.phase_since = None
        self.arrived_since = None
        self.last_sent = None
        self.hold = None
        self.tilt_start = None
        self.offset_deg = 0.
        self.peak_model = None
        self.hold_model_peak = None
        self.paths = {}
        self.goals = {}
        self.path_index = 0
        self.metrics = {}
        self.last_token = None
        self.stable_anchor = None
        self.stable_since = None
        self.align_left = None
        self.align_target = None
        self.align_j6 = None
        self.full_joints = None
        self.prepare_joints = None
        self.prepare_relative = None
        self.full_target = None
        self.return_goal_phase = 'descend'
        self.right_grip_holding = False

    @property
    def active(self):
        return self.phase != 'idle'

    def tips(self, state):
        poses = [self.fk.pose(state[s:s + 6]) for s in (0, 7)]
        return poses, [fingertip_xyz(p) for p in poses]

    def geometry(self, state):
        poses, tips = self.tips(state)
        delta = tips[1] + RIGHT_BASE_XYZ_MM - tips[0]
        left_rotation = Rotation.from_euler('xyz', poses[0][3:], degrees=True)
        relative = (left_rotation.inv()
                    * Rotation.from_euler('xyz', poses[1][3:], degrees=True))
        error = (Rotation.from_euler('xyz', self.config.relative_rpy_deg, degrees=True).inv() * relative).magnitude()
        return dict(distance_mm=float(np.linalg.norm(delta)), horizontal_mm=float(np.linalg.norm(delta[:2])),
                    height_mm=float(delta[2]), delta_xyz_mm=delta.tolist(),
                    right_tip_in_left_tip_xyz_mm=left_rotation.inv().apply(delta).tolist(),
                    angle_deg=float(np.rad2deg(error)), relative_rpy_deg=relative.as_euler('xyz', degrees=True).tolist(),
                    left_pose=list(map(float, poses[0])), right_pose=list(map(float, poses[1])))

    def observe(self, physical, previous, model, token):
        """Use executed model intent, never duplicated 100 Hz rows as evidence counts."""
        self.metrics = self.geometry(physical)
        if self.active or token == self.last_token:
            return
        self.last_token = token
        cfg, now = self.config, self.clock()
        for side, s in enumerate((0, 7)):
            grip, feedback = float(model[0, s + 6]), float(physical[s + 6])
            if grip >= cfg.open_threshold:
                self.open_seen[side] = True
                self.pickup[side] = None
                self.lifted[side] = False
            if self.pickup[side] is None and self.open_seen[side] and grip <= cfg.close_threshold:
                self.pickup[side] = np.asarray(physical[s:s + 6]).copy()
                self.close_opening[side] = feedback
                self.open_seen[side] = False
            if self.pickup[side] is not None:
                lift = (fingertip_xyz(self.fk.pose(physical[s:s + 6]))[2]
                        - fingertip_xyz(self.fk.pose(self.pickup[side]))[2])
                # The left receiving container may stay on the table; only
                # the right pouring container must be lifted after closing.
                if feedback <= self.close_opening[side] - cfg.contact_delta and (side == 0 or lift >= cfg.lift_mm):
                    self.lifted[side] = True
        region = True
        for name in ('distance_mm', 'horizontal_mm', 'height_mm'):
            setting = 'align_' + name if cfg.relative_align and name != 'height_mm' else name
            lo, hi = getattr(cfg, setting)
            region = region and lo <= self.metrics[name] <= hi
        region = region and self.metrics['angle_deg'] <= cfg.angle_deg
        ready = all(self.lifted) and all(model[0, i] < cfg.open_threshold for i in (6, 13))
        if not region or not ready:
            self.region_since = None
            self.stable_anchor = self.stable_since = None
            self.reason = 'outside_pour_region' if ready else 'waiting_for_grasp_and_lift'
            return
        if self.region_since is None:
            self.region_since = now
        if cfg.relative_align:
            stable = self.observe_stability(physical, now)
            self.reason = 'waiting_for_stable_pour_pose'
            if now - self.region_since >= cfg.region_seconds and stable:
                self.latch(physical, previous)
                self.align_left = self.hold[:6].copy()
                self.phase, self.reason = 'align_pending', 'pour_region_and_stable_pose'
                self.phase_since = now
            return
        count = max(2, int(np.ceil(cfg.trend_seconds * self.hz)) + 1)
        q = np.rad2deg(model[:count, 12])
        changes = np.diff(q)
        significant = changes[np.abs(changes) > .05]
        trend = len(q) > 1 and q[-1] - q[0] >= cfg.trend_deg
        trend = trend and len(significant) > 0 and np.mean(significant > 0) >= .7
        self.metrics['predicted_j6_change_deg'] = float(q[-1] - q[0])
        self.reason = 'waiting_for_tilt_trend'
        if now - self.region_since >= cfg.region_seconds and trend:
            self.phase, self.reason = 'tilting', 'pour_region_and_tilt_trend'
            self.phase_since = now
            self.latch(physical, previous)
            self.peak_model = float(model[0, 12])

    def latch(self, physical, previous):
        self.tilt_start = np.asarray(physical, dtype=np.float32).copy()
        self.hold = self.tilt_start.copy()
        self.hold[[6, 13]] = np.asarray(previous)[[6, 13]]
        self.last_sent = self.hold.copy()

    def apply_right_grip_cap(self, action, model_grip):
        """Cap only the right gripping command; preserve raw model release intent."""
        cap = self.config.right_grip_command_cap
        if cap is None:
            return action
        model_grip = float(model_grip)
        if self.phase in ('release', 'done'):
            self.right_grip_holding = False
        elif self.active:
            self.right_grip_holding = True
        elif model_grip >= self.config.open_threshold:
            self.right_grip_holding = False
        elif model_grip <= self.config.close_threshold:
            self.right_grip_holding = True
        if self.right_grip_holding:
            action = np.asarray(action, dtype=np.float32).copy()
            action[13] = min(model_grip, cap)
        self.metrics.update(right_grip_cap_active=self.right_grip_holding,
                            right_grip_model=model_grip, right_grip_command=float(action[13]))
        return action

    def observe_stability(self, physical, now):
        poses, tips = self.tips(physical)
        rotations = [Rotation.from_euler('xyz', p[3:], degrees=True) for p in poses]
        if self.stable_anchor is None:
            self.stable_anchor, self.stable_since = (tips, rotations), now
        old_tips, old_rotations = self.stable_anchor
        distance = max(float(np.linalg.norm(tips[s] - old_tips[s])) for s in (0, 1))
        angle = max(float(np.rad2deg((old_rotations[s].inv() * rotations[s]).magnitude())) for s in (0, 1))
        if distance > self.config.align_stable_mm or angle > self.config.align_stable_deg:
            self.stable_anchor, self.stable_since = (tips, rotations), now
        self.metrics.update(stable_position_change_mm=distance, stable_angle_change_deg=angle,
                            stable_elapsed_s=now - self.stable_since)
        return now - self.stable_since >= self.config.align_stable_seconds

    def next(self, physical, model):
        self.metrics['right_j6_feedback_deg'] = float(np.rad2deg(physical[12]))
        self.metrics['right_j6_model_deg'] = float(np.rad2deg(model[0, 12]))
        if not self.active:
            return None, self.record()
        now, cfg = self.clock(), self.config
        action = self.hold.copy()
        if self.phase == 'tilting':
            # Freeze the pouring geometry; enhance only J6, once per command.
            taught_full = cfg.relative_align and cfg.full_pose
            travel_deg = abs(cfg.prepare_j6_backoff_deg) if taught_full else cfg.extra_deg
            self.offset_deg = min(travel_deg, self.offset_deg + cfg.tilt_speed_deg_s / self.hz)
            q = float(model[0, 12])
            # Model absolute J6 belongs to the original IK pose. After explicit
            # alignment, tilt relative to the aligned wrist instead of chasing it.
            self.peak_model = self.align_j6 if cfg.relative_align else max(self.peak_model, q)
            # A slower rate-limited arm must finish the requested tilt even if
            # the next model chunk already predicts a return. Do not chase its
            # decreasing target or interpret it as an executed pour.
            if taught_full:
                desired = float(self.full_joints[5])
            else:
                desired = np.clip(self.peak_model + np.deg2rad(self.offset_deg),
                                  PIPER_JOINT_MIN[5], PIPER_JOINT_MAX[5])
            self.metrics['right_j6_goal_deg'] = float(np.rad2deg(desired))
            step = np.deg2rad(cfg.tilt_speed_deg_s) / self.hz
            action[12] = self.last_sent[12] + np.clip(desired - self.last_sent[12], -step, step)
            reached = abs(float(physical[12]) - desired) <= np.deg2rad(2.)
            if taught_full:
                self.metrics.update(self.full_pose_error(physical))
                reached = reached and self.at_goal(physical, 'full')
            full = self.offset_deg >= travel_deg and abs(float(action[12]) - desired) < 1e-5
            if full and reached:
                self.arrived_since = now if self.arrived_since is None else self.arrived_since
            else:
                self.arrived_since = None
            if self.arrived_since is not None and now - self.arrived_since >= cfg.settle_seconds:
                self.hold = action.copy()
                self.phase, self.phase_since = 'holding', None
                self.reason = 'enhanced_tilt_reached'
                self.hold_model_peak = q
                self.reversing_since = None
        elif self.phase == 'holding':
            q = float(model[0, 12])
            # Count a new reversal only after reaching the enhanced tilt.
            self.hold_model_peak = max(q, self.peak_model if self.hold_model_peak is None else self.hold_model_peak)
            reversing = not cfg.relative_align and np.rad2deg(self.hold_model_peak - q) >= cfg.return_trend_deg
            self.reversing_since = ((now if self.reversing_since is None else self.reversing_since)
                                    if reversing else None)
            if self.reversing_since is not None and now - self.reversing_since >= cfg.return_seconds:
                self.phase = 'return_pending'
                self.reason = 'sustained_model_return'
            elif self.phase_since is not None and now - self.phase_since >= cfg.hold_seconds:
                self.phase = 'return_pending'
                self.reason = 'timed_pour_complete'
        elif self.phase in self.paths:
            path = self.paths[self.phase]
            action[JOINTS] = path[min(self.path_index, len(path) - 1)][JOINTS]
            if self.phase == 'align':
                self.metrics.update(self.alignment_error(physical))
        elif self.phase == 'aligned':
            self.metrics.update(self.alignment_error(physical))
        elif self.phase == 'dwell':
            if not self.at_goal(physical, self.return_goal_phase):
                self.phase_since = None
            elif self.phase_since is not None and now - self.phase_since >= cfg.release_delay_s:
                self.phase = 'release'
                action[[6, 13]] = [cfg.left_opening_mm / 100, cfg.right_opening_mm / 100]
        elif self.phase in ('release', 'done'):
            action[[6, 13]] = [cfg.left_opening_mm / 100, cfg.right_opening_mm / 100]
        self.metrics['right_j6_command_deg'] = float(np.rad2deg(action[12]))
        return action, self.record()

    def commit(self, physical, sent):
        if not self.active:
            return
        self.last_sent = np.asarray(sent).copy()
        now = self.clock()
        if self.phase == 'holding' and self.phase_since is None:
            self.phase_since = now
        elif self.phase in self.paths:
            path = self.paths[self.phase]
            waypoint = path[min(self.path_index, len(path) - 1)]
            if np.max(np.abs(sent[JOINTS] - waypoint[JOINTS])) < 1e-5:
                self.path_index = min(self.path_index + 1, len(path))
            arrived = self.path_index == len(path) and self.at_goal(physical, self.phase)
            self.arrived_since = (now if self.arrived_since is None else self.arrived_since) if arrived else None
            if self.arrived_since is not None and now - self.arrived_since >= self.config.settle_seconds:
                self.hold = sent.copy()
                if self.phase == 'align':
                    self.align_j6 = float(sent[12])
                    self.peak_model = self.align_j6
                    self.offset_deg = 0.
                    self.phase = 'aligned' if self.config.align_only else 'tilting'
                    self.reason = 'relative_pose_reached'
                else:
                    self.phase = {'untilt': 'approach', 'approach': 'descend', 'descend': 'dwell',
                                  'joint_return': 'dwell'}[self.phase]
                self.path_index = 0
                self.arrived_since = None
                self.phase_since = now
        elif self.phase == 'dwell' and self.phase_since is None and self.at_goal(physical, self.return_goal_phase):
            self.phase_since = now
        elif self.phase == 'release':
            desired = np.array([self.config.left_opening_mm, self.config.right_opening_mm]) / 100
            if np.max(np.abs(np.asarray(physical)[[6, 13]] - desired)) <= .05:
                self.phase = 'done'

    def at_goal(self, physical, phase):
        if phase == 'joint_return':
            error = np.asarray(physical)[JOINTS] - self.paths[phase][-1, JOINTS]
            self.metrics['return_joint_error_deg'] = float(np.rad2deg(np.max(np.abs(error))))
            if self.metrics['return_joint_error_deg'] > 2.:
                return False
        position_tolerance = (self.config.return_position_tolerance_mm
                              if phase in ('approach', 'descend', 'joint_return') else self.config.arrival_mm)
        orientation_tolerance = (self.config.return_orientation_tolerance_deg
                                 if phase in ('approach', 'descend', 'joint_return') else self.config.arrival_deg)
        for side, start in enumerate((0, 7)):
            actual = self.fk.pose(physical[start:start + 6])
            xyz, rotation = self.goals[phase][side]
            angle = (rotation.inv() * Rotation.from_euler('xyz', actual[3:], degrees=True)).magnitude()
            if (np.linalg.norm(fingertip_xyz(actual) - xyz) > position_tolerance
                    or np.rad2deg(angle) > orientation_tolerance):
                return False
        if phase == 'align':
            error = self.alignment_error(physical)
            return (error['align_position_error_mm'] <= self.config.arrival_mm
                    and error['align_orientation_error_deg'] <= self.config.arrival_deg)
        if phase == 'full':
            error = self.full_pose_error(physical)
            return (error['full_position_error_mm'] <= self.config.arrival_mm
                    and error['full_orientation_error_deg'] <= self.config.arrival_deg)
        return True

    def alignment_error(self, physical):
        geometry = self.geometry(physical)
        rotation = Rotation.from_euler('xyz', geometry['relative_rpy_deg'], degrees=True)
        xyz, rpy = (self.prepare_relative if self.prepare_relative is not None else
                    (self.config.align_xyz_mm, self.config.align_rpy_deg))
        target_rotation = Rotation.from_euler('xyz', rpy, degrees=True)
        return dict(align_position_error_mm=float(np.linalg.norm(
                        np.asarray(geometry['right_tip_in_left_tip_xyz_mm']) - xyz)),
                    align_orientation_error_deg=float(np.rad2deg((target_rotation.inv() * rotation).magnitude())))

    def full_pose_error(self, physical):
        geometry = self.geometry(physical)
        actual = Rotation.from_euler('xyz', geometry['relative_rpy_deg'], degrees=True)
        desired = Rotation.from_euler('xyz', self.config.align_rpy_deg, degrees=True)
        return dict(full_position_error_mm=float(np.linalg.norm(
                        np.asarray(geometry['right_tip_in_left_tip_xyz_mm']) - self.config.align_xyz_mm)),
                    full_orientation_error_deg=float(np.rad2deg((desired.inv() * actual).magnitude())))

    def plan_pending(self, physical):
        """Only called at batch boundaries, never inside the timed send loop."""
        if self.phase in ('align_pending', 'align_planning_failed'):
            self.plan_alignment(physical)
        else:
            self.plan_return(physical)

    def plan_alignment(self, physical):
        if self.phase not in ('align_pending', 'align_planning_failed'):
            return
        # The left anchor stays fixed across planning retries and inference.
        self.hold[:6] = self.align_left
        self.hold[7:13] = np.asarray(physical)[7:13]
        left_pose = self.fk.pose(self.align_left)
        left_rotation = Rotation.from_euler('xyz', left_pose[3:], degrees=True)
        right_xyz = (fingertip_xyz(left_pose) + left_rotation.apply(self.config.align_xyz_mm)
                     - RIGHT_BASE_XYZ_MM)
        right_rotation = left_rotation * Rotation.from_euler('xyz', self.config.align_rpy_deg, degrees=True)
        self.align_target = (right_xyz, right_rotation)
        targets = [(fingertip_xyz(left_pose), left_rotation), self.align_target]
        right_min = PIPER_JOINT_MIN.astype(float).copy()
        right_max = PIPER_JOINT_MAX.astype(float).copy()
        right_min[3] = np.deg2rad(self.config.right_j4_min_deg)
        try:
            full_targets = list(targets)
            final_joints = None
            if self.config.full_pose:
                full, _, _ = solve_ik(self.fk, self.hold[7:13], right_xyz,
                                     right_rotation.as_euler('xyz', degrees=True),
                                     position_tolerance_mm=self.config.arrival_mm,
                                     orientation_tolerance_deg=self.config.arrival_deg,
                                     prefer_local_solution=True, joint_min=right_min, joint_max=right_max)
                prepare = full.copy()
                prepare[5] += np.deg2rad(self.config.prepare_j6_backoff_deg)
                if np.any(prepare < right_min - 1e-7) or np.any(prepare > right_max + 1e-7):
                    raise ValueError('preparation J6 backoff exceeds joint limits; full pose/backoff not clipped')
                self.full_joints, self.prepare_joints = full, prepare
                self.full_target = self.align_target
                pose = self.fk.pose(prepare)
                self.align_target = (fingertip_xyz(pose), Rotation.from_euler('xyz', pose[3:], degrees=True))
                targets[1] = self.align_target
                prep_state = self.hold.copy()
                prep_state[7:13] = prepare
                geometry = self.geometry(prep_state)
                self.prepare_relative = (geometry['right_tip_in_left_tip_xyz_mm'], geometry['relative_rpy_deg'])
                final_joints = {1: prepare}
            path = self.cartesian_path(self.hold, targets, moving_sides=(1,), final_joints=final_joints,
                                       joint_bounds={1: (right_min, right_max)})
            self.paths, self.goals = {'align': path}, {'align': targets}
            if self.config.full_pose:
                self.goals['full'] = full_targets
            self.phase, self.reason, self.path_index = 'align', 'relative_path_ready', 0
            self.arrived_since = None
        except (ValueError, RuntimeError) as exc:
            self.phase, self.reason = 'align_planning_failed', str(exc)

    def plan_return(self, physical):
        """Return directly to recorded joints; explicit XYZ overrides retain Cartesian return."""
        if self.phase not in ('return_pending', 'planning_failed'):
            return
        self.hold[JOINTS] = np.asarray(physical)[JOINTS]
        if all(getattr(self.config, name) is None for name in
               ('left_place_xyz_mm', 'right_place_xyz_mm', 'left_place_z_mm', 'right_place_z_mm')):
            target = self.hold.copy()
            target[:6], target[7:13] = self.pickup
            duration = (float(np.max(np.abs(target[JOINTS] - self.hold[JOINTS])))
                        / np.deg2rad(self.config.joint_speed_deg_s))
            count = max(1, int(np.ceil(duration * self.hz)))
            path = np.asarray([self.hold + alpha * (target - self.hold)
                               for alpha in np.linspace(0., 1., count + 1)[1:]], dtype=np.float32)
            poses, tips = self.tips(target)
            self.paths = {'joint_return': path}
            self.goals = {'joint_return': [(tips[s], Rotation.from_euler('xyz', poses[s][3:], degrees=True))
                                           for s in (0, 1)]}
            self.return_goal_phase = 'joint_return'
            self.phase, self.reason, self.path_index = 'joint_return', 'recorded_pickup_joints_ready', 0
            self.arrived_since = None
            return
        self.return_goal_phase = 'descend'
        leg = 'untilt'
        try:
            paths, goals = {}, {}
            starts = self.hold.copy()
            # First undo only the wrist tilt, with the other eleven joints
            # fixed. Never require the complete pickup rotation at this XYZ.
            untilted = starts.copy()
            untilted[12] = self.align_j6 if self.align_j6 is not None else self.tilt_start[12]
            if not PIPER_JOINT_MIN[5] <= untilted[12] <= PIPER_JOINT_MAX[5]:
                raise ValueError('preparation J6 return target outside joint limits')
            speed = min(self.config.tilt_speed_deg_s, self.config.joint_speed_deg_s)
            count = max(1, int(np.ceil(abs(float(untilted[12] - starts[12])) / np.deg2rad(speed) * self.hz)))
            paths['untilt'] = np.asarray([starts + a * (untilted - starts)
                                         for a in np.linspace(0., 1., count + 1)[1:]], dtype=np.float32)
            poses, tips = self.tips(untilted)
            goals['untilt'] = [(tips[s], Rotation.from_euler('xyz', poses[s][3:], degrees=True)) for s in (0, 1)]
            starts = untilted
            _, tips = self.tips(starts)
            destinations, rotations = [], []
            for side, name in enumerate(('left', 'right')):
                pose = self.fk.pose(self.pickup[side])
                dest = fingertip_xyz(pose)
                override = getattr(self.config, name + '_place_xyz_mm')
                if override is not None:
                    dest = np.asarray(override, dtype=float).copy()
                z = getattr(self.config, name + '_place_z_mm')
                if z is not None:
                    dest[2] = z
                destinations.append(dest)
                rotations.append(Rotation.from_euler('xyz', pose[3:], degrees=True))
            for phase in ('approach', 'descend'):
                leg = phase
                targets = []
                for side in (0, 1):
                    xyz = destinations[side].copy()
                    if phase == 'approach':
                        xyz[2] = max(tips[side][2], destinations[side][2] + self.config.clearance_mm)
                    targets.append((xyz, rotations[side]))
                paths[phase] = self.cartesian_path(
                    starts, targets, orientation_tolerance_deg=self.config.return_orientation_tolerance_deg,
                    position_tolerance_mm=self.config.return_position_tolerance_mm)
                goals[phase] = targets
                starts = paths[phase][-1].copy()
            self.paths, self.goals = paths, goals
            self.phase, self.reason, self.path_index = 'untilt', 'return_path_ready', 0
            self.arrived_since = None
        except (ValueError, RuntimeError) as exc:
            self.phase, self.reason = 'planning_failed', f'{leg}: {exc}'

    def cartesian_path(self, start, targets, *, moving_sides=(0, 1), final_joints=None, joint_bounds=None,
                       orientation_tolerance_deg=None, position_tolerance_mm=None):
        cfg = self.config
        orientation_tolerance = cfg.arrival_deg if orientation_tolerance_deg is None else orientation_tolerance_deg
        position_tolerance = cfg.arrival_mm if position_tolerance_mm is None else position_tolerance_mm
        poses, tips = self.tips(start)
        rotations = [Rotation.from_euler('xyz', p[3:], degrees=True) for p in poses]
        lengths = [np.linalg.norm(targets[s][0] - tips[s]) for s in (0, 1)]
        angles = [np.rad2deg((rotations[s].inv() * targets[s][1]).magnitude()) for s in (0, 1)]
        segments = max(1, int(np.ceil(max(*lengths) / 5.)), int(np.ceil(max(*angles) / 3.)))
        slerps = [Slerp([0, 1], Rotation.concatenate([rotations[s], targets[s][1]])) for s in (0, 1)]
        result, previous = [], start.copy()
        for fraction in np.linspace(0., 1., segments + 1)[1:]:
            waypoint = previous.copy()
            for side, s in enumerate((0, 7)):
                if side not in moving_sides:
                    continue
                xyz = tips[side] + fraction * (targets[side][0] - tips[side])
                rotation = slerps[side](fraction)

                def residual(q, rotation=rotation, xyz=xyz):
                    pose = self.fk.pose(q)
                    error = (rotation.inv() * Rotation.from_euler('xyz', pose[3:], degrees=True)).as_rotvec()
                    return np.r_[(fingertip_xyz(pose) - xyz) / 100., error]

                lower, upper = ((joint_bounds or {}).get(side,
                                (PIPER_JOINT_MIN.astype(float), PIPER_JOINT_MAX.astype(float))))
                solved = least_squares(residual, np.clip(previous[s:s + 6], lower, upper),
                                       bounds=(lower, upper), max_nfev=100)
                q = solved.x.astype(np.float32)
                if fraction == 1. and final_joints is not None and side in final_joints:
                    q = final_joints[side].copy()
                error = residual(q)
                if np.any(q < lower - 1e-7) or np.any(q > upper + 1e-7):
                    raise ValueError('Cartesian waypoint exceeds configured planning joint limits')
                if (np.linalg.norm(error[:3]) * 100 > position_tolerance
                        or np.rad2deg(np.linalg.norm(error[3:])) > orientation_tolerance):
                    raise ValueError(f'{("left", "right")[side]} return IK failed at fraction={fraction:.3f}')
                waypoint[s:s + 6] = q
            duration = max(max(lengths) / segments / cfg.move_speed_mm_s,
                           float(np.max(np.abs(waypoint[JOINTS] - previous[JOINTS]))) / np.deg2rad(cfg.joint_speed_deg_s))
            count = max(1, int(np.ceil(duration * self.hz)))
            result.extend(previous + alpha * (waypoint - previous) for alpha in np.linspace(0, 1, count + 1)[1:])
            previous = waypoint
        return np.asarray(result, dtype=np.float32)

    def record(self):
        return dict(phase=self.phase, reason=self.reason, offset_deg=self.offset_deg,
                    path_index=self.path_index, lifted=list(self.lifted),
                    align_goal_xyz_mm=list(self.config.align_xyz_mm) if self.config.relative_align else None,
                    align_goal_rpy_deg=list(self.config.align_rpy_deg) if self.config.relative_align else None,
                    align_target_right_base_xyz_mm=None if self.align_target is None else self.align_target[0].tolist(),
                    aligned_j6_deg=None if self.align_j6 is None else float(np.rad2deg(self.align_j6)),
                    full_pose_mode=self.config.relative_align and self.config.full_pose,
                    prepare_j6_backoff_deg=self.config.prepare_j6_backoff_deg,
                    right_j4_min_deg=self.config.right_j4_min_deg,
                    right_grip_command_cap=self.config.right_grip_command_cap,
                    full_j6_deg=None if self.full_joints is None else float(np.rad2deg(self.full_joints[5])),
                    prepare_goal_xyz_mm=None if self.prepare_relative is None else list(self.prepare_relative[0]),
                    prepare_goal_rpy_deg=None if self.prepare_relative is None else list(self.prepare_relative[1]),
                    place_target_xyz_mm=[p[0].tolist() for p in self.goals[self.return_goal_phase]]
                    if self.return_goal_phase in self.goals else None,
                    pickup_joints=[None if p is None else p.tolist() for p in self.pickup], **self.metrics)


def add_arguments(parser):
    parser.add_argument('--pour-assist', action=argparse.BooleanOptionalAction, default=False,
                        help='water_pouring only: enhance tilt, return to pickup positions, then release')
    for name, value in asdict(PourConfig()).items():
        if isinstance(value, bool):
            parser.add_argument('--pour-' + name.replace('_', '-'), action=argparse.BooleanOptionalAction, default=value)
            continue
        nargs = 3 if name.endswith(('xyz_mm', 'rpy_deg')) else 2 if isinstance(value, tuple) else None
        parser.add_argument('--pour-' + name.replace('_', '-'), type=float, default=value, **({'nargs': nargs} if nargs else {}))


def config_from_args(args):
    if not args.pour_assist:
        if args.pour_right_grip_command_cap is not None:
            raise ValueError('--pour-right-grip-command-cap requires --pour-assist')
        if args.pour_relative_align or args.pour_align_only:
            raise ValueError('--pour-relative-align/--pour-align-only requires --pour-assist')
        return None
    if args.task != 'water_pouring':
        raise ValueError('--pour-assist requires --task water_pouring')
    # Other task-specific owners already reject water_pouring in their parsers.
    return PourConfig(**{name: getattr(args, 'pour_' + name) for name in asdict(PourConfig())})
