"""Plan a measured, elevated posture adjustment before mixer visual alignment."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .mixer_servo import JOINT_MIN, JOINT_MAX, solve_ik, tcp_pose, rotation_distance


def plan_placement_preparation(seed, config):
    """Elevate if needed, then follow a checked joint path to the saved ready pose."""
    c = config
    q = np.asarray(seed, float).copy()
    start, R = tcp_pose(q, c.tcp_offset_mm)
    elevated = start.copy()
    elevated[2] = max(start[2], c.prepare_min_tcp_z_mm)
    goal_q = np.asarray(c.placement_prepare_joints, float)
    report = dict(status='planned', reason='right_placement_confirmed',
                  target_joints_rad=goal_q.tolist(), lift_mm=float(elevated[2]-start[2]))
    path = []

    def fail(reason):
        report.update(status='failed', reason=reason)
        return [], report

    for k in range(1, math.ceil((elevated[2]-start[2])/5.)+1):
        position = start.copy()
        position[2] = min(elevated[2], start[2]+5*k)
        next_q = solve_ik(q, position, R, c.tcp_offset_mm)
        if next_q is None or np.max(abs(next_q-q)) > .2:
            return fail('placement_prepare_lift_ik_failed')
        for t in np.linspace(0., 1., 11):
            point, _ = tcp_pose(q+t*(next_q-q), c.tcp_offset_mm)
            if np.linalg.norm(point[:2]-start[:2]) > .5 or point[2] < start[2]-.3:
                return fail('placement_prepare_lift_path_invalid')
        q = next_q
        path.append(('placement_prepare', q.copy(), R.copy()))
    # This saved pose is 106 mm above the base table reference. Check the
    # complete transfer above clearance, including interpolated joint commands.
    count = max(1, math.ceil(np.max(abs(goal_q-q))/np.deg2rad(4.)))
    origin = q.copy()
    for k in range(1, count+1):
        next_q = origin+(goal_q-origin)*k/count
        for t in np.linspace(0., 1., 11):
            point, _ = tcp_pose(q+t*(next_q-q), c.tcp_offset_mm)
            if point[2] < c.prepare_min_tcp_z_mm-.3:
                return fail('placement_prepare_transfer_below_clearance')
        q = next_q
        _, rotation = tcp_pose(q, c.tcp_offset_mm)
        path.append(('placement_prepare', q.copy(), rotation))
    report['waypoint_count'] = len(path)
    report['target_tcp_mm'] = tcp_pose(goal_q, c.tcp_offset_mm)[0].tolist()
    return path, report


def plan_preparation(seed, config):
    """Return a checked waypoint list; planning failure never starts a partial lift.

    The finger first rises along base Z at the captured orientation. Only at the
    elevated point may orientation change, within the configured angular cap.
    J4's target is a bound on the entire-arm IK, never an isolated joint command.
    """
    c = config
    seed = np.asarray(seed, float)
    margin = float(np.rad2deg(min(seed[3]-JOINT_MIN[3], JOINT_MAX[3]-seed[3])))
    report = dict(status='planning', reason='', initial_j4_margin_deg=margin,
                  requested_j4_margin_deg=c.prepare_j4_margin_deg,
                  lift_mm=0., rotation_deg=0., waypoint_count=0)

    def fail(reason):
        report.update(status='failed', reason=reason)
        return [], report

    if margin >= c.prepare_j4_margin_deg:
        report.update(status='skipped', reason='j4_margin_already_available')
        return [], report
    start, R0 = tcp_pose(seed, c.tcp_offset_mm)
    elevated = start.copy()
    elevated[2] = max(start[2], c.prepare_min_tcp_z_mm)
    report['lift_mm'] = float(elevated[2]-start[2])
    lower, upper = JOINT_MIN.copy(), JOINT_MAX.copy()
    lower[3] += np.deg2rad(c.prepare_j4_margin_deg)
    upper[3] -= np.deg2rad(c.prepare_j4_margin_deg)

    def residual(q):
        p, R = tcp_pose(q, c.tcp_offset_mm)
        return np.r_[100.*(p-elevated),
                     100.*Rotation.from_matrix(R0 @ R.T).as_rotvec(), .001*(q-seed)]

    result = least_squares(residual, np.clip(seed, lower+1e-9, upper-1e-9),
                           bounds=(lower, upper), max_nfev=100,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    qp = result.x
    point, Rp = tcp_pose(qp, c.tcp_offset_mm)
    angle = rotation_distance(R0, Rp)
    report['rotation_deg'] = float(np.rad2deg(angle))
    report['planned_j4_margin_deg'] = float(np.rad2deg(min(qp[3]-JOINT_MIN[3], JOINT_MAX[3]-qp[3])))
    if not np.isfinite(qp).all() or np.linalg.norm(point-elevated) > .05:
        return fail('prepare_posture_position_unreachable')
    if angle > np.deg2rad(c.prepare_max_rotation_deg):
        return fail('prepare_posture_rotation_budget')

    waypoints = []
    q = seed.copy()

    def append(phase, position, rotation):
        nonlocal q
        candidate = solve_ik(q, position, rotation, c.tcp_offset_mm)
        if candidate is None or np.max(abs(candidate-q)) > .2:
            return False
        # Verify interpolated commands too: straight joint ramps must not dip
        # below the starting height or swing the fingertip away during rotation.
        previous, _ = tcp_pose(q, c.tcp_offset_mm)
        for fraction in (.25, .5, .75, 1.):
            mid, _ = tcp_pose(q+fraction*(candidate-q), c.tcp_offset_mm)
            if np.linalg.norm(mid[:2]-start[:2]) > .5:
                return False
            if mid[2] < min(previous[2], position[2])-.3:
                return False
        q = candidate
        waypoints.append((phase, q.copy(), rotation.copy()))
        return True

    count = math.ceil(report['lift_mm']/5.)
    for k in range(1, count+1):
        position = start+(elevated-start)*k/count
        if not append('prepare_lift', position, R0):
            return fail('prepare_vertical_lift_ik_failed')
    count = max(1, math.ceil(np.rad2deg(angle)/2.))
    change = Rotation.from_matrix(Rp @ R0.T).as_rotvec()
    for k in range(1, count+1):
        rotation = Rotation.from_rotvec(change*k/count).as_matrix() @ R0
        if not append('prepare_posture', elevated, rotation):
            return fail('prepare_rotation_path_ik_failed')
    actual_margin = float(np.rad2deg(min(q[3]-JOINT_MIN[3], JOINT_MAX[3]-q[3])))
    if actual_margin < c.prepare_j4_margin_deg-.1:
        return fail('prepare_j4_margin_not_reached')
    report.update(status='planned', waypoint_count=len(waypoints),
                  target_tcp_mm=elevated.tolist(), target_joints_rad=q.tolist(),
                  planned_j4_margin_deg=actual_margin)
    return waypoints, report
