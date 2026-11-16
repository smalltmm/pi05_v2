"""Timed, open-loop rotate-then-lower preparation for downward table grasps."""

import copy

import numpy as np

from .grasp_dip import DipPlan
from .grasp_dip_cartesian import solve_translation

MOVE_SPEED_MM_S = 50.0
ROTATE_SPEED_DEG_S = 45.0
JOINT_SPEED_DEG_S = 90.0
ENDPOINT_HOLD_SECONDS = .15


def _segment(helper, q, target, side, hz, *, rotate):
    origin = helper._xyz(q)
    delta, reason = solve_translation(helper, q, target, side)
    if reason is not None:
        raise ValueError(reason)
    end = helper.apply_offset(q, delta)
    # Quintic interpolation has maximum normalized slope 1.875. Account for
    # that peak when choosing the number of rows at the actual control rate.
    seconds = 1.875 * max(
        np.linalg.norm(target - origin) / MOVE_SPEED_MM_S,
        helper._tilt(q) / ROTATE_SPEED_DEG_S if rotate else 0.,
        np.max(np.abs(end - q)) / np.deg2rad(JOINT_SPEED_DEG_S),
    )
    count = max(1, int(np.ceil(seconds * hz)))
    for _ in range(4):
        rows = []
        for t in np.linspace(0., 1., count + 1)[1:]:
            fraction = t ** 3 * (10. + t * (-15. + 6. * t))
            xyz = origin if rotate else origin + fraction * (target - origin)
            offset, reason = solve_translation(
                helper, q, xyz, side, orientation_fraction=fraction if rotate else 1.)
            if reason is not None:
                raise ValueError(reason)
            rows.append(helper.apply_offset(q, offset))
        rows = np.asarray(rows, dtype=np.float32)
        peak = np.max(np.abs(np.diff(np.vstack([q, rows]), axis=0))) * hz
        ratio = peak / np.deg2rad(JOINT_SPEED_DEG_S)
        if ratio <= 1.001:
            return rows
        count = int(np.ceil(count * ratio * 1.01))
    raise ValueError("vertical grasp preparation exceeds joint speed")


def preparation_path(helper, joints, target_z, side, hz):
    q = np.asarray(joints, dtype=np.float32)
    xyz = helper._xyz(q)
    # Rotate at constant XYZ before descending; tool heading remains fixed.
    rotate = _segment(helper, q, xyz, side, hz, rotate=True)
    target = xyz.copy()
    target[0] += helper.config.forward_mm[side]
    target[2] = min(float(target_z), xyz[2])
    descend = _segment(helper, rotate[-1], target, side, hz, rotate=False)
    hold_count = max(1, int(np.ceil(ENDPOINT_HOLD_SECONDS * hz)))
    path = np.vstack([rotate, descend, np.repeat(descend[-1:], hold_count, axis=0)])
    return path, {"rotate_rows": len(rotate), "descend_rows": len(descend),
                  "hold_rows": hold_count, "control_hz": hz,
                  "sequence": "rotate_at_current_height_then_lower_then_close"}


def prepare(runtime, physical, previous):
    helper = runtime.helper
    cfg = helper.config
    remaining = runtime.actions[runtime.row:]
    paths = {}
    events = []
    candidates = {}
    for side, start in enumerate((0, 7)):
        target_z = cfg.target_z_mm[side]
        arm = helper._states[side]
        if (target_z is None or arm.phase != "idle"
                or not (arm.armed or previous[start + 6] > cfg.close_threshold)
                or physical[start + 6] <= cfg.close_threshold):
            continue
        closes = np.flatnonzero(remaining[:, start + 6] <= np.float32(cfg.close_threshold))
        if len(closes):
            if closes[0] > cfg.pre_steps:
                continue
            candidate = remaining[int(closes[0])]
            source = "executed_prefix"
        else:
            closes = np.flatnonzero(runtime.lookahead[:, start + 6] <= np.float32(cfg.close_threshold))
            if (not len(closes) or len(remaining) > cfg.pre_steps
                    or remaining[0, start + 6] >= previous[start + 6] - 1e-4):
                continue
            candidate = runtime.lookahead[int(closes[0])]
            source = "prediction_tail"
        q = physical[start:start + 6]
        xyz = helper._xyz(q)
        limits = cfg.z_ranges[side]
        if not limits[0] <= xyz[2] <= limits[1]:
            continue
        event = {"side": ("left", "right")[side], "status": "applied", "start_row": 0,
                 "target_z_mm": target_z, "physical_z_mm": float(xyz[2]), "trigger_source": source}
        try:
            if not limits[0] <= helper._xyz(candidate[start:start + 6])[2] <= limits[1]:
                raise ValueError("close_height_outside_range")
            if not helper._points_down(helper._tilt(q)):
                raise ValueError("physical_orientation_not_downward")
            if not helper._points_down(helper._tilt(candidate[start:start + 6])):
                raise ValueError("close_orientation_not_downward")
            paths[side], timing = preparation_path(helper, q, target_z, side, runtime.control_hz)
        except ValueError as exc:
            event.update(status="skipped", reason=str(exc))
            runtime._event(event)
            continue
        event.update(timing)
        candidates[side] = candidate
        events.append(event)
    if not paths:
        return None
    count = max(len(path) for path in paths.values())
    rows = np.repeat(previous[None, :], count, axis=0)
    states = [copy.deepcopy(helper._states) for _ in rows]
    for side, path in paths.items():
        start = side * 7
        final = helper.fk.pose(path[-1])
        close_q = candidates[side][start:start + 6]
        for j in range(count):
            q = path[min(j, len(path) - 1)]
            rows[j, start:start + 6] = q
            rows[j, start + 6] = max(float(physical[start + 6]), float(previous[start + 6]))
            state = states[j][side]
            state.phase, state.armed, state.closed, state.prepared = "hold", False, False, True
            state.prepare_grip = float(rows[j, start + 6])
            state.offset = q - physical[start:start + 6]
            state.grasp_offset = path[-1] - close_q
            state.grasp_z = float(helper._xyz(close_q)[2])
            state.target_x, state.target_y = map(float, final[:2])
            state.target_heading_deg = float(final[5])
    # Validate the resumed grasp before sending any of the added movement.
    preview = type(helper)(cfg, helper.fk, down_tilt=helper.down_tilt)
    preview._states = copy.deepcopy(states[-1])
    preview.plan(remaining, previous_action=rows[-1], physical_state=rows[-1])
    runtime.preparing_sides = list(paths)
    for event in events:
        runtime._event(event)
    return DipPlan(rows, states, events)
