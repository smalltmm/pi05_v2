"""Relative geometry and monitor transaction tests, without hardware motion."""
import json
import signal
import sys
import numpy as np
import pytest
from pi05_assists.grasp_dip import make_piper_fk
from pi05_assists.pipette_press import DEFAULT_REFERENCE, PipettePress, PipettePressConfig
from pi05_assists.relative_handover import pose_error

def captured():
    return np.array(json.loads(DEFAULT_REFERENCE.read_text())['state_14'], dtype=np.float32)

@pytest.mark.parametrize('hz', [10, 100])
def test_speed_streaming_finishes_without_feedback_arrival(hz):
    state = captured()
    durations = []
    for speed in (30.0, 90.0):
        press = PipettePress(PipettePressConfig(move_speed_mm_s=240, joint_speed_deg_s=speed), make_piper_fk(), hz)
        goal = state.copy()
        goal[5] += np.deg2rad(30)
        press.path = [state + a * (goal - state) for a in np.linspace(0, 1, 21)[1:]]
        press.labels = ['approach'] * len(press.path)
        press.tip_targets = [press.tips(row)[0] for row in press.path]
        press._time_path(state)
        durations.append(len(press.path) / hz)
        steps = np.diff(np.vstack([state, *press.path]), axis=0)
        assert np.max(np.abs(steps[:, :6])) <= np.deg2rad(speed) / hz + 1e-06
        np.testing.assert_array_equal(press.path[-1], goal)
        np.testing.assert_array_equal(np.array(press.path)[:, 6:], np.tile(state[6:], (len(press.path), 1)))
        press.phase = 'approach'
        press.anchor = press.tips(state)[1]
        press.target = press.tip_targets[-1]
        feedback = state.copy()
        feedback[7] += 0.2
        for _ in range(len(press.path)):
            press.commit(feedback, press.path[press.index], cycle=0)
        assert press.phase == 'completed'
    assert durations[1] < durations[0]

def test_speed_cli_and_validation(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['monitor', '--checkpoint', '/unused', '--task', 'manipulate_pipette', '--right-grip-hold', '--pipette-press', '--pipette-press-move-speed-mm-s', '240', '--pipette-press-joint-speed-deg-s', '90'])
    args = __import__('pi05_assists.options', fromlist=['parse_options']).parse_options([x for x in sys.argv[1:] if x not in ('--checkpoint', '/unused')])
    assert args.pipette_press_move_speed_mm_s == 240
    assert args.pipette_press_joint_speed_deg_s == 90
    for value in (0, -1, float('nan')):
        with pytest.raises(ValueError, match='speeds'):
            PipettePressConfig(move_speed_mm_s=value)

@pytest.mark.parametrize('hz', [10, 100])
def test_translation_speed_and_phase_boundaries(hz, monkeypatch):
    press = PipettePress(PipettePressConfig(move_speed_mm_s=40, joint_speed_deg_s=90), make_piper_fk(), hz)
    state = captured()

    def tips(row):
        (left, right) = (np.eye(4), np.eye(4))
        left[0, 3] = row[0] * 1000
        return (left, right)
    monkeypatch.setattr(press, 'tips', tips)
    goal = state.copy()
    goal[0] += 0.1
    press.path = [goal, state.copy()]
    press.labels = ['down', 'up']
    press.tip_targets = [tips(row)[0] for row in press.path]
    press._time_path(state)
    xyz = np.array([tips(row)[0][:3, 3] for row in [state, *press.path]])
    assert np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)) <= 40 / hz + 0.0001
    assert len(press.path) / hz == pytest.approx(5.0, abs=2 / hz)
    bottom = press.labels.index('up') - 1
    np.testing.assert_array_equal(press.path[bottom], goal)
    np.testing.assert_array_equal(press.path[-1], state)

@pytest.mark.parametrize('ready_only', [True, False])
def test_live_wrist_limit_failure_uses_continuous_clearance_path(ready_only):
    physical = np.array([-0.17156586, 1.18003452, -1.41254735, -0.98708838, 0.82513928, 0.99982929, 0.998, 0.74317867, 1.70014262, -0.6701017, 0, -1.02264082, 0, 0.375], dtype=np.float32)
    press = PipettePress(PipettePressConfig(ready_only=ready_only, depth_mm=20, step_mm=4), make_piper_fk())
    assert press.plan(physical, cycle=0), press.reason
    assert press.approach_method == 'joint_clearance'
    assert press.approach_fallback_reason == 'pipette IK branch discontinuity'
    assert press.labels[0] == 'reconfigure'
    rows = np.stack([physical, *press.path])
    assert np.max(np.abs(np.diff(rows[:, :6], axis=0))) <= 0.03001
    previous_tip = press.tips(physical)[0]
    for (row, goal) in zip(press.path, press.tip_targets, strict=True):
        np.testing.assert_allclose(row[7:13], physical[7:13])
        (actual_tip, _) = press.tips(row)
        (position_error, orientation_error) = pose_error(actual_tip, goal)
        assert position_error <= 1
        assert orientation_error <= 2
        assert np.linalg.norm(actual_tip[:3, 3] - previous_tip[:3, 3]) <= 4.01
        previous_tip = actual_tip
    end = 'ready_hold' if ready_only else 'completed'
    for _ in range(3 * len(press.path)):
        (action, record) = press.next(physical)
        press.commit(physical, action, cycle=0)
        physical = action.copy()
        if press.phase == end:
            break
    assert press.phase == end
    assert record['approach_method'] == 'joint_clearance'
    np.testing.assert_allclose(press.tip_targets[-1], press.target, atol=1e-06)
    if ready_only:
        assert 'down' not in press.labels
    else:
        down = [goal for (goal, label) in zip(press.tip_targets, press.labels, strict=True) if label == 'down']
        assert down[-1][2, 3] == pytest.approx(press.target[2, 3] - 20)

def test_arrival_tolerance_accepts_feedback_quantization_without_timeout():
    physical = np.array([-0.17156586, 1.18003452, -1.41254735, -0.98708838, 0.82513928, 0.99982929, 0.998, 0.74317867, 1.70014262, -0.6701017, 0, -1.02264082, 0, 0.375], dtype=np.float32)
    press = PipettePress(PipettePressConfig(ready_only=True, step_mm=4), make_piper_fk())
    assert press.plan(physical, cycle=0), press.reason
    for index in range(8):
        (action, _) = press.next(physical)
        press.commit(physical, action, cycle=0)
        physical = action.copy()
        if index == 7:
            assert press.phase != 'idle'
    assert press.phase in ('reconfigure', 'approach', 'ready_hold')

def test_reconfiguration_rejects_joint_path_that_dips_below_endpoints(monkeypatch):
    press = PipettePress(PipettePressConfig(), make_piper_fk())
    state = captured()
    state[0] = 0
    target_q = state[:6].copy()
    target_q[0] = 0.2

    def tips(row):
        (left, right) = (np.eye(4), np.eye(4))
        alpha = row[0] / 0.2
        left[2, 3] = 100 - 400 * alpha * (1 - alpha)
        right[1, 3] = -600
        return (left, right)
    monkeypatch.setattr(press, 'tips', tips)
    with pytest.raises(ValueError, match='height/separation'):
        press._joint_approach(state, target_q)
    assert press.path == []

def test_relative_target_vertical_stroke_return_and_fixed_donor():
    press = PipettePress(PipettePressConfig(), make_piper_fk())
    state = captured()
    state[7] += 0.02
    (left, right) = press.tips(state)
    assert press.plan(state, cycle=0), press.reason
    np.testing.assert_allclose(press.target, right @ press.relative)
    down = [row for (row, label) in zip(press.tip_targets, press.labels, strict=True) if label == 'down']
    assert down[-1][2, 3] == pytest.approx(press.target[2, 3] - 5)
    for goal in down:
        np.testing.assert_allclose(goal[:2, 3], press.target[:2, 3])
        np.testing.assert_allclose(goal[:3, :3], press.target[:3, :3])
    for (row, goal) in zip(press.path, press.tip_targets, strict=True):
        np.testing.assert_allclose(row[7:13], state[7:13])
        assert row[13] == 0
        assert row[6] == press.left_grip
        (error, angle) = pose_error(press.tips(row)[0], goal)
        assert error <= 1
        assert angle <= 2
    np.testing.assert_allclose(press.tip_targets[-1], press.target)
    physical = state.copy()
    for _ in range(3 * len(press.path)):
        (action, _) = press.next(physical)
        press.commit(physical, action, cycle=0)
        physical = action
        if not press.active:
            break
    assert press.phase == 'completed'
    assert not press.observe(physical, right_holding=True, feedback={}, cycle=100)

def test_gate_requires_holding_and_stable_next_chunk_not_feedback_samples():
    press = PipettePress(PipettePressConfig(), make_piper_fk())
    state = captured()
    model = np.tile(state, (25, 1))
    feedback = {s: {'stamp': 1, 'valid': True} for s in ('left', 'right')}
    assert not press.observe(state, right_holding=False, feedback=feedback, cycle=0, model_chunk=model)
    bad_feedback = {s: {'valid': False} for s in ('left', 'right')}
    assert not press.observe(state, right_holding=True, feedback=bad_feedback, cycle=1, model_chunk=model)
    moving = model.copy()
    moving[12, 7] += 0.2
    assert not press.observe(state, right_holding=True, feedback=feedback, cycle=2, model_chunk=moving)
    assert press.reason == 'waiting_for_stable_model_chunk'
    assert press.trigger_metrics['model_right_span_mm'] > 3
    assert not press.observe(state, right_holding=True, feedback=feedback, cycle=2, model_chunk=model)
    model[:, 0] += np.linspace(0, 0.3, len(model))
    model[:, 13] = np.linspace(0, 1, len(model))
    state[7] += 0.1
    assert press.observe(state, right_holding=True, feedback=feedback, cycle=3, model_chunk=model)
    assert press.trigger_metrics['model_right_span_mm'] == pytest.approx(0)
    assert press.reason == 'model_chunk_stable'

def test_model_stability_checks_orientation_with_fixed_fingertip():
    press = PipettePress(PipettePressConfig(), make_piper_fk())
    state = captured()
    model = np.tile(state, (25, 1))
    model[10, 12] += np.deg2rad(6)
    feedback = {s: {'valid': True} for s in ('left', 'right')}
    assert not press.observe(state, right_holding=True, feedback=feedback, cycle=0, model_chunk=model)
    assert press.trigger_metrics['model_right_span_mm'] < 0.01
    assert press.trigger_metrics['model_right_span_deg'] > 3

def test_offsets_in_right_frame_and_no_press_option():
    fk = make_piper_fk()
    base = PipettePress(PipettePressConfig(), fk)
    press = PipettePress(PipettePressConfig(offset_xyz_mm=(0, 0, 2), depth_mm=0), fk)
    np.testing.assert_allclose(press.relative[:3, 3] - base.relative[:3, 3], [0, 0, 2])
    assert press.plan(captured(), cycle=0), press.reason
    assert all((pose_error(goal, press.target)[0] < 1e-08 for (goal, label) in zip(press.tip_targets, press.labels, strict=True) if label in ('down', 'up')))

def test_ready_only_holds_target_and_ignores_configured_depth():
    press = PipettePress(PipettePressConfig(ready_only=True, depth_mm=20, step_mm=4), make_piper_fk())
    physical = captured()
    assert press.plan(physical, cycle=0), press.reason
    assert set(press.labels) == {'approach'}
    for _ in range(3 * len(press.path)):
        (action, _) = press.next(physical)
        press.commit(physical, action, cycle=0)
        physical = action.copy()
    assert press.phase == 'ready_hold'
    for cycle in range(10):
        (action, record) = press.next(physical)
        press.commit(physical, action, cycle=cycle)
        assert press.active
        assert record['depth_mm'] == 0
        np.testing.assert_allclose(action, press.path[-1])
        np.testing.assert_allclose(record['waypoint_in_left_base'], press.target)

def test_failed_plan_discards_partial_path_and_waits_full_cycle(monkeypatch):
    from pi05_assists import pipette_press as module
    press = PipettePress(PipettePressConfig(), make_piper_fk())

    def fail(*args, **kwargs):
        raise RuntimeError('IK unavailable')
    monkeypatch.setattr(module, 'solve_ik', fail)
    assert not press.plan(captured(), cycle=4)
    assert press.phase == 'idle'
    assert press.path == []
    assert press.reason == 'IK unavailable'
    assert press.retry_after_cycle == 5
    assert not press.observe(captured(), right_holding=True, feedback={}, cycle=5)

def test_blocked_descent_keeps_commanding_current_waypoint():
    press = PipettePress(PipettePressConfig(), make_piper_fk())
    assert press.plan(captured(), cycle=0)
    press.index = press.labels.index('down') + 3
    press.phase = 'down'
    physical = press.path[press.labels.index('down')].copy()
    blocked = press.path[press.index].copy()
    for attempt in range(40):
        (action, record) = press.next(physical)
        press.commit(physical, action, cycle=0)
        assert press.phase == 'down'
        assert record['wait_frames'] == attempt
        np.testing.assert_allclose(action, blocked)

def test_explicit_task_and_hold_flags(monkeypatch):
    base = ['monitor', '--checkpoint', '/unused', '--task', 'manipulate_pipette']
    monkeypatch.setattr(sys, 'argv', base)
    assert not __import__('pi05_assists.options', fromlist=['parse_options']).parse_options([x for x in sys.argv[1:] if x not in ('--checkpoint', '/unused')]).pipette_press
    monkeypatch.setattr(sys, 'argv', [*base, '--pipette-press'])
    with pytest.raises(SystemExit):
        __import__('pi05_assists.options', fromlist=['parse_options']).parse_options([x for x in sys.argv[1:] if x not in ('--checkpoint', '/unused')])
    monkeypatch.setattr(sys, 'argv', [*base, '--pipette-press', '--right-grip-hold'])
    assert __import__('pi05_assists.options', fromlist=['parse_options']).parse_options([x for x in sys.argv[1:] if x not in ('--checkpoint', '/unused')]).pipette_press
