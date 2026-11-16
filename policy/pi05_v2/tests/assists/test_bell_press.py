import numpy as np
import pytest
from pi05_assists.bell_press import BellPress, BellPressConfig, BellPressRuntime

def fk(q):
    return np.array([0.0, 0.0, 160.0 - 100.0 * q[1]])
fk.down_tilt_deg = lambda q: 0.0

def actions(zs, grip=0.03):
    a = np.zeros((len(zs), 14), dtype=np.float32)
    a[:, 8] = (160.0 - np.asarray(zs)) / 100.0
    a[:, 13] = grip
    a[:, 6] = 0.8
    return a

def helper(**kwargs):
    return BellPress(BellPressConfig(z_range_mm=(120.0, 180.0), target_z_mm=130.0, max_drop_mm=50.0, **kwargs), fk)

def execute(h, a, physical, *, tail=None, feedback=True):
    rt = BellPressRuntime(h, a, lookahead=tail)
    sent = []
    previous = physical.copy()
    for _ in range(100):
        step = rt.next(physical, previous)
        if step is None:
            return (rt, sent, physical)
        sent.append(step)
        rt.commit(step)
        previous = step.action.copy()
        if feedback:
            physical = previous.copy()
    pytest.fail('press loop did not finish')

def test_closed_gripper_press_ramps_holds_and_releases_on_model_lift():
    h = helper()
    original = actions([150.0, 145.0, 140.0, 145.0, 155.0, 160.0])
    (rt, sent, _) = execute(h, original, actions([155.0])[0])
    down = [s for s in sent if s.kind == 'bell_press_down']
    hold = [s for s in sent if s.kind == 'bell_press_hold']
    assert len(down) == 2 and len(hold) == 2
    assert [s.source_row for s in down] == [2, 2]
    np.testing.assert_allclose([fk(s.action[7:13])[2] for s in down], [135.0, 130.0], atol=0.001)
    np.testing.assert_allclose([fk(s.action[7:13])[2] for s in hold], 130.0, atol=0.001)
    np.testing.assert_allclose([s.action[13] for s in sent], 0.03)
    np.testing.assert_allclose([s.action[:7] for s in sent], np.tile(original[0, :7], (len(sent), 1)))
    np.testing.assert_array_equal(sent[-1].action, original[-1])
    assert [e['status'] for e in rt.events].count('press_command_sent') == 1

def test_no_retrigger_while_low_across_inference_then_rearm_after_feedback_rise():
    h = helper(rearm_rise_mm=20.0)
    (_, sent, physical) = execute(h, actions([140.0, 140.0]), actions([145.0])[0])
    assert len([s for s in sent if s.kind == 'bell_press_down']) == 2
    (rt, sent, physical) = execute(h, actions([140.0, 140.0]), physical)
    assert all((s.kind == 'model' for s in sent))
    assert not h.state.armed
    (_, _, physical) = execute(h, actions([150.0, 160.0, 160.0, 160.0]), physical)
    assert h.state.armed
    (rt, sent, _) = execute(h, actions([145.0, 140.0, 140.0]), physical)
    assert len([s for s in sent if s.kind == 'bell_press_down']) == 2

def test_tail_low_point_is_only_a_hint_and_not_executed():
    h = helper()
    physical = actions([145.0])[0]
    (rt, sent, physical) = execute(h, actions([140.0]), physical, tail=actions([140.0, 155.0]))
    assert len(sent) == 5
    assert all((s.source_row == 0 for s in sent))
    assert fk(sent[-1].action[7:13])[2] == pytest.approx(130.0, abs=0.001)
    (rt, sent, _) = execute(h, actions([140.0, 150.0]), physical)
    assert all((s.kind == 'model' for s in sent))

def test_new_inference_low_pose_does_not_accumulate_press_offset():
    h = helper()
    (_, _, physical) = execute(h, actions([140.0, 140.0]), actions([145.0])[0])
    (rt, sent, _) = execute(h, actions([135.0, 132.0, 130.0]), physical)
    assert all((s.kind == 'model' for s in sent))
    np.testing.assert_allclose([fk(s.action[7:13])[2] for s in sent], 130.0, atol=0.001)

def test_small_lift_crossing_drop_limit_releases_instead_of_pausing():
    h = BellPress(BellPressConfig(z_range_mm=(130.0, 180.0), target_z_mm=130.0, max_drop_mm=10.0), fk)
    (_, _, physical) = execute(h, actions([139.9, 139.9]), actions([140.5])[0])
    model = actions([140.6, 140.8, 141.0])
    (rt, sent, _) = execute(h, model, physical)
    assert all((s.kind == 'model' for s in sent))
    correction = np.array([s.action[8] - model[i, 8] for (i, s) in enumerate(sent)])
    np.testing.assert_allclose(correction, [0.099 * 2 / 3, 0.099 / 3, 0.0], atol=1e-06)
    assert rt.events[0]['status'] == 'releasing'
    assert rt.events[0]['reason'] == 'target_drop_exceeds_limit'
    assert all((fk(s.action[7:13])[2] >= fk(model[i, 7:13])[2] - 10.0 for (i, s) in enumerate(sent)))

def test_active_solver_failure_falls_back_to_original_model_and_waits_for_rearm(monkeypatch):
    h = helper()
    (_, _, physical) = execute(h, actions([140.0, 140.0]), actions([145.0])[0])
    monkeypatch.setattr(h.solver, 'solve_joint2', lambda *args: (None, 'target_z_unreachable_within_j2_limit'))
    model = actions([139.0, 139.0])
    (rt, sent, _) = execute(h, model, physical)
    np.testing.assert_array_equal(np.stack([s.action for s in sent]), model)
    assert rt.events[0]['status'] == 'fallback_to_model'
    assert not h.state.armed

@pytest.mark.parametrize('condition', ['open_feedback', 'open_model', 'high', 'sideways', 'still', 'rising'])
def test_ineligible_approaches_do_not_press(condition):
    h = helper()
    physical = actions([145.0])[0]
    a = actions([140.0, 140.0])
    if condition == 'open_feedback':
        physical[13] = 0.8
    elif condition == 'open_model':
        a[:, 13] = 0.8
    elif condition == 'high':
        physical[8] = -0.3
        a[:, 8] = -0.3
    elif condition == 'sideways':
        h.solver.down_tilt = lambda q: 90.0
    elif condition == 'still':
        physical = a[0].copy()
    elif condition == 'rising':
        physical = actions([135.0])[0]
        a = actions([140.0, 145.0])
    (rt, sent, _) = execute(h, a, physical, feedback=False)
    assert all((s.kind == 'model' for s in sent))
    np.testing.assert_array_equal(np.stack([s.action for s in sent]), a)

def test_no_contact_or_arrival_wait_even_when_feedback_cannot_reach_target():
    h = helper()
    (rt, sent, _) = execute(h, actions([140.0, 140.0]), actions([145.0])[0], feedback=False)
    assert len(sent) == 6
    assert sum((s.kind == 'bell_press_hold' for s in sent)) == 2
    assert any((e['status'] == 'hold_complete' for e in rt.events))

def test_already_low_model_is_not_pressed_again():
    h = helper()
    a = actions([130.0, 129.9])
    (rt, sent, _) = execute(h, a, actions([132.0])[0])
    np.testing.assert_array_equal(np.stack([s.action for s in sent]), a)
    assert any((e['status'] == 'target_already_satisfied' for e in rt.events))

def test_abandoned_press_does_not_change_committed_state():
    h = helper()
    rt = BellPressRuntime(h, actions([140.0, 140.0]))
    physical = actions([145.0])[0]
    step = rt.next(physical, physical)
    assert step.kind == 'bell_press_down'
    assert h.state.armed and h.state.offset == 0
    assert rt.row == rt.inserted_index == 0

def test_max_drop_and_joint_limits_still_apply():
    h = BellPress(BellPressConfig(z_range_mm=(120.0, 180.0), target_z_mm=130.0), fk)
    (rt, sent, _) = execute(h, actions([140.0, 140.0]), actions([145.0])[0])
    assert all((s.kind == 'model' for s in sent))
    assert rt.events[0]['reason'] == 'max_drop_exceeded'

def dual_actions(left_z, right_z, left_grip=0.02, right_grip=0.03):
    result = actions(right_z, right_grip)
    result[:, 1] = (160.0 - np.asarray(left_z)) / 100.0
    result[:, 6] = left_grip
    return result

def execute_dual(helper, model, physical):
    from pi05_assists.bell_press import DualBellPressRuntime
    runtime = DualBellPressRuntime(helper, model)
    previous = physical.copy()
    sent = []
    for _ in range(100):
        step = runtime.next(physical, previous)
        if step is None:
            return (runtime, sent, physical)
        runtime.commit(step)
        sent.append(step)
        physical = previous = step.action.copy()
    pytest.fail('dual press did not finish')

@pytest.mark.parametrize('enabled', [(0,), (1,), (0, 1)])
def test_each_enabled_arm_reaches_its_own_target_and_disabled_arm_is_held(enabled):
    from pi05_assists.bell_press import DualBellPress
    configs = {side: BellPressConfig(z_range_mm=(120.0, 180.0), target_z_mm=130.0 + side * 5, max_drop_mm=50.0) for side in enabled}
    helper = DualBellPress(configs, fk)
    model = dual_actions([140.0, 140.0], [145.0, 145.0])
    physical = dual_actions([150.0], [150.0])[0]
    (runtime, sent, _) = execute_dual(helper, model, physical)
    assert len(sent) == 6
    assert [s.source_row for s in sent] == [0] * 5 + [1]
    for side in enabled:
        assert fk(sent[1].action[side * 7:side * 7 + 6])[2] == pytest.approx(configs[side].target_z_mm, abs=0.001)
        assert not helper.arms[side].state.armed
        assert any((e['side'] == ('left', 'right')[side] and e['status'] == 'press_command_sent' for e in runtime.events))
    for side in set((0, 1)) - set(enabled):
        np.testing.assert_array_equal(np.stack([s.action[side * 7:side * 7 + 7] for s in sent[:4]]), np.tile(physical[side * 7:side * 7 + 7], (4, 1)))
        np.testing.assert_array_equal(sent[-1].action[side * 7:side * 7 + 7], model[-1, side * 7:side * 7 + 7])
    np.testing.assert_allclose([s.action[[6, 13]] for s in sent], np.tile([0.02, 0.03], (6, 1)))

def test_dual_press_staggered_trigger_and_independent_rearm_across_chunks():
    from pi05_assists.bell_press import DualBellPress
    configs = {side: BellPressConfig(z_range_mm=(120.0, 180.0), target_z_mm=130.0, max_drop_mm=50.0) for side in (0, 1)}
    helper = DualBellPress(configs, fk)
    model = dual_actions([140.0, 140.0, 140.0], [150.0, 145.0, 140.0])
    physical = dual_actions([145.0], [155.0])[0]
    (runtime, sent, physical) = execute_dual(helper, model, physical)
    starts = [(e['side'], e['source_row']) for e in runtime.events if e['status'] == 'press_command_sent']
    assert starts == [('left', 0), ('right', 2)]
    assert len(sent) == 11
    right_steps = [s for s in sent if s.kind != 'model' and s.source_row == 2]
    np.testing.assert_allclose([fk(s.action[:6])[2] for s in right_steps], 130.0, atol=0.001)
    lift = dual_actions([150.0, 160.0, 160.0, 160.0], [140.0] * 4)
    (_, _, physical) = execute_dual(helper, lift, physical)
    assert helper.arms[0].state.armed
    assert not helper.arms[1].state.armed
    (runtime, sent, _) = execute_dual(helper, dual_actions([140.0, 140.0], [140.0, 140.0]), physical)
    assert [e['side'] for e in runtime.events if e['status'] == 'press_command_sent'] == ['left']

def test_one_arm_fallback_does_not_cancel_other_arm_hold(monkeypatch):
    from pi05_assists.bell_press import DualBellPress
    configs = {side: BellPressConfig(z_range_mm=(120.0, 180.0), target_z_mm=130.0, max_drop_mm=50.0) for side in (0, 1)}
    helper = DualBellPress(configs, fk)
    (_, _, physical) = execute_dual(helper, dual_actions([140.0, 140.0], [140.0, 140.0]), dual_actions([145.0], [145.0])[0])
    monkeypatch.setattr(helper.arms[0].solver, 'solve_joint2', lambda *args: (None, 'max_drop_exceeded'))
    model = dual_actions([139.0, 139.0], [139.0, 139.0])
    (runtime, sent, _) = execute_dual(helper, model, physical)
    np.testing.assert_array_equal(np.stack([s.action[:7] for s in sent]), model[:, :7])
    np.testing.assert_allclose([fk(s.action[7:13])[2] for s in sent], 130.0, atol=0.001)
    assert any((e['side'] == 'left' and e['status'] == 'fallback_to_model' for e in runtime.events))

@pytest.mark.parametrize('distance,steps', [(5.0, 1), (10.0, 1), (15.0, 2), (25.0, 3), (40.0, 4), (80.0, 8)])
@pytest.mark.parametrize('nonlinear', [False, True])
def test_default_press_uses_one_control_period_per_ten_mm_including_remainder(distance, steps, nonlinear):

    def geometry(q):
        return np.array([0.0, 0.0, 160.0 - 100.0 * (q[1] ** 2 if nonlinear else q[1])])
    geometry.down_tilt_deg = lambda q: 0.0
    cfg = BellPressConfig(z_range_mm=(50.0, 180.0), target_z_mm=150.0 - distance, max_drop_mm=100.0)
    h = BellPress(cfg, geometry)
    physical = actions([150.0])[0]
    model = actions([149.0, 149.0])
    if nonlinear:
        physical[8] = np.sqrt(physical[8])
        model[:, 8] = np.sqrt(model[:, 8])
    (rt, sent, _) = execute(h, model, physical)
    down = [s for s in sent if s.kind == 'bell_press_down']
    assert len(down) == steps
    expected = 150.0 - np.minimum(np.arange(1, steps + 1) * 10.0, distance)
    np.testing.assert_allclose([geometry(s.action[7:13])[2] for s in down], expected, atol=0.001)
    assert rt.events[0]['down_seconds'] == pytest.approx(steps * 0.1)
    assert rt.events[0]['down_speed_mm_s'] == 100.0
    assert len([s for s in sent if s.kind == 'bell_press_hold']) == 2

@pytest.mark.parametrize('speed', [0, -1, float('nan'), float('inf')])
def test_invalid_descent_speed_rejected(speed):
    with pytest.raises(ValueError, match='down-speed'):
        BellPressConfig(z_range_mm=(130.0, 180.0), target_z_mm=130.0, down_speed_mm_s=speed)
