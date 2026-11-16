import numpy as np
import pytest
from pi05_assists.grasp_dip import GraspDip, GraspDipConfig, make_piper_fk
from pi05_assists.grasp_dip_runtime import GraspDipRuntime

def fk(q):
    return np.array([100.0, 0.0, 160.0 - 100.0 * q[1] + 100.0 * q[4]])

def actions(grips):
    rows = np.zeros((len(grips), 14), dtype=np.float32)
    rows[:, 6] = 0.8
    rows[:, 13] = grips
    return rows

def helper(**kwargs):
    cfg = GraspDipConfig(target_z_mm=(None, 120.0), z_ranges=(None, (120.0, 180.0)), max_drop_mm=130.0, **kwargs)
    return GraspDip(cfg, fk, down_tilt=lambda q: 0.0)

def runtime(h, rows, physical, previous=None, tail=None):
    previous = physical if previous is None else previous
    plan = h.plan(rows, previous_action=previous, physical_state=physical)
    return GraspDipRuntime(h, rows, plan, lookahead=tail)

def execute(rt, physical, previous=None):
    previous = physical.copy() if previous is None else previous.copy()
    sent = []
    for _ in range(100):
        step = rt.next(physical, previous)
        if step is None:
            return (sent, physical, previous)
        sent.append((step.kind, step.source_row, step.action.copy()))
        physical = step.action.copy()
        previous = step.action.copy()
        rt.commit(step)
    pytest.fail('runtime did not finish')

def test_tail_hint_prepares_without_executing_tail_and_carries_to_next_chunk():
    h = helper()
    physical = actions([0.9])[0]
    rows = actions([0.7, 0.4, 0.175])
    tail = actions([0.05])
    tail[0, 0] = 0.5
    rt = runtime(h, rows, physical, tail=tail)
    (sent, physical, previous) = execute(rt, physical)
    assert [s[0] for s in sent] == ['grasp_dip_prepare'] * 3 + ['model'] * 3
    np.testing.assert_allclose([s[2][13] for s in sent[:3]], 0.9)
    np.testing.assert_allclose([fk(s[2][7:13])[2] for s in sent[:3]], [146.6667, 133.3333, 120.0], atol=0.001)
    np.testing.assert_array_equal([s[2][0] for s in sent], 0.0)
    assert [s[1] for s in sent] == [0, 0, 0, 0, 1, 2]
    next_rows = actions([0.05, 0.05])
    next_rows[:, 11] = 0.02
    second = runtime(h, next_rows, physical, previous)
    (sent, _, _) = execute(second, physical, previous)
    assert sent[0][0] == 'model'
    assert fk(sent[0][2][7:13])[2] == pytest.approx(120.0, abs=0.001)
    assert sent[0][2][13] == pytest.approx(0.05)

def test_first_row_closed_in_new_chunk_gets_preparation_even_without_tail():
    h = helper()
    physical = actions([0.307])[0]
    previous = actions([0.175])[0]
    rt = runtime(h, actions([0.001, 0.001]), physical, previous)
    (sent, _, _) = execute(rt, physical, previous)
    assert len(sent) == 5
    assert all((s[2][13] == pytest.approx(0.307) for s in sent[:3]))
    assert sent[3][2][13] == pytest.approx(0.001)
    assert fk(sent[3][2][7:13])[2] == pytest.approx(120.0, abs=0.001)

@pytest.mark.parametrize('model_z', [120.0, 119.9, 118.0, 120.0005])
def test_prepared_grasp_accepts_model_already_at_or_below_target(model_z):
    h = helper()
    physical = actions([0.9])[0]
    rt = runtime(h, actions([0.7, 0.4, 0.175]), physical, tail=actions([0.05]))
    (_, physical, previous) = execute(rt, physical)
    next_rows = actions([0.175, 0.05, 0.05])
    next_rows[:, 8] = (160.0 - model_z) / 100.0
    second = runtime(h, next_rows, physical, previous)
    (sent, _, _) = execute(second, physical, previous)
    np.testing.assert_array_equal(np.stack([s[2] for s in sent]), next_rows)
    assert second.plan.events[0]['status'] == 'satisfied'
    assert h._states[1].closed
    assert h._states[1].grasp_offset == 0.0
    lifted = actions([0.05, 0.05])
    (_, physical, previous) = execute(runtime(h, lifted, physical, previous), physical, previous)
    np.testing.assert_array_equal(previous, lifted[-1])

def test_feedback_is_rechecked_after_initial_height_was_outside_range():
    h = helper()
    physical = actions([0.9])[0]
    physical[11] = 0.6
    rt = runtime(h, actions([0.8, 0.6, 0.4, 0.05]), physical)
    first = rt.next(physical, physical)
    assert first.kind == 'model'
    rt.commit(first)
    physical = first.action.copy()
    second = rt.next(physical, physical)
    assert second.kind == 'grasp_dip_prepare'
    assert rt.events[-1]['physical_z_mm'] == pytest.approx(160.0)

def test_preparation_waits_for_feedback_then_timeout_falls_back_to_model():
    h = helper()
    physical = actions([0.7])[0]
    rt = runtime(h, actions([0.05]), physical)
    previous = physical.copy()
    for _ in range(3):
        step = rt.next(physical, previous)
        assert step.kind == 'grasp_dip_prepare'
        rt.commit(step)
        previous = step.action.copy()
    for _ in range(rt.MAX_SETTLE_STEPS):
        step = rt.next(physical, previous)
        assert step.kind == 'grasp_dip_wait'
        assert step.action[13] > 0.1
        rt.commit(step)
    step = rt.next(physical, previous)
    assert step.kind == 'model'
    assert rt.events[-1]['status'] == 'fallback'
    assert rt.row == 0

def test_preparation_resumes_only_after_feedback_is_close_to_target():
    h = helper()
    physical = actions([0.7])[0]
    rt = runtime(h, actions([0.05]), physical)
    previous = physical.copy()
    for _ in range(3):
        step = rt.next(physical, previous)
        rt.commit(step)
        previous = step.action.copy()
    waiting = rt.next(physical, previous)
    assert waiting.kind == 'grasp_dip_wait'
    rt.commit(waiting)
    step = rt.next(previous, previous)
    assert step.kind == 'model'
    assert step.action[13] <= 0.1
    assert rt.events[-1]['status'] == 'ready'

def test_uncommitted_preparation_does_not_advance_helper_or_model_cursor():
    h = helper()
    physical = actions([0.7])[0]
    rt = runtime(h, actions([0.05]), physical)
    step = rt.next(physical, physical)
    assert step.kind == 'grasp_dip_prepare'
    assert rt.row == rt.prep_row == 0
    assert h._states[1].phase == 'idle'

def test_already_closed_gripper_is_not_reopened():
    h = helper()
    physical = actions([0.05])[0]
    rt = runtime(h, actions([0.001]), physical, actions([0.175])[0])
    assert rt.next(physical, physical).kind == 'model'

@pytest.mark.parametrize('failure', ['drop', 'tilt', 'joint_limit'])
def test_fallback_keeps_remaining_constraints(failure):
    h = helper()
    if failure == 'drop':
        h = GraspDip(GraspDipConfig(target_z_mm=(None, 120.0), z_ranges=(None, (120.0, 180.0))), fk, down_tilt=lambda q: 0.0)
    elif failure == 'tilt':
        h.down_tilt = lambda q: 90.0
    else:
        h.fk = lambda q: np.array([0.0, 0.0, 160.0])
    physical = actions([0.7])[0]
    rt = runtime(h, actions([0.05]), physical)
    assert rt.next(physical, physical).kind == 'model'
    assert rt.events[0]['reason'] == {'drop': 'max_drop_exceeded', 'tilt': 'physical_orientation_not_downward', 'joint_limit': 'target_z_unreachable_within_j2_limit'}[failure]

def test_both_arms_prepare_and_other_joints_are_held():
    h = GraspDip(GraspDipConfig(target_z_mm=(120.0, 130.0), z_ranges=((120.0, 180.0), (120.0, 180.0)), max_drop_mm=130.0), fk, down_tilt=lambda q: 0.0)
    physical = actions([0.7])[0]
    physical[[0, 7]] = [0.1, 0.2]
    rows = actions([0.05])
    rows[0, 6] = 0.05
    (sent, _, _) = execute(runtime(h, rows, physical), physical)
    for (_, _, row) in sent[:3]:
        np.testing.assert_allclose(row[[0, 7, 6, 13]], [0.1, 0.2, 0.8, 0.7])
    assert fk(sent[2][2][:6])[2] == pytest.approx(120.0, abs=0.001)
    assert fk(sent[2][2][7:13])[2] == pytest.approx(130.0, abs=0.001)

def test_sdk_low_grasp_reaches_120_without_extra_angle_cap():
    sdk_fk = make_piper_fk()
    q = np.array([0.14, 2.25, -1.82, 0.046, 1.19, 0.414], dtype=np.float32)
    h = GraspDip(GraspDipConfig(target_z_mm=(None, 120.0), z_ranges=(None, (120.0, 180.0)), max_drop_mm=130.0), sdk_fk)
    (delta, reason) = h.solve_joint2(q, 120.0, 1)
    assert reason is None
    assert delta > 0.05
    q[1] += delta
    assert 0 <= q[1] <= 3.14
    assert sdk_fk(q)[2] == pytest.approx(120.0, abs=0.001)

def test_cancelled_prediction_releases_preparation_on_lift_or_reopen():
    for lift in (True, False):
        h = helper()
        physical = actions([0.7])[0]
        rt = runtime(h, actions([0.4, 0.175]), physical, tail=actions([0.05]))
        (_, physical, previous) = execute(rt, physical)
        cancelled = actions([0.175] * 4 if lift else [0.99] * 4)
        if lift:
            cancelled[:, 11] = 0.2
        second = runtime(h, cancelled, physical, previous)
        (sent, _, _) = execute(second, physical, previous)
        np.testing.assert_allclose([s[2][8] for s in sent], [0.4 * 2 / 3, 0.4 / 3, 0.0, 0.0], atol=1e-06)
