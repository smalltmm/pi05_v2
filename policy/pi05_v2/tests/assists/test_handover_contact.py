from types import SimpleNamespace
import numpy as np
import pytest
from .test_handover import assist, state

def feedback(n, valid=True):
    return {side: dict(stamp=float(n), valid=valid) for side in ('left', 'right')}

def setup_contact():
    h = assist(gripper_cap=0.2, left_closed_threshold=0.3, right_closed_threshold=0.3, right_open_threshold=0.35)
    s = state(left_grip=0.6, right_grip=0.24)
    previous = s.copy()
    previous[[6, 13]] = [0.2, 0.2]
    model = np.tile(s, (10, 1))
    model[:, [6, 13]] = [0.5, 0.6]
    return (h, s, previous, model)

def ready_contact():
    (h, s, previous, model) = setup_contact()
    for (n, value) in enumerate([0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4], 1):
        s[6] = value
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
    assert h.state.phase == 'ready'
    assert step.record['confirmation'] == 'contact'
    assert not step.record['left_threshold_stable']
    return (h, s, previous, model)

def test_contact_waits_for_fresh_inference_then_ramps_until_model_target_reached():
    (h, s, previous, model) = ready_contact()
    step = h.next(s, previous, model, feedback=feedback(8))
    h.commit(step)
    assert step.kind == 'handover_hold'
    h.begin_inference_result()
    for n in range(9, 24):
        model[:, 0] = 0.1
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        assert step.action[13] <= min(previous[13], s[13]) + 0.050001
        np.testing.assert_array_equal(step.action[:6], h.state.hold[:6])
        assert step.action[6] == pytest.approx(0.2)
        if s[13] > 0.35 and s[13] < 0.58:
            assert h.state.phase == 'releasing'
        if h.state.phase == 'completed':
            break
        previous = step.action
        s[13] = previous[13]
    assert h.state.phase == 'completed'
    assert 'handover_completed' in step.record['events']

@pytest.mark.parametrize('failure', ['stale', 'invalid', 'no_feedback', 'moving', 'opening_command', 'disabled'])
def test_contact_does_not_confirm_without_required_evidence(failure):
    (h, s, previous, model) = setup_contact()
    if failure == 'disabled':
        from dataclasses import replace
        h.config = replace(h.config, contact_confirmation=False)
    for n in range(1, 22):
        s[6] = 0.6 if n == 1 else 0.4
        if failure == 'moving':
            s[6] = 0.4 + 0.03 * (n % 2)
        fb = None if failure == 'no_feedback' else feedback(1 if failure == 'stale' else n, failure != 'invalid')
        if failure == 'opening_command':
            previous[6] = 0.5
        step = h.next(s, previous, model, feedback=fb)
        h.commit(step)
        assert step.state.phase == 'waiting'
        previous = step.action.copy()

@pytest.mark.parametrize('lost', ['invalid', 'reopen'])
def test_contact_loss_reblocks_even_after_old_open_threshold(lost):
    (h, s, previous, model) = ready_contact()
    h.begin_inference_result()
    for n in range(8, 14):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
        s[13] = previous[13]
    assert s[13] > 0.35
    if lost == 'reopen':
        s[6] = 0.45
    step = h.next(s, previous, model, feedback=feedback(14, valid=lost != 'invalid'))
    h.commit(step)
    assert step.state.phase == 'waiting'
    assert step.action[13] == pytest.approx(0.2)
    assert 'left_closure_lost' in step.record['events']
    if lost == 'invalid':
        for n in range(15, 23):
            step = h.next(s, step.action, model, feedback=feedback(n))
            h.commit(step)
        assert h.state.phase == 'ready'

def test_contact_uses_sent_action_and_does_not_commit_failed_send():
    (h, s, previous, model) = setup_contact()
    for n in range(1, 4):
        s[6] = 0.6 if n == 1 else 0.4
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
    for _ in range(8):
        step = h.next(s, previous, model, feedback=feedback(4))
        assert step.state.phase == 'waiting'
    assert len(h.state.contact_samples) == 3
    previous[6] = 0.39
    model[:, 6] = 0.0
    step = h.next(s, previous, model, feedback=feedback(5))
    assert step.record['left_contact_frames'] == 0
    assert step.record['left_action_state_gap'] == pytest.approx(0.01, abs=1e-06)

def test_already_blocked_gripper_needs_only_one_stable_window():
    from dataclasses import replace
    (h, s, previous, model) = setup_contact()
    h.config = replace(h.config, contact_close_delta=0.9, contact_settle_frames=100)
    s[6] = 0.4
    for n in range(1, 6):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
        assert step.state.phase == ('waiting' if n < 5 else 'ready')
    assert step.record['confirmation'] == 'contact'
    assert not step.record['left_observed_closing']

@pytest.mark.parametrize('releasing', [False, True])
def test_recording_15_rebound_recovers_without_another_shrink(releasing):
    h = assist()
    s = state(left_grip=0.135, right_grip=0.076)
    previous = s.copy()
    previous[[6, 13]] = 0.0
    model = np.tile(s, (10, 1))
    model[:, 13] = 0.6
    for n in range(1, 6):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
    assert h.state.phase == 'ready'
    if releasing:
        h.begin_inference_result()
    s[6] = 0.148
    step = h.next(s, previous, model, feedback=feedback(6))
    h.commit(step)
    assert h.state.phase == 'waiting'
    assert h.state.contact_reference is None
    previous = step.action
    s[6] = 0.151
    for n in range(7, 12):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
    assert h.state.phase == 'ready'
    assert not step.record['left_observed_closing']
    assert not step.record['left_threshold_stable']
    h.begin_inference_result()
    step = h.next(s, previous, model, feedback=feedback(12))
    assert step.kind == 'handover_release'
    assert step.action[13] == pytest.approx(0.05)

def test_duplicate_valid_feedback_neither_counts_nor_clears_contact():
    (h, s, previous, model) = setup_contact()
    s[6] = 0.4
    for n in range(1, 6):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
        for _ in range(3):
            step = h.next(s, previous, model, feedback=feedback(n))
            h.commit(step)
            assert step.record['left_contact_frames'] == n
            assert not step.record['left_feedback_fresh']
            assert h.state.phase == ('waiting' if n < 5 else 'ready')
            previous = step.action
    h.begin_inference_result()
    fb = feedback(6)
    fb['left']['stamp'] = 5.0
    step = h.next(s, previous, model, feedback=fb)
    h.commit(step)
    assert step.kind == 'handover_release'
    assert step.record['left_contact_stable']
    assert step.action[13] > previous[13]

def test_invalid_feedback_clears_partial_contact_evidence():
    (h, s, previous, model) = setup_contact()
    s[6] = 0.4
    for n in range(1, 4):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
    step = h.next(s, previous, model, feedback=feedback(4, valid=False))
    h.commit(step)
    assert step.record['left_contact_frames'] == 0
    previous = step.action
    for n in range(5, 10):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
        assert h.state.phase == ('waiting' if n < 9 else 'ready')

def test_duplicate_valid_feedback_preserves_threshold_confirmation():
    h = assist()
    s = state(left_grip=0.05)
    previous = s.copy()
    model = np.tile(s, (10, 1))
    model[:, 13] = 0.6
    for n in range(1, 4):
        step = h.next(s, previous, model, feedback=feedback(n))
        h.commit(step)
        previous = step.action
        repeated = h.next(s, previous, model, feedback=feedback(n))
        h.commit(repeated)
        assert len(h.state.samples) == n
    assert h.state.confirmation == 'threshold'
    h.begin_inference_result()
    step = h.next(s, previous, model, feedback=feedback(3))
    assert step.kind == 'handover_release'
