import numpy as np
import pytest
from pi05_assists.grip_hold import GripHoldConfig, RightGripHold, config_from_args, LeftGripHold, LeftGripHoldConfig, left_config_from_args

def sample(grip=0.8, z=100.0):
    s = np.zeros(14, dtype=np.float32)
    (s[13], s[9]) = (grip, z)
    return s

def helper():
    return RightGripHold(GripHoldConfig(), lambda q: np.array([0.0, 0.0, q[2]]))

def advance(h, grip=0.15, z=100.0, model=0.15, previous=0.1, token=(0, 0), phase='idle', region=False):
    step = h.next(sample(grip, z), sample(previous), model, token=token, handover=dict(phase=phase, in_handover_region=region))
    h.commit(step)
    return step

def lifted():
    h = helper()
    advance(h, grip=0.8)
    advance(h)
    advance(h, z=105.0)
    step = advance(h, z=115.0)
    assert step.force_close
    return h

def test_requires_actual_closure_stability_and_lift_not_just_action():
    h = helper()
    advance(h, grip=0.8)
    assert not advance(h, grip=0.8, z=150.0, model=0.0).force_close
    assert not advance(h, z=100.0).force_close
    assert not advance(h, z=115.0).force_close
    step = advance(h, z=116.0)
    assert step.force_close and step.state.baseline == pytest.approx(0.15)

def test_starting_closed_does_not_imply_grasp_and_low_grasp_does_not_imply_lift():
    h = helper()
    for z in [100.0, 110.0, 120.0, 130.0]:
        assert not advance(h, z=z).force_close
    advance(h, grip=0.8)
    for _ in range(5):
        assert not advance(h).force_close
    assert h.state.phase == 'closing'

def test_unstable_or_reopened_gripper_does_not_confirm_grasp():
    h = helper()
    advance(h, grip=0.8)
    for (g, z) in [(0.2, 100.0), (0.1, 120.0), (0.2, 125.0), (0.8, 140.0)]:
        assert not advance(h, grip=g, z=z).force_close
    assert h.state.phase == 'idle'

def test_small_model_opening_is_suppressed_across_chunks():
    h = lifted()
    for c in range(4):
        step = advance(h, model=0.28, token=(c, 1), z=130.0)
        assert step.force_close and step.state.release_count == 0

def test_release_needs_absolute_relative_and_consecutive_distinct_rows():
    h = lifted()
    h.state.baseline = 0.24
    for r in range(3):
        assert advance(h, model=0.36, token=(1, r)).force_close
    assert advance(h, model=0.5, token=(2, 0)).state.release_count == 1
    for _ in range(5):
        assert advance(h, model=0.5, token=(2, 0)).state.release_count == 1
    assert advance(h, model=0.2, token=(2, 1)).state.release_count == 0
    assert advance(h, model=0.5, token=(2, 2)).force_close
    assert advance(h, model=0.5, token=(3, 0)).force_close
    step = advance(h, model=0.5, token=(3, 1))
    assert not step.force_close and step.record['events'] == ['released_by_model']

def test_release_waits_for_physical_open_before_rearming():
    h = lifted()
    for r in range(3):
        advance(h, model=0.8, token=(1, r))
    for r in range(5):
        assert not advance(h, z=150.0, token=(2, r)).force_close
    assert h.state.phase == 'released'
    advance(h, grip=0.8)
    advance(h)
    advance(h, z=110.0)
    assert advance(h, z=120.0).force_close

def test_handover_region_blocks_model_release_until_receiver_ready():
    h = lifted()
    for r in range(5):
        assert advance(h, model=0.9, token=(1, r), region=True).force_close
    for phase in ['waiting', 'ready']:
        assert advance(h, model=0.9, token=(2, 0), phase=phase).force_close
    step = advance(h, model=0.9, phase='releasing', region=True)
    assert not step.force_close
    assert step.record['events'] == ['released_by_handover']

def test_speculation_does_not_advance_evidence_until_command_sent():
    h = lifted()
    for _ in range(5):
        step = h.next(sample(), sample(0.0), 0.9, token=(1, 0), handover=dict(phase='idle', in_handover_region=False))
        assert step.state.release_count == 1
    assert h.state.release_count == 0

def left_step(h, model=0.2, *, token=(0, 0), phase='completed', complete=False, feedback=0.1, commit=True):
    physical = sample()
    physical[6] = feedback
    step = h.next(physical, model, token=token, handover=dict(phase=phase, events=['handover_completed'] if complete else []))
    if commit:
        h.commit(step)
    return step

def test_left_requires_completion_event_then_holds_after_separation():
    h = LeftGripHold(LeftGripHoldConfig())
    for phase in ['idle', 'waiting', 'ready', 'releasing', 'completed']:
        assert not left_step(h, phase=phase).force_close
    assert left_step(h, complete=True, model=0.9).force_close
    assert h.state.release_count == 0
    for c in range(1, 5):
        step = left_step(h, model=0.3, phase='idle', token=(c, 0))
        assert step.force_close and step.state.release_count == 0

def test_left_requires_absolute_relative_and_consecutive_distinct_opening_rows():
    h = LeftGripHold(LeftGripHoldConfig())
    left_step(h, complete=True, feedback=0.24)
    for r in range(1, 4):
        assert left_step(h, model=0.36, token=(0, r)).state.release_count == 0
    assert left_step(h, model=0.5, token=(1, 0)).state.release_count == 1
    for _ in range(5):
        assert left_step(h, model=0.5, token=(1, 0)).state.release_count == 1
    assert left_step(h, model=0.2, token=(1, 1)).state.release_count == 0
    assert left_step(h, model=0.5, token=(1, 2)).force_close
    assert left_step(h, model=0.5, token=(2, 0)).force_close
    step = left_step(h, model=0.5, token=(2, 1))
    assert not step.force_close and step.record['events'] == ['released_by_model']
    for phase in ['completed', 'idle', 'waiting', 'ready']:
        assert not left_step(h, phase=phase).force_close
    step = left_step(h, complete=True, feedback=0.08, token=(4, 0))
    assert step.force_close and step.state.baseline == pytest.approx(0.08)

def test_left_does_not_commit_on_failed_send_or_release_during_new_handover():
    h = LeftGripHold(LeftGripHoldConfig())
    left_step(h, complete=True, commit=False)
    assert h.state.phase == 'idle'
    left_step(h, complete=True)
    for _ in range(5):
        assert left_step(h, model=0.9, token=(1, 0), commit=False).state.release_count == 1
    assert h.state.release_count == 0
    left_step(h, model=0.9, token=(1, 0))
    for phase in ['waiting', 'ready', 'releasing']:
        step = left_step(h, model=0.9, phase=phase, token=(2, 0))
        assert step.force_close and step.state.release_count == 0

def test_capped_hold_preserves_smaller_model_actions_and_release_evidence():
    h = RightGripHold(GripHoldConfig(closed_threshold=0.3, command_cap=0.25), lambda q: np.array([0.0, 0.0, q[2]]))
    advance(h, grip=0.8)
    advance(h, grip=0.26, previous=0.25)
    advance(h, grip=0.26, previous=0.25, z=110.0)
    step = advance(h, grip=0.26, previous=0.25, z=120.0, model=0.28)
    assert step.force_close and step.record['hold_command'] == 0.25
    step = advance(h, grip=0.26, model=0.18, previous=0.25, z=120.0, token=(1, 0))
    assert step.record['hold_command'] == 0.18
    for r in range(3):
        step = advance(h, grip=0.26, model=0.5, token=(2, r))
    assert not step.force_close and step.record['hold_command'] is None
    left = LeftGripHold(LeftGripHoldConfig(command_cap=0.25))
    assert left_step(left, complete=True, model=0.5, feedback=0.26).record['hold_command'] == 0.25
    assert left_step(left, model=0.18, token=(1, 0)).record['hold_command'] == 0.18
    for r in range(3):
        step = left_step(left, model=0.5, token=(2, r))
    assert not step.force_close and step.record['hold_command'] is None

def test_pipette_hold_ignores_even_sustained_model_opening():
    h = RightGripHold(GripHoldConfig(closed_threshold=0.45, open_threshold=0.55, model_release_enabled=False), lambda q: np.array([0.0, 0.0, q[2]]))
    advance(h, grip=0.8)
    advance(h, grip=0.38, previous=0.02)
    advance(h, grip=0.38, previous=0.02, z=110.0)
    step = advance(h, grip=0.38, previous=0.02, z=120.0, model=0.9)
    assert step.force_close and step.record['hold_command'] == 0.0
    for n in range(50):
        step = advance(h, grip=0.38, previous=0.0, model=0.99, z=130.0, token=(n, 0))
        assert step.force_close and step.state.release_count == 0
        assert step.record['events'] == []
