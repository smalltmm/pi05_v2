"""Forward/down grasp corrections without CAN or real motion."""
import numpy as np
import pytest
from pi05_assists.grasp_dip import GraspDip
from pi05_assists.grasp_dip import GraspDipConfig
from pi05_assists.grasp_dip_runtime import GraspDipRuntime

class FK:

    def __call__(self, q):
        return np.array([1000 * q[0], 1000 * q[2], 250 - 100 * q[1]])

    def pose(self, q):
        return np.r_[self(q), [180, 0, 0]]

    def down_tilt_deg(self, q):
        return 0.0

def setup(forward=25.0):
    h = GraspDip(GraspDipConfig(target_z_mm=(None, 130), z_ranges=(None, (130, 180)), close_threshold=0.3, open_threshold=0.4, max_drop_mm=132, forward_mm=(0.0, forward)), FK())
    state = np.array([0.3, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8] * 2, dtype=np.float32)
    return (h, state)

def test_forward_and_down_ramp_hold_fade_preserve_grippers():
    (h, state) = setup()
    actions = np.tile(state, (8, 1))
    actions[3:, 13] = 0.2
    actions[5:, 8] = 0.8
    plan = h.plan(actions, previous_action=state, physical_state=state)
    assert plan.events[0]['status'] == 'applied'
    np.testing.assert_array_equal(plan.actions[:, :7], actions[:, :7])
    np.testing.assert_array_equal(plan.actions[:, 13], actions[:, 13])
    poses = np.array([h._xyz(row[7:13]) for row in plan.actions])
    np.testing.assert_allclose(poses[2:5, 0], 325.0, atol=0.01)
    np.testing.assert_allclose(poses[2:5, 2], 130.0, atol=0.01)
    assert poses[0, 0] < poses[1, 0] < poses[2, 0]
    np.testing.assert_allclose(plan.actions[-1], actions[-1], atol=1e-06)
    assert h._states[1].phase == 'idle'

def test_boundary_waits_for_x_and_replan_does_not_add_another_25_mm():
    (h, physical) = setup()
    actions = np.tile(physical, (3, 1))
    actions[:, 13] = 0.2
    initial = h.plan(actions, previous_action=physical, physical_state=physical)
    runtime = GraspDipRuntime(h, actions, initial)
    previous = physical.copy()
    for _ in range(3):
        step = runtime.next(physical, previous)
        assert step.kind == 'grasp_dip_prepare'
        assert step.action[13] == pytest.approx(0.8)
        runtime.commit(step)
        previous = step.action.copy()
    physical[8] = previous[8]
    step = runtime.next(physical, previous)
    assert step.kind == 'grasp_dip_wait'
    assert step.action[13] == pytest.approx(0.8)
    runtime.commit(step)
    physical = previous.copy()
    new_actions = np.tile(physical, (3, 1))
    new_actions[:, 13] = 0.2
    new_plan = h.plan(new_actions, previous_action=physical, physical_state=physical)
    np.testing.assert_allclose(h._xyz(new_plan.actions[0, 7:13]), [325, -1000, 130], atol=0.01)
    step = runtime.next(physical, previous)
    assert step.kind == 'model'
    np.testing.assert_allclose(h._xyz(step.action[7:13]), [325, -1000, 130], atol=0.01)
    assert step.action[13] == pytest.approx(0.2)

def test_unreachable_forward_is_rejected_before_preparation():
    (h, state) = setup(5000.0)
    actions = np.tile(state, (4, 1))
    actions[3:, 13] = 0.2
    plan = h.plan(actions, previous_action=state, physical_state=state)
    assert plan.events[0]['reason'] == 'cartesian_target_unreachable'
    np.testing.assert_array_equal(plan.actions, actions)

@pytest.mark.parametrize('forward', [0.0, -25.0])
def test_zero_is_legacy_negative_is_base_minus_x(forward):
    (h, state) = setup(forward)
    (offset, reason) = h.solve_correction(state[7:13], 130, 1)
    assert reason is None
    pose = h._xyz(h.apply_offset(state[7:13], offset))
    np.testing.assert_allclose(pose, [300 + forward, -1000, 130], atol=0.01)

def test_recorded_piper_grasp_reaches_forward_down_target_with_original_rotation():
    from scipy.spatial.transform import Rotation
    from pi05_assists.grasp_dip import make_piper_fk
    (h, state) = setup()
    h.fk = make_piper_fk()
    h.down_tilt = h.fk.down_tilt_deg
    q = np.array([0.1270774156, 2.037096024, -1.329487085, -0.0483630747, 0.895458639, 0.225932866])
    pose = h.fk.pose(q)
    (delta, reason) = h.solve_correction(q, 130.0, 1)
    assert reason is None
    corrected = h.apply_offset(q, delta)
    result = h.fk.pose(corrected)
    np.testing.assert_allclose(result[:3], [pose[0] + 25, pose[1], 130], atol=1.0)
    angle = (Rotation.from_euler('xyz', pose[3:], degrees=True).inv() * Rotation.from_euler('xyz', result[3:], degrees=True)).magnitude()
    assert np.rad2deg(angle) < 3.0
    physical = state.copy()
    physical[7:13] = q
    actions = np.tile(physical, (4, 1))
    actions[3:, 13] = 0.2
    plan = h.plan(actions, previous_action=physical, physical_state=physical)
    assert plan.events[0]['status'] == 'applied'
    runtime = GraspDipRuntime(h, actions, plan)
    previous = physical.copy()
    for _ in range(10):
        step = runtime.next(physical, previous)
        assert step is not None
        assert not runtime.fallback
        runtime.commit(step)
        physical = previous = step.action.copy()
        if step.kind == 'model':
            break
    assert step.kind == 'model'
    assert any((e['status'] == 'ready' for e in runtime.events))
