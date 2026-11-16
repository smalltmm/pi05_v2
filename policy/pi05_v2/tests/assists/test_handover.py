import numpy as np
import pytest
from pi05_assists.handover import HandoverAssist, HandoverConfig, rotation

class FK:

    def __call__(self, q):
        return self.pose(q)[:3]

    def pose(self, q):
        return np.array([0.0, 1000 * q[0], 150.0, *np.rad2deg(q[3:6])])

def state(left_grip=0.8, right_grip=0.05):
    s = np.zeros(14, dtype=np.float32)
    (s[0], s[7]) = (-0.3, 0.3)
    (s[6], s[13]) = (left_grip, right_grip)
    return s

def assist(**kwargs):
    return HandoverAssist(HandoverConfig(relative_rpy_deg=(0.0, 0.0, 0.0), **kwargs), FK())

def opening(s):
    actions = np.tile(s, (4, 1))
    actions[:, 13] = [0.05, 0.05, 0.9, 0.9]
    actions[:, 0] += 0.1
    return actions

def test_common_frame_distance_uses_600mm_translation():
    h = assist()
    record = h.poses(state())
    assert record['distance_mm'] == pytest.approx(0.0, abs=0.001)
    assert record['relative_orientation_error_deg'] == pytest.approx(0.0)
    assert record['right_base_pose_xyz_mm_rpy_deg'][1] == pytest.approx(300.0)
    assert np.asarray(record['right_pose_in_left_base'])[1, 3] == pytest.approx(-300.0)
    assert record['in_handover_region']

def test_rotation_error_uses_so3_not_subtracted_euler_angles():
    h = HandoverAssist(HandoverConfig(relative_rpy_deg=(2.0, 0.0, 0.0)), FK())
    s = state()
    (s[3], s[10]) = (np.deg2rad(179), np.deg2rad(-179))
    assert h.poses(s)['relative_orientation_error_deg'] < 0.001
    h = HandoverAssist(HandoverConfig(right_base_rpy_deg=(0.0, 0.0, 90.0), relative_rpy_deg=(0.0, 0.0, 90.0)), FK())
    record = h.poses(state())
    expected = rotation((0.0, 0.0, 90.0)) @ np.array([0.0, 300.0, 150.0]) + [0.0, -600.0, 0.0]
    np.testing.assert_allclose(np.array(record['right_pose_in_left_base'])[:3, 3], expected, atol=0.001)
    assert record['relative_orientation_error_deg'] < 0.001

@pytest.mark.parametrize('case', ['far', 'orientation', 'not_opening', 'already_open'])
def test_nonhandover_actions_remain_untouched(case):
    h = assist()
    s = state()
    actions = opening(s)
    if case == 'far':
        s[7] = -0.2
    elif case == 'orientation':
        s[10] = 1.0
    elif case == 'not_opening':
        actions[:, 13] = 0.05
    else:
        s[13] = 0.8
    step = h.next(s, s, actions)
    assert step.action is None
    assert step.kind == 'model'
    assert h.state.phase == 'idle'

def test_hold_is_latched_across_chunks_until_stable_then_uses_fresh_inference():
    h = assist()
    s = state()
    initial = s.copy()
    previous = s.copy()
    model = opening(s)
    step = h.next(s, previous, model)
    assert h.state.phase == 'idle'
    h.commit(step)
    assert step.kind == 'handover_hold'
    np.testing.assert_array_equal(step.action[[0, 7]], initial[[0, 7]])
    assert step.action[6] == 0.0 and step.action[13] == pytest.approx(0.05)
    previous = step.action
    for value in [0.09, 0.07, 0.07, 0.07]:
        h.begin_inference_result()
        s[6] = value
        step = h.next(s, previous, model + 0.01)
        assert step.kind == 'handover_hold'
        np.testing.assert_array_equal(step.action, previous)
        h.commit(step)
    assert h.state.phase == 'ready'
    step = h.next(s, previous, model)
    assert step.kind == 'handover_hold'
    h.commit(step)
    h.begin_inference_result()
    latest = model.copy()
    latest[:, 0] = 0.15
    latest[:, 13] = 0.9
    step = h.next(s, previous, latest)
    assert step.kind == 'handover_release'
    assert step.action[0] == pytest.approx(0.15)
    assert step.action[13] == pytest.approx(0.9)
    assert step.action[6] == 0.0
    h.commit(step)
    s[13] = 0.9
    step = h.next(s, step.action, latest)
    h.commit(step)
    assert h.state.phase == 'completed'
    assert h.next(s, step.action, latest).action is None

def test_lost_closure_before_donor_release_reblocks_right():
    h = assist(stable_frames=1)
    s = state(left_grip=0.05)
    step = h.next(s, s, opening(s))
    h.commit(step)
    h.begin_inference_result()
    s[6] = 0.3
    step = h.next(s, step.action, opening(s))
    assert step.kind == 'handover_hold'
    assert step.action[13] == pytest.approx(0.05)
    assert 'left_closure_lost' in step.record['events']

def test_no_timeout_release_and_rearm_only_after_separation():
    h = assist(stable_frames=1)
    s = state()
    previous = s.copy()
    for _ in range(100):
        h.begin_inference_result()
        step = h.next(s, previous, opening(s))
        h.commit(step)
        previous = step.action
        assert step.kind == 'handover_hold'
    s[6] = 0.05
    step = h.next(s, previous, opening(s))
    h.commit(step)
    h.begin_inference_result()
    s[13] = 0.9
    step = h.next(s, previous, opening(s))
    h.commit(step)
    assert h.state.phase == 'completed'
    s[13] = 0.05
    assert h.next(s, s, opening(s)).action is None
    s[7] = -0.2
    step = h.next(s, s, opening(s))
    h.commit(step)
    assert h.state.phase == 'idle' and 'rearmed' in step.record['events']

def test_capped_handover_updates_model_grippers_while_holding_joints():
    h = assist(gripper_cap=0.25, left_closed_threshold=0.3, right_closed_threshold=0.3, right_open_threshold=0.35)
    s = state(left_grip=0.8, right_grip=0.26)
    previous = s.copy()
    previous[13] = 0.25
    model = opening(s)
    (model[0, 6], model[0, 13]) = (0.6, 0.5)
    step = h.next(s, previous, model)
    h.commit(step)
    assert step.kind == 'handover_hold'
    np.testing.assert_allclose(step.action[[6, 13]], [0.25, 0.25])
    initial_joints = step.action[:6].copy()
    s[6] = 0.26
    (model[0, 6], model[0, 13]) = (0.18, 0.16)
    model[:, 0] += 0.1
    for _ in range(3):
        step = h.next(s, step.action, model)
        h.commit(step)
        np.testing.assert_allclose(step.action[[6, 13]], [0.18, 0.16])
        np.testing.assert_array_equal(step.action[:6], initial_joints)
    assert h.state.phase == 'ready'
    h.begin_inference_result()
    (model[0, 6], model[0, 13]) = (0.5, 0.6)
    step = h.next(s, step.action, model)
    h.commit(step)
    assert step.kind == 'handover_release'
    np.testing.assert_allclose(step.action[[6, 13]], [0.25, 0.6])
    s[13] = 0.6
    step = h.next(s, step.action, model)
    assert 'handover_completed' in step.record['events']
    np.testing.assert_allclose(step.action[[6, 13]], [0.25, 0.6])
