"""Offline checks for the item_assembly dual-arm grasp assist."""
import sys
import numpy as np
import pytest
from pi05_assists.grasp_dip import GraspDip, GraspDipConfig
from pi05_assists.grasp_dip_runtime import GraspDipRuntime
from pi05_assists.grip_hold import AssemblyGripConfig, AssemblyLeftGripHold, AssemblyRightGripHold

def advance_hold(hold, state, index, value, token, *, allow_close=True):
    if index == 6:
        step = hold.next(state, value, token=token, allow_close=allow_close)
    else:
        step = hold.next(state, state, value, token=token, allow_close=allow_close)
    hold.commit(step)
    return step

class PoseFK:

    def pose(self, q):
        return np.array([1000 * q[0], 1000 * q[2], 250 - 100 * q[1], 165 + 50 * q[3], 0, 0], dtype=float)

    def __call__(self, q):
        return self.pose(q)[:3]

    def down_tilt_deg(self, q):
        return abs(180 - self.pose(q)[3])

def test_both_arms_reach_target_height_and_vertical_rotation_before_closing():
    fk = PoseFK()
    helper = GraspDip(GraspDipConfig(target_z_mm=(130, 130), z_ranges=((130, 180), (130, 180)), close_threshold=0.4, open_threshold=0.6, max_drop_mm=132, max_tilt_deg=20, vertical_down=True), fk)
    physical = np.array([0.1, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
    rows = np.tile(physical, (5, 1))
    rows[:, 6] = rows[:, 13] = [0.8, 0.7, 0.5, 0.3, 0.2]
    plan = helper.plan(rows, previous_action=physical, physical_state=physical)
    assert [event['status'] for event in plan.events] == ['applied', 'applied']
    for start in (0, 7):
        q = plan.actions[2, start:start + 6]
        assert fk(q)[2] == pytest.approx(130, abs=0.1)
        assert fk.down_tilt_deg(q) <= 5
        assert np.count_nonzero(np.abs(q - physical[start:start + 6]) > 0.01) >= 2
    np.testing.assert_array_equal(plan.actions[:, [6, 13]], rows[:, [6, 13]])
    runtime = GraspDipRuntime(helper, rows, plan)
    assert runtime.next(physical, physical).kind == 'model'

def test_close_waits_for_both_arms_to_reach_the_vertical_pose():
    fk = PoseFK()
    helper = GraspDip(GraspDipConfig(target_z_mm=(130, 130), z_ranges=((130, 180), (130, 180)), close_threshold=0.4, open_threshold=0.6, max_drop_mm=132, max_tilt_deg=20, vertical_down=True), fk)
    physical = np.array([0.1, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
    rows = np.tile(physical, (4, 1))
    rows[:, 6] = rows[:, 13] = [0.8, 0.7, 0.5, 0.3]
    plan = helper.plan(rows, previous_action=physical, physical_state=physical)
    runtime = GraspDipRuntime(helper, rows, plan)
    previous = physical.copy()
    for _ in range(3):
        step = runtime.next(physical, previous)
        assert step.kind == 'model'
        runtime.commit(step)
        previous = step.action.copy()
        physical[7:] = previous[7:]
    waiting = runtime.next(physical, previous)
    assert waiting.kind == 'grasp_dip_wait'
    assert waiting.action[6] > 0.4
    assert waiting.action[13] > 0.4
    runtime.commit(waiting)
    physical = previous.copy()
    closing = runtime.next(physical, previous)
    assert closing.kind == 'model'
    assert closing.action[6] == pytest.approx(0.3)
    assert closing.action[13] == pytest.approx(0.3)

def test_first_row_close_inserts_progressive_vertical_preparation():
    fk = PoseFK()
    helper = GraspDip(GraspDipConfig(target_z_mm=(130, 130), z_ranges=((130, 180), (130, 180)), close_threshold=0.4, open_threshold=0.6, max_drop_mm=132, max_tilt_deg=20, vertical_down=True), fk)
    physical = np.array([0.1, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
    rows = np.tile(physical, (2, 1))
    rows[:, 6] = rows[:, 13] = 0.3
    plan = helper.plan(rows, previous_action=physical, physical_state=physical)
    runtime = GraspDipRuntime(helper, rows, plan)
    first = runtime.next(physical, physical)
    assert first.kind == 'grasp_dip_prepare'
    tilts = [fk.down_tilt_deg(row[:6]) for row in first.plan.actions]
    assert tilts[0] > tilts[1] > tilts[2]
    assert tilts[2] < 5
    assert np.all(first.plan.actions[:, [6, 13]] >= 0.8)
    previous = physical.copy()
    sent = []
    for _ in range(8):
        step = runtime.next(physical, previous)
        if step is None:
            break
        sent.append(step)
        runtime.commit(step)
        physical = previous = step.action.copy()
    assert [step.kind for step in sent] == ['grasp_dip_prepare'] * 3 + ['model'] * 2
    assert sent[3].action[6] == pytest.approx(0.3)
    assert sent[3].action[13] == pytest.approx(0.3)
    assert not runtime.fallback

def test_fixed_gripper_hold_waits_for_close_and_requires_deliberate_release():
    config = AssemblyGripConfig()
    state = np.zeros(14, dtype=np.float32)
    for (hold, index) in ((AssemblyLeftGripHold(config), 6), (AssemblyRightGripHold(config), 13)):
        state[index] = 0.8
        assert not advance_hold(hold, state, index, 0.3, (0, 0), allow_close=False).force_close
        assert not advance_hold(hold, state, index, 0.8, (0, 0)).force_close
        closed = advance_hold(hold, state, index, 0.3, (0, 1))
        assert closed.force_close
        assert closed.record['hold_command'] == 0.2
        for model in (0.1, 0.3, 0.45, 0.55):
            assert advance_hold(hold, state, index, model, (1, int(model * 100))).record['hold_command'] == 0.2
        assert advance_hold(hold, state, index, 0.8, (2, 0)).state.release_count == 1
        assert advance_hold(hold, state, index, 0.8, (2, 0)).state.release_count == 1
        assert advance_hold(hold, state, index, 0.8, (2, 1)).force_close
        assert not advance_hold(hold, state, index, 0.8, (2, 2)).force_close

def test_unreachable_vertical_pose_skips_dip_without_modifying_grippers():

    class FixedOrientationFK(PoseFK):

        def pose(self, q):
            pose = super().pose(q)
            pose[3] = 165
            return pose
    fk = FixedOrientationFK()
    helper = GraspDip(GraspDipConfig(target_z_mm=(130, None), z_ranges=((130, 180), None), close_threshold=0.4, open_threshold=0.6, max_drop_mm=132, max_tilt_deg=20, vertical_down=True), fk)
    physical = np.array([0.1, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8] * 2, dtype=np.float32)
    rows = np.tile(physical, (4, 1))
    rows[3, 6] = 0.3
    plan = helper.plan(rows, previous_action=physical, physical_state=physical)
    assert plan.events[0]['status'] == 'skipped'
    assert plan.events[0]['reason'] == 'cartesian_orientation_unreachable'
    np.testing.assert_array_equal(plan.actions, rows)

def test_recorded_assembly_closing_poses_have_full_joint_solutions():
    from pi05_assists.grasp_dip import make_piper_fk
    fk = make_piper_fk()
    helper = GraspDip(GraspDipConfig(target_z_mm=(130, 130), z_ranges=((130, 180), (130, 180)), close_threshold=0.4, open_threshold=0.6, max_drop_mm=132, max_tilt_deg=20, vertical_down=True), fk)
    poses = ((0, [-0.2809, 2.0343, -1.2716, -0.075, 0.6672, -0.2171]), (1, [0.3578, 2.1323, -1.4575, 0.0495, 0.9253, 0.2583]))
    for (side, values) in poses:
        original = np.array(values)
        (correction, reason) = helper.solve_correction(original, 130, side)
        assert reason is None
        solved = original + correction
        np.testing.assert_allclose(fk.pose(solved)[:3], [*fk.pose(original)[:2], 130], atol=1)
        assert fk.down_tilt_deg(solved) < 1
