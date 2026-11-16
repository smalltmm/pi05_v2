"""Offline item_assembly insertion planning and trigger checks."""
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pi05_assists.grasp_dip import make_piper_fk
from pi05_assists.item_assembly import AssemblyInsertConfig, ItemAssemblyInsert
from pi05_assists.kinematics import fingertip_xyz

def test_new_relative_alignment_uses_right_in_left_direction_and_plans():
    cfg = AssemblyInsertConfig()
    assist = ItemAssemblyInsert(cfg, make_piper_fk(), grip_command=0.2)
    (left, rotation) = assist.alignment_goal()
    (_, right) = assist._tips(assist.start_state)
    right_rotation = Rotation.from_euler('xyz', assist.fk.pose(assist.start_state[7:13])[3:], degrees=True)
    np.testing.assert_allclose(rotation.inv().apply(right - left), cfg.right_in_left_xyz_mm, atol=1e-06)
    np.testing.assert_allclose((rotation.inv() * right_rotation).as_euler('xyz', degrees=True), cfg.right_in_left_rpy_deg, atol=1e-06)
    assert assist.plan(assist.start_state, cycle=0), assist.reason
    index = assist.labels.index('insert_left') - 1
    (actual_left, actual_right) = assist._tips(assist.path[index])
    actual_rotation = Rotation.from_euler('xyz', assist.fk.pose(assist.path[index][:6])[3:], degrees=True)
    assert np.linalg.norm(actual_left - left) <= cfg.position_tolerance_mm
    assert np.rad2deg((rotation.inv() * actual_rotation).magnitude()) <= cfg.orientation_tolerance_deg
    assert np.linalg.norm(actual_right - actual_left) == pytest.approx(229.75, abs=3)
    assert assist.final_distance_mm == pytest.approx(213, abs=3)

def test_captured_start_plans_both_arm_arrival_then_left_only_insertion():
    fk = make_piper_fk()
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False), fk, grip_command=0.2)
    current = assist.start_state.copy()
    current[0] += 0.06
    current[8] += 0.03
    assert assist.plan(current, cycle=0), assist.reason
    assert assist.labels[0] == 'zero_j1'
    assert 'move_start' in assist.labels
    assert 'align_left' in assist.labels
    assert assist.labels[-1] == 'insert_left'
    assert all((np.allclose(row[[6, 13]], 0.2) for row in assist.path))
    first_insert = assist.labels.index('insert_left')
    for row in assist.path[first_insert:]:
        np.testing.assert_allclose(row[7:13], assist.start_state[7:13], atol=1e-06)
    distances = []
    for row in assist.path[first_insert:]:
        (left, right) = assist._tips(row)
        distances.append(np.linalg.norm(right - left))
        actual = Rotation.from_euler('xyz', fk.pose(row[:6])[3:], degrees=True)
        fixed = Rotation.from_euler('xyz', fk.pose(row[7:13])[3:], degrees=True)
        assert np.rad2deg((actual.inv() * fixed).magnitude()) <= 5
    assert np.all(np.diff(distances) < 0)
    assert distances[-1] == pytest.approx(213, abs=3)
    assert fingertip_xyz(fk.pose(assist.path[-1][:6]))[2] > 0
    physical = current.copy()
    for _ in range(2 * len(assist.path) + 4):
        (action, record) = assist.next(physical)
        assist.commit(physical, action)
        physical = action.copy()
        if assist.phase == 'done':
            break
    assert assist.phase == 'done'
    (held, _) = assist.next(physical)
    np.testing.assert_allclose(held, assist.path[-1])
    assert record['tip_distance_mm'] >= 213

def test_j1_zero_leg_from_recorded_pickup_keeps_other_joints_and_grips_fixed():
    state = np.array([-0.49434707, 1.93824053, -1.30267882, 0.0, 0.96843082, -0.31297243, 0.402, 0.48860493, 2.08758831, -1.55283689, 0.02085668, 1.12622607, 0.15919147, 0.401], dtype=np.float32)
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False), make_piper_fk(), grip_command=0.2)
    assert assist.plan(state, cycle=0), assist.reason
    count = assist.labels.index('move_start')
    leg = np.asarray(assist.path[:count])
    fixed = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(leg[:, fixed], np.tile(state[fixed], (len(leg), 1)))
    np.testing.assert_allclose(leg[:, [6, 13]], 0.2)
    np.testing.assert_array_equal(leg[-1, [0, 7]], [0, 0])
    assert np.max(np.abs(np.diff(np.vstack([state, leg])[:, [0, 7]], axis=0))) <= 0.030001
    for (row, label) in zip(assist.path, assist.labels, strict=True):
        if label in ('zero_j1', 'move_start'):
            (left, right) = assist._tips(row)
            assert np.linalg.norm(right - left) >= 180
    assist.phase = 'zero_j1'
    assist.index = count - 1
    assist.commit(state, leg[-1])
    assert assist.phase == 'zero_j1'
    assist.commit(leg[-1], leg[-1])
    assert assist.phase == 'move_start'

def test_offsets_persist_along_fixed_taught_axis():
    cfg = AssemblyInsertConfig(relative_align=False, offset_xyz_mm=(10.0, 0.0, 5.0), offset_rpy_deg=(3.0, 0.0, 0.0))
    assist = ItemAssemblyInsert(cfg, make_piper_fk(), grip_command=0.2)
    (left, right) = assist._tips(assist.start_state)
    axis = (right - left) / np.linalg.norm(right - left)
    right_rotation = Rotation.from_euler('xyz', assist.fk.pose(assist.start_state[7:13])[3:], degrees=True)
    offset = right_rotation.apply(cfg.offset_xyz_mm)
    expected_rotation = right_rotation * Rotation.from_euler('xyz', cfg.offset_rpy_deg, degrees=True)
    assert assist.plan(assist.start_state, cycle=0), assist.reason
    np.testing.assert_allclose(assist.insert_direction, axis)
    projection = np.eye(3) - np.outer(axis, axis)
    assert np.linalg.norm(projection @ offset) > cfg.position_tolerance_mm
    for (row, label) in zip(assist.path, assist.labels, strict=True):
        if label not in ('zero_j1', 'move_start'):
            np.testing.assert_array_equal(row[7:13], assist.start_state[7:13])
        if label != 'insert_left':
            continue
        tip = assist._tips(row)[0]
        assert np.linalg.norm(projection @ (tip - left - offset)) <= cfg.position_tolerance_mm
        actual = Rotation.from_euler('xyz', assist.fk.pose(row[:6])[3:], degrees=True)
        assert np.rad2deg((expected_rotation.inv() * actual).magnitude()) <= cfg.orientation_tolerance_deg
    assert assist.final_distance_mm == pytest.approx(cfg.stop_tip_distance_mm, abs=cfg.position_tolerance_mm)

def test_invalid_offset_and_unreachable_offset_leave_no_partial_path():
    with pytest.raises(ValueError, match='offsets'):
        AssemblyInsertConfig(relative_align=False, offset_xyz_mm=(0, float('nan'), 0))
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False, offset_xyz_mm=(1000, 0, 0)), make_piper_fk(), grip_command=0.2)
    assert not assist.plan(assist.start_state, cycle=0)
    assert assist.path == []

def test_trigger_needs_two_holds_lift_contact_and_new_feedback():

    class FK:

        def pose(self, joints):
            return np.array([0, 0, 100 + 100 * joints[1], 180, 0, 0], dtype=float)
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False), FK(), grip_command=0.2)
    state = assist.start_state.copy()
    state[[1, 8]] = 0.5
    state[[6, 13]] = 0.42

    def feedback(stamp, opening=0.42):
        return {side: {'valid': True, 'stamp': stamp, 'opening': opening} for side in ('left', 'right')}
    assert not assist.observe(state, left_holding=False, right_holding=True, cycle=0, feedback=feedback(1))
    assert not assist.observe(state, left_holding=True, right_holding=True, cycle=0, feedback=feedback(2))
    raised = state.copy()
    raised[[1, 8]] += 0.5
    assert not assist.observe(raised, left_holding=True, right_holding=True, cycle=0, feedback=feedback(3, 0.2))
    assert not assist.observe(raised, left_holding=True, right_holding=True, cycle=0, feedback=feedback(4))
    assert not assist.observe(raised, left_holding=True, right_holding=True, cycle=0, feedback=feedback(4))
    assert not assist.observe(raised, left_holding=True, right_holding=True, cycle=0, feedback=feedback(5))
    assert assist.observe(raised, left_holding=True, right_holding=True, cycle=0, feedback=feedback(6))

def test_unreachable_travel_reverts_to_model_without_partial_path():
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False, max_insert_travel_mm=10), make_piper_fk(), grip_command=0.2)
    assert not assist.plan(assist.start_state, cycle=0)
    assert assist.phase == 'idle'
    assert assist.path == []
    assert 'outside' in assist.reason

def test_start_requires_both_pickup_lifts_complete():
    assist = ItemAssemblyInsert(AssemblyInsertConfig(relative_align=False), make_piper_fk(), grip_command=0.2)
    state = assist.start_state.copy()
    for stamp in range(5):
        feedback = {side: dict(valid=True, stamp=stamp, opening=0.35) for side in ('left', 'right')}
        assert not assist.observe(state, left_holding=True, right_holding=True, cycle=0, feedback=feedback, pickup_ready=False)
    for stamp in range(5, 8):
        feedback = {side: dict(valid=True, stamp=stamp, opening=0.35) for side in ('left', 'right')}
        ready = assist.observe(state, left_holding=True, right_holding=True, cycle=0, feedback=feedback, pickup_ready=True)
    assert ready
