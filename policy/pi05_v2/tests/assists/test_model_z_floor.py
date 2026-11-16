"""Offline checks with SDK FK; never opens CAN or sends motion."""
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pi05_assists.grasp_dip import make_piper_fk
from pi05_assists.model_z_floor import ModelZFloor

@pytest.mark.parametrize('minimum', [120.0, 145.0])
def test_both_arms_clamp_height_preserving_xy_rotation_and_grippers(minimum):
    fk = make_piper_fk()
    row = np.array([0.1, 1.8, -0.5, 0.1, 0.6, 0.1, 0.3, -0.1, 2.0, -0.5, 0.1, 0.6, 0.1, 0.7], dtype=np.float32)
    original = row.copy()
    floor = ModelZFloor(fk, minimum)
    result = floor.apply(np.tile(row, (10, 1)))
    for start in (0, 7):
        before = fk.pose(row[start:start + 6])
        after = fk.pose(result[0, start:start + 6])
        assert before[2] < minimum
        assert minimum <= after[2] <= minimum + 0.1
        np.testing.assert_allclose(after[:2], before[:2], atol=0.1)
        rotation_error = Rotation.from_euler('xyz', before[3:], degrees=True).inv() * Rotation.from_euler('xyz', after[3:], degrees=True)
        assert rotation_error.magnitude() < np.deg2rad(0.1)
    np.testing.assert_array_equal(result[:, [6, 13]], np.tile(row[[6, 13]], (10, 1)))
    np.testing.assert_array_equal(row, original)
    assert len(floor.cache) == 2
    np.testing.assert_array_equal(floor.apply(row[None]), result[:1])
    assert len(floor.cache) == 2
    floor.begin_chunk()
    assert not floor.cache

def test_above_floor_arm_is_unchanged():
    fk = make_piper_fk()
    row = np.array([0, 1.2, -1, 0, 1.2, 0, 0.3, 0.1, 1.8, -0.5, 0.1, 0.6, 0.1, 0.7], dtype=np.float32)
    assert fk.pose(row[:6])[2] > 120
    result = ModelZFloor(fk, 120).apply(row[None])
    np.testing.assert_array_equal(result[0, :7], row[:7])
    assert fk.pose(result[0, 7:13])[2] >= 120

def test_unreachable_floor_does_not_return_low_targets():
    row = np.array([0, 1.2, -1, 0, 1.2, 0, 0.3] * 2, dtype=np.float32)
    with pytest.raises(ValueError, match='model Z floor IK failed'):
        ModelZFloor(make_piper_fk(), 10000).apply(row[None])
