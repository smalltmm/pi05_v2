"""Pure numerical regression tests migrated from latest local helpers."""
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pi05_assists import config as timing
from pi05_assists.grip_hold import GripHoldConfig, RightGripHold, TableLeftGripHold
from pi05_assists.table_release import TableReleaseConfig, TableReleaseLower


class FK:

    def pose(self, q):
        return np.r_[1000.0 * np.asarray(q)[[0, 3, 4]] + [100.0, 200.0, 200.0], 180.0, 0.0, 0.0]

    def __call__(self, q):
        return self.pose(q)[:3]

    def down_tilt_deg(self, q):
        return 0.0

@pytest.mark.parametrize('side,roll,pitch,yaw', [('left', -171.547, 16.434, 168.001), ('right', 171.188, 7.389, -178.401)])
@pytest.mark.parametrize('initial_z', [180.0, 140.0])
@pytest.mark.parametrize('hz', [10, 100])
def test_endpoint_roll_correction_preserves_pitch_yaw_before_lowering(monkeypatch, side, roll, pitch, yaw, initial_z, hz):
    monkeypatch.setattr(timing, 'CONTROL_DT', 1.0 / hz)
    base_roll = -160.0 if side == 'left' else 160.0
    target_roll = -180.0 if side == 'left' else 180.0

    class RollFK(FK):

        def pose(self, q):
            pose = super().pose(q)
            pose[3:] = Rotation.from_euler('xyz', [base_roll + np.rad2deg(q[5]), pitch, yaw], degrees=True).as_euler('xyz', degrees=True)
            return pose
    fk = RollFK()
    physical = np.zeros(14, dtype=np.float32)
    start = 0 if side == 'left' else 7
    physical[start + 4] = (initial_z - 200.0) / 1000.0
    physical[start + 5] = np.deg2rad(roll - base_roll)
    assist = TableReleaseLower(TableReleaseConfig(), fk, side)
    (path, _, end_z) = assist._path(physical)
    poses = np.array([fk.pose(q) for q in path])
    np.testing.assert_allclose(poses[:, :2], np.tile([100.0, 200.0], (len(path), 1)), atol=0.04)
    np.testing.assert_allclose(poses[-1, 2], min(initial_z, 155.0), atol=0.04)
    assert end_z == min(initial_z, 155.0)
    assert abs(abs(poses[-1, 3]) - 180.0) < 0.04
    np.testing.assert_allclose(poses[:, 4:], np.tile([pitch, yaw], (len(path), 1)), atol=0.04)
    assert np.rad2deg(path[-1][5]) == pytest.approx(target_roll - base_roll, abs=0.04)
    assert abs(path[-1][5]) > 0.1
    lower_rows = poses[:, 2] < initial_z - 0.1
    assert np.all(np.abs(np.abs(poses[lower_rows, 3]) - 180.0) < 0.04)

@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('hz', [10, 100])
@pytest.mark.parametrize('initial_z', [200.0, 140.0])
def test_lower_before_open_across_chunks_without_feedback_wait(monkeypatch, side, hz, initial_z):
    monkeypatch.setattr(timing, 'CONTROL_DT', 1.0 / hz)
    fk = FK()
    assist = TableReleaseLower(TableReleaseConfig(), fk, side)
    hold = (TableLeftGripHold if side == 'left' else RightGripHold)(GripHoldConfig(), fk)
    (hold.state.phase, hold.state.baseline) = ('holding', 0.2)
    physical = np.zeros(14, dtype=np.float32)
    start = 0 if side == 'left' else 7
    (physical[start + 4], physical[start + 6]) = ((initial_z - 200.0) / 1000.0, 0.2)
    previous = physical.copy()
    commands = []
    phases = []
    for i in range(400):
        cycle = i // 8
        if side == 'left':
            step = hold.next(physical, 0.8, token=(cycle, i % 8), previous=previous)
        else:
            step = hold.next(physical, previous, 0.8, token=(cycle, i % 8))
        step = assist.next(step, hold.state, physical, previous, cycle=cycle)
        target = previous.copy()
        if step.force_close:
            target[start + 6] = step.record['hold_command']
        arm = getattr(step, 'arm_action', None)
        if arm is not None:
            target[start:start + 7] = arm
        hold.commit(step)
        assist.commit(step)
        commands.append(target.copy())
        phases.append(step.record['release_lower']['phase'])
        previous = target
        if target[start + 6] > 0.5:
            assert step.record['events'] == ['released_after_lowering']
            break
    else:
        pytest.fail('lowering never completed')
    commands = np.array(commands)
    last_lowering = max((i for (i, phase) in enumerate(phases) if phase == 'lowering'))
    assert len(commands) - 1 - last_lowering == int(np.ceil(0.2 * hz))
    np.testing.assert_allclose(commands[last_lowering:, start:start + 6], np.tile(commands[last_lowering, start:start + 6], (len(commands) - last_lowering, 1)))
    np.testing.assert_allclose(commands[:-1, start + 6], 0.0)
    pose = fk.pose(commands[-1, start:start + 6])
    np.testing.assert_allclose(pose, [100.0, 200.0, min(initial_z, 155.0), 180.0, 0.0, 0.0], atol=0.03)
    np.testing.assert_allclose(commands[-1, start:start + 6], commands[-2, start:start + 6])
    other = 7 if start == 0 else 0
    np.testing.assert_array_equal(commands[:, other:other + 7], np.tile(physical[other:other + 7], (len(commands), 1)))
    replay = hold.next(physical, 0.1, token=(cycle, 99)) if side == 'left' else hold.next(physical, previous, 0.1, token=(cycle, 99))
    replay = assist.next(replay, hold.state, physical, previous, cycle=cycle)
    np.testing.assert_allclose(replay.arm_action, commands[-1, start:start + 7])
    assist.commit(replay)
    fresh = hold.next(physical, 0.1, token=(cycle + 1, 0)) if side == 'left' else hold.next(physical, previous, 0.1, token=(cycle + 1, 0))
    fresh = assist.next(fresh, hold.state, physical, previous, cycle=cycle + 1)
    assert getattr(fresh, 'arm_action', None) is None

def test_failed_ik_does_not_commit_opening(monkeypatch):
    assist = TableReleaseLower(TableReleaseConfig(), FK(), 'left')
    hold = TableLeftGripHold(GripHoldConfig(release_frames=1), FK())
    (hold.state.phase, hold.state.baseline) = ('holding', 0.2)
    state = np.zeros(14)

    def fail(*args):
        raise ValueError('IK failed')
    monkeypatch.setattr(assist, '_solve', fail)
    with pytest.raises(ValueError, match='IK failed'):
        assist.next(hold.next(state, 0.8, token=(0, 0)), hold.state, state, state, cycle=0)
    assert hold.state.phase == 'holding'
    assert assist.state.phase == 'idle'
