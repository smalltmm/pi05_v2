"""Pure numerical regression tests migrated from latest local helpers."""
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pi05_assists import config as timing
from pi05_assists.basket_release import BasketReleaseAdvance, BasketReleaseConfig
from pi05_assists.grip_hold import BasketLeftGripHold, BasketLeftGripConfig


class FK:

    def __init__(self, rpy=(0.0, 90.0, 0.0)):
        self.rpy = rpy

    def pose(self, q):
        return np.r_[1000.0 * np.asarray(q)[[0, 3, 4]] + [0.0, 0.0, 300.0], self.rpy]

    def __call__(self, q):
        return self.pose(q)[:3]

@pytest.mark.parametrize('rpy', [(0.0, 90.0, 0.0), (90.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
def test_move_in_tool_frame_then_open_without_feedback_wait(monkeypatch, rpy):
    monkeypatch.setattr(timing, 'CONTROL_DT', 0.1)
    fk = FK(rpy)
    assist = BasketReleaseAdvance(BasketReleaseConfig(), fk)
    hold = BasketLeftGripHold(BasketLeftGripConfig(close_threshold=0.5, open_threshold=0.6))
    hold.state.phase = 'holding'
    hold.state.baseline = 0.4
    physical = np.zeros(14)
    physical[6] = 0.35
    previous = physical.copy()
    sent = []
    for i in range(12):
        step = hold.next(physical, 0.8, token=(0, i))
        step = assist.next(step, hold.state, physical, previous, cycle=0)
        target = previous.copy()
        if step.force_close:
            target[6] = step.record['hold_command']
        if getattr(step, 'arm_action', None) is not None:
            target[:7] = step.arm_action
        hold.commit(step)
        assist.commit(step)
        sent.append(target.copy())
        previous = target
    sent = np.array(sent)
    opened = np.flatnonzero(sent[:, 6] > 0.6)[0]
    assert opened == 7
    direction = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()[:, 2]
    expected = fk(physical[:6]) + 15 * direction
    np.testing.assert_allclose(fk(sent[opened - 1, :6]), expected, atol=0.02)
    np.testing.assert_allclose(sent[opened:, :6], np.tile(sent[opened - 1, :6], (len(sent) - opened, 1)))
    np.testing.assert_allclose(sent[:opened, 6], 0.35)
    np.testing.assert_allclose(sent[opened:, 6], 0.8)

def test_failed_plan_does_not_commit_release(monkeypatch):
    assist = BasketReleaseAdvance(BasketReleaseConfig(), FK())
    hold = BasketLeftGripHold(BasketLeftGripConfig(release_frames=1))
    hold.state.phase = 'holding'
    hold.state.baseline = 0.4
    state = np.zeros(14)

    def unreachable(*args, **kwargs):
        raise ValueError('cartesian_target_unreachable')
    monkeypatch.setattr(sys.modules[BasketReleaseAdvance.__module__], 'solve_ik', unreachable)
    step = hold.next(state, 0.8, token=(0, 0))
    with pytest.raises(ValueError, match='cartesian_target_unreachable'):
        assist.next(step, hold.state, state, state, cycle=0)
    assert hold.state.phase == 'holding'
    assert assist.state.phase == 'idle'
