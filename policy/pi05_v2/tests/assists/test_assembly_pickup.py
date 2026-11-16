"""Assembly pickup never resumes tilted model targets before vertical clearance."""
import numpy as np
from scipy.spatial.transform import Rotation
from pi05_assists.assembly_pickup import AssemblyPickupLift
from pi05_assists.grasp_dip import make_piper_fk
from pi05_assists.grip_hold import AssemblyGripConfig
from pi05_assists.item_assembly import AssemblyInsertConfig
from pi05_assists.kinematics import fingertip_xyz

def test_each_grasp_holds_then_lifts_without_xy_or_rotation_change():
    fk = make_piper_fk()
    q = np.array([0.0, 1.5, -1.0, 0.0, -0.5, 0.0], dtype=np.float32)
    state = np.r_[q, 0.6, q, 0.6].astype(np.float32)
    helper = AssemblyPickupLift(AssemblyInsertConfig(trigger_lift_mm=40), fk, AssemblyGripConfig(close_threshold=0.6, open_threshold=0.7), 100)
    model = state.copy()
    model[[1, 8]] += 0.3
    feedback = lambda stamp, opening: {s: dict(valid=True, stamp=stamp, opening=opening) for s in ('left', 'right')}
    target = helper.next(state, model, (True, False), feedback(1, 0.6), 0)
    np.testing.assert_array_equal(target[:6], q)
    np.testing.assert_array_equal(target[7:13], model[7:13])
    assert helper.arms[0]['phase'] == 'closing'
    state[6] = 0.3
    initial = fk.pose(q)
    for i in range(2, 1000):
        target = helper.next(state, model, (True, False), feedback(i, 0.3), 0)
        helper.commit(state, target)
        state[:7] = target[:7]
        state[6] = 0.3
        pose = fk.pose(state[:6])
        np.testing.assert_allclose(fingertip_xyz(pose)[:2], fingertip_xyz(initial)[:2], atol=0.5)
        err = (Rotation.from_euler('xyz', initial[3:], degrees=True).inv() * Rotation.from_euler('xyz', pose[3:], degrees=True)).magnitude()
        assert np.rad2deg(err) <= 1.0
        if helper.arms[0]['phase'] == 'raised':
            break
    assert helper.arms[0]['phase'] == 'raised', helper.record()
    assert not helper.ready
    assert abs(fingertip_xyz(fk.pose(state[:6]))[2] - fingertip_xyz(initial)[2] - 40) < 0.5
    held = helper.next(state, model, (True, True), feedback(1001, 0.6), 1)
    np.testing.assert_allclose(held[:6], state[:6])
    np.testing.assert_array_equal(held[7:13], q)
    assert helper.arms[1]['phase'] == 'closing'

def test_failed_vertical_ik_holds_grasp_and_never_releases_to_model(monkeypatch):
    helper = AssemblyPickupLift(AssemblyInsertConfig(), make_piper_fk(), AssemblyGripConfig(), 100)
    state = np.array([0.0, 1.5, -1.0, 0.0, -0.5, 0.0, 0.5] * 2, dtype=np.float32)
    model = state + 0.1
    from pi05_assists import assembly_pickup

    def fail(*args, **kwargs):
        raise RuntimeError('unreachable lift')
    monkeypatch.setattr(assembly_pickup, 'solve_ik', fail)
    for i in range(5):
        opening = 0.5 if i == 0 else 0.3
        feedback = {s: dict(valid=True, stamp=i, opening=opening) for s in ('left', 'right')}
        target = helper.next(state, model, (True, True), feedback, 0)
        np.testing.assert_array_equal(target[:6], state[:6])
        np.testing.assert_array_equal(target[7:13], state[7:13])
    assert not helper.ready
    assert helper.arms[0]['reason'] == 'unreachable lift'
