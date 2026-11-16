"""Pure numerical regression tests migrated from latest local helpers."""
import json
import numpy as np
import pytest
from pi05_assists.pipette_press import DEFAULT_REFERENCE, PipettePress, PipettePressConfig
from pi05_assists.grasp_dip import make_piper_fk
def captured():
    return np.array(json.loads(DEFAULT_REFERENCE.read_text())["state_14"], dtype=np.float32)


@pytest.mark.parametrize('hz', [10, 100])
def test_faster_approach_preserves_down_up_duration_and_endpoints(hz, monkeypatch):
    from collections import Counter
    state = captured()
    results = []
    for approach_speed in (None, 180.0):
        press = PipettePress(PipettePressConfig(move_speed_mm_s=16.0, joint_speed_deg_s=20.0, approach_speed_mm_s=approach_speed, approach_joint_speed_deg_s=60.0 if approach_speed else None), make_piper_fk(), hz)

        def tips(row):
            (left, right) = (np.eye(4), np.eye(4))
            left[:3, 3] = row[:3] * 1000
            return (left, right)
        monkeypatch.setattr(press, 'tips', tips)
        ready = state.copy()
        ready[0] += 0.18
        bottom = ready.copy()
        bottom[2] -= 0.045
        press.path = [ready, bottom, ready.copy()]
        press.labels = ['approach', 'down', 'up']
        press.tip_targets = [tips(row)[0] for row in press.path]
        press._time_path(state)
        counts = Counter(press.labels)
        results.append(counts)
        np.testing.assert_allclose(press.path[counts['approach'] - 1], ready)
        np.testing.assert_allclose(press.path[counts['approach'] + counts['down'] - 1], bottom)
        np.testing.assert_allclose(press.path[-1], ready)
    assert results[1]['approach'] < results[0]['approach']
    assert results[1]['down'] == results[0]['down']
    assert results[1]['up'] == results[0]['up']

def test_three_presses_retract_before_horizontal_offset_and_return_to_origin():
    press = PipettePress(PipettePressConfig(count=3, retry_offset_xyz_mm=(0, 0, 3), move_speed_mm_s=40, joint_speed_deg_s=20), make_piper_fk())
    state = captured()
    assert press.plan(state, cycle=0), press.reason
    transitions = [i for (i, label) in enumerate(press.labels) if i == 0 or label != press.labels[i - 1]]
    downs = [i for i in transitions if press.labels[i] == 'down']
    assert len(downs) == 3
    (_, right) = press.tips(state)
    lateral = right[:3, :3] @ np.array([0.0, 0.0, 3.0])
    lateral[2] = 0
    for (attempt, start) in enumerate(downs):
        end = next((i for i in range(start + 1, len(press.labels)) if press.labels[i] != 'down')) - 1
        expected = press.target[:3, 3] + (0, 1, -1)[attempt] * lateral
        expected[2] -= press.config.depth_mm
        np.testing.assert_allclose(press.tip_targets[end][:3, 3], expected, atol=1e-06)
        if attempt:
            shift = start - 1
            np.testing.assert_allclose(press.tip_targets[shift][2, 3], press.target[2, 3])
    for i in transitions:
        if press.labels[i] == 'approach' and i:
            assert press.labels[i - 1] == 'up'
            assert press.tip_targets[i - 1][2, 3] == pytest.approx(press.target[2, 3])
    np.testing.assert_allclose(press.tip_targets[-1], press.target, atol=1e-06)
    np.testing.assert_allclose(np.asarray(press.path)[:, 7:13], np.tile(state[7:13], (len(press.path), 1)))
    press.phase = press.labels[0]
    for _ in press.path:
        (command, record) = press.next(state)
        press.commit(state, command, cycle=0)
    assert press.phase == 'completed' and record['press_attempt'] == 3

def test_ready_only_ignores_repeat_count():
    press = PipettePress(PipettePressConfig(count=3, ready_only=True), make_piper_fk())
    assert press.plan(captured(), cycle=0)
    assert 'down' not in press.labels

def test_triangle_covers_two_axes_with_exact_radius_even_for_tilted_wrist():
    from scipy.spatial.transform import Rotation
    press = PipettePress(PipettePressConfig(count=3, pattern='triangle', spread_mm=6), make_piper_fk())
    for rpy in [(20, 40, 15), (0, 0, 0), (90, 0, 0)]:
        right = np.eye(4)
        right[:3, :3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
        offsets = np.array(press.press_offsets(right))
        np.testing.assert_allclose(offsets[:, 2], 0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 6)
        np.testing.assert_allclose(offsets.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose([np.linalg.norm(offsets[i] - offsets[(i + 1) % 3]) for i in range(3)], np.sqrt(3) * 6)

def test_triangle_path_retracts_before_switching_vertices():
    press = PipettePress(PipettePressConfig(count=3, pattern='triangle', spread_mm=6, move_speed_mm_s=40, joint_speed_deg_s=20), make_piper_fk())
    state = captured()
    assert press.plan(state, cycle=0), press.reason
    ends = [i for (i, l) in enumerate(press.labels[:-1]) if l == 'down' and press.labels[i + 1] != 'down']
    offsets = press.press_offsets(press.anchor)
    assert len(ends) == 3
    for (i, offset) in zip(ends, offsets):
        expected = press.target[:3, 3] + offset
        expected[2] -= press.config.depth_mm
        np.testing.assert_allclose(press.tip_targets[i][:3, 3], expected, atol=1e-06)
    for i in range(1, len(press.labels)):
        if press.labels[i] == 'approach' and press.labels[i - 1] == 'up':
            assert press.tip_targets[i - 1][2, 3] == pytest.approx(press.target[2, 3])
    np.testing.assert_allclose(press.tip_targets[-1], press.target)
