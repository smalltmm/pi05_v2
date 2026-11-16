import dataclasses
import numpy as np
import pytest
from pi05_assists.grasp_dip import GraspDip, GraspDipConfig, down_tilt_from_rpy_deg, make_piper_fk
from pi05_assists.timing import ActionSchedule

def make_assist(cfg, position_fk, *, down_tilt=lambda q: 0.0):
    return GraspDip(cfg, position_fk, down_tilt=down_tilt)

def fk(q):
    return np.array([100.0, 0.0, 150.0 - 100.0 * q[1] + 100.0 * q[4]])

def config(**kwargs):
    kwargs.setdefault('target_z_mm', (149.5, 149.5))
    return GraspDipConfig(z_ranges=((100.0, 160.0), (100.0, 160.0)), **kwargs)

def actions(grips):
    result = np.zeros((len(grips), 14), dtype=np.float32)
    result[:, 6] = grips
    result[:, 13] = 0.6
    return result

def plan(helper, values, previous=None):
    state = actions([0.6])[0] if previous is None else previous
    return helper.plan(values, previous_action=state, physical_state=state)

def test_ramp_starts_before_close_and_holds_closed_rows():
    helper = make_assist(config(target_z_mm=(149.4, 149.1)), fk)
    base = actions([0.6, 0.6, 0.5, 0.3, 0.15, 0.05, 0.05, 0.05])
    original = base.copy()
    result = plan(helper, base)
    np.testing.assert_allclose(result.actions[:, 1], [0, 0, 0.002, 0.004, 0.006, 0.006, 0.006, 0.006])
    np.testing.assert_array_equal(base, original)
    np.testing.assert_array_equal(result.actions[:, [i for i in range(14) if i != 1]], base[:, [i for i in range(14) if i != 1]])
    assert result.events[0]['close_row'] == 5
    assert result.events[0]['close_drop_mm'] == pytest.approx(0.6)

def test_both_arms_have_independent_offsets_and_transitions():
    helper = make_assist(config(target_z_mm=(149.4, 149.1)), fk)
    base = actions([0.6, 0.3, 0.05, 0.05, 0.05, 0.05])
    base[:, 13] = [0.6, 0.6, 0.6, 0.3, 0.15, 0.05]
    result = plan(helper, base)
    np.testing.assert_allclose(result.actions[:, 1], [0.003, 0.006, 0.006, 0.006, 0.006, 0.006])
    np.testing.assert_allclose(result.actions[:, 8], [0, 0, 0.003, 0.006, 0.009, 0.009])
    assert len(result.events) == 2

def test_hold_across_inferences_then_fade_on_lift_without_accumulation():
    helper = make_assist(config(target_z_mm=(149.4, None)), fk)
    first = plan(helper, actions([0.6, 0.3, 0.05]))
    helper.commit(first, 2)
    next_values = actions([0.05] * 4)
    second = plan(helper, next_values, first.actions[-1])
    np.testing.assert_allclose(second.actions[:, 1], 0.006)
    helper.commit(second, 3)
    lift = actions([0.05] * 4)
    lift[:, 4] = 0.2
    third = plan(helper, lift, second.actions[-1])
    np.testing.assert_allclose(third.actions[:, 1], [0.004, 0.002, 0.0, 0.0], atol=1e-08)
    helper.commit(third, 3)
    np.testing.assert_array_equal(plan(helper, next_values, third.actions[-1]).actions, next_values)

def test_release_on_reopen_and_rearm_for_a_second_grasp():
    helper = make_assist(config(target_z_mm=(149.4, None)), fk)
    first = plan(helper, actions([0.6, 0.05]))
    helper.commit(first, 1)
    second = plan(helper, actions([0.6] * 3), first.actions[-1])
    np.testing.assert_allclose(second.actions[:, 1], [0.004, 0.002, 0], atol=1e-08)
    helper.commit(second, 2)
    third = plan(helper, actions([0.6, 0.05]), second.actions[-1])
    np.testing.assert_allclose(third.actions[:, 1], 0.006)

def test_exact_float32_thresholds_are_inclusive():
    helper = make_assist(config(), fk)
    first = plan(helper, actions([0.2, 0.1]))
    helper.commit(first, 1)
    result = plan(helper, actions([0.2, 0.2, 0.2]), first.actions[-1])
    np.testing.assert_allclose(result.actions[:, 1], [0.005 * 2 / 3, 0.005 / 3, 0], atol=1e-08)

def test_release_can_span_an_inference_boundary():
    helper = make_assist(config(target_z_mm=(149.4, None)), fk)
    first = plan(helper, actions([0.6, 0.05]))
    helper.commit(first, 1)
    release = plan(helper, actions([0.6]), first.actions[-1])
    helper.commit(release, 0)
    next_plan = plan(helper, actions([0.6, 0.6, 0.6]), release.actions[-1])
    np.testing.assert_allclose(next_plan.actions[:, 1], [0.002, 0, 0], atol=1e-08)

def test_abandoned_prediction_does_not_change_state():
    helper = make_assist(config(), fk)
    first = plan(helper, actions([0.6, 0.6, 0.6, 0.3, 0.05]))
    helper.commit(first, 0)
    values = actions([0.6] * 4)
    np.testing.assert_array_equal(plan(helper, values, first.actions[0]).actions, values)
    plan(helper, actions([0.6, 0.05]))
    np.testing.assert_array_equal(plan(helper, values).actions, values)

def test_closed_first_row_skips_instead_of_inventing_opening_or_downward_jump():
    values = actions([0.05] * 4)
    result = plan(make_assist(config(), fk), values)
    np.testing.assert_array_equal(result.actions, values)
    assert result.events[0]['reason'] == 'no_open_control_row_before_close'

@pytest.mark.parametrize('cfg,pose,reason', [(config(), {4: 0.2}, 'close_height_outside_range'), (config(), {1: 0.1}, 'target_not_below_model_z'), (config(target_z_mm=(146.0, None)), {}, 'max_drop_exceeded'), (config(), {1: 0.499}, 'target_not_below_model_z'), (dataclasses.replace(config(), z_ranges=((-200, 160), (100, 160)), target_z_mm=(-164.5, None)), {1: 3.139}, 'target_z_unreachable_within_j2_limit')])
def test_ineligible_dips_leave_chunk_unchanged(cfg, pose, reason):
    values = actions([0.6, 0.3, 0.05, 0.05])
    for (index, value) in pose.items():
        values[:, index] = value
    result = plan(make_assist(cfg, fk), values)
    np.testing.assert_array_equal(result.actions, values)
    assert result.events[0]['reason'] == reason

def test_horizontal_motion_does_not_block_target_z_or_hold():

    def sideways(q):
        return np.array([2000 * q[1], 0, 150 - 100 * q[1]])
    values = actions([0.6, 0.05])
    helper = make_assist(config(), sideways)
    result = plan(helper, values)
    assert result.events[0]['status'] == 'applied'
    assert result.events[0]['corrected_close_z_mm'] == pytest.approx(149.5)
    assert np.linalg.norm(sideways(result.actions[-1, :6]) - sideways(values[-1, :6])) > 5
    helper.commit(result, 1)
    held = plan(helper, actions([0.05, 0.05]), result.actions[-1])
    np.testing.assert_allclose(held.actions[:, 1], 0.005)

def test_hysteresis_ignores_closed_jitter_until_open_threshold():
    values = actions([0.11, 0.09, 0.12, 0.08])
    previous = actions([0.05])[0]
    result = plan(make_assist(config(), fk), values, previous)
    assert not result.events
    np.testing.assert_array_equal(result.actions, values)

def test_closing_outside_executed_prefix_does_not_trigger():
    raw = actions([0.6] * 30 + [0.05] * 20)
    selected = ActionSchedule(mode='sample', legacy_steps=25, sample_start_step=2).make_plan(raw, state=raw[0])
    result = plan(make_assist(config(), fk), selected)
    assert not result.events
    np.testing.assert_array_equal(result.actions, selected)

def test_active_dip_rejects_new_unsafe_targets_before_sending():
    helper = make_assist(config(), fk)
    first = plan(helper, actions([0.6, 0.05]))
    helper.commit(first, 1)
    values = actions([0.05] * 3)
    values[:, 1] = 3.139
    with pytest.raises(ValueError, match='active grasp dip rejected: joint2_limit'):
        plan(helper, values, first.actions[-1])

def test_sdk_fk_units_and_j2_direction():
    sdk_fk = make_piper_fk()
    np.testing.assert_allclose(sdk_fk(np.zeros(6)), [56.128, 0, 213.266], atol=0.02)
    q = np.array([0, 2.0, -1.6, 0, 0.8, 0], dtype=np.float32)
    moved = q.copy()
    moved[1] += 0.005
    assert 0 < sdk_fk(q)[2] - sdk_fk(moved)[2] < 3

def test_target_height_gives_different_angles_for_different_poses():

    def nonlinear_fk(q):
        return np.array([100.0, 0, 150 - 100 * np.sin(q[1]) + 50 * q[4]])
    helper = make_assist(config(target_z_mm=(149.4, None)), nonlinear_fk)
    offsets = []
    for (joint2, joint5) in ((0.0, 0.0), (0.002, 0.01), (0.004, 0.02)):
        q = np.array([0, joint2, 0, 0, joint5, 0], dtype=np.float32)
        (delta, reason) = helper.solve_joint2(q, 149.4, 0)
        assert reason is None
        corrected = q.copy()
        corrected[1] += delta
        assert nonlinear_fk(corrected)[2] == pytest.approx(149.4, abs=0.001)
        np.testing.assert_array_equal(corrected[[0, 2, 3, 4, 5]], q[[0, 2, 3, 4, 5]])
        offsets.append(delta)
    assert len({round(x, 6) for x in offsets}) == 3

def test_varying_poses_reach_same_z_at_last_open_and_first_closed_row():
    helper = make_assist(config(target_z_mm=(149.4, None)), fk)
    base = actions([0.6, 0.4, 0.2, 0.05, 0.05])
    base[:, 1] = [0, 0.001, 0.002, 0.003, 0.004]
    result = plan(helper, base)
    assert fk(result.actions[2, :6])[2] == pytest.approx(149.4, abs=0.001)
    assert fk(result.actions[3, :6])[2] == pytest.approx(149.4, abs=0.001)
    assert result.events[0]['j2_offset_rad'] == pytest.approx(0.003, abs=1e-06)
    assert result.actions[4, 1] - base[4, 1] == pytest.approx(0.003, abs=1e-06)

def test_sdk_solver_reaches_requested_z_with_only_joint2_changed():
    sdk_fk = make_piper_fk()
    q = np.array([0, 2.0, -1.6, 0, 0.8, 0], dtype=np.float32)
    target = float(sdk_fk(q)[2] - 2.0)
    helper = make_assist(GraspDipConfig(target_z_mm=(target, None), z_ranges=((target, target + 3), None)), sdk_fk)
    (delta, reason) = helper.solve_joint2(q, target, 0)
    assert reason is None
    moved = q.copy()
    moved[1] += delta
    assert sdk_fk(moved)[2] == pytest.approx(target, abs=0.001)
    assert 0 <= moved[1] <= 3.14

def test_solver_supports_negative_solution_and_rejects_unreachable_target():

    def reversed_fk(q):
        return np.array([0, 0, 100 + 100 * q[1]])
    helper = make_assist(config(target_z_mm=(149.4, None)), reversed_fk)
    q = np.array([0, 0.5, 0, 0, 0, 0], dtype=np.float32)
    (delta, reason) = helper.solve_joint2(q, 149.4, 0)
    assert reason is None
    assert delta == pytest.approx(-0.006, abs=1e-06)
    flat = make_assist(config(), lambda q: np.array([0, 0, 150]))
    (delta, reason) = flat.solve_joint2(q, 149.5, 0)
    assert delta is None
    assert reason == 'target_z_unreachable_within_j2_limit'

def test_target_zero_is_an_actual_height_not_a_disable_flag():
    cfg = GraspDipConfig(target_z_mm=(None, 0.0), z_ranges=(None, (0.0, 3.0)))
    assert cfg.target_z_mm[1] == 0.0
    assert GraspDipConfig(target_z_mm=(0.0, None), z_ranges=((128, 160), None)).target_z_mm[0] == 0.0

def test_target_can_require_more_than_old_angle_limit():
    cfg = config(target_z_mm=(120.0, None), max_drop_mm=40)
    base = actions([0.6, 0.4, 0.05])
    result = plan(make_assist(cfg, fk), base)
    assert result.events[0]['status'] == 'applied'
    assert result.events[0]['j2_offset_rad'] == pytest.approx(0.3)
    assert result.events[0]['corrected_close_z_mm'] == pytest.approx(120)

@pytest.mark.parametrize('physical_z', [140.0, 170.0])
def test_feedback_outside_range_does_not_dip_even_with_low_prediction(physical_z):
    cfg = GraspDipConfig(target_z_mm=(149.4, None), z_ranges=((149.5, 160), None))
    helper = make_assist(cfg, fk)
    base = actions([0.6, 0.3, 0.05])
    physical = base[0].copy()
    physical[4] = (physical_z - 150) / 100
    result = helper.plan(base, previous_action=base[0], physical_state=physical)
    np.testing.assert_array_equal(result.actions, base)
    assert result.events[0]['reason'] == 'physical_height_outside_range'

def test_trigger_window_is_not_a_floor_and_does_not_cancel_active_grasp():
    cfg = GraspDipConfig(target_z_mm=(149.4, None), z_ranges=((149.5, 160), None))
    helper = make_assist(cfg, fk)
    first = plan(helper, actions([0.6, 0.3, 0.05]))
    assert first.events[0]['status'] == 'applied'
    assert fk(first.actions[1, :6])[2] == pytest.approx(149.4, abs=0.001)
    assert fk(first.actions[2, :6])[2] == pytest.approx(149.4, abs=0.001)
    helper.commit(first, 2)
    following = plan(helper, actions([0.05, 0.05]), first.actions[-1])
    np.testing.assert_allclose(following.actions[:, 1], 0.006, atol=1e-06)

def test_both_arms_gate_against_their_own_current_height():
    helper = make_assist(config(target_z_mm=(149.4, 149.1)), fk)
    base = actions([0.6, 0.3, 0.05])
    base[:, 13] = base[:, 6]
    physical = base[0].copy()
    physical[11] = 0.2
    result = helper.plan(base, previous_action=base[0], physical_state=physical)
    assert result.actions[-1, 1] == pytest.approx(0.006, abs=1e-06)
    np.testing.assert_array_equal(result.actions[:, 7:], base[:, 7:])
    assert result.events[1]['reason'] == 'physical_height_outside_range'

@pytest.mark.parametrize('rpy,expected', [([180, 0, 0], 0), ([-180, 0, 170], 0), ([0, 0, 0], 180), ([90, 0, 45], 90), ([0, 90, 30], 90), ([180, 20, -170], 20), ([180, -20, 170], 20), ([150, 0, 0], 30), ([-150, 0, 180], 30)])
def test_tool_axis_angle_handles_degrees_yaw_and_euler_wrap(rpy, expected):
    assert down_tilt_from_rpy_deg(rpy) == pytest.approx(expected, abs=1e-06)

def test_sdk_fk_supplies_orientation_as_well_as_position():
    sdk_fk = make_piper_fk()
    assert sdk_fk.down_tilt_deg(np.zeros(6)) == pytest.approx(95.0, abs=1e-06)
    np.testing.assert_allclose(sdk_fk(np.zeros(6)), [56.128, 0, 213.266], atol=0.02)

@pytest.mark.parametrize('angle,applied', [(0, True), (20, True), (30, True), (30.01, False), (90, False), (180, False)])
def test_downward_angle_threshold(angle, applied):
    helper = make_assist(config(), fk, down_tilt=lambda q: angle)
    base = actions([0.6, 0.3, 0.05])
    result = plan(helper, base)
    assert result.events[0]['physical_down_tilt_deg'] == angle
    if applied:
        assert result.events[0]['status'] == 'applied'
        assert result.actions[-1, 1] > 0
    else:
        assert result.events[0]['reason'] == 'physical_orientation_not_downward'
        np.testing.assert_array_equal(result.actions, base)

@pytest.mark.parametrize('stage,reason', [('physical', 'physical_orientation_not_downward'), ('close', 'close_orientation_not_downward'), ('approach', 'approach_orientation_not_downward'), ('corrected', 'corrected_orientation_not_downward')])
def test_all_grasp_postures_are_checked_before_applying_correction(stage, reason):
    base = actions([0.6, 0.3, 0.05])
    physical = base[0].copy()
    if stage == 'physical':
        physical[0] = 0.5
    elif stage == 'close':
        base[2, 0] = 0.5
    elif stage == 'approach':
        base[1, 0] = 0.5
    angle = (lambda q: 29.9 + 100 * q[1]) if stage == 'corrected' else lambda q: 90 * q[0]
    helper = make_assist(config(), fk, down_tilt=angle)
    result = helper.plan(base, previous_action=base[0], physical_state=physical)
    assert result.events[0]['reason'] == reason
    np.testing.assert_array_equal(result.actions, base)

@pytest.mark.parametrize('source', ['feedback', 'prediction'])
def test_turning_sideways_releases_existing_offset_smoothly(source):
    helper = make_assist(config(target_z_mm=(149.4, None)), fk, down_tilt=lambda q: 90 * q[0])
    first = plan(helper, actions([0.6, 0.05]))
    helper.commit(first, 1)
    base = actions([0.05] * 3)
    physical = first.actions[-1].copy()
    if source == 'feedback':
        physical[0] = 0.5
    else:
        base[:, 0] = 0.5
    following = helper.plan(base, previous_action=first.actions[-1], physical_state=physical)
    assert following.events[0]['reason'] == 'orientation_not_downward'
    np.testing.assert_allclose(following.actions[:, 1], [0.004, 0.002, 0], atol=1e-08)

def test_right_sideways_grasp_does_not_block_left_downward_grasp():
    helper = make_assist(config(), fk, down_tilt=lambda q: 90 * q[0])
    base = actions([0.6, 0.3, 0.05])
    base[:, 13] = base[:, 6]
    physical = base[0].copy()
    physical[7] = 1.0
    result = helper.plan(base, previous_action=base[0], physical_state=physical)
    assert result.events[0]['status'] == 'applied'
    assert result.events[1]['reason'] == 'physical_orientation_not_downward'
    np.testing.assert_array_equal(result.actions[:, 7:], base[:, 7:])

def test_orientation_callback_cannot_be_silently_omitted():
    with pytest.raises(ValueError, match='orientation FK callback'):
        GraspDip(config(), fk)

@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0, 181.0])
def test_invalid_fk_orientation_fails_closed(bad):
    helper = make_assist(config(), fk, down_tilt=lambda q: bad)
    with pytest.raises(ValueError, match='finite downward angle'):
        plan(helper, actions([0.6, 0.05]))
