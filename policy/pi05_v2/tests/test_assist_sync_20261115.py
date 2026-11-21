"""Latest command timing and release ownership at the competition boundary."""

import copy

import numpy as np
import pytest

from final_policy import load_options
from pi05_assists.grip_hold import GripHoldState
from pi05_assists.runtime import AssistRuntime


@pytest.mark.parametrize("task,fps,horizon,prefix,executed", [
    ("click_bell", 25, 50, 25, 10),
    ("handle_basket", 10, 16, 8, 8),
    ("water_pouring", 10, 16, 8, 8),
    ("table_rearrangement", 10, 16, 8, 8),
    ("items_handover", 10, 16, 10, 10),
    ("drawer_open_place", 10, 16, 6, 6),
    ("mixer_operating", 10, 16, 8, 8),
    ("item_assembly", 10, 16, 8, 8),
    ("manipulate_pipette", 10, 16, 8, 8),
    ("sample_loading", 25, 50, 25, 10),
])
def test_task_timing_matches_latest_command(task, fps, horizon, prefix, executed):
    runtime = AssistRuntime(load_options({"task_name": task}))
    args = runtime.args
    assert (args.model_action_fps, args.model_action_horizon, args.legacy_steps) == (
        fps, horizon, prefix
    )
    assert args.control_hz == 10
    rows = np.zeros((horizon, 14), dtype=np.float32)
    assert len(runtime.schedule.make_plan(rows, state=rows[0])) == executed


class FK:
    def __init__(self, rpy=(180., 0., 0.)):
        self.rpy = rpy

    def pose(self, q):
        return np.r_[1000 * np.asarray(q)[[0, 3, 4]] + [100, 200, 200], self.rpy]

    def __call__(self, q):
        return self.pose(q)[:3]

    def down_tilt_deg(self, q):
        return 0.


def test_basket_release_advances_then_opens_across_chunks_only_after_commit():
    runtime = AssistRuntime(load_options({
        "task_name": "handle_basket",
        "pi05_assist_options": ("--legacy-steps 3 --basket-left-release-advance "
                               "--basket-left-release-forward-mm 50 "
                               "--basket-left-release-speed-mm-s 100")
    }))
    assert runtime.dip is None and runtime.basket_release is not None
    runtime.basket_release.fk = FK((0., 90., 0.))
    runtime.left.state = GripHoldState(phase="holding", baseline=.35)
    physical = np.zeros(14, dtype=np.float32)
    physical[[6, 13]] = [.35, .2]
    raw = np.tile(physical, (16, 1))
    raw[:, 6] = .8
    commands = []
    for _ in range(14):
        if runtime.needs_chunk:
            runtime.begin_chunk(raw, physical)
        phase = runtime.basket_release.state.phase
        index = runtime.basket_release.state.index
        action = runtime.propose(physical)
        assert runtime.basket_release.state.phase == phase
        assert runtime.basket_release.state.index == index
        runtime.commit()
        commands.append(action)
        if action[6] > .6:
            break
    commands = np.asarray(commands)
    assert runtime.cycle >= 2  # Release path survives fresh inference boundaries.
    assert runtime.steps == len(commands)
    np.testing.assert_allclose(commands[:-1, 6], .35)
    np.testing.assert_allclose(commands[-1, 6], .8)
    np.testing.assert_allclose(commands[-1, 0], .05, atol=1e-5)
    np.testing.assert_array_equal(commands[:, 7:], np.tile(physical[7:], (len(commands), 1)))


def test_table_release_lowers_both_arms_and_dwells_before_opening():
    runtime = AssistRuntime(load_options({"task_name": "table_rearrangement"}))
    # Isolate release from grasp preparation and floor projection.
    runtime.dip = runtime.floor = None
    runtime.dip_config = None
    for helper in (*runtime.table_releases.values(), runtime.left, runtime.right):
        helper.fk = FK()
    for helper in (runtime.left, runtime.right):
        helper.state = GripHoldState(phase="holding", baseline=.2)
    physical = np.zeros(14, dtype=np.float32)
    physical[[6, 13]] = .2
    raw = np.tile(physical, (16, 1))
    raw[:, [6, 13]] = .8
    commands, phases = [], []
    for _ in range(40):
        if runtime.needs_chunk:
            runtime.begin_chunk(raw, physical)
        before = {side: copy.copy(helper.state) for side, helper in runtime.table_releases.items()}
        action = runtime.propose(physical)
        for side, helper in runtime.table_releases.items():
            assert helper.state.phase == before[side].phase
            assert helper.state.index == before[side].index
        runtime.commit()
        commands.append(action)
        phases.append(runtime.table_releases["left"].state.phase)
        if np.all(action[[6, 13]] > .5):
            break
    else:
        pytest.fail("table release did not open")
    commands = np.asarray(commands)
    assert runtime.cycle > 0 and runtime.steps == len(commands)
    np.testing.assert_array_equal(commands[:-1, [6, 13]], 0.)
    np.testing.assert_allclose(commands[-1, [6, 13]], .8)
    np.testing.assert_allclose(commands[-1, [4, 11]], -.06, atol=1e-5)
    last_lowering = max(i for i, phase in enumerate(phases) if phase == "lowering")
    assert len(commands) - 1 - last_lowering == 6  # 0.6 s at 10 Hz, not 100 Hz.


def test_pour_grip_cap_uses_raw_release_intent_and_assisted_release(monkeypatch):
    runtime = AssistRuntime(load_options({"task_name": "water_pouring"}))
    assert runtime.pour.config.angle_deg == 90
    assert runtime.pour.config.right_grip_command_cap == .3
    physical = np.zeros(14, dtype=np.float32)
    physical[[6, 13]] = .8
    raw = np.tile(physical, (16, 1))
    intentions = [.9, .55, .7, .2, .79, .8, .9, .4]
    raw[:8, 13] = intentions
    monkeypatch.setattr(runtime.pour, "observe", lambda *a: None)
    monkeypatch.setattr(runtime.pour, "plan_pending", lambda *a: None)
    monkeypatch.setattr(runtime.pour, "next", lambda *a: (None, {}))
    monkeypatch.setattr(runtime.pour, "commit", lambda *a: None)
    runtime.begin_chunk(raw, physical)
    sent = []
    for _ in intentions:
        sent.append(runtime.propose(physical))
        runtime.commit()
    np.testing.assert_allclose(np.asarray(sent)[:, 13], [.9, .3, .3, .2, .3, .8, .9, .3])
    np.testing.assert_array_equal(raw[:8, 13], np.asarray(intentions, dtype=np.float32))
    runtime.pour.phase = "release"
    opened = physical.copy()
    opened[[6, 13]] = 1
    monkeypatch.setattr(runtime.pour, "next", lambda *a: (opened.copy(), {}))
    raw[:, 13] = .2
    runtime.begin_chunk(raw, physical)
    action = runtime.propose(physical)
    np.testing.assert_array_equal(action[[6, 13]], [1, 1])


def test_pipette_approach_and_press_configuration_and_hold_gate(monkeypatch):
    runtime = AssistRuntime(load_options({"task_name": "manipulate_pipette"}))
    cfg = runtime.pipette.config
    assert (cfg.depth_mm, cfg.step_mm, cfg.count, cfg.pattern) == (61, 4, 3, "triangle")
    assert (cfg.approach_speed_mm_s, cfg.approach_joint_speed_deg_s) == (180, 60)
    assert (cfg.move_speed_mm_s, cfg.joint_speed_deg_s, runtime.pipette.hz) == (32, 40, 10)
    physical = np.array([0., 1.5, -1., 0., -.5, 0., .8] * 2, dtype=np.float32)
    raw = np.tile(physical, (16, 1))
    raw[:, 13] = .55  # Above dip close=.5, below grip hold close=.6.
    runtime.begin_chunk(raw, physical)
    runtime.dip_runtime = None
    runtime.right.state = GripHoldState(saw_open=True, peak_open=.8)
    runtime.previous = physical.copy()
    runtime.previous[13] = .55
    action = runtime.propose(physical)
    assert runtime.pending["right"].state.phase != "closing"
    assert action[13] == pytest.approx(.55)


def test_new_defaults_leave_unrequested_helpers_disabled():
    table = AssistRuntime(load_options({"task_name": "table_rearrangement"}))
    assert not table.args.grasp_dip_vertical_down
    basket = AssistRuntime(load_options({"task_name": "handle_basket"}))
    assert not basket.args.basket_left_grasp_dip
    mixer = AssistRuntime(load_options({"task_name": "mixer_operating"}))
    assert mixer.dip is mixer.pipette is mixer.handover is None
