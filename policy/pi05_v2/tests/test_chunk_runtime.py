"""One-observation block execution and command-driven assistance contracts."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from final_policy import FinalPolicy, load_options
from pi05_assists.chunk_runtime import ChunkRuntime, ChunkPlanner, GripIntent, CLOSE_FRAMES, MODEL_FALLBACK_TASKS
from pi05_assists.config import TASK_PROMPTS
from pi05_assists.handover import HandoverAssist, HandoverConfig
from pi05_assists.bell_press import DualBellPress, BellPressConfig


def state():
    return np.array([.3, 1., -1., 0., 0., 0., .8] * 2, np.float32)


class NoObservation:
    def __getitem__(self, key):
        raise AssertionError("cached execution read an observation")

    def __iter__(self):
        raise AssertionError("cached execution iterated an observation")


class Client:
    strict_action_dim = True

    def __init__(self, actions):
        self.actions, self.calls, self.resets = actions, [], 0

    def infer(self, obs, prompt):
        self.calls.append(obs)
        return self.actions.copy()

    def reset(self):
        self.resets += 1


def policy(task, q=None):
    q = state() if q is None else q
    return FinalPolicy(Client(np.tile(q, (50, 1))), load_options({"task_name": task}))


@pytest.mark.parametrize("task", TASK_PROMPTS)
def test_all_profiles_consume_only_one_observation_per_block(task):
    model = policy(task)
    first = model.next_action({"observation/state": state()})
    block = model.runtime.control.copy()
    np.testing.assert_array_equal(first, block[0])
    assert model.runtime.steps == 0
    assert model.runtime.planner.cycle == -1  # speculative state not published
    model.commit_action()
    for expected in block[1:]:
        np.testing.assert_array_equal(model.next_action(NoObservation()), expected)
        model.commit_action()
    assert len(model.client.calls) == 1
    assert model.runtime.steps == len(block)
    assert model.runtime.needs_chunk
    assert model.runtime.planner.cycle == 0
    new = state()
    new[0] += .05
    model.next_action({"observation/state": new})
    np.testing.assert_array_equal(model.client.calls[-1]["observation/state"], new)
    assert len(model.client.calls) == 2


@pytest.mark.parametrize("task", ["water_pouring", "sample_loading"])
def test_completed_assistance_resumes_fresh_model_blocks_without_relatching(task, monkeypatch):
    model = policy(task)
    planner = model.runtime.planner
    completed_pose = state()
    completed_pose[[6, 13]] = 1.
    planner.terminal = completed_pose.copy()
    planner.grips = [GripIntent(True, .05), GripIntent(True, .05)]
    if task == "sample_loading":
        planner.handover_done = True
        planner.helpers.handover.state.phase = "completed"
        planner.helpers.placement.phase = "done"
    else:
        planner.helpers.pour.phase = "done"

    def no_assistance(*args, **kwargs):
        pytest.fail("completed assistance restarted")

    monkeypatch.setattr(ChunkPlanner, "_plan_rows", no_assistance)
    for cycle in range(2):
        measured = state()
        measured[0] += cycle * .05
        raw = np.tile(state(), (50, 1))
        raw[:, 0] += .1 + cycle * .1 + np.arange(50) * .002
        raw[:, [6, 13]] = [.7, .9]
        model.client.actions = raw
        _, expected = model.runtime.planner.model_actions(raw, measured)
        np.testing.assert_array_equal(model.next_action({"observation/state": measured}), expected[0])
        np.testing.assert_array_equal(model.runtime.control, expected)
        assert all(r["kind"] == "model" for r in model.runtime.records)
        assert len(model.client.calls) == cycle + 1
        model.commit_action()
        for action in expected[1:]:
            np.testing.assert_array_equal(model.next_action(NoObservation()), action)
            model.commit_action()
        np.testing.assert_array_equal(model.runtime.planner.terminal, completed_pose)
        if task == "sample_loading":
            assert model.runtime.planner.handover_done
            assert model.runtime.planner.helpers.placement.phase == "done"
        else:
            assert model.runtime.planner.helpers.pour.phase == "done"
    model.reset()
    assert model.runtime.planner.terminal is None


@pytest.mark.parametrize("task", TASK_PROMPTS)
def test_acknowledgement_reset_and_limit(task):
    model = policy(task)
    model.runtime.steps = 997
    model.next_action({"observation/state": state()})
    assert len(model.runtime.control) == 3
    with pytest.raises(RuntimeError, match="commit_action"):
        model.next_action(NoObservation())
    model.commit_action()
    for _ in range(2):
        model.next_action(NoObservation())
        model.commit_action()
    with pytest.raises(StopIteration):
        model.next_action(NoObservation())
    model.reset()
    assert model.runtime.steps == 0 and model.runtime.needs_chunk
    assert model.runtime.planner.cycle == -1
    with pytest.raises(RuntimeError, match="No proposed"):
        model.commit_action()


@pytest.mark.parametrize("task", sorted(MODEL_FALLBACK_TASKS))
@pytest.mark.parametrize("error", [ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError])
def test_failed_assist_runs_whole_model_block_and_retries_transactionally(task, error, monkeypatch):
    model = policy(task)
    raw = model.client.actions
    raw[:, 0] += np.arange(len(raw)) * .003
    _, expected = model.runtime.planner.model_actions(raw, state())
    original = ChunkPlanner.plan
    calls = []
    def fail(planner, raw, measured, **kwargs):
        calls.append(measured.copy())
        planner.handover_done = True
        planner.terminal = state()
        raw[:] = 0.  # A failed planner may have modified its own inputs.
        raise error("failed path")
    monkeypatch.setattr(ChunkPlanner, "plan", fail)
    for attempt in range(2):
        measured = state()
        measured[0] += .01 * attempt
        np.testing.assert_allclose(model.next_action({"observation/state": measured}), expected[0])
        assert not model.runtime.pending.get("recovery_hold")
        np.testing.assert_allclose(model.runtime.control, expected)
        assert not model.runtime.planner.handover_done
        assert all(r["kind"] == "model_fallback" for r in model.runtime.records)
        assert all(r["assist_error"] == "failed path" for r in model.runtime.records)
        model.commit_action()
        for action in expected[1:]:
            np.testing.assert_allclose(model.next_action(NoObservation()), action)
            model.commit_action()
        assert len(model.client.calls) == len(calls) == attempt + 1
        assert model.runtime.steps == len(expected) * (attempt + 1)
        assert model.runtime.planner.terminal is None
        assert not model.runtime.planner.handover_done
    monkeypatch.setattr(ChunkPlanner, "plan", original)
    model.next_action({"observation/state": state()})
    assert len(model.client.calls) == 3
    assert all(r["kind"] != "model_fallback" for r in model.runtime.records)


@pytest.mark.parametrize("profile", ["mixer_operating", "legacy"])
def test_mixer_is_model_only_even_with_legacy_assist_flags(profile):
    args = load_options({"task_name": "mixer_operating"})
    if profile == "legacy":
        args.mixer_servo = args.right_grip_hold = True
    runtime = ChunkRuntime(args)
    q = state()
    q[13] = .05
    raw = np.tile(q, (16, 1))
    raw[:, 0] += np.arange(16) * .01
    runtime.begin_chunk(raw, q)
    np.testing.assert_array_equal(runtime.control, raw[:8])
    assert runtime.planner.helpers is None


def bell_fk(q):
    return np.array([0., 0., 200. - 100. * q[1]])


bell_fk.down_tilt_deg = lambda q: 0.


def test_bell_inserts_complete_press_and_holds_without_reading_feedback():
    args = load_options({"task_name": "click_bell"})
    args.model_action_fps, args.legacy_steps, args.sample_start_step = 10, 6, 0
    model = FinalPolicy(Client(np.zeros((6, 14), np.float32)), args)
    cfg = BellPressConfig(z_range_mm=(155, 200), target_z_mm=155, max_drop_mm=45)
    model.runtime.planner.helpers.bell = DualBellPress({1: cfg}, bell_fk)
    q = state()
    q[8], q[13] = .2, .0
    raw = np.tile(q, (6, 1))
    raw[:, 8] = [.25, .3, .35, .3, .2, .1]
    model.client.actions = raw
    model.next_action({"observation/state": q})
    records = model.runtime.records
    block = model.runtime.control.copy()
    hold = [i for i, r in enumerate(records) if r["kind"] == "bell_press_hold"]
    assert len(hold) == 2
    assert len(block) > len(raw)
    assert all(bell_fk(block[i, 7:13])[2] == pytest.approx(155, abs=.01) for i in hold)
    for i in range(len(block)):
        if i:
            model.next_action(NoObservation())
        model.commit_action()
    assert model.runtime.steps == len(block) and len(model.client.calls) == 1


class HandoverFK:
    def pose(self, q):
        return np.array([0., 1000 * q[0], 150., *np.rad2deg(q[3:6])])


def make_planner(task):
    planner = ChunkPlanner(load_options({"task_name": task}))
    planner.helpers.dip = None  # Isolate the task sequence from pickup geometry.
    return planner


def sequence_start(planner, q):
    planner.actions, planner.records = [], []
    planner.cursor, planner.budget, planner.source_row = q.copy(), 1000, 0


@pytest.mark.parametrize("task", ["items_handover", "sample_loading"])
def test_handover_closes_left_before_opening_right_and_never_checks_feedback(task, monkeypatch):
    p = make_planner(task)
    q = state()
    q[6], q[13] = .8, 0.
    sequence_start(p, q)
    helper = p.helpers.handover
    monkeypatch.setattr(helper, "next", lambda *a, **k: pytest.fail("feedback state machine called"))
    if task == "sample_loading":
        monkeypatch.setattr(helper, "tips", lambda q: (np.eye(4), np.eye(4)))
        monkeypatch.setattr(helper, "plan_approach", lambda q, right: (np.tile(q[:6], (2, 1)), np.eye(4)))
        placement = p.helpers.placement

        def start(q, previous):
            placement.hold = q.copy()
            placement.paths = [np.tile(q[:6], (2, 1)), np.tile(q[:6], (2, 1))]
            placement.phase = "approach"

        monkeypatch.setattr(placement, "start", start)
        monkeypatch.setattr(placement, "arrival", lambda *a: pytest.fail("arrival check called"))
    raw = np.tile(q, (8, 1))
    raw[:, 13] = .9
    p._handover(raw)
    actions = np.asarray(p.actions)
    kinds = [r["kind"] for r in p.records]
    left = [i for i, kind in enumerate(kinds) if kind == "handover_close_left"]
    right = [i for i, kind in enumerate(kinds) if kind == "handover_open_right"]
    assert len(left) == CLOSE_FRAMES and max(left) < min(right)
    np.testing.assert_array_equal(actions[left, 13], 0.)
    assert np.all(actions[right, 6] <= .2)
    assert np.all(np.diff(actions[right, 13]) >= 0)
    assert p.handover_done and p.grips[0].closed and not p.grips[1].closed
    if task == "sample_loading":
        assert kinds.index("placement_approach") > max(right)
        assert kinds[-1] == "placement_release" and actions[-1, 6] == 1.
        assert p.terminal is not None


def test_handover_trigger_uses_model_intent_and_does_not_release_donor_early():
    p = make_planner("items_handover")
    p.helpers.handover_observation = None
    p.helpers.handover = HandoverAssist(HandoverConfig(relative_rpy_deg=(0., 0., 0.)), HandoverFK())
    q = state()
    q[[0, 7]], q[[6, 13]] = [-.3, .3], [.8, .05]
    p.grips[1] = GripIntent(True, .05)
    raw = np.tile(q, (50, 1))
    raw[:, 13] = .9
    actions, records = p.plan(raw, q, budget=1000)
    assert actions[0, 13] <= .05
    assert any(r["kind"] == "handover_open_right" for r in records)
    assert p.handover_done


@pytest.mark.parametrize("close_row", [0, 4, 9])
@pytest.mark.parametrize("transfer", [False, True])
def test_items_handover_pauses_four_frames_after_right_close_before_motion_or_transfer(close_row, transfer, monkeypatch):
    model = policy("items_handover")
    planner = model.runtime.planner
    planner.helpers.dip = None
    raw = np.tile(state(), (16, 1))
    raw[:, 0] += np.arange(16) * .01
    raw[:, 7] += np.arange(16) * .02
    raw[close_row:, 13] = .1
    model.client.actions = raw
    monkeypatch.setattr(ChunkPlanner, "_handover_requested",
                        lambda p, row: transfer and p.grips[1].closed and not p.handover_done)

    def handover(p, remaining):
        action = p.cursor.copy()
        action[0] += .1
        p._emit(action, "handover_test")
        p.handover_done = True

    monkeypatch.setattr(ChunkPlanner, "_handover", handover)
    model.next_action({"observation/state": state()})
    block = model.runtime.control.copy()
    kinds = [r["kind"] for r in model.runtime.records]
    dwell = [i for i, kind in enumerate(kinds) if kind == "grasp_dwell"]
    assert dwell == list(range(close_row + 1, close_row + 5))
    np.testing.assert_array_equal(block[dwell], np.tile(block[close_row], (4, 1)))
    assert block[close_row, 13] == 0.
    if transfer:
        assert kinds[-1] == "handover_test" and len(block) == close_row + 6
    elif close_row < 9:
        assert block[close_row + 5, 7] == pytest.approx(raw[close_row + 1, 7])
    model.commit_action()
    for action in block[1:]:
        np.testing.assert_array_equal(model.next_action(NoObservation()), action)
        model.commit_action()
    assert len(model.client.calls) == 1
    model.next_action({"observation/state": state()})
    assert all(r["kind"] != "grasp_dwell" for r in model.runtime.records)


def test_post_handover_observation_motion_starts_at_block_state():
    p = make_planner("items_handover")
    p.handover_done = True
    p.grips[0] = GripIntent(True, 0.)
    q = state()
    q[13] = 1.
    raw = np.tile(q, (16, 1))
    raw[:, 7:13] += .3
    actions, _ = p.plan(raw, q, budget=1000)
    step = p.helpers.handover_observation.step
    assert np.max(np.abs(actions[0, 7:13] - q[7:13])) <= step + 1e-6
    assert np.max(np.abs(np.diff(actions[:, 7:13], axis=0))) <= step + 1e-6
    np.testing.assert_array_equal(actions[:, 13], 1.)


def test_table_release_paths_finish_before_gripper_opens(monkeypatch):
    p = make_planner("table_rearrangement")
    q = state()
    q[6] = 0.
    p.grips[0] = GripIntent(True, 0.)
    helper = p.helpers.table_releases["left"]
    path = np.tile(q[:6], (3, 1))
    path[:, 0] += [.01, .02, .03]
    result = (path, 180., 140.)
    monkeypatch.setattr(helper, "_path", lambda q: result)
    raw = np.tile(q, (50, 1))
    raw[:, 6] = .9
    actions, records = p.plan(raw, q, budget=1000)
    release = np.array([i for i, r in enumerate(records) if r["kind"] == "release_path"])
    assert len(release) >= 4
    held = actions[release[0] - 1, 6]
    np.testing.assert_array_equal(actions[release[:-1], 6], held)
    assert actions[release[-1], 6] == pytest.approx(.9)
    np.testing.assert_allclose(actions[release[-1], :6], path[-1])


@pytest.mark.parametrize("legacy_flags", [False, True])
def test_assembly_is_model_only_even_with_legacy_assist_flags(legacy_flags):
    args = load_options({"task_name": "item_assembly"})
    if legacy_flags:
        args.grasp_dip = args.assembly_grasp_assist = args.assembly_insert = True
        args.gripper_offset_compensation = True
    model = FinalPolicy(Client(np.tile(state(), (16, 1))), args)
    raw = model.client.actions
    raw[:, 0] += np.arange(16) * .01
    raw[:, 7] -= np.arange(16) * .01
    raw[:8, 6] = [.8, .1, .1, .8, .8, .1, .8, .8]
    raw[:8, 13] = raw[:8, 6][::-1]
    first = model.next_action({"observation/state": state()})
    np.testing.assert_array_equal(first, raw[0])
    np.testing.assert_array_equal(model.runtime.control, raw[:8])
    assert model.runtime.planner.helpers is None
    for row in range(8):
        if row:
            np.testing.assert_array_equal(model.next_action(NoObservation()), raw[row])
        model.commit_action()
    assert len(model.client.calls) == 1 and model.runtime.steps == 8
    assert model.runtime.planner.terminal is None
    assert model.runtime.needs_chunk
    model.next_action({"observation/state": state()})
    assert len(model.client.calls) == 2


def test_grasp_preparation_is_finite_and_close_does_not_wait_for_contact(monkeypatch):
    from pi05_assists.grasp_dip import GraspDip, GraspDipConfig
    from pi05_assists.grasp_dip_runtime import GraspDipRuntime

    p = ChunkPlanner(load_options({"task_name": "manipulate_pipette"}))
    h = p.helpers
    h.pipette = None

    def fk(q):
        return np.array([100., 0., 160. - 100. * q[1]])

    h.dip = GraspDip(GraspDipConfig(
        target_z_mm=(None, 120.), z_ranges=(None, (120., 180.)),
        close_threshold=.5, open_threshold=.6, max_drop_mm=132.,
    ), fk, down_tilt=lambda q: 0.)
    monkeypatch.setattr(GraspDipRuntime, "_close_in_place", lambda *a: pytest.fail("contact check called"))
    monkeypatch.setattr(GraspDipRuntime, "_next", lambda *a: pytest.fail("arrival check called"))
    q = np.zeros(14, dtype=np.float32)
    q[[6, 13]] = .8
    raw = np.tile(q, (16, 1))
    raw[:, 13] = .1
    actions, records = p.plan(raw, q, budget=1000)
    kinds = [r["kind"] for r in records]
    assert kinds.count("grasp_prepare") == 3
    assert kinds.count("grasp_dwell") == 2
    assert kinds.count("grasp_close") == CLOSE_FRAMES
    assert len(actions) == 8 + 3 + 2 + CLOSE_FRAMES
    close = kinds.index("grasp_close")
    assert actions[close, 13] == 0.
    assert fk(actions[close, 7:13])[2] == pytest.approx(120, abs=.01)
    assert p.grips[1].closed


def test_pipette_consumes_preplanned_path_without_feedback(monkeypatch):
    p = make_planner("manipulate_pipette")
    p.grips[1] = GripIntent(True, 0.)
    q = state()
    q[13] = 0.
    h = p.helpers.pipette
    expected = np.tile(q, (5, 1))
    expected[:, 0] += np.arange(5) * .01

    def plan(q, cycle):
        h.path = expected
        return True

    monkeypatch.setattr(h, "plan", plan)
    monkeypatch.setattr(h, "observe", lambda *a, **k: pytest.fail("feedback trigger called"))
    monkeypatch.setattr(h, "commit", lambda *a, **k: pytest.fail("feedback commit called"))
    actions, records = p.plan(np.tile(q, (16, 1)), q, budget=1000)
    np.testing.assert_allclose(actions, expected)
    assert all(r["kind"] == "pipette_press" for r in records)


def test_pour_preserves_timed_hold_and_return_without_arrival_checks(monkeypatch):
    p = make_planner("water_pouring")
    q = state()
    q[[6, 13]] = 0.
    sequence_start(p, q)
    h = p.helpers.pour
    h.pickup = [q[:6].copy(), q[7:13].copy()]

    def align(current):
        h.paths = {"align": np.tile(current, (2, 1))}
        h.full_joints = current[7:13].copy()
        h.full_joints[5] = .1
        h.phase = "align"

    def return_path(current):
        h.paths = {"joint_return": np.tile(current, (2, 1))}
        h.phase = "joint_return"

    monkeypatch.setattr(h, "plan_alignment", align)
    monkeypatch.setattr(h, "plan_return", return_path)
    monkeypatch.setattr(h, "at_goal", lambda *a: pytest.fail("arrival confirmation called"))
    p._pour(np.tile(q, (8, 1)))
    kinds = [r["kind"] for r in p.records]
    assert kinds.count("pour_hold") == 15
    assert kinds.index("pour_tilt") < kinds.index("pour_hold") < kinds.index("pour_joint_return")
    assert kinds[-1] == "pour_release"
    np.testing.assert_array_equal(p.actions[-1][[6, 13]], 1.)


@pytest.fixture
def adapter(monkeypatch):
    # Deterministic timing hooks for adapter contract tests.
    import final_policy
    timing = ModuleType("policy.inference_timing")
    timing.start_inference = lambda device: 0.
    timing.finish_inference = lambda started, timings, device: timings.append(0.)
    package = ModuleType("policy")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "policy", package)
    monkeypatch.setitem(sys.modules, "policy.inference_timing", timing)
    path = Path(final_policy.__file__).with_name("deploy_policy.py")
    spec = importlib.util.spec_from_file_location("chunk_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_action_adapter_does_not_encode_cached_observations(adapter, monkeypatch):
    model = policy("drawer_open_place")
    calls = []

    def encode(obs):
        calls.append(obs)
        return {"observation/state": state()}

    monkeypatch.setattr(adapter, "encode_obs", encode)
    adapter.get_action(model, "first")
    adapter.commit_action(model)
    for _ in range(5):
        adapter.get_action(model, NoObservation())
        adapter.commit_action(model)
    assert calls == ["first"]


def test_eval_executes_whole_block_and_still_honors_early_success(adapter, monkeypatch):
    import torch

    for success_at in (None, 3):
        model = policy("drawer_open_place")
        encodings = []
        monkeypatch.setattr(adapter, "encode_obs", lambda obs: encodings.append(obs) or {"observation/state": state()})
        monkeypatch.setattr(adapter, "encode_action", lambda action, env, **kw: action)

        class Env:
            unwrapped = SimpleNamespace(device="cpu")
            steps = 0

            def step(self, action):
                self.steps += 1
                return NoObservation(), 0, False, False, {"elapsed_steps": torch.tensor(self.steps)}

            def get_wrapper_attr(self, name):
                return lambda: self.steps == success_at

        env = Env()
        _, _, _, timings = adapter._eval_final(env, model, "initial")
        assert len(timings) == len(model.client.calls) == 1
        assert env.steps == (success_at or 6)
        assert model.runtime.steps == env.steps
        assert encodings == ["initial"]


def test_factory_never_selects_mixer_servo(adapter, monkeypatch, tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "assets" / "test").mkdir(parents=True)
    (tmp_path / "assets" / "test" / "norm_stats.json").write_text("{}")
    (tmp_path / "src" / "openpi").mkdir(parents=True)
    monkeypatch.setattr(adapter, "PI05JaxWorkerClient", lambda args: Client(np.tile(state(), (16, 1))))
    model = adapter.get_model(dict(
        checkpoint_path=str(tmp_path), openpi_root=str(tmp_path),
        task_name="mixer_operating", pi05_final_mode=True,
        pi05_assist_options="--mixer-servo --right-grip-hold --mixer-servo-config /unused/legacy.json",
    ))
    assert type(model) is FinalPolicy and model.runtime.planner.helpers is None


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("height,tilt,allowed", [
    (131., 0., False), (132., 20., True), (180., 20., True),
    (181., 0., False), (150., 21., False), (150., 0., True),
])
def test_table_grip_uses_dip_pose_limits(side, height, tilt, allowed, monkeypatch):
    p = ChunkPlanner(load_options({"task_name": "table_rearrangement"}))
    dip = p.helpers.dip
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    start = side * 7
    model[start:start + 2] = [height, tilt]
    model[start + 6] = .2
    # The gate must inspect the model pose, not a corrected command or obs.
    target = model.copy()
    target[start:start + 2] = [1000., 90.]
    closed, _ = p._grips(target, model)
    assert closed[side] == allowed
    assert p.grips[side].closed == allowed
    assert target[start + 6] == pytest.approx(0. if allowed else .2)
    if allowed:
        model[start:start + 2] = [1000., 90.]
        target = model.copy()
        p._grips(target, model)
        assert p.grips[side].closed and target[start + 6] == 0.


def test_table_grip_uses_custom_dip_limits_and_requires_enabled_dip(monkeypatch):
    from dataclasses import replace
    p = ChunkPlanner(load_options({"task_name": "table_rearrangement"}))
    dip = p.helpers.dip
    dip.config = replace(dip.config, z_ranges=((200., 220.), (300., 320.)), max_tilt_deg=10.)
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    model[[0, 1, 6, 7, 8, 13]] = [210., 10., .25, 310., 11., .25]
    closed, _ = p._grips(model.copy(), model)
    assert closed == [True, False]
    p.helpers.dip = None
    closed, _ = p._grips(model.copy(), model)
    assert not closed[1] and not p.grips[1].closed


@pytest.mark.parametrize("legacy_flag", [False, True])
def test_basket_release_has_no_advance_or_arm_hold(legacy_flag):
    options = {"task_name": "handle_basket"}
    if legacy_flag:
        options["pi05_assist_options"] = "--basket-left-release-advance"
    args = load_options(options)
    assert args.basket_left_release_advance == legacy_flag
    p = ChunkPlanner(args)
    q = state()
    raw = np.tile(q, (16, 1))
    raw[:, 0] += np.arange(16) * .01
    raw[:8, 6] = [.4, .8, .3, .8, .8, .8, .8, .8]
    for _ in range(2):
        actions, records = p.plan(raw, q, budget=1000)
        assert len(actions) == 8
        assert all(r["kind"] == "model" for r in records)
        np.testing.assert_allclose(actions[:, :6], raw[:8, :6])
        np.testing.assert_allclose(actions[:, 7:13], raw[:8, 7:13])
        # Initial small opening is not latched; open -> close still holds.
        assert actions[2, 6] == pytest.approx(.3)
        np.testing.assert_allclose(actions[3:, 6], raw[3:8, 6])
        q = actions[-1].copy()
        raw[:, 0] += .1


@pytest.mark.parametrize("height,tilt,grip,allowed", [
    (129., 0., .4, False), (130., 20., .45, True),
    (180., 20., .4, True), (181., 0., .4, False),
    (150., 21., .4, False), (150., 0., .46, False),
])
def test_pipette_grip_requires_dip_pose_before_latching(height, tilt, grip, allowed, monkeypatch):
    p = ChunkPlanner(load_options({"task_name": "manipulate_pipette"}))
    dip = p.helpers.dip
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    model[[7, 8, 13]] = [height, tilt, grip]
    target = model.copy()
    target[[7, 8]] = [150., 0.]  # Only the predicted pose controls the gate.
    closed, _ = p._grips(target, model)
    assert closed[1] == allowed
    assert p.grips[1].closed == allowed
    assert target[13] == pytest.approx(0. if allowed else grip)
    if allowed:
        model[[7, 8, 13]] = [250., 90., .8]
        target = model.copy()
        p._grips(target, model)
        assert p.grips[1].closed and target[13] == 0.


def test_pipette_grip_reuses_custom_dip_parameters(monkeypatch):
    from dataclasses import replace
    p = ChunkPlanner(load_options({"task_name": "manipulate_pipette"}))
    dip = p.helpers.dip
    dip.config = replace(dip.config, z_ranges=(None, (200., 220.)), max_tilt_deg=10.)
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    model[13] = .4
    for height, tilt, expected in [(150., 0., False), (210., 11., False), (210., 10., True)]:
        model[[7, 8]] = [height, tilt]
        closed, _ = p._grips(model.copy(), model)
        assert closed[1] == expected


@pytest.mark.parametrize("height,tilt,grip,allowed", [
    (129., 0., .24, False), (130., 20., .24, True),
    (180., 20., .24, True), (181., 0., .24, False),
    (150., 21., .24, False), (150., 0., .26, False),
])
def test_sample_loading_right_grip_requires_dip_pose(height, tilt, grip, allowed, monkeypatch):
    p = ChunkPlanner(load_options({"task_name": "sample_loading"}))
    dip = p.helpers.dip
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    model[[7, 8, 13]] = [height, tilt, grip]
    target = model.copy()
    target[[7, 8]] = [150., 0.]
    closed, _ = p._grips(target, model)
    assert closed[1] == allowed
    assert target[13] == pytest.approx(.2 if allowed else grip)
    # The receiving left arm is not restricted to the donor's pickup region.
    assert p._grip_pose_allowed(model, 0)
    if allowed:
        model[[7, 8, 13]] = [250., 90., .8]
        target = model.copy()
        p._grips(target, model)
        assert p.grips[1].closed and target[13] == pytest.approx(.2)


def test_sample_loading_grip_reuses_custom_dip_limits(monkeypatch):
    from dataclasses import replace
    p = ChunkPlanner(load_options({"task_name": "sample_loading"}))
    dip = p.helpers.dip
    dip.config = replace(dip.config, z_ranges=(None, (200., 220.)), max_tilt_deg=10.)
    monkeypatch.setattr(dip, "_xyz", lambda q: np.array([0., 0., q[0]]))
    monkeypatch.setattr(dip, "_tilt", lambda q: q[1])
    model = state()
    model[13] = .24
    for height, tilt, expected in [(150., 0., False), (210., 11., False), (210., 10., True)]:
        model[[7, 8]] = [height, tilt]
        closed, _ = p._grips(model.copy(), model)
        assert closed[1] == expected


def test_sample_preparation_close_stays_closed_before_model_lift(monkeypatch):
    from pi05_assists.grasp_dip import GraspDip, GraspDipConfig
    args = load_options({"task_name": "sample_loading"})
    args.model_action_fps, args.legacy_steps, args.sample_start_step = 10, 8, 0
    p = ChunkPlanner(args)
    p.helpers.handover = None
    def fk(q):
        return np.array([100., 0., 160. - 100. * q[1]])
    p.helpers.dip = GraspDip(GraspDipConfig(
        target_z_mm=(None, 130.), z_ranges=(None, (130., 180.)),
        close_threshold=.25, open_threshold=.4, max_drop_mm=132.,
    ), fk, down_tilt=lambda q: 0.)
    q = np.zeros(14, dtype=np.float32)
    q[[6, 13]] = .8
    raw = np.tile(q, (8, 1))
    raw[2, 13] = .1
    raw[3:, 8] = -.2  # Model lifts and reopens after its predicted closure.
    actions, records = p.plan(raw, q, budget=1000)
    kinds = [r['kind'] for r in records]
    close = kinds.index('grasp_close')
    assert fk(actions[close, 7:13])[2] == pytest.approx(130., abs=.01)
    assert np.all(actions[close:, 13] <= .1 + 1e-6)
    assert kinds.count('grasp_close') == 5  # Close command plus 0.4 s at 10 Hz.
    np.testing.assert_array_equal(actions[close:close + 5],
                                  np.tile(actions[close], (5, 1)))
    assert any(fk(action[7:13])[2] > 130.01 for action in actions[close + 5:])
    assert p.grips[1].closed


@pytest.mark.parametrize("close_row", [2, 9])
def test_sample_model_close_dwells_before_lift_and_only_once(close_row, monkeypatch):
    args = load_options({"task_name": "sample_loading"})
    args.model_action_fps, args.legacy_steps, args.sample_start_step = 10, 10, 0
    p = ChunkPlanner(args)
    p.helpers.dip = None
    p.helpers.handover = None
    monkeypatch.setattr(p, '_grip_pose_allowed', lambda model, side: True)
    q = state()
    raw = np.tile(q, (50, 1))
    raw[close_row:, 13] = .1
    raw[close_row + 1:, 8] -= .2
    actions, records = p.plan(raw, q, budget=1000)
    kinds = [r['kind'] for r in records]
    assert [i for i, kind in enumerate(kinds) if kind == 'grasp_close'] == list(range(close_row + 1, close_row + 5))
    assert actions[close_row, 13] == pytest.approx(.1)
    np.testing.assert_array_equal(actions[close_row:close_row + 5],
                                  np.tile(actions[close_row], (5, 1)))
    if close_row < 9:
        assert actions[close_row + 5, 8] == pytest.approx(raw[close_row + 1, 8])
    following, records = p.plan(raw, actions[-1], budget=1000)
    assert all(r['kind'] != 'grasp_close' for r in records)
    assert np.all(following[:, 13] <= .1)


def test_sample_handover_freezes_donor_and_waits_before_receiver(monkeypatch):
    p = make_planner('sample_loading')
    p.helpers.placement = None
    p.grips[1] = GripIntent(True, .1)
    q = state()
    q[13] = .1
    raw = np.tile(q, (50, 1))
    raw[:, 0] += .2
    raw[:, 7] += .1
    monkeypatch.setattr(p, '_handover_requested', lambda model: True)
    helper = p.helpers.handover
    monkeypatch.setattr(helper, 'tips', lambda q: (np.eye(4), np.eye(4)))
    def approach(q, right):
        path = np.tile(q[:6], (3, 1))
        path[:, 0] += [.01, .02, .03]
        return path, np.eye(4)
    monkeypatch.setattr(helper, 'plan_approach', approach)
    actions, records = p.plan(raw, q, budget=1000)
    kinds = [r['kind'] for r in records]
    assert kinds[0] == 'handover_position_right'
    assert kinds.count('handover_right_dwell') == helper.relative_config.stable_frames
    first_left = kinds.index('handover_open_left')
    np.testing.assert_allclose(actions[:first_left, :7], np.tile(q[:7], (first_left, 1)))
    np.testing.assert_allclose(actions[:, 7:13], np.tile(actions[0, 7:13], (len(actions), 1)))
    assert kinds.index('handover_close_left') < kinds.index('handover_open_right')


@pytest.mark.parametrize("task", sorted(MODEL_FALLBACK_TASKS))
def test_failed_assist_fallback_respects_remaining_budget(task, monkeypatch):
    model = policy(task)
    model.runtime.steps = 998
    def fail(*args, **kwargs):
        raise ValueError("failed trajectory")
    monkeypatch.setattr(ChunkPlanner, "plan", fail)
    model.next_action({"observation/state": state()})
    assert len(model.runtime.control) == 2
    model.commit_action()
    model.next_action(NoObservation())
    model.commit_action()
    with pytest.raises(StopIteration):
        model.next_action(NoObservation())
    assert model.runtime.steps == 1000 and len(model.client.calls) == 1


def test_pipette_rejected_plan_falls_back_without_publishing_failed_phase(monkeypatch):
    from pi05_assists.pipette_press import PipettePress
    model = policy("manipulate_pipette")
    model.runtime.planner.grips[1] = GripIntent(True, .1)
    monkeypatch.setattr(ChunkPlanner, "_pipette_ready", lambda *args: True)
    def reject(helper, *args, **kwargs):
        helper.reason = "unreachable press goal"
        helper.phase = "deferred"
        return False
    monkeypatch.setattr(PipettePress, "plan", reject)
    model.next_action({"observation/state": state()})
    assert all(r["kind"] == "model_fallback" for r in model.runtime.records)
    np.testing.assert_allclose(model.runtime.control[:, 13], .8)
    assert model.runtime._planned.helpers.pipette.phase == "idle"
    assert "unreachable" in model.runtime.records[0]["assist_error"]


def test_invalid_model_is_not_sent_as_auxiliary_failure_fallback():
    from final_policy import PolicyFailure
    model = policy("click_bell")
    model.client.actions[:, 0] = np.nan
    with pytest.raises(PolicyFailure, match="nonfinite"):
        model.next_action({"observation/state": state()})
    assert model.runtime.pending is None and model.runtime.steps == 0
