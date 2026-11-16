"""Submission contract, feedback, lifecycle, limits and no-hardware checks."""

from pathlib import Path
import ast
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from final_policy import FinalPolicy, load_options
from pi05_assists.options import parse_options
from pi05_assists.runtime import AssistRuntime
from pi05_assists.grasp_dip import make_piper_fk
from pi05_assists.config import TASK_PROMPTS
import deploy_policy


class Client:
    strict_action_dim = True

    def __init__(self, rows):
        self.rows = rows
        self.seen = []
        self.resets = 0

    def infer(self, obs, prompt):
        self.seen.append((obs, prompt))
        return self.rows.copy()

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def state():
    return np.array([0.3, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8] * 2, np.float32)


def observation(q):
    obs = {"observation/state": q}
    for key in (
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
    ):
        obs[key] = np.zeros((8, 8, 3), np.uint8)
    return obs


def plain_policy(task="items_handover"):
    args = load_options({"task_name": task})
    return FinalPolicy(Client(np.tile(state(), (args.model_action_horizon, 1))), args)


@pytest.mark.parametrize("task", list(TASK_PROMPTS))
def test_all_task_profiles_construct_and_produce_finite_actions(task):
    model = plain_policy(task)
    q = state()
    for _ in range(3):
        action = model.next_action(observation(q))
        assert action.shape == (14,) and action.dtype == np.float32
        assert np.isfinite(action).all()
        model.commit_action()
    assert model.runtime.steps == 3


def test_each_frame_uses_measured_obs_and_only_commit_advances():
    model = plain_policy()
    q = state()
    a = model.next_action(observation(q))
    assert model.runtime.steps == 0
    np.testing.assert_array_equal(model.runtime.pending["current"], q)
    with pytest.raises(RuntimeError, match="commit_action"):
        model.next_action(observation(q))
    model.commit_action()
    q[0] += 0.1
    model.next_action(observation(q))
    np.testing.assert_array_equal(model.runtime.pending["current"], q)
    assert model.runtime.steps == 1
    model.commit_action()
    assert len(model.client.seen) == 1
    model.reset()
    assert model.runtime.steps == 0 and model.runtime.previous is None
    assert model.runtime.needs_chunk and model.client.resets == 1


def test_offsets_apply_once_and_do_not_modify_physical_feedback():
    args = parse_options(
        [
            "--task",
            "mixer_operating",
            "--gripper-offset-compensation",
            "--left-gripper-offset",
            ".1",
            "--right-gripper-offset",
            ".2",
        ]
    )
    model = FinalPolicy(Client(np.tile(state(), (50, 1))), args)
    q = state()
    a = model.next_action(observation(q))
    np.testing.assert_allclose(
        model.client.seen[0][0]["observation/state"][[6, 13]], [0.9, 1.0]
    )
    np.testing.assert_allclose(a[[6, 13]], [0.7, 0.6])
    np.testing.assert_array_equal(q, state())
    np.testing.assert_array_equal(model.runtime.pending["current"], q)


def test_1000_frames_including_assists_and_reset():
    model = plain_policy()
    model.runtime.steps = 999
    model.next_action(observation(state()))
    model.commit_action()
    with pytest.raises(StopIteration):
        model.next_action(observation(state()))
    assert model.runtime.clock() == 100.0
    model.reset()
    assert model.runtime.clock() == 0


def test_25hz_prefix_samples_to_ten_rows_and_10hz_prefix_keeps_six():
    for task, count in [("items_handover", 10), ("drawer_open_place", 6)]:
        model = plain_policy(task)
        model.next_action(observation(state()))
        assert len(model.runtime.control) == count


@pytest.mark.parametrize(
    "options", [["--control-hz", "100"], ["--model-action-fps", "0"]]
)
def test_invalid_rates_fail_before_inference(options):
    with pytest.raises(SystemExit):
        parse_options(options)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.2, 1.3])
def test_observation_values_pass_through_without_validation_or_replacement(value, caplog):
    model = plain_policy("drawer_open_place")
    q = state()
    q[[0, 6, 13]] = value
    original = q.copy()
    for _ in range(4):
        model.runtime.row = len(model.runtime.control)
        action = model.next_action(observation(q))
        np.testing.assert_array_equal(model.client.seen[-1][0]["observation/state"], original)
        np.testing.assert_array_equal(model.runtime.pending["current"], original)
        np.testing.assert_array_equal(action, state())
        model.commit_action()
    assert model.runtime.steps == 4
    np.testing.assert_array_equal(q, original)
    assert not caplog.records


@pytest.mark.parametrize("task", list(TASK_PROMPTS))
def test_gripper_observation_overshoot_is_preserved_without_stopping(task):
    model = plain_policy(task)
    q = state()
    q[[6, 13]] = [-0.2, 1.3]
    original = q.copy()
    action = model.next_action(observation(q))
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(model.runtime.pending["current"][[6, 13]], original[[6, 13]])
    np.testing.assert_array_equal(q, original)
    model.commit_action()
    assert model.runtime.steps == 1


def test_gripper_calibration_saturates_normalized_values():
    from pi05_assists.config import GripperCalibration

    calibration = GripperCalibration()
    assert calibration.model_to_mm(-0.2) == 0
    assert calibration.model_to_mm(1.3) == 100


@pytest.mark.parametrize("during_propose", [False, True])
def test_floor_failure_holds_then_reinfers_without_committing_failed_helpers(monkeypatch, during_propose):
    model = plain_policy("table_rearrangement")
    q = state()
    if during_propose:
        model.next_action(observation(q))
        model.commit_action()
    before = model.runtime.steps
    previous_runtime = model.runtime

    def fail(actions):
        raise ValueError("model Z floor IK failed")

    monkeypatch.setattr(previous_runtime.floor, "apply", fail)
    target = model.next_action(observation(q))
    np.testing.assert_allclose(target, q)
    assert model.runtime.steps == before
    assert model.runtime.pending["recovery_hold"]
    model.commit_action()
    assert model.runtime.steps == before + 1
    assert "IK failed" in model.runtime.last_record["recovery"]
    assert model.runtime.needs_chunk
    inferred = len(model.client.seen)
    model.next_action(observation(q))
    assert len(model.client.seen) == inferred + 1
    assert not model.runtime.pending.get("recovery_hold")
    model.commit_action()
    assert model.runtime.steps == before + 2


def test_nonfinite_model_output_holds_and_recovers():
    model = plain_policy("drawer_open_place")
    model.client.rows[0, 0] = np.nan
    target = model.next_action(observation(state()))
    np.testing.assert_allclose(target, state())
    model.commit_action()
    model.client.rows[0, 0] = state()[0]
    model.next_action(observation(state()))
    assert not model.runtime.pending.get("recovery_hold")
    model.commit_action()
    assert model.runtime.steps == 2


def test_repeated_rejected_chunks_hold_one_frame_then_recover(monkeypatch):
    model = plain_policy("drawer_open_place")

    def reject(*args, **kwargs):
        model.runtime.row = len(model.runtime.control)
        return None

    monkeypatch.setattr(model.runtime, "propose", reject)
    target = model.next_action(observation(state()))
    np.testing.assert_allclose(target, state())
    assert len(model.client.seen) == 2
    assert model.runtime.steps == 0
    model.commit_action()
    model.next_action(observation(state()))
    assert len(model.client.seen) == 3
    assert not model.runtime.pending.get("recovery_hold")


def test_repeated_numeric_failure_in_eval_still_stops_at_frame_limit():
    model = plain_policy("drawer_open_place")
    model.client.rows[0, 0] = np.nan
    model.runtime.steps = 999
    env = Env()
    _, info, truncated, _ = deploy_policy.eval(env, model, env.obs())
    assert truncated
    assert env.count == 1
    assert info["pi05_control_steps"] == model.runtime.steps == 1000


def test_no_hardware_dependencies_in_submission_helpers():
    banned = {
        "real_robot",
        "piper_sdk",
        "can",
        "cv2",
        "pyrealsense2",
        "socket",
        "serial",
    }
    for path in (ROOT / "pi05_assists").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not {name.split(".")[0] for name in names} & banned, str(path)
    assert not {"real_robot", "piper_sdk", "can", "pyrealsense2"} & sys.modules.keys()


def test_pure_fk_matches_recorded_sdk_reference():
    # Every calibration has captured states; numerical FK determinism is tested
    # independently against the unmodified SDK implementation during migration.
    fk = make_piper_fk()
    pose = fk.pose(np.zeros(6))
    np.testing.assert_allclose(pose[:3], [56.12758099, 0.0, 213.26502882], atol=0.01)


class Env:
    def __init__(self, stop=None, truncate=False):
        self.count = 0
        self.stop = stop
        self.truncate = truncate
        self.unwrapped = SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(14,)), device="cpu"
        )
        self.device = "cpu"

    def obs(self):
        return {
            "robot": {"qpos": state()[None, :]},
            "sensor": {
                key: {"color": np.zeros((1, 8, 8, 3), np.uint8)}
                for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
        }

    def step(self, action):
        self.count += 1
        assert tuple(action.shape) == (1, 14)
        return self.obs(), 0, False, self.truncate and self.count == self.stop, {}

    def get_wrapper_attr(self, name):
        return lambda: self.count == self.stop and not self.truncate


def test_official_eval_stops_success_and_commits_exactly_executed_frames():
    env = Env(stop=2)
    model = plain_policy()
    obs, info, truncated, timings = deploy_policy.eval(env, model, env.obs())
    assert info["pi05_step_success"] and not truncated
    assert env.count == model.runtime.steps == len(timings) == 2


def test_official_eval_finishes_one_chunk_and_reinfers_next_call():
    env = Env()
    model = plain_policy()
    deploy_policy.eval(env, model, env.obs())
    assert env.count == 10 and len(model.client.seen) == 1
    deploy_policy.eval(env, model, env.obs())
    assert env.count == 20 and len(model.client.seen) == 2


def test_official_eval_preserves_truncation_and_failure_does_not_commit():
    env = Env(stop=1, truncate=True)
    model = plain_policy()
    assert deploy_policy.eval(env, model, env.obs())[2]
    assert model.runtime.steps == 1
    model.reset()

    def failed(action):
        raise RuntimeError("executor unavailable")

    env.step = failed
    with pytest.raises(RuntimeError, match="executor unavailable"):
        deploy_policy.eval(env, model, env.obs())
    assert model.runtime.steps == 0


def test_bell_insertions_keep_model_cursor_and_count_execution_frames(monkeypatch):
    import pi05_assists.runtime as runtime_module
    from pi05_assists.bell_press import BellPressConfig, DualBellPress

    def fk(q):
        return np.array([0.0, 0.0, 160.0 - 100.0 * q[1]])

    fk.down_tilt_deg = lambda q: 0.0
    args = parse_options(
        ["--task", "click_bell", "--schedule", "sequential", "--legacy-steps", "2"]
    )
    rt = AssistRuntime(args)
    rt.bell = DualBellPress(
        {
            1: BellPressConfig(
                z_range_mm=(120.0, 180.0), target_z_mm=130.0, max_drop_mm=50.0
            )
        },
        fk,
    )
    q = np.zeros(14, np.float32)
    q[6], q[13], q[8] = 0.8, 0.03, 0.15
    rows = np.tile(q, (2, 1))
    rows[:, 8] = 0.2
    rt.begin_chunk(rows, q)
    kinds = []
    source_rows = []
    for _ in range(30):
        action = rt.propose(q)
        if action is None:
            break
        kinds.append(rt.pending["bell"].kind)
        source_rows.append(rt.row)
        # This idealized executor follows the command; the API still receives
        # the measured state on the following call.
        rt.commit()
        q = action
    assert rt.needs_chunk
    assert kinds.count("bell_press_down") == 2
    assert kinds.count("bell_press_hold") == 2
    assert rt.steps == len(kinds) > len(rows)
    assert source_rows[:4] == [0, 0, 0, 0]


def test_reset_discards_pending_assist_and_prompt_is_official():
    model = plain_policy("mixer_operating")
    model.next_action(observation(state()))
    assert model.client.seen[0][1] == TASK_PROMPTS["mixer_operating"]
    model.reset()
    assert model.runtime.pending is None
    model.next_action(observation(state()))
    model.commit_action()
    assert model.runtime.steps == 1


@pytest.mark.parametrize(
    "left,right", [(0.0, 1.0), (0.2, 0.5), (1.0, 0.0), (-0.1, 1.1)]
)
def test_competition_gripper_units_and_input_ownership(left, right):
    q = state()
    q[[6, 13]] = [left, right]
    saved = q.copy()
    encoded = deploy_policy.encode_action(q, Env(), strict_dim=True)
    actual = encoded.cpu().numpy()[0]
    np.testing.assert_allclose(actual[[6, 13]], np.clip([left, right], 0, 1) * 0.05)
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(actual[joints], saved[joints])
    np.testing.assert_array_equal(q, saved)
    assert encoded.dtype == deploy_policy.torch.float32 and tuple(encoded.shape) == (
        1,
        14,
    )


@pytest.mark.parametrize("final_mode", [False, True])
def test_env_step_receives_metres_while_policy_feedback_stays_normalized(final_mode):
    env = Env(stop=2)
    args = parse_options(["--task", "drawer_open_place"])
    client = Client(np.tile(state(), (50, 1)))
    if final_mode:
        model = FinalPolicy(client, args)
    else:
        model = client
        model.task_name = "drawer_open_place"
    original_step = env.step
    sent = []

    def step(action):
        sent.append(action.cpu().numpy()[0].copy())
        return original_step(action)

    env.step = step
    deploy_policy.eval(env, model, env.obs())
    assert len(sent) == 2
    np.testing.assert_allclose(np.array(sent)[:, [6, 13]], 0.04)
    np.testing.assert_allclose(client.seen[0][0]["observation/state"][[6, 13]], 0.8)
    np.testing.assert_allclose(client.rows[:, [6, 13]], 0.8)
    if final_mode:
        np.testing.assert_allclose(model.runtime.previous[[6, 13]], 0.8)
        assert model.runtime.steps == 2


def test_single_action_api_converts_once_and_commit_keeps_normalized_target():
    args = parse_options(["--task", "drawer_open_place"])
    q = state()
    q[[6, 13]] = [0.2, 1.0]
    model = FinalPolicy(Client(np.tile(q, (50, 1))), args)
    action = deploy_policy.get_action(model, Env().obs())
    np.testing.assert_allclose(action[[6, 13]], [0.01, 0.05])
    np.testing.assert_allclose(model.runtime.pending["target"][[6, 13]], [0.2, 1.0])
    deploy_policy.commit_action(model)
    np.testing.assert_allclose(model.runtime.previous[[6, 13]], [0.2, 1.0])
    # Repeated calls must not multiply the retained model chunk by 0.05 again.
    action = deploy_policy.get_action(model, Env().obs())
    np.testing.assert_allclose(action[[6, 13]], [0.01, 0.05])
