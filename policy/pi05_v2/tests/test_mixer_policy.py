"""Mixer-only profile and observation/action integration, without hardware."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_policy import load_options
from mixer_policy import MixerFinalPolicy
from pi05_assists.grip_hold import GripHoldState


def state():
    return np.array([
        .4776, 1.6136, -.8825, -.9539, .9555, .7692, .8,
        .5790478587, 1.8236421347, -1.5691033602,
        .0564788543, .7552912235, .0156206964, .2,
    ], dtype=np.float32)


def observation(q):
    return {
        "observation/state": q,
        "observation/left_wrist_image": np.zeros((480, 640, 3), np.uint8),
    }


class Client:
    strict_action_dim = True

    def __init__(self):
        self.calls = self.resets = 0

    def infer(self, obs, prompt):
        self.calls += 1
        return np.tile(state(), (16, 1))

    def reset(self):
        self.resets += 1


def model():
    return MixerFinalPolicy(Client(), load_options({"task_name": "mixer_operating"}))


def test_mixer_full17_profile_and_config_overrides():
    policy = model()
    args, config = policy.args, policy.mixer.config
    assert (args.model_action_fps, args.model_action_horizon, args.legacy_steps) == (10, 16, 8)
    assert args.right_grip_hold_closed_threshold == .75
    assert args.right_grip_hold_open_threshold == .8
    assert config.prepare_after_right_release and config.placement_trigger == "pose"
    assert config.prepare_timeout_s == 30 and config.prepare_orientation_tolerance_deg == 5
    assert config.trigger_max_tcp_z_mm == 130 and config.trigger_radius_px is None
    assert config.prepare_speed_mm_s == 120
    assert config.floor_approach_base_dir == (0, -1, 0)


def test_owned_mixer_frames_skip_model_keep_right_hold_count_and_reset():
    policy = model()
    policy.mixer.phase = "done"
    policy.mixer.hold_target = state()
    policy.right_hold_command = .1
    for _ in range(3):
        target = policy.next_action(observation(state()))
        assert target[13] == pytest.approx(.1)
        policy.commit_action()
    assert policy.client.calls == 0
    assert policy.runtime.steps == 3 and policy.runtime.row == 0
    policy.reset()
    assert policy.runtime.steps == 0 and policy.mixer.phase == "model"
    assert policy.client.resets == 1 and policy.right_hold_command is None


def test_pose_gate_takes_over_after_pickup_without_new_model_chunks():
    policy = model()
    policy.runtime.right.state = GripHoldState(
        phase="holding", baseline=.2, saw_open=True, peak_open=.8,
    )
    phases = []
    for _ in range(8):
        target = policy.next_action(observation(state()))
        assert target.shape == (14,) and np.isfinite(target).all()
        phases.append(policy.mixer.phase)
        policy.commit_action()
    assert "placement_prepare" in phases
    assert policy.mixer.owns_motion and policy.client.calls == 1
    assert policy.runtime.steps == 8


def test_model_stage_keeps_left_stationary_until_pose_gate():
    policy = model()
    q = state()
    q[0] += .1
    target = policy.next_action(observation(q))
    np.testing.assert_allclose(target[:7], q[:7])
    assert policy.mixer.phase == "model"
    policy.commit_action()
