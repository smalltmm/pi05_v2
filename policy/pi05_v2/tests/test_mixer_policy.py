"""Mixer-only profile and observation/action integration, without hardware."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_policy import load_options
from mixer_policy import MixerFinalPolicy


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


def test_mixer_profile_is_model_only():
    policy = model()
    args = policy.args
    assert (args.model_action_fps, args.model_action_horizon, args.legacy_steps) == (10, 16, 8)
    assert not args.mixer_servo and not args.right_grip_hold
    assert policy.runtime.planner.helpers is None


def test_cached_mixer_frames_need_no_images_or_state():
    policy = model()
    target = policy.next_action(observation(state()))
    np.testing.assert_array_equal(target, state())
    policy.commit_action()
    for _ in range(7):
        np.testing.assert_array_equal(policy.next_action(None), state())
        policy.commit_action()
    assert policy.client.calls == 1 and policy.runtime.steps == 8
    assert policy.runtime.needs_chunk


def test_mixer_reset_discards_unexecuted_cached_actions():
    policy = model()
    policy.next_action(observation(state()))
    policy.reset()
    assert policy.runtime.steps == 0 and policy.runtime.pending is None
    assert policy.runtime.needs_chunk and policy.client.resets == 1
    policy.next_action(observation(state()))
    assert policy.client.calls == 2


def test_model_stage_preserves_model_arm_and_gripper_commands():
    policy = model()
    q = state()
    q[0] += .1
    q[13] = .9
    target = policy.next_action(observation(q))
    np.testing.assert_array_equal(target, state())
    policy.commit_action()
