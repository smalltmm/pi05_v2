"""Integration coverage for synchronized observation-only task helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from final_policy import load_options
from pi05_assists.options import parse_options
from pi05_assists.runtime import AssistRuntime
from pi05_assists.grip_hold import GripHoldState

ROOT = Path(__file__).resolve().parents[1]


def state():
    return np.array([0.0, 1.5, -1.0, 0.0, -0.5, 0.0, 0.6] * 2, dtype=np.float32)


@pytest.mark.parametrize(
    "profile", list(json.loads((ROOT / "assist_profiles.json").read_text()))
)
def test_every_current_and_saved_profile_initializes(profile):
    task = json.loads((ROOT / "assist_profiles.json").read_text())[profile]["task"]
    rt = AssistRuntime(
        load_options({"task_name": task, "pi05_assist_profile": profile})
    )
    assert rt.args.control_hz == 10


def test_latest_full_commands_and_previous_profiles():
    for task in ("manipulate_pipette", "mixer_operating", "sample_loading"):
        rt = AssistRuntime(load_options({"task_name": task}))
        assert not rt.args.gripper_offset_compensation
        if task == "sample_loading":
            assert rt.args.model_action_horizon == 50 and rt.args.model_action_fps == 25
            assert rt.dip is not None and rt.handover is not None and rt.placement is not None
        else:
            assert rt.args.model_action_horizon == 16 and rt.args.model_action_fps == 10
            if task == "manipulate_pipette":
                assert rt.dip is not None and rt.pipette is not None
                assert rt.pipette.config.count == 3 and rt.pipette.config.depth_mm == 61
            else:
                assert rt.dip is rt.pipette is rt.handover is None
                assert rt.args.legacy_steps == 8
        previous = AssistRuntime(load_options(
            {"task_name": task, "pi05_assist_profile": task + "_previous_20261113"}
        ))
        assert previous.dip is previous.pipette is previous.handover is None
        old = load_options(
            {"task_name": task, "pi05_assist_profile": task + "_previous_20261111"}
        )
        assert old.model_action_horizon == 50
    rt = AssistRuntime(load_options({"task_name": "items_handover"}))
    assert rt.handover_observation is not None and rt.args.legacy_steps == 10


def test_observation_pose_only_changes_model_input_and_right_command_after_transfer(
    monkeypatch,
):
    rt = AssistRuntime(load_options({"task_name": "items_handover"}))
    q = state()
    np.testing.assert_array_equal(rt.model_state(q), q)
    rt.begin_chunk(np.tile(q, (16, 1)), q)
    rt.handover.state.phase = "completed"
    # Avoid rearming solely due to the synthetic test pose's large separation.
    rt.handover.keep_completed = True
    rt.handover.state.completion_mode = "contact"
    action = rt.propose(q)
    np.testing.assert_allclose(action[:7], q[:7])
    assert action[13] == 1.0
    assert np.max(np.abs(action[7:13] - q[7:13])) <= np.deg2rad(120) / 10 + 1e-6
    assert rt.handover_observation.last is not None
    rt.commit()
    supplied = rt.model_state(q)
    np.testing.assert_array_equal(supplied[:7], q[:7])
    np.testing.assert_array_equal(supplied[7:], rt.handover_observation.model_right)
    # Physical state remains the actual obs, even after model input substitution.
    rt.propose(q)
    np.testing.assert_array_equal(rt.pending["current"], q)
    assert rt.handover.state.phase == "completed"
    rt.commit()
    assert rt.steps == 2


def test_pipette_trigger_gets_unsampled_unclipped_full_raw_chunk(monkeypatch):
    args = parse_options(
        [
            "--task",
            "manipulate_pipette",
            "--right-grip-hold",
            "--pipette-press",
            "--legacy-steps",
            "25",
            "--sample-start-step",
            "2",
        ]
    )
    rt = AssistRuntime(args)
    q = state()
    raw = np.tile(q, (50, 1))
    raw[1, 7] = 4.0  # Deliberately outside limits and skipped by nearest-row sampling.
    raw[40, 7] = -4.0  # Unexecuted tail must also participate in the trigger.
    seen = []

    def observe(physical, **kwargs):
        seen.append(kwargs["model_chunk"].copy())
        return False

    monkeypatch.setattr(rt.pipette, "observe", observe)
    rt.begin_chunk(raw, q)
    rt.propose(q)
    np.testing.assert_array_equal(seen[0], raw)
    rt.commit()
    rt.propose(q)
    assert len(seen) == 1  # Trigger evaluation only at inference boundaries.


def test_assembly_pickup_owns_grasp_before_insertion_and_commits_once(monkeypatch):
    rt = AssistRuntime(load_options({
        "task_name": "item_assembly",
        "pi05_assist_options": '--grasp-dip --left-grasp-dip-z-range-mm 130 180 --left-grasp-dip-target-z-mm 130 --right-grasp-dip-z-range-mm 130 180 --right-grasp-dip-target-z-mm 130 --grasp-dip-max-drop-mm 132 --grasp-dip-max-tilt-deg 30 --grasp-dip-close-threshold 0.6 --grasp-dip-open-threshold 0.7 --assembly-trigger-lift-mm 40 --assembly-pickup-lift-speed-mm-s 40 --assembly-grasp-assist --assembly-insert --assembly-stop-tip-distance-mm 213 --assembly-step-mm 3'
    }))
    q = state()
    rt.begin_chunk(np.tile(q, (16, 1)), q)
    # A confirmed helper grasp makes pickup own that arm; insertion must wait.
    rt.left.state = GripHoldState(phase="holding")
    rt.right.state = GripHoldState(phase="holding")
    seen = []
    monkeypatch.setattr(
        rt.assembly,
        "observe",
        lambda *a, **kw: seen.append(kw["pickup_ready"]) or False,
    )
    commits = []
    original = rt.assembly_pickup.commit

    def commit(physical, sent):
        commits.append(sent.copy())
        original(physical, sent)

    monkeypatch.setattr(rt.assembly_pickup, "commit", commit)
    action = rt.propose(q)
    assert seen == [False]
    assert rt.assembly_pickup.owns(0) and rt.assembly_pickup.owns(1)
    np.testing.assert_array_equal(action[:6], q[:6])
    np.testing.assert_array_equal(action[7:13], q[7:13])
    assert not commits and rt.steps == 0
    rt.commit()
    assert len(commits) == 1 and rt.steps == 1
    assert "assembly_pickup" in rt.last_record


def test_new_options_validate_without_hardware_parser():
    args = parse_options(
        [
            "--task",
            "manipulate_pipette",
            "--right-grip-hold",
            "--pipette-press",
            "--pipette-press-depth-mm",
            "55",
            "--pipette-press-move-speed-mm-s",
            "180",
            "--pipette-press-joint-speed-deg-s",
            "60",
            "--model-z-floor",
        ]
    )
    rt = AssistRuntime(args)
    assert rt.pipette.timed_motion and rt.pipette.hz == 10 and rt.floor is not None
    with pytest.raises(SystemExit):
        parse_options(["--task", "click_bell", "--handover-right-observe"])
