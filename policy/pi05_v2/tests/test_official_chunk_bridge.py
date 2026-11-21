"""CPU-only contract checks against the unchanged organizer bridge sources.

Set RSC_BASELINE_DIR to the unzipped official adaptation baseline. Synthetic
model/state fixtures do not measure real model inference or station latency.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

baseline_dir = os.environ.get("RSC_BASELINE_DIR")
if not baseline_dir:
    pytest.skip("Organizer baseline required; set RSC_BASELINE_DIR", allow_module_level=True)
BASELINE = Path(baseline_dir).expanduser().resolve()
if not (BASELINE / "rsc_real/chunks.py").is_file():
    raise FileNotFoundError(f"RSC_BASELINE_DIR does not contain the organizer bridge: {BASELINE}")
sys.path.insert(0, str(BASELINE))

import deploy_policy as adapter
from final_policy import FinalPolicy, load_options
from pi05_assists.config import TASK_PROMPTS
from pi05_assists.chunk_delivery import station_tensor
from rsc_real.finalist10_pi05_serial import Finalist10SerialBridge
from rsc_real.chunks import ChunkBridge, ChunkGuard, intercept_eval


def state():
    return np.array([.3, 1., -1., 0., 0., 0., .8] * 2, np.float32)


def payload(q):
    return {"observation.state": q.copy(), **{
        "observation.images." + name: np.zeros((32, 32, 3), np.uint8)
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}


class Client:
    strict_action_dim = True

    def __init__(self, args):
        self.rows = np.tile(state(), (args.model_action_horizon, 1))
        self.seen = []

    def reset(self):
        self.seen.clear()

    def infer(self, obs, prompt):
        self.seen.append(obs["observation/state"].copy())
        return self.rows.copy()


def setup(task="mixer_operating", bridge_type=ChunkBridge):
    args = load_options({"task_name": task})
    model = FinalPolicy(Client(args), args)
    bridge = bridge_type(adapter, model, TASK_PROMPTS[task], timeout=3)
    return model, bridge


def wait_committed(model, frames):
    import time
    deadline = time.monotonic() + 3
    while model.runtime.steps != frames:
        if time.monotonic() >= deadline:
            raise AssertionError("Measured feedback was not committed")
        time.sleep(.001)


def test_public_eval_is_accepted_by_unmodified_official_adapter():
    _, metadata = intercept_eval(adapter.eval)
    assert metadata["kind"] == "precomputed-action-loop-v1"
    model, bridge = setup()
    assert type(bridge) is ChunkBridge
    assert bridge.adaptation == metadata
    bridge.close()


def test_organizer_sources_unchanged():
    manifest = json.loads((BASELINE / "BASELINE.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((BASELINE / path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("task", TASK_PROMPTS)
def test_all_tasks_deliver_one_response_per_block_with_real_feedback(task):
    import time
    model, bridge = setup(task)
    q = state()
    requests = frames = 0
    # Verify the station's supplied chunk guard accepts the delivered units.
    lower, upper = np.full(14, -10.), np.full(14, 10.)
    lower[[6, 13]], upper[[6, 13]] = 0., 1.
    guard = ChunkGuard("audit", lower, upper)
    try:
        while frames < 1000:
            block = bridge.infer(payload(q))
            requests += 1
            assert 1 <= len(block) <= 1000 - frames
            if frames == 0:
                assert len(block) > 1
                assert model.runtime.steps == 0  # Declaration is not execution.
            np.testing.assert_array_equal(model.client.seen[-1], q)
            np.testing.assert_allclose(block[:, [6, 13]], .8, atol=1e-6)
            guard.accept({"attempt_id": "audit", "seq": requests-1,
                          "actions": block}, time.monotonic())
            for action in block:
                guard.execute(action, lambda command: None)
                frames += 1
                q = state()
                q[0] += (frames % 7 + 1) * .00001  # Distinct measured value.
                bridge.feedback(payload(q))
        bridge.finish()
        assert bridge.steps == model.runtime.steps == guard.steps == frames == 1000
        assert requests == len(model.client.seen)
        assert requests <= 167  # 6/8/10-frame profiles, not 1000 requests.
        assert not bridge.worker.is_alive()
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


@pytest.mark.parametrize("task,total,bridge_type,expected_requests", [
    ("mixer_operating", 16, Finalist10SerialBridge, 16),
    ("mixer_operating", 16, ChunkBridge, 2),
    ("click_bell", 446, Finalist10SerialBridge, 446),
    ("click_bell", 446, ChunkBridge, 45),
])
def test_station_request_count_differs_from_model_inference_count(task, total, bridge_type, expected_requests):
    model, bridge = setup(task, bridge_type=bridge_type)
    frames = requests = 0
    try:
        while frames < total:
            block = bridge.infer(payload(state()))
            requests += 1
            for _ in block[:total-frames]:
                frames += 1
                bridge.feedback(payload(state()))
        if bridge_type is ChunkBridge and frames % len(block):
            wait_committed(model, frames)
            bridge.close()  # Official finish() requires the entire declared block.
        else:
            bridge.finish()
        assert requests == expected_requests
        assert frames == model.runtime.steps == total
        if bridge_type is ChunkBridge:
            assert len(model.client.seen) == expected_requests
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


@pytest.mark.parametrize("executed", [0, 1, 3, 8])
def test_official_close_stops_unexecuted_suffix_without_fake_feedback(executed):
    model, bridge = setup()
    try:
        block = bridge.infer(payload(state()))
        assert len(block) == 8
        for _ in range(executed):
            bridge.feedback(payload(state()))
        wait_committed(model, executed)
        if executed == len(block):
            bridge.finish()
        else:
            bridge.close()
        assert model.runtime.steps == bridge.steps == executed
        assert len(model.client.seen) == 1
        assert not bridge.worker.is_alive()
        assert bridge.outputs.empty()
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


def test_missing_feedback_rejects_new_chunk_without_committing():
    model, bridge = setup()
    try:
        bridge.infer(payload(state()))
        bridge.timeout = .05
        with pytest.raises(RuntimeError, match="Previous chunk feedback"):
            bridge.infer(payload(state()))
        assert model.runtime.steps == 0
        bridge.close()
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


def test_unit_conversion_preserves_boundaries_and_rejects_wrong_units():
    q = state()
    q[[6, 13]] = [0., 1.]
    metres = adapter.to_competition_action(q)
    np.testing.assert_allclose(station_tensor(metres[None, :])[0], q)
    with pytest.raises(ValueError, match="finger displacement"):
        station_tensor(q[None, :])


def test_changed_action_after_declaration_is_rejected():
    model, bridge = setup()
    q = state()
    command = q[None, :].copy()
    try:
        bridge.action_chunk([command])
        command[0, 0] += .1
        with pytest.raises(ValueError, match="mutated"):
            bridge.step(command)
        assert bridge.steps == model.runtime.steps == 0
    finally:
        bridge.close()


def test_assisted_long_block_is_delivered_and_acknowledged_without_reinference(monkeypatch):
    from pi05_assists.chunk_runtime import ChunkPlanner
    def plan(planner, raw, measured, *, budget):
        # Isolate delivery from trajectory geometry; emulate a long assist plan.
        rows = np.tile(state(), (min(123, budget), 1))
        return rows, [{"kind": "audit_assist", "source_row": 0} for _ in rows]
    monkeypatch.setattr(ChunkPlanner, "plan", plan)
    model, bridge = setup("click_bell")
    try:
        block = bridge.infer(payload(state()))
        assert len(block) == 123
        for _ in block:
            bridge.feedback(payload(state()))
        bridge.finish()
        assert model.runtime.steps == 123 and len(model.client.seen) == 1
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


def test_assist_failure_delivers_complete_model_block_then_retries(monkeypatch):
    from pi05_assists.chunk_runtime import ChunkPlanner
    calls = []
    original = ChunkPlanner.plan
    def fail(planner, *args, **kwargs):
        calls.append(planner.cycle)
        planner.handover_done = True
        raise RuntimeError("injected assist failure")
    monkeypatch.setattr(ChunkPlanner, "plan", fail)
    model, bridge = setup("click_bell")
    model.client.rows[:, 0] += np.arange(len(model.client.rows)) * .002
    _, expected = model.runtime.planner.model_actions(model.client.rows, state())
    try:
        block = bridge.infer(payload(state()))
        assert len(block) == len(expected) == 10
        np.testing.assert_allclose(block, expected, atol=1e-6)
        for _ in block:
            bridge.feedback(payload(state()))
        # The next block is a fresh planning attempt, not a one-frame recovery.
        block = bridge.infer(payload(state()))
        assert len(block) == 10 and len(calls) == len(model.client.seen) == 2
        assert not model.runtime.planner.handover_done
        for _ in block:
            bridge.feedback(payload(state()))
        monkeypatch.setattr(ChunkPlanner, "plan", original)
        bridge.infer(payload(state()))
        assert all(r["kind"] != "model_fallback" for r in model.runtime.records)
        bridge.close()  # Third block was declared but never executed.
        assert model.runtime.steps == 20
    finally:
        if not bridge.stopped.is_set():
            bridge.close()


@pytest.mark.parametrize("bridge_type", [Finalist10SerialBridge, ChunkBridge])
@pytest.mark.parametrize("fault", ["nonfinite", "worker_error"])
def test_unrecoverable_failure_uses_organizer_error_channel_promptly(bridge_type, fault):
    import time
    model, bridge = setup(bridge_type=bridge_type)
    if fault == "nonfinite":
        model.client.rows[0, 0] = np.nan
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("worker exited")
        model.client.infer = fail
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="PolicyFailure"):
            bridge.infer(payload(state()))
        assert time.monotonic() - started < 1
        assert bridge.steps == model.runtime.steps == 0
        assert model.runtime.pending is None
        assert model.failure is not None
    finally:
        bridge.close()
    assert not bridge.worker.is_alive()


def test_failure_after_completed_chunk_preserves_executed_count():
    model, bridge = setup()
    try:
        block = bridge.infer(payload(state()))
        for _ in block:
            bridge.feedback(payload(state()))
        wait_committed(model, len(block))
        model.client.rows[0, 0] = np.nan
        with pytest.raises(RuntimeError, match="PolicyFailure"):
            bridge.infer(payload(state()))
        assert bridge.steps == model.runtime.steps == len(block)
        assert model.runtime.pending is None
    finally:
        bridge.close()
