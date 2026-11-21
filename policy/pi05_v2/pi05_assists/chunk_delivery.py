"""Prepare complete blocks for the organizer's unchanged ChunkBridge AST adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


FINGER_TRAVEL = np.float32(.05)


def station_tensor(metres):
    """Preserve the organizer's metres -> normalized wire conversion exactly once."""
    values = metres.detach().cpu().numpy() if hasattr(metres, "detach") else np.asarray(metres)
    if values.shape != (1, 14) or not np.isfinite(values).all():
        raise ValueError("Expected one finite metre action [1,14]")
    if np.any((values[:, [6, 13]] < 0) | (values[:, [6, 13]] > FINGER_TRAVEL)):
        raise ValueError("PI0.5 emitted an out-of-range finger displacement")
    result = metres.clone() if hasattr(metres, "clone") else values.copy()
    result[:, [6, 13]] /= float(FINGER_TRAVEL)
    return result


@dataclass
class PreparedBlock:
    model: object
    env: object
    scalar_bool: object
    task_succeeded: object
    actions: list = field(default_factory=list)
    targets: np.ndarray | None = None
    timings: list = field(default_factory=list)
    index: int = 0
    truncated_done: bool = False

    def after_step(self, terminated, truncated, info):
        """Acknowledge execution, then propose the next cached row only if continuing."""
        if self.targets is not None:
            self.model.commit_action()
        self.index += 1
        self.truncated_done = self.scalar_bool(truncated)
        success = self.task_succeeded(self.env)
        info = dict(info) if isinstance(info, dict) else {}
        info["pi05_step_success"] = success
        if self.targets is not None:
            info["pi05_control_steps"] = self.model.runtime.steps
            if self.model.runtime.steps >= 1000 and not success:
                self.truncated_done = True
        stop = self.scalar_bool(terminated) or self.truncated_done or success
        if self.targets is not None and not stop and self.index < len(self.targets):
            # The next action is already planned. Do not encode feedback or
            # invoke model.next_action here (it could infer at a bad boundary).
            proposed = self.model.runtime.propose()
            if proposed is None or not np.array_equal(proposed, self.targets[self.index]):
                raise RuntimeError("Cached action changed after block preparation")
        return info, self.truncated_done, stop


def prepare_eval_chunk(env, model, obs, *, encode_obs, encode_action,
                       start_inference, finish_inference, scalar_bool, task_succeeded):
    block = PreparedBlock(model, env, scalar_bool, task_succeeded)
    final_mode = hasattr(model, "next_action")
    if final_mode and model.runtime.steps >= 1000:
        block.truncated_done = True
        return block
    needs_inference = not final_mode or model.runtime.needs_chunk
    started = start_inference(env.unwrapped.device) if needs_inference else None
    prompt = getattr(env, "_current_instruction", None)
    if final_mode:
        encoded = encode_obs(obs) if needs_inference else None
        first = model.next_action(encoded, prompt)
        runtime = model.runtime
        if runtime.pending.get("recovery_hold"):
            targets = first[None, :].copy()
        else:
            targets = runtime.control[runtime.row:].copy()
            if not np.array_equal(targets[0], first):
                raise RuntimeError("First action differs from cached chunk")
        if not 1 <= len(targets) <= 1000 - runtime.steps:
            raise ValueError("Chunk exceeds remaining episode budget")
        block.targets = targets
    else:
        targets = model.infer(encode_obs(obs), prompt=str(prompt or model.task_name.replace("_", " ")))
    block.actions = [encode_action(row, env, strict_dim=model.strict_action_dim) for row in targets]
    # This exact metadata is installed by the supplied ChunkBridge.__init__.
    # Its generic step publishes commands unchanged, unlike the finalist serial
    # bridge which performs the same conversion in its step method. Simulation
    # and serial paths continue to receive metre commands.
    adaptation = getattr(env, "adaptation", None)
    if isinstance(adaptation, dict) and adaptation.get("kind") == "precomputed-action-loop-v1":
        block.actions = [station_tensor(action) for action in block.actions]
    if needs_inference:
        finish_inference(started, block.timings, env.unwrapped.device)
    return block
