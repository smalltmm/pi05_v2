"""Finals observation/action API, with no physical robot dependencies."""

from pathlib import Path
import json
import logging
import shlex
import time

import numpy as np

try:
    from .pi05_assists.options import parse_options
    from .pi05_assists.runtime import AssistRuntime, clip_actions
    from .pi05_assists.config import TASK_PROMPTS
except ImportError:
    from pi05_assists.options import parse_options
    from pi05_assists.runtime import AssistRuntime, clip_actions
    from pi05_assists.config import TASK_PROMPTS


def load_options(usr_args):
    root = Path(__file__).resolve().parent
    profiles = json.loads((root / "assist_profiles.json").read_text())
    task = str(usr_args.get("task_name") or "")
    profile = str(usr_args.get("pi05_assist_profile") or task)
    if profile not in profiles:
        raise ValueError(
            f"Unknown task/profile {profile!r}; available: {list(profiles)}"
        )
    entry = profiles[profile]
    if entry["task"] != task:
        raise ValueError(f"Profile {profile} belongs to {entry['task']}, not {task}")
    tokens = [
        s.replace("${ASSIST_ROOT}", str(root / "pi05_assists"))
        for s in entry["options"]
    ]
    extra = usr_args.get("pi05_assist_options") or []
    tokens.extend(shlex.split(extra) if isinstance(extra, str) else extra)
    if task == "mixer_operating":
        try:
            from .mixer_policy import parse_mixer_options
        except ImportError:
            from mixer_policy import parse_mixer_options
        args = parse_mixer_options(tokens)
    else:
        args = parse_options(tokens)
    if args.task != task:
        raise ValueError("pi05_assist_options cannot override task_name")
    return args


class FinalPolicy:
    """One episode per instance. Call commit_action after each executed frame.

    Observations: existing encode_obs result (three RGB images and state [14]).
    Actions: absolute radians and normalized gripper openings (0 closed, 1 open).
    No action is executed here; the official evaluator owns execution/timing.
    """

    def __init__(self, client, args):
        self.client = client
        self.args = args
        self.task_name = args.task
        self.strict_action_dim = client.strict_action_dim
        self.runtime = AssistRuntime(args)
        self.last_inference_seconds = 0.0

    def next_action(self, encoded_obs, prompt=None, *, feedback=None):
        if self.runtime.pending is not None:
            raise RuntimeError(
                "Call commit_action after official execution before requesting another action"
            )
        if self.runtime.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        state = encoded_obs["observation/state"]
        # A helper can reject a stale chunk. Re-infer once from the measured obs.
        for _ in range(2):
            if self.runtime.needs_chunk:
                observation = dict(encoded_obs)
                observation["observation/state"] = self.runtime.model_state(state)
                started = time.perf_counter()
                raw = self.client.infer(
                    observation,
                    str(prompt or self.args.prompt or TASK_PROMPTS[self.task_name]),
                )
                self.last_inference_seconds = time.perf_counter() - started
                # Structural API errors are not recoverable numeric fluctuations.
                raw = np.asarray(raw, dtype=np.float32)
                if raw.ndim != 2 or raw.shape[1] != 14 or len(raw) == 0:
                    raise ValueError("Expected nonempty model actions [T,14]")
                try:
                    self.runtime.begin_chunk(raw, state)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    return self._recovery_hold(state, str(exc))
            try:
                action = self.runtime.propose(state, feedback=feedback)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                return self._recovery_hold(state, str(exc))
            if action is not None:
                return action
        return self._recovery_hold(state, "Assistance rejected two chunks")

    def _recovery_hold(self, state, reason):
        """Discard failed plans, hold one frame, then infer from fresh feedback."""
        logging.getLogger(__name__).warning("Policy recovery: %s; holding one frame", reason)
        old = self.runtime
        runtime = AssistRuntime(self.args)
        runtime.steps = old.steps
        runtime.cycle = max(0, old.cycle)
        runtime.previous = old.previous
        target = clip_actions(state[None, :])[0]
        runtime.pending = dict(
            current=state.copy(), target=target.copy(), recovery_hold=True,
            record={"recovery": reason},
        )
        self.runtime = runtime
        return target.copy()

    def commit_action(self):
        self.runtime.commit()

    def reset(self):
        self.client.reset()
        self.runtime = AssistRuntime(self.args)
        self.last_inference_seconds = 0.0

    def close(self):
        self.client.close()
