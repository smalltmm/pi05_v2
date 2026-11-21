"""Finals observation/action API, with no physical robot dependencies."""

from pathlib import Path
import json
import shlex
import time

import numpy as np

try:
    from .pi05_assists.options import parse_options
    from .pi05_assists.chunk_runtime import ChunkRuntime
    from .pi05_assists.config import TASK_PROMPTS
except ImportError:
    from pi05_assists.options import parse_options
    from pi05_assists.chunk_runtime import ChunkRuntime
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


class PolicyFailure(RuntimeError):
    """Unrecoverable episode error, forwarded by the organizer bridge's error channel."""


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
        self.runtime = ChunkRuntime(args)
        self.last_inference_seconds = 0.0
        self.failure = None

    def next_action(self, encoded_obs, prompt=None, *, feedback=None):
        if self.failure is not None:
            raise PolicyFailure(self.failure)
        if self.runtime.pending is not None:
            raise RuntimeError(
                "Call commit_action after official execution before requesting another action"
            )
        if self.runtime.steps >= 1000:
            raise StopIteration("RoboSyn episode reached 1000 executed frames")
        # Cached frames never access observations (not even joint positions).
        if not self.runtime.needs_chunk:
            return self.runtime.propose()
        try:
            state = encoded_obs["observation/state"]
            observation = dict(encoded_obs)
            observation["observation/state"] = self.runtime.model_state(state)
            started = time.perf_counter()
            raw = self.client.infer(
                observation,
                str(prompt or self.args.prompt or TASK_PROMPTS[self.task_name]),
            )
            self.last_inference_seconds = time.perf_counter() - started
            raw = np.asarray(raw, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != 14 or len(raw) == 0:
                raise ValueError("Expected nonempty model actions [T,14]")
            if not np.isfinite(raw).all():
                raise ValueError("Model actions contain nonfinite values")
            # Recoverable auxiliary failures already fall back to a valid model
            # block inside ChunkRuntime. Anything escaping here cannot continue.
            self.runtime.begin_chunk(raw, state)
            return self.runtime.propose()
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            raise PolicyFailure(self.failure) from exc

    def commit_action(self):
        self.runtime.commit()

    def reset(self):
        self.client.reset()
        self.runtime = ChunkRuntime(self.args)
        self.last_inference_seconds = 0.0
        self.failure = None

    def close(self):
        self.client.close()
