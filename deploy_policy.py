"""RoboSyn policy adapter for an isolated JAX OpenPI PI0.5 worker."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


_IMAGE_KEYS = {
    "observation/image": "cam_high",
    "observation/left_wrist_image": "cam_left_wrist",
    "observation/right_wrist_image": "cam_right_wrist",
}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 0:
        return array
    return array[0]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _scalar_bool(value: Any) -> bool:
    array = _to_numpy(value).reshape(-1)
    return bool(array.size and array[0])


def _extract_image(obs: dict, sensor_name: str) -> np.ndarray:
    image = _first_env(obs["sensor"][sensor_name]["color"])
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image for {sensor_name}, got {image.shape}.")

    # RoboSyn observations may be either CHW or HWC. OpenPI expects HWC.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]

    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def encode_obs(obs: dict) -> dict[str, np.ndarray]:
    state = _first_env(obs["robot"]["qpos"])
    state = np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))
    if state.size != 14:
        state_indices = np.asarray((0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22), dtype=np.int64)
        if state.size <= int(state_indices.max()):
            raise ValueError(
                f"Expected a 14-dim policy state or a legacy state with at least "
                f"{int(state_indices.max()) + 1} dims, got {state.shape}."
            )
        state = np.ascontiguousarray(state[state_indices])
    encoded = {"observation/state": state}
    for openpi_key, sensor_name in _IMAGE_KEYS.items():
        encoded[openpi_key] = _extract_image(obs, sensor_name)
    return encoded


def encode_action(action: np.ndarray, env, *, strict_dim: bool) -> torch.Tensor:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.size < env_action_dim:
        raise ValueError(
            f"PI0.5 returned {action_array.size} action values, but the environment "
            f"requires {env_action_dim}."
        )
    if strict_dim and action_array.size != env_action_dim:
        raise ValueError(
            f"PI0.5 returned action_dim={action_array.size}, while the environment "
            f"requires action_dim={env_action_dim}. Set strict_action_dim=false only "
            "if the extra dimensions are intentionally unused."
        )
    action_array = np.ascontiguousarray(action_array[:env_action_dim])
    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=env.unwrapped.device
    )
    return action_tensor.unsqueeze(0)


class PI05JaxWorkerClient:
    """Runs OpenPI in a dependency-isolated subprocess."""

    def __init__(self, usr_args: dict):
        policy_root = Path(__file__).resolve().parent
        self.checkpoint_path = Path(usr_args["checkpoint_path"]).expanduser().resolve()
        self.openpi_root = (
            Path(
                usr_args.get("openpi_root")
                or os.environ.get("OPENPI_ROOT")
                or policy_root
            )
            .expanduser()
            .resolve()
        )
        self.python_bin = str(
            usr_args.get("pi05_python")
            or os.environ.get("PI05_PYTHON")
            or self.openpi_root / ".venv" / "bin" / "python"
        )
        self.train_config_name = str(
            usr_args.get("train_config_name", "pi05_robosyn_robotwin_piper_jax_full")
        )
        self.pytorch_device = str(usr_args.get("pytorch_device", "cuda"))
        self.exec_steps = int(usr_args.get("pi05_exec_steps", 10))
        self.num_inference_steps = int(usr_args.get("pi05_num_inference_steps", 10))
        self.compile_mode = str(usr_args.get("pi05_compile_mode", "none"))
        self.strict_action_dim = _as_bool(usr_args.get("strict_action_dim", True))
        self.debug_actions = _as_bool(usr_args.get("pi05_debug_actions", False))
        self.task_name = str(usr_args.get("task_name", "click_bell"))
        worker_gpu = usr_args.get("pi05_cuda_visible_devices")
        if worker_gpu is None or str(worker_gpu).strip() == "":
            worker_gpu = usr_args.get("gpu_id", 0)
        self.cuda_visible_devices = str(worker_gpu)

        if self.exec_steps <= 0:
            raise ValueError("pi05_exec_steps must be positive.")
        if self.num_inference_steps <= 0:
            raise ValueError("pi05_num_inference_steps must be positive.")

        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        python_paths = [
            str(self.openpi_root / "src"),
            str(self.openpi_root / "packages" / "openpi-client" / "src"),
        ]
        worker_env["PYTHONPATH"] = os.pathsep.join(python_paths)
        worker_env["PYTHONNOUSERSITE"] = "1"
        worker_env.pop("PYTHONHOME", None)
        worker_env.setdefault("OPENPI_DATA_HOME", str(self.openpi_root / "runtime_assets" / "openpi"))
        worker_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        worker_env.setdefault("JAX_COMPILATION_CACHE_DIR", str(self.openpi_root / ".cache" / "jax"))

        worker_cmd = [
            self.python_bin,
            str(policy_root / "pi05_worker.py"),
            "--openpi-root",
            str(self.openpi_root),
            "--checkpoint-path",
            str(self.checkpoint_path),
            "--train-config-name",
            self.train_config_name,
            "--device",
            self.pytorch_device,
            "--num-inference-steps",
            str(self.num_inference_steps),
            "--compile-mode",
            self.compile_mode,
        ]
        self.proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        atexit.register(self.close)
        ready = self._read_response()
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(f"Unexpected PI0.5 worker startup response: {ready}")
        print(
            "PI0.5 JAX worker ready: "
            f"config={self.train_config_name}, checkpoint={self.checkpoint_path}, "
            f"device={self.pytorch_device}, CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}",
            flush=True,
        )

    def _read_response(self) -> dict:
        if self.proc.stdout is None:
            raise RuntimeError("PI0.5 worker stdout is unavailable.")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return_code = self.proc.poll()
                raise RuntimeError(
                    "PI0.5 worker exited before returning a response "
                    f"(return code: {return_code})."
                )
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                # Some OpenPI dependencies log to stdout during model creation.
                print(f"[PI0.5 worker] {line}", flush=True)
                continue
            if not isinstance(response, dict) or "ok" not in response:
                print(f"[PI0.5 worker] {line}", flush=True)
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Unknown PI0.5 worker error."))
            return response

    def _rpc(self, payload: dict) -> dict:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError("PI0.5 worker is not running.")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return self._read_response()

    def infer(self, obs: dict[str, np.ndarray], prompt: str) -> np.ndarray:
        fd, obs_path = tempfile.mkstemp(prefix="pi05_obs_", suffix=".npz")
        os.close(fd)
        try:
            np.savez(obs_path, **obs, prompt=np.asarray(prompt))
            response = self._rpc(
                {
                    "cmd": "infer",
                    "obs_path": obs_path,
                    "max_actions": self.exec_steps,
                }
            )
        finally:
            try:
                os.remove(obs_path)
            except FileNotFoundError:
                pass

        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected a [time, action_dim] action chunk, got {actions.shape}."
            )
        actions = actions[: self.exec_steps]
        if self.debug_actions and actions.size:
            print(
                "[PI0.5 action debug] "
                f"shape={actions.shape}, first={np.round(actions[0], 5).tolist()}, "
                f"last={np.round(actions[-1], 5).tolist()}",
                flush=True,
            )
        return actions

    def reset(self) -> None:
        self._rpc({"cmd": "reset"})

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _validate_checkpoint(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    if not (checkpoint_path / "params").is_dir():
        raise FileNotFoundError(f"Missing JAX checkpoint params directory: {checkpoint_path / 'params'}")
    norm_files = list((checkpoint_path / "assets").glob("*/norm_stats.json"))
    if not norm_files:
        raise FileNotFoundError(f"No norm_stats.json was found under {checkpoint_path / 'assets'}")

def get_model(usr_args: dict) -> PI05JaxWorkerClient:
    policy_root = Path(__file__).resolve().parent
    checkpoint = usr_args.get("checkpoint_path")
    if not checkpoint:
        raise ValueError(
            "checkpoint_path must be set in deploy_policy.yml or passed with "
            "--overrides checkpoint_path=..."
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    _validate_checkpoint(checkpoint_path)
    openpi_root = (
        Path(
            usr_args.get("openpi_root")
            or os.environ.get("OPENPI_ROOT")
            or policy_root
        )
        .expanduser()
        .resolve()
    )
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"OpenPI source directory is invalid: {openpi_root}")
    return PI05JaxWorkerClient(usr_args)


def _task_succeeded(env) -> bool:
    try:
        result = env.get_wrapper_attr("is_task_success")()
    except (AttributeError, RuntimeError):
        return False
    return _scalar_bool(result)


def eval(env, model: PI05JaxWorkerClient, obs):
    prompt = getattr(env, "_current_instruction", None)
    if not prompt:
        prompt = model.task_name.replace("_", " ")

    infer_started = time.perf_counter()
    encoded_obs = encode_obs(obs)
    actions = model.infer(encoded_obs, prompt=str(prompt))
    action_tensors = [
        encode_action(action, env, strict_dim=model.strict_action_dim)
        for action in actions
    ]
    infer_elapsed = time.perf_counter() - infer_started

    final_obs = obs
    info: dict = {}
    truncated_done = False
    for action_tensor in action_tensors:
        final_obs, _reward, terminated, truncated, info = env.step(action_tensor)
        terminated_done = _scalar_bool(terminated)
        truncated_done = _scalar_bool(truncated)
        step_success = _task_succeeded(env)
        if not isinstance(info, dict):
            info = {}
        info["pi05_step_success"] = step_success
        stop_signal = terminated_done or truncated_done or step_success
        if stop_signal:
            break

    # RoboSynChallenge's evaluation loop expects four return values:
    # observation, info, truncated, and per-inference timing samples.
    # Success stops the chunk, but must not be reported as a time-limit truncation.
    return final_obs, info, truncated_done, [infer_elapsed]


def reset_model(model: PI05JaxWorkerClient) -> None:
    model.reset()


def close_model(model: PI05JaxWorkerClient) -> None:
    model.close()
