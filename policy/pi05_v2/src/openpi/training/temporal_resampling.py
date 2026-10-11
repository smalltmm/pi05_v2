"""Timestamp-aware action resampling for LeRobot datasets.

The LeRobot ``delta_timestamps`` API deliberately only supports offsets that
are integer multiples of the source frame period.  This module keeps the
observation lookup native and resamples only the action trajectory in physical
time, before OpenPI's robot-specific transforms and normalization.
"""

from collections.abc import Sequence
import re
from typing import Any

import numpy as np


def make_target_times(action_horizon: int, target_fps: float) -> np.ndarray:
    """Return ``[0, 1/target_fps, ..., (horizon-1)/target_fps]``."""
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    return np.arange(action_horizon, dtype=np.float64) / float(target_fps)


def _flatten_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    if isinstance(names, Sequence):
        result: list[str] = []
        for name in names:
            result.extend(_flatten_names(name))
        return result
    return [str(names)]


def _contiguous_groups(indices: Sequence[int], size: int) -> list[tuple[int, ...]]:
    groups = []
    indices = sorted(indices)
    for start in range(0, len(indices), size):
        group = tuple(indices[start : start + size])
        if len(group) == size and group == tuple(range(group[0], group[0] + size)):
            groups.append(group)
    return groups


def _rotation_groups(names: Sequence[str]) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Find contiguous quaternion and rotation-6D fields from feature names."""
    lower = [name.lower() for name in names]
    quat_indices = [i for i, name in enumerate(lower) if "quaternion" in name or "quat" in name]
    rot6d_indices = [i for i, name in enumerate(lower) if "rot6d" in name or "rotation6d" in name]
    return _contiguous_groups(quat_indices, 4), _contiguous_groups(rot6d_indices, 6)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(q0.astype(np.float64, copy=False))
    q1 = _normalize_quaternion(q1.astype(np.float64, copy=False))
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = _normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    # The linear branch is both more stable near zero and the exact limit of
    # SLERP there.
    safe = sin_theta > 1e-7
    w0 = np.sin((1.0 - alpha) * theta) / np.where(safe, sin_theta, 1.0)
    w1 = np.sin(alpha * theta) / np.where(safe, sin_theta, 1.0)
    spherical = _normalize_quaternion(w0 * q0 + w1 * q1)
    return np.where(safe, spherical, linear)


def _rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    # The convention used by the common rotation_6d_to_matrix implementation:
    # the first two 3-vectors are the first two columns of the matrix.
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to scalar-first quaternions."""
    # This branch-free formulation is stable enough for action trajectories and
    # avoids introducing a scipy dependency into the data loader.
    m = matrix
    qw = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2])) / 2.0
    qx = np.sign(m[..., 2, 1] - m[..., 1, 2]) * np.sqrt(
        np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])
    ) / 2.0
    qy = np.sign(m[..., 0, 2] - m[..., 2, 0]) * np.sqrt(
        np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])
    ) / 2.0
    qz = np.sign(m[..., 1, 0] - m[..., 0, 1]) * np.sqrt(
        np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])
    ) / 2.0
    return _normalize_quaternion(np.stack((qw, qx, qy, qz), axis=-1))


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _slerp_rot6d(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _matrix_to_quaternion(_rot6d_to_matrix(x0))
    q1 = _matrix_to_quaternion(_rot6d_to_matrix(x1))
    return _matrix_to_rot6d(_quaternion_to_matrix(_slerp_quaternion(q0, q1, alpha)))


def infer_interpolation_groups(
    feature_names: Sequence[str] | None,
    *,
    zoh_indices: Sequence[int] = (),
) -> tuple[set[int], list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Infer gripper (ZOH), quaternion, and rot6d groups from LeRobot names."""
    names = list(feature_names or ())
    zoh = set(zoh_indices)
    zoh.update(i for i, name in enumerate(names) if re.search(r"gripper|grip", name, re.IGNORECASE))
    # Some ALOHA/CobotMagic conversions label the two grippers as the seventh
    # joint of each arm. OpenPI's Aloha delta-action mask confirms that indices
    # 6 and 13 are grippers for this 14-D layout.
    if len(names) == 14:
        for index in (6, 13):
            if re.search(r"joint[_ -]?7$", names[index], re.IGNORECASE):
                zoh.add(index)
    quaternions, rot6d = _rotation_groups(names)
    return zoh, quaternions, rot6d


def resample_trajectory(
    values: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    zoh_indices: Sequence[int] = (),
) -> np.ndarray:
    """Interpolate a vector trajectory at arbitrary timestamps.

    Continuous dimensions use linear interpolation. Gripper dimensions use
    zero-order hold. Quaternion and rotation-6D groups use SLERP (rotation-6D
    is converted through a rotation matrix and quaternion), never componentwise
    interpolation.
    """
    values = np.asarray(values)
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"values must be [num_frames, action_dim], got {values.shape}")
    if values.shape[0] != source_timestamps.size:
        raise ValueError("values and source_timestamps must have matching frame counts")
    if source_timestamps.size == 0:
        raise ValueError("source_timestamps cannot be empty")
    if np.any(np.diff(source_timestamps) < 0):
        raise ValueError("source_timestamps must be sorted")

    zoh, quaternion_groups, rot6d_groups = infer_interpolation_groups(feature_names, zoh_indices=zoh_indices)
    out = np.empty((target_timestamps.size, values.shape[1]), dtype=np.result_type(values.dtype, np.float32))
    for target_i, timestamp in enumerate(target_timestamps):
        right = int(np.searchsorted(source_timestamps, timestamp, side="left"))
        # LeRobot timestamps are stored as float32. At long episode times,
        # rounding can exceed 1e-7 and make an exact grid hit hold the previous
        # gripper sample. Snap to either neighboring timestamp within its
        # float32 precision, while retaining genuine between-frame targets.
        tolerance = max(1e-7, 2 * np.finfo(np.float32).eps * max(1.0, abs(timestamp)))
        neighbors = [i for i in (right - 1, right) if 0 <= i < source_timestamps.size]
        nearest = min(neighbors, key=lambda i: abs(source_timestamps[i] - timestamp))
        if abs(source_timestamps[nearest] - timestamp) <= tolerance:
            left = right = nearest
            alpha = 0.0
        elif right <= 0:
            left = right = 0
            alpha = 0.0
        elif right >= source_timestamps.size:
            left = right = source_timestamps.size - 1
            alpha = 0.0
        else:
            left = right - 1
            denominator = source_timestamps[right] - source_timestamps[left]
            alpha = 0.0 if denominator <= 0 else float((timestamp - source_timestamps[left]) / denominator)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        out[target_i] = (1.0 - alpha) * values[left] + alpha * values[right]
        if left == right:
            out[target_i] = values[left]
        if zoh:
            out[target_i, sorted(zoh)] = values[left, sorted(zoh)]
        if left == right:
            continue
        for group in quaternion_groups:
            out[target_i, list(group)] = _slerp_quaternion(values[left, list(group)], values[right, list(group)], alpha)
        for group in rot6d_groups:
            out[target_i, list(group)] = _slerp_rot6d(values[left, list(group)], values[right, list(group)], alpha)
    return out


class TemporalResampledDataset:
    """Wrap a native LeRobotDataset and resample only future actions.

    The wrapped dataset is constructed without ``delta_timestamps``. Its
    ``__getitem__`` therefore decodes exactly one native observation frame.
    Action rows are fetched from the parquet table only for a small bracket
    around the requested target horizon; no video frames are queried for them.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        action_horizon: int,
        target_action_fps: float,
        action_keys: Sequence[str] = ("action",),
        zoh_indices: Sequence[int] = (),
        observation_dataset: Any | None = None,
        emit_padding_mask: bool = False,
    ):
        if target_action_fps <= 0:
            raise ValueError(f"target_action_fps must be positive, got {target_action_fps}")
        if getattr(dataset, "delta_timestamps", None) is not None:
            raise ValueError("TemporalResampledDataset requires a LeRobotDataset with delta_timestamps=None")
        self._dataset = dataset
        self._observation_dataset = dataset if observation_dataset is None else observation_dataset
        self.action_horizon = action_horizon
        self.target_action_fps = float(target_action_fps)
        self.action_keys = tuple(action_keys)
        self.zoh_indices = tuple(zoh_indices)
        self.emit_padding_mask = emit_padding_mask
        self.meta = dataset.meta
        self.source_fps = float(dataset.meta.fps)
        self.source_dataset = getattr(dataset, "repo_id", None)
        self._target_times = make_target_times(action_horizon, target_action_fps)
        episode_ids = getattr(dataset, "episodes", None)
        if episode_ids is None:
            meta_episodes = getattr(self.meta, "episodes", None)
            episode_ids = meta_episodes.keys() if meta_episodes is not None else range(
                len(dataset.episode_data_index["from"])
            )
        self._episode_positions = {int(episode_id): position for position, episode_id in enumerate(episode_ids)}

    def __len__(self) -> int:
        return len(self._observation_dataset)

    @staticmethod
    def _scalar(value: Any) -> float:
        value = np.asarray(value)
        return float(value.reshape(-1)[0])

    def _feature_names(self, key: str) -> list[str]:
        feature = self.meta.features.get(key, {})
        names = _flatten_names(feature.get("names"))
        indices = getattr(self._dataset, "feature_name_indices", None)
        if indices is not None and names:
            names = [names[int(index)] for index in indices]
        return names

    def _read_bracket(self, indices: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        selected = self._dataset.hf_dataset.select(indices.tolist())
        timestamps = np.asarray([self._scalar(x) for x in selected["timestamp"]], dtype=np.float64)
        columns = {}
        for key in self.action_keys:
            columns[key] = np.stack([np.asarray(x) for x in selected[key]], axis=0)
            adapter = getattr(self._dataset, "action_adapter", None)
            if adapter is not None:
                columns[key] = adapter(columns[key])
        return timestamps, columns

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if index < 0:
            index += len(self)
        item = dict(self._observation_dataset[index])
        episode_index = int(self._scalar(item["episode_index"]))
        episode_position = self._episode_positions.get(episode_index, episode_index)
        episode_start = int(self._dataset.episode_data_index["from"][episode_position])
        episode_end = int(self._dataset.episode_data_index["to"][episode_position])
        local_index = index - episode_start

        # The canonical-frequency path must preserve native values exactly:
        # float32 timestamps such as 0.0399999991 should not turn a direct
        # lookup into a numerically different interpolation.
        if np.isclose(self.source_fps, self.target_action_fps, rtol=0.0, atol=1e-8):
            native_offsets = local_index + np.arange(self.action_horizon, dtype=np.int64)
            pad = native_offsets >= (episode_end - episode_start)
            native_indices = np.minimum(episode_start + native_offsets, episode_end - 1)
            _, columns = self._read_bracket(native_indices)
            for key in self.action_keys:
                item[key] = columns[key]
                if self.emit_padding_mask:
                    item[f"{key}_is_pad"] = pad.astype(bool)
            return item

        # LeRobot validates that timestamps are synchronized to native fps. Use
        # fps only to select a compact bracket, then compute alpha from the
        # actual timestamps read from parquet.
        predicted = local_index + self._target_times * self.source_fps
        bracket_start = max(episode_start, episode_start + int(np.floor(predicted.min())) - 2)
        bracket_end = min(episode_end, episode_start + int(np.ceil(predicted.max())) + 3)
        indices = np.arange(bracket_start, max(bracket_start + 1, bracket_end), dtype=np.int64)
        source_timestamps, columns = self._read_bracket(indices)
        current_timestamp = self._scalar(item["timestamp"])
        target_timestamps = current_timestamp + self._target_times
        pad = target_timestamps >= source_timestamps[-1] + 1e-8

        for key in self.action_keys:
            item[key] = resample_trajectory(
                columns[key],
                source_timestamps,
                target_timestamps,
                feature_names=self._feature_names(key),
                zoh_indices=self.zoh_indices,
            )
            if self.emit_padding_mask:
                item[f"{key}_is_pad"] = pad.astype(bool)

        return item
