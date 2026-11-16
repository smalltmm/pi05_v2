"""Action-chunk timing and safe absolute-trajectory resampling."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .config import CONTROL_HZ
from .config import GRIPPER_INDICES
from .config import JOINT_INDICES
from .config import LEGACY_EXECUTION_STEPS
from .config import MODEL_ACTION_FPS

LOGGER = logging.getLogger(__name__)


def _interp_zero_order(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    indexes = np.searchsorted(source_times, target_times, side="right") - 1
    indexes = np.clip(indexes, 0, len(source_times) - 1)
    return values[indexes]


def resample_absolute_trajectory(
    actions: np.ndarray,
    *,
    source_fps: float = MODEL_ACTION_FPS,
    target_hz: float = CONTROL_HZ,
    target_steps: int | None = None,
) -> np.ndarray:
    """Interpolate an absolute [time, 14] trajectory to the control clock.

    Joint positions are linearly interpolated. Piper grippers are absolute
    position values but are held with zero-order hold, matching the training
    configuration's ``action_zoh_indices=(6, 13)``. This function must not be
    called on raw delta actions; convert deltas to absolute positions first.
    """

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"expected absolute actions with shape [T, 14], got {values.shape}")
    if len(values) == 0 or source_fps <= 0 or target_hz <= 0:
        raise ValueError("trajectory and frequencies must be positive")
    if target_steps is None:
        target_steps = max(1, int(np.ceil(len(values) / float(source_fps) * target_hz)))
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")

    source_times = np.arange(len(values), dtype=np.float64) / float(source_fps)
    target_times = np.arange(target_steps, dtype=np.float64) / float(target_hz)
    out = np.empty((target_steps, values.shape[1]), dtype=np.float32)
    for index in JOINT_INDICES:
        out[:, index] = np.interp(target_times, source_times, values[:, index]).astype(np.float32)
    out[:, list(GRIPPER_INDICES)] = _interp_zero_order(values[:, list(GRIPPER_INDICES)], source_times, target_times)
    return out


def delta_to_absolute(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert OpenPI-style joint deltas to absolute positions.

    OpenPI's ``DeltaActions`` subtracts the current state from every future
    joint target, so conversion is ``state + delta`` per row, not cumulative
    integration. Gripper dimensions remain absolute.
    """

    values = np.asarray(actions, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 14 or current.shape != (14,):
        raise ValueError(f"expected actions [T,14] and state [14], got {values.shape} and {current.shape}")
    out = values.copy()
    out[:, list(JOINT_INDICES)] += current[list(JOINT_INDICES)]
    return out


def absolute_to_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert absolute positions to the model's joint-delta convention."""

    values = np.asarray(actions, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 14 or current.shape != (14,):
        raise ValueError(f"expected actions [T,14] and state [14], got {values.shape} and {current.shape}")
    out = values.copy()
    out[:, list(JOINT_INDICES)] -= current[list(JOINT_INDICES)]
    return out


def median_downsample_trajectory(actions: np.ndarray, *, source_fps: float = MODEL_ACTION_FPS,
                                 target_hz: float = CONTROL_HZ,
                                 source_steps: int = LEGACY_EXECUTION_STEPS,
                                 target_steps: int = CONTROL_HZ) -> np.ndarray:
    """Median-aggregate the first one-second model segment to control rate."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) < source_steps:
        raise ValueError(f"expected at least [{source_steps}, 14] absolute actions, got {values.shape}")
    source = values[:source_steps]
    times = np.arange(source_steps, dtype=np.float64) / float(source_fps)
    edges = np.arange(target_steps + 1, dtype=np.float64) / float(target_hz)
    out = np.empty((target_steps, 14), dtype=np.float32)
    for index in range(target_steps):
        rows = source[(times >= edges[index]) & (times < edges[index + 1])]
        if len(rows) == 0:
            nearest = min(int(round((edges[index] + edges[index + 1]) * 0.5 * source_fps)), source_steps - 1)
            rows = source[nearest : nearest + 1]
        out[index] = np.median(rows, axis=0).astype(np.float32)
    return out


def sample_absolute_trajectory(
    actions: np.ndarray,
    *,
    source_fps: float = MODEL_ACTION_FPS,
    target_hz: float = CONTROL_HZ,
    source_steps: int | None = None,
    start_model_step: int = 2,
) -> np.ndarray:
    """Take nearest model rows on the target control clock, without averaging.

    For a 25 Hz model and a 10 Hz Piper loop this selects approximately every
    2-3 model rows. The default offset starts at model row 2, so a 25-row
    prefix produces 10 control rows and ends at model row 24. The output
    duration follows the selected source prefix: 10 model rows become 4
    control rows, 25 become 10, and 50 become 20. This is valid for absolute
    actions; delta actions must be converted to absolute positions first.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"expected absolute actions with shape [T, 14], got {values.shape}")
    if len(values) == 0 or source_fps <= 0 or target_hz <= 0:
        raise ValueError("trajectory and frequencies must be positive")
    if source_steps is None:
        source_steps = len(values)
    if source_steps <= 0 or source_steps > len(values):
        raise ValueError(f"source_steps must be in [1, {len(values)}], got {source_steps}")
    if start_model_step < 0 or start_model_step >= source_steps:
        raise ValueError(f"start_model_step must be in [0, {source_steps - 1}], got {start_model_step}")
    source = values[:source_steps]
    target_steps = max(1, int(np.ceil(source_steps * target_hz / source_fps)))
    target_times = (float(start_model_step) / float(source_fps)) + np.arange(target_steps, dtype=np.float64) / float(target_hz)
    indexes = np.rint(target_times * float(source_fps)).astype(np.int64)
    indexes = np.clip(indexes, 0, source_steps - 1)
    return source[indexes].copy()


@dataclass(frozen=True)
class ActionSchedule:
    """Describe how a model chunk is presented to the 10 Hz Piper loop."""

    mode: str = "sequential"
    legacy_steps: int = LEGACY_EXECUTION_STEPS
    source_fps: float = MODEL_ACTION_FPS
    control_hz: float = CONTROL_HZ
    sample_start_step: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"sequential", "resample", "sample", "median"}:
            raise ValueError("action schedule mode must be 'sequential', 'resample', 'sample' or 'median'")
        if (self.legacy_steps <= 0 or not np.isfinite([self.source_fps, self.control_hz]).all()
                or self.source_fps <= 0 or self.control_hz <= 0):
            raise ValueError("schedule steps and frequencies must be positive")
        if self.sample_start_step < 0:
            raise ValueError("sample_start_step must be non-negative")

    def make_plan(self, actions: np.ndarray, *, state: np.ndarray, action_semantics: str = "absolute") -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 14:
            raise ValueError(f"model action chunk must have shape [T, 14], got {values.shape}")
        if len(values) == 0:
            raise ValueError("model returned an empty action chunk")
        if action_semantics == "delta":
            values = delta_to_absolute(values, state)
        elif action_semantics != "absolute":
            raise ValueError("action_semantics must be 'absolute' or 'delta'")

        if self.mode == "sequential":
            count = min(self.legacy_steps, len(values))
            plan = values[:count].copy()
        elif self.mode == "sample":
            plan = sample_absolute_trajectory(
                values,
                source_fps=self.source_fps,
                target_hz=self.control_hz,
                source_steps=min(self.legacy_steps, len(values)),
                start_model_step=min(self.sample_start_step, min(self.legacy_steps, len(values)) - 1),
            )
        elif self.mode == "median":
            plan = median_downsample_trajectory(
                values,
                source_fps=self.source_fps,
                target_hz=self.control_hz,
                source_steps=min(self.legacy_steps, len(values)),
                target_steps=int(round(self.control_hz)),
            )
        else:
            plan = resample_absolute_trajectory(
                values,
                source_fps=self.source_fps,
                target_hz=self.control_hz,
            )

        LOGGER.info(
            "action schedule: model=%d steps @ %.2f Hz (%.3f s rows), control=%d steps @ %.2f Hz (%.3f s rows), mode=%s",
            len(values),
            self.source_fps,
            1.0 / self.source_fps,
            len(plan),
            self.control_hz,
            1.0 / self.control_hz,
            self.mode,
        )
        if self.mode == "sequential":
            LOGGER.warning(
                "sequential mode keeps the first %d model rows but sends them at %.1f Hz: %.3f s model span vs %.3f s real span",
                len(plan),
                self.control_hz,
                len(plan) / self.source_fps,
                len(plan) / self.control_hz,
            )
        elif self.mode == "sample":
            LOGGER.info(
                "nearest-row sampling uses model rows %d..%d (%.3f s prefix) as %d control rows (%.3f s); no interpolation or median",
                min(self.sample_start_step, min(self.legacy_steps, len(values)) - 1),
                min(self.legacy_steps, len(values)) - 1,
                min(self.legacy_steps, len(values)) / self.source_fps,
                len(plan),
                len(plan) / self.control_hz,
            )
        elif self.mode == "median":
            LOGGER.info(
                "median aggregation converts the first %d model rows (%.3f s) to %d control rows (%.3f s)",
                min(self.legacy_steps, len(values)),
                min(self.legacy_steps, len(values)) / self.source_fps,
                len(plan),
                len(plan) / self.control_hz,
            )
        return plan
