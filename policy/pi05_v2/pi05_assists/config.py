"""Numeric policy conventions, joint limits and gripper calibration; no device I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTROL_HZ = 10
CONTROL_DT = 0.1
MODEL_ACTION_FPS = 25.0
MODEL_ACTION_DT = 1.0 / MODEL_ACTION_FPS
MODEL_ACTION_HORIZON = 50
LEGACY_EXECUTION_STEPS = 25
STATE_DIM = 14
GRIPPER_INDICES = (6, 13)
JOINT_INDICES = tuple(i for i in range(STATE_DIM) if i not in GRIPPER_INDICES)


def _gripper_offsets(offset: float | tuple[float, float]) -> np.ndarray:
    """Accept a shared offset or a (left, right) pair in normalized units."""
    offsets = np.asarray(offset, dtype=np.float32)
    if offsets.shape not in ((), (2,)) or not np.all(np.isfinite(offsets)) or np.any((offsets < 0) | (offsets > 1)):
        raise ValueError("gripper compensation offsets must be finite values in [0, 1], shared or (left, right)")
    return offsets


def compensate_gripper_state(state: np.ndarray, offset: float | tuple[float, float], *, enabled: bool) -> np.ndarray:
    """Add each gripper's offset to model input without changing physical feedback."""
    values = np.asarray(state, dtype=np.float32).copy().reshape(STATE_DIM)
    if enabled:
        values[list(GRIPPER_INDICES)] = np.clip(values[list(GRIPPER_INDICES)] + _gripper_offsets(offset), 0.0, 1.0)
    return values


def compensate_gripper_action(actions: np.ndarray, offset: float | tuple[float, float], *, enabled: bool) -> np.ndarray:
    """Apply an optional negative normalized offset to model gripper actions."""
    values = np.asarray(actions, dtype=np.float32).copy()
    if values.ndim < 1 or values.shape[-1] != STATE_DIM:
        raise ValueError(f"actions must end with dimension {STATE_DIM}, got {values.shape}")
    if enabled:
        values[..., list(GRIPPER_INDICES)] = np.clip(values[..., list(GRIPPER_INDICES)] - _gripper_offsets(offset), 0.0, 1.0)
    return values

TASK_PROMPTS = {
    "click_bell": "Click the bell",
    "drawer_open_place": "Open the drawer, pick the tomato with the other arm then place it in the drawer.",
    "handle_basket": "Pick the bottle then place it to the basket, carry the basket with the other arm.",
    "item_assembly": "Pick up two centrifuge tubes from the table and dock them horizontally.",
    "items_handover": "Pick up the pen, hand it over to the other arm and then place it in to the pen holder",
    "manipulate_pipette": "Use the pipette on the table to transfer liquid from one beaker to another",
    "mixer_operating": "Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm.",
    "sample_loading": "Pick up the test tube, and it to the other arm, and insert it to the rack.",
    "table_rearrangement": "Pick the fork and the spoon, place them next to the plate.",
    "water_pouring": "Pick the cup and the bottle with the other arm, pour the water from bottle to cup",
}

# Piper SDK JointCtrl limits, in radians. These are used as a conservative
# host-side check before a command is sent to the arm.
PIPER_JOINT_MIN = np.asarray((-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439), dtype=np.float32)
PIPER_JOINT_MAX = np.asarray((2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439), dtype=np.float32)


@dataclass(frozen=True)
class GripperCalibration:
    """Map model gripper values to Piper's 0.001 mm command units.

    ``normalized`` follows the common OpenPI convention (0=closed, 1=open).
    ``piper_mm`` treats model values as millimetres. The real-robot command
    path requires this choice to be explicit because the checkpoint metadata
    does not encode a hardware-unit conversion.
    """

    mode: str = "normalized"
    closed_mm: float = 0.0
    # The two connected Piper grippers report a usable stroke of about 100 mm
    # (the right side was observed at 98.7 mm after a hardware restart).
    open_mm: float = 100.0

    def __post_init__(self) -> None:
        if self.mode not in {"normalized", "piper_mm"}:
            raise ValueError("gripper mode must be 'normalized' or 'piper_mm'")
        if not np.isfinite(self.closed_mm) or not np.isfinite(self.open_mm):
            raise ValueError("gripper calibration values must be finite")
        if self.open_mm <= self.closed_mm:
            raise ValueError("open_mm must be greater than closed_mm")

    def model_to_mm(self, value: float) -> float:
        value = float(value)
        if self.mode == "normalized":
            # Saturate normalized openings, including observation overshoot.
            if not np.isfinite(value):
                raise ValueError("normalized gripper value must be finite")
            value = float(np.clip(value, 0.0, 1.0))
            # Preserve the continuous normalized interval. A model value of
            # 0 means fully closed and 1 means fully open; intermediate
            # values map linearly to the calibrated Piper stroke.
            return self.closed_mm + value * (self.open_mm - self.closed_mm)
        if not self.closed_mm <= value <= self.open_mm:
            raise ValueError(f"piper_mm gripper value is outside [{self.closed_mm}, {self.open_mm}]: {value}")
        return value

    def mm_to_model(self, value_mm: float) -> float:
        value_mm = float(value_mm)
        # Piper feedback can overshoot the configured endpoint by a few
        # tenths of a millimetre due to encoder quantisation/backlash.
        endpoint_tolerance = 10.0
        # Piper feedback can exceed the nominal command range on some
        # gripper firmware (the SDK accepts feedback up to 150 mm). Treat a
        # Positive overshoot is saturated-open feedback. Small negative
        # readings are treated as closed-end encoder noise while clearly
        # invalid readings are still rejected.
        if value_mm < self.closed_mm - endpoint_tolerance or value_mm > 150.0:
            raise ValueError(f"Piper gripper feedback is outside calibration range: {value_mm} mm")
        value_mm = float(np.clip(value_mm, self.closed_mm, self.open_mm))
        if self.mode == "piper_mm":
            return value_mm
        return (value_mm - self.closed_mm) / (self.open_mm - self.closed_mm)
