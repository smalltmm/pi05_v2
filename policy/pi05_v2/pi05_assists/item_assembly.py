"""Feedback-gated dual-arm approach and left-arm insertion for item_assembly."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import PIPER_JOINT_MAX, PIPER_JOINT_MIN
from .kinematics import fingertip_xyz, solve_ik


DEFAULT_START = Path(__file__).parent / "calibration/item_assembly_assembly_start_pose_20261106_142043_utc.json"
RIGHT_BASE_IN_LEFT_MM = np.array([0.0, -600.0, 0.0])


@dataclass(frozen=True)
class AssemblyInsertConfig:
    start_pose: Path = DEFAULT_START
    stop_tip_distance_mm: float = 213.0
    step_mm: float = 5.0
    max_insert_travel_mm: float = 200.0
    joint_step_rad: float = 0.03
    trigger_lift_mm: float = 40.0
    pickup_lift_speed_mm_s: float = 40.0
    contact_gap: float = 0.05
    stable_frames: int = 3
    arrival_tolerance_rad: float = 0.02
    position_tolerance_mm: float = 3.0
    orientation_tolerance_deg: float = 5.0
    offset_xyz_mm: tuple = (0., 0., 0.)
    offset_rpy_deg: tuple = (0., 0., 0.)
    relative_align: bool = True
    right_in_left_xyz_mm: tuple = (-229.71, -4.22, 1.29)
    right_in_left_rpy_deg: tuple = (9.06, 2.22, 2.22)

    def __post_init__(self):
        for value in (self.offset_xyz_mm, self.offset_rpy_deg,
                      self.right_in_left_xyz_mm, self.right_in_left_rpy_deg):
            if np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                raise ValueError("assembly offsets must contain three finite values")
        values = (self.stop_tip_distance_mm, self.step_mm, self.max_insert_travel_mm,
                  self.joint_step_rad, self.trigger_lift_mm, self.pickup_lift_speed_mm_s, self.arrival_tolerance_rad,
                  self.position_tolerance_mm, self.orientation_tolerance_deg)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("assembly insertion distances, steps and tolerances must be finite and positive")
        if not 100 <= self.stop_tip_distance_mm <= 400 or self.step_mm > 20:
            raise ValueError("assembly insertion requires stop distance 100..400 mm and step <= 20 mm")
        if self.joint_step_rad > 0.1 or self.max_insert_travel_mm > 300:
            raise ValueError("assembly joint step must be <= 0.1 rad and travel <= 300 mm")
        if not np.isfinite(self.contact_gap) or not 0 <= self.contact_gap < 1:
            raise ValueError("assembly contact gap must be in [0, 1)")
        if self.stable_frames < 1:
            raise ValueError("assembly stable frames must be positive")


class ItemAssemblyInsert:
    def __init__(self, config: AssemblyInsertConfig, fk, *, grip_command: float):
        self.config, self.fk, self.grip_command = config, fk, grip_command
        data = json.loads(config.start_pose.read_text())
        if data.get("task") != "item_assembly" or data.get("phase") != "assembly" or data.get("pose_role") != "start":
            raise ValueError("assembly start reference must be an item_assembly assembly start pose")
        self.start_state = np.asarray(data["state_14"], dtype=np.float32)
        if self.start_state.shape != (14,) or not np.isfinite(self.start_state).all():
            raise ValueError("assembly start reference needs finite state_14")
        for start in (0, 7):
            joints = self.start_state[start:start + 6]
            if np.any(joints < PIPER_JOINT_MIN) or np.any(joints > PIPER_JOINT_MAX):
                raise ValueError("assembly start joints exceed configured Piper limits")
        self.phase = "idle"
        self.path: list[np.ndarray] = []
        self.labels: list[str] = []
        self.index = 0
        self.close_z: list[float | None] = [None, None]
        self.stable_count = 0
        self.last_attempt_cycle = -1
        self.reason = None
        self.wait_count = 0
        self.start_distance_mm = None
        self.final_distance_mm = None
        self.last_feedback_stamps = None
        self.insert_direction = None
        self.aligned_tip = None
        self.goal_rpy = None

    @property
    def active(self):
        return self.phase not in ("idle", "failed")

    def settings(self):
        return {**asdict(self.config), "start_pose": str(self.config.start_pose),
                "right_base_in_left_mm": RIGHT_BASE_IN_LEFT_MM.tolist(),
                "grip_command": self.grip_command, "offset_frame": "right_tool",
                "insertion_axis": "taught_start_tip_line_in_left_base"}

    def _pose(self, q):
        pose = np.asarray(self.fk.pose(q), dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("assembly FK must return finite XYZ/RPY")
        return pose

    def _tips(self, state):
        left = fingertip_xyz(self._pose(state[:6]))
        right = fingertip_xyz(self._pose(state[7:13])) + RIGHT_BASE_IN_LEFT_MM
        return left, right

    def alignment_goal(self):
        left_tip, right_tip = self._tips(self.start_state)
        right_rotation = Rotation.from_euler("xyz", self._pose(self.start_state[7:13])[3:], degrees=True)
        if self.config.relative_align:
            # Given T_left_right, solve T_base_left = T_base_right * inv(T_left_right).
            relative = Rotation.from_euler("xyz", self.config.right_in_left_rpy_deg, degrees=True)
            rotation = right_rotation * relative.inv()
            tip = right_tip - rotation.apply(self.config.right_in_left_xyz_mm)
        else:
            tip, rotation = left_tip, right_rotation
        tip = tip + right_rotation.apply(self.config.offset_xyz_mm)
        correction = Rotation.from_euler("xyz", self.config.offset_rpy_deg, degrees=True)
        rotation = right_rotation * correction * right_rotation.inv() * rotation
        return tip, rotation

    def observe(self, physical, *, left_holding: bool, right_holding: bool, cycle: int,
                feedback: dict | None = None, pickup_ready: bool | None = None) -> bool:
        if self.active or self.last_attempt_cycle == cycle:
            return False
        if pickup_ready is False:
            return False
        state = np.asarray(physical, dtype=float)
        if state.shape != (14,) or not np.isfinite(state).all():
            return False
        holding = (left_holding, right_holding)
        for side, start in enumerate((0, 7)):
            if not holding[side]:
                self.close_z[side] = None
                self.stable_count = 0
                continue
            z = float(self._pose(state[start:start + 6])[2])
            self.close_z[side] = z if self.close_z[side] is None else min(self.close_z[side], z)
        if not all(holding) or any(z is None for z in self.close_z):
            return False
        if feedback is None or not all(feedback.get(side, {}).get("valid", False) for side in ("left", "right")):
            self.stable_count = 0
            return False
        stamps = tuple(feedback[side].get("stamp") for side in ("left", "right"))
        if any(stamp is None or not np.isfinite(stamp) for stamp in stamps):
            self.stable_count = 0
            return False
        if stamps == self.last_feedback_stamps:
            return False
        self.last_feedback_stamps = stamps
        lifted = pickup_ready is True or all(
            self._pose(state[start:start + 6])[2] - self.close_z[side] >= self.config.trigger_lift_mm
            for side, start in enumerate((0, 7)))
        contact = all(float(feedback[side]["opening"]) - self.grip_command >= self.config.contact_gap
                      for side in ("left", "right"))
        self.stable_count = self.stable_count + 1 if lifted and contact else 0
        return self.stable_count >= self.config.stable_frames

    def _add(self, left_q, right_q, label):
        action = self.start_state.copy()
        action[:6], action[7:13] = left_q, right_q
        action[[6, 13]] = self.grip_command
        for start in (0, 7):
            q = action[start:start + 6]
            if np.any(q < PIPER_JOINT_MIN - 1e-6) or np.any(q > PIPER_JOINT_MAX + 1e-6):
                raise RuntimeError(f"assembly {label} exceeds joint limits")
        if self.path:
            indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            if np.max(np.abs(action[indices] - self.path[-1][indices])) > 1.2 * self.config.joint_step_rad:
                raise RuntimeError(f"assembly {label} exceeds per-step joint change")
        left, right = self._tips(action)
        if min(left[2], right[2]) < 0 or np.linalg.norm(right - left) < 180:
            raise RuntimeError(f"assembly {label} violates fingertip height/separation guard")
        self.path.append(action)
        self.labels.append(label)

    def plan(self, physical, *, cycle: int) -> bool:
        self.last_attempt_cycle = cycle
        self.path, self.labels = [], []
        self.reason = None
        stage = "zero_j1"
        try:
            current = np.asarray(physical, dtype=np.float32).reshape(14)
            if not np.isfinite(current).all():
                raise ValueError("non-finite assembly start feedback")
            # Clear the inward base rotations before moving to the taught pose.
            # Only J1 moves in this leg; both grasps and J2..J6 stay fixed.
            q0 = np.r_[current[:6], current[7:13]]
            zero_j1 = q0.copy()
            zero_j1[[0, 6]] = 0.
            count = max(1, int(np.ceil(np.max(np.abs(zero_j1 - q0)) / self.config.joint_step_rad)))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                q = q0 + fraction * (zero_j1 - q0)
                self._add(q[:6], q[6:], "zero_j1")

            # Reach the captured assembly start with both grippers held shut.
            stage = "move_start"
            q0 = zero_j1
            q1 = np.r_[self.start_state[:6], self.start_state[7:13]]
            count = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / self.config.joint_step_rad)))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                q = q0 + fraction * (q1 - q0)
                self._add(q[:6], q[6:], "move_start")

            left_q = self.start_state[:6].copy()
            stage = "align_left"
            right_q = self.start_state[7:13].copy()
            left_pose = self._pose(left_q)
            left_tip, right_tip = self._tips(self.start_state)
            distance = float(np.linalg.norm(right_tip - left_tip))
            # Freeze the taught assembly axis BEFORE applying the offset so a
            # transverse correction is preserved throughout insertion.
            direction = (right_tip - left_tip) / distance
            aligned_tip, goal_rotation = self.alignment_goal()
            goal_rpy = goal_rotation.as_euler("xyz", degrees=True)
            delta = right_tip - aligned_tip
            axial = float(np.dot(delta, direction))
            lateral_sq = float(np.dot(delta - axial * direction, delta - axial * direction))
            stop_sq = self.config.stop_tip_distance_mm ** 2
            if lateral_sq >= stop_sq:
                raise RuntimeError("assembly offset line cannot reach requested stop tip distance")
            travel = axial - np.sqrt(stop_sq - lateral_sq)
            if travel <= 0 or travel > self.config.max_insert_travel_mm:
                raise RuntimeError(f"assembly insertion travel {travel:.1f} mm outside (0, max]")
            self.start_distance_mm = float(np.linalg.norm(delta))
            self.insert_direction, self.aligned_tip, self.goal_rpy = direction, aligned_tip, goal_rpy
            start_rotation = Rotation.from_euler("xyz", left_pose[3:], degrees=True)
            goal_q, _, _ = solve_ik(self.fk, left_q, aligned_tip, goal_rpy,
                                    position_tolerance_mm=self.config.position_tolerance_mm,
                                    orientation_tolerance_deg=self.config.orientation_tolerance_deg,
                                    prefer_local_solution=True)
            angle = float(np.rad2deg((start_rotation.inv() * goal_rotation).magnitude()))
            align_steps = max(1, int(np.ceil(angle / 3)),
                              int(np.ceil(np.linalg.norm(aligned_tip - left_tip) / self.config.step_mm)),
                              int(np.ceil(np.max(np.abs(goal_q - left_q)) / self.config.joint_step_rad)))
            turn = (start_rotation.inv() * goal_rotation).as_rotvec()

            def move_left(target_tip, target_rotation, label, depth=0):
                nonlocal left_q
                new_q, _, _ = solve_ik(self.fk, left_q, target_tip,
                                      target_rotation.as_euler("xyz", degrees=True),
                                      position_tolerance_mm=self.config.position_tolerance_mm,
                                      orientation_tolerance_deg=self.config.orientation_tolerance_deg,
                                      prefer_local_solution=True)
                if np.max(np.abs(new_q - left_q)) > self.config.joint_step_rad:
                    if depth >= 8:
                        raise RuntimeError(f"assembly {label} cannot connect IK joints continuously")
                    pose = self._pose(left_q)
                    actual_rotation = Rotation.from_euler("xyz", pose[3:], degrees=True)
                    midpoint_rotation = actual_rotation * Rotation.from_rotvec(
                        .5 * (actual_rotation.inv() * target_rotation).as_rotvec())
                    move_left((fingertip_xyz(pose) + target_tip) / 2, midpoint_rotation, label, depth + 1)
                    move_left(target_tip, target_rotation, label, depth + 1)
                    return
                self._add(new_q, right_q, label)
                left_q = new_q

            for fraction in np.linspace(0, 1, align_steps + 1)[1:]:
                rotation = start_rotation * Rotation.from_rotvec(fraction * turn)
                align_tip = left_tip + fraction * (aligned_tip - left_tip)
                move_left(align_tip, rotation, "align_left")

            steps = int(np.ceil(travel / self.config.step_mm))
            stage = "insert_left"
            for move in np.linspace(0, travel, steps + 1)[1:]:
                target_tip = aligned_tip + move * direction
                move_left(target_tip, goal_rotation, "insert_left")
            final_left, final_right = self._tips(self.path[-1])
            self.final_distance_mm = float(np.linalg.norm(final_right - final_left))
            if abs(self.final_distance_mm - self.config.stop_tip_distance_mm) > self.config.position_tolerance_mm:
                raise RuntimeError("assembly final tip distance exceeds tolerance")
        except (ValueError, RuntimeError) as exc:
            self.path, self.labels = [], []
            self.reason = f"{stage}: {exc}"
            self.phase = "idle"
            return False
        self.phase, self.index, self.wait_count = self.labels[0], 0, 0
        return True

    def next(self, physical):
        if not self.active:
            raise RuntimeError("assembly insertion is not active")
        action = self.path[min(self.index, len(self.path) - 1)].copy()
        left, right = self._tips(physical)
        record = {"phase": self.phase, "path_index": self.index, "path_length": len(self.path),
                  "left_tip_in_left_base_mm": left.tolist(), "right_tip_in_left_base_mm": right.tolist(),
                  "tip_distance_mm": float(np.linalg.norm(right - left)),
                  "target_tip_distance_mm": self.config.stop_tip_distance_mm,
                  "insert_direction_in_left_base": None if self.insert_direction is None else self.insert_direction.tolist(),
                  "aligned_left_tip_in_left_base_mm": None if self.aligned_tip is None else self.aligned_tip.tolist(),
                  "left_goal_rpy_deg": None if self.goal_rpy is None else self.goal_rpy.tolist(),
                  "reason": self.reason}
        return action, record

    def commit(self, physical, sent):
        if self.phase == "done":
            return
        goal = self.path[self.index]
        q_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        reached = (np.max(np.abs(np.asarray(physical)[q_idx] - goal[q_idx])) <= self.config.arrival_tolerance_rad
                   and np.allclose(np.asarray(sent)[q_idx], goal[q_idx], atol=1e-5))
        if reached:
            self.index += 1
            self.wait_count = 0
            self.phase = self.labels[self.index] if self.index < len(self.path) else "done"
        else:
            self.wait_count += 1
            if self.wait_count >= 20:
                self.phase = "failed"
                self.reason = "assembly waypoint did not settle within 20 control frames"


def add_arguments(parser):
    parser.add_argument("--assembly-insert", action=argparse.BooleanOptionalAction, default=False,
                        help="item_assembly only: move to captured start, align left to right, insert left incrementally")
    for name, value in asdict(AssemblyInsertConfig()).items():
        if isinstance(value, bool):
            parser.add_argument("--assembly-" + name.replace("_", "-"),
                                action=argparse.BooleanOptionalAction, default=value)
            continue
        if isinstance(value, tuple):
            parser.add_argument("--assembly-" + name.replace("_", "-"), type=float, nargs=3, default=value)
            continue
        parser.add_argument("--assembly-" + name.replace("_", "-"),
                            type=Path if name == "start_pose" else int if name == "stable_frames" else float,
                            default=value)


def config_from_args(args):
    if not args.assembly_insert:
        return None
    if args.task != "item_assembly" or not args.assembly_grasp_assist:
        raise ValueError("--assembly-insert requires --task item_assembly --assembly-grasp-assist")
    return AssemblyInsertConfig(**{name: getattr(args, "assembly_" + name) for name in asdict(AssemblyInsertConfig())})
