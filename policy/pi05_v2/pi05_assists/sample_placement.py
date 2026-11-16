"""Feedback-gated sample-loading placement after a confirmed handover.

Uses the standalone fingertip IK/path planner, but the monitor owns all CAN
commands, timing, recording and interrupts. No second robot connection is made.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .config import GripperCalibration
from .kinematics import SAMPLE_LOADING_RELEASE_RPY_DEG
from .kinematics import fingertip_xyz
from .kinematics import plan_tip_line
from .kinematics import solve_ik


@dataclass(frozen=True)
class PlacementConfig:
    x_mm: float = 432.0
    y_mm: float = -200.0
    approach_z_mm: float = 220.0
    release_z_mm: float = 100.0
    opening_mm: float = 100.0
    release_delay_s: float = 0.1
    position_tolerance_mm: float = 10.0
    orientation_tolerance_deg: float = 10.0

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("sample placement parameters must be finite")
        if self.release_delay_s < 0:
            raise ValueError("sample placement release delay must be nonnegative")
        if self.release_z_mm >= self.approach_z_mm:
            raise ValueError("sample placement release Z must be below approach Z")
        if not 0 <= self.opening_mm <= GripperCalibration().open_mm:
            raise ValueError("sample placement opening must be in [0, 100] mm")
        if self.position_tolerance_mm <= 0 or not 0 < self.orientation_tolerance_deg < 180:
            raise ValueError("sample placement tolerances must be positive; angle must be below 180 degrees")


class SamplePlacement:
    def __init__(self, config, fk, *, clock=None):
        self.clock = clock or (lambda: time.monotonic())
        self.config, self.fk = config, fk
        self.phase = "idle"
        self.hold = None
        self.paths = []
        self.goals = []
        self.index = 0
        self.reason = None
        self.release_frames = 0
        self.arrival_frames = 0
        self.planning_seconds = None
        self.dwell_started = None

    @property
    def active(self):
        return self.phase != "idle"

    def start(self, physical, previous):
        """Latch both arms and plan BOTH legs before issuing any placement motion."""
        if self.active:
            return
        self.hold = np.asarray(physical, dtype=np.float32).copy()
        # Preserve the last actual grip command, not the larger contact opening.
        self.hold[6] = min(float(previous[6]), 0.25)
        self.hold[13] = previous[13]
        self.phase = "planning_failed"
        planning_started = time.perf_counter()
        try:
            paths, goals = [], []
            joints = self.hold[:6].copy()
            for z in (self.config.approach_z_mm, self.config.release_z_mm):
                xyz = np.array([self.config.x_mm, self.config.y_mm, z])
                goal, _, _ = solve_ik(
                    self.fk,
                    joints,
                    xyz,
                    SAMPLE_LOADING_RELEASE_RPY_DEG,
                    position_tolerance_mm=self.config.position_tolerance_mm,
                    orientation_tolerance_deg=self.config.orientation_tolerance_deg,
                    prefer_local_solution=True,
                )
                paths.append(
                    plan_tip_line(self.fk, joints, goal, xyz, position_tolerance_mm=self.config.position_tolerance_mm)
                )
                goals.append((goal, xyz))
                joints = goal
            self.paths, self.goals = paths, goals
            self.phase = "approach"
        except (ValueError, RuntimeError) as exc:
            self.reason = str(exc)
        finally:
            self.planning_seconds = time.perf_counter() - planning_started

    def arrival(self, physical):
        leg = 0 if self.phase == "approach" else 1
        goal, xyz = self.goals[leg]
        pose = self.fk.pose(physical[:6])
        error = float(np.linalg.norm(fingertip_xyz(pose) - xyz))
        angle = float(
            np.rad2deg(
                (
                    Rotation.from_euler("xyz", SAMPLE_LOADING_RELEASE_RPY_DEG, degrees=True).inv()
                    * Rotation.from_euler("xyz", pose[3:], degrees=True)
                ).magnitude()
            )
        )
        return (
            (
                np.max(np.abs(physical[:6] - goal)) <= 0.02
                and error <= self.config.position_tolerance_mm
                and angle <= self.config.orientation_tolerance_deg
            ),
            error,
            angle,
        )

    def next(self, physical):
        action = self.hold.copy()
        record = {
            "planning_seconds": self.planning_seconds,
            "phase": self.phase,
            "path_index": self.index,
            "reason": self.reason,
            "fingertip_xyz_mm": fingertip_xyz(self.fk.pose(physical[:6])).tolist(),
            "gripper_feedback_mm": GripperCalibration().model_to_mm(float(physical[6])),
        }
        if self.phase in ("approach", "descend"):
            leg = 0 if self.phase == "approach" else 1
            path = self.paths[leg]
            action[:6] = path[min(self.index, len(path) - 1)]
            reached, error, angle = self.arrival(physical)
            record.update(arrived=bool(reached), position_error_mm=error, orientation_error_deg=angle)
        elif self.phase == "dwell":
            action[:6] = self.goals[1][0]
            arrived, error, angle = self.arrival(physical)
            elapsed = 0.0 if self.dwell_started is None else self.clock() - self.dwell_started
            record.update(
                arrived=bool(arrived),
                position_error_mm=error,
                orientation_error_deg=angle,
                release_delay_s=self.config.release_delay_s,
                dwell_elapsed_s=elapsed,
            )
            if arrived and self.dwell_started is not None and elapsed >= self.config.release_delay_s:
                action[6] = GripperCalibration().mm_to_model(self.config.opening_mm)
                record["phase"] = "release"
        elif self.phase in ("release", "done"):
            action[:6] = self.goals[1][0]
            action[6] = GripperCalibration().mm_to_model(self.config.opening_mm)
        record["gripper_command_mm"] = GripperCalibration().model_to_mm(float(action[6]))
        return action, record

    def commit(self, physical, sent):
        """Advance only after the monitor successfully sends and records this step."""
        if self.phase in ("approach", "descend"):
            leg = 0 if self.phase == "approach" else 1
            path = self.paths[leg]
            waypoint = path[min(self.index, len(path) - 1)]
            # Do not skip a waypoint when speed clipping or hardware lag occurs.
            if np.max(np.abs(physical[:6] - waypoint)) <= 0.02 and np.allclose(sent[:6], waypoint, atol=1e-5):
                self.index += 1
            reached = self.index >= len(path) and self.arrival(physical)[0]
            self.arrival_frames = self.arrival_frames + 1 if reached else 0
            if self.arrival_frames >= 3:
                self.phase = "descend" if leg == 0 else "dwell"
                if leg == 1:
                    self.dwell_started = self.clock()
                self.index = self.arrival_frames = 0
        elif self.phase == "dwell":
            if not self.arrival(physical)[0]:
                self.dwell_started = None
            elif self.dwell_started is None:
                self.dwell_started = self.clock()
            elif self.clock() - self.dwell_started >= self.config.release_delay_s and np.isclose(
                sent[6], GripperCalibration().mm_to_model(self.config.opening_mm)
            ):
                self.phase = "release"
                self.release_frames = 1
        elif self.phase == "release":
            self.release_frames += 1
            opening = GripperCalibration().model_to_mm(float(physical[6]))
            if self.release_frames >= 10 and abs(opening - self.config.opening_mm) <= 5.0:
                self.phase = "done"


def add_arguments(parser):
    parser.add_argument(
        "--sample-placement",
        action="store_true",
        help="sample_loading only: after handover, place the left fingertip using Cartesian IK",
    )
    for name, value in vars(PlacementConfig()).items():
        parser.add_argument("--sample-placement-" + name.replace("_", "-"), type=float, default=value)


def config_from_args(args):
    if not args.sample_placement:
        return None
    if args.task != "sample_loading" or not args.handover_assist:
        raise ValueError("--sample-placement requires --task sample_loading --handover-assist")
    if args.bell_press:
        raise ValueError("--sample-placement cannot be combined with --bell-press")
    return PlacementConfig(**{name: getattr(args, "sample_placement_" + name) for name in vars(PlacementConfig())})
