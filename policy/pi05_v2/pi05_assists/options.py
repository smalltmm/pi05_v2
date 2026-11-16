"""Task-specific policy configuration, independent of robot drivers."""

import argparse
from .handover_observation import (
    add_arguments as add_observation_arguments,
    validate_args as validate_observation_args,
)
import numpy as np
from .model_z_floor import ModelZFloor
from . import config as control_timing
from .config import (
    CONTROL_DT,
    CONTROL_HZ,
    GripperCalibration,
    TASK_PROMPTS,
    compensate_gripper_action,
    compensate_gripper_state,
)
from .grasp_dip import (
    DipPlan,
    GraspDip,
    add_arguments as add_grasp_dip_arguments,
    config_from_args,
    make_piper_fk,
)
from .grasp_dip_runtime import GraspDipRuntime
from .basket_dip import (
    BasketDipHold,
    add_arguments as add_basket_dip_arguments,
    config_from_args as basket_dip_config_from_args,
)
from .basket_release import (
    BasketReleaseAdvance,
    add_arguments as add_basket_release_arguments,
    config_from_args as basket_release_config_from_args,
)
from .table_release import (
    TableReleaseLower,
    add_arguments as add_table_release_arguments,
    config_from_args as table_release_config_from_args,
)
from .bell_press import (
    DualBellPress,
    DualBellPressRuntime,
    add_arguments as add_bell_press_arguments,
    config_from_args as bell_config_from_args,
)
from .relative_handover import (
    RelativeHandover,
    add_arguments as add_relative_arguments,
    config_from_args as relative_config_from_args,
)
from .sample_placement import (
    SamplePlacement,
    add_arguments as add_placement_arguments,
    config_from_args as placement_config_from_args,
)
from .item_assembly import (
    ItemAssemblyInsert,
    add_arguments as add_assembly_arguments,
    config_from_args as assembly_insert_config_from_args,
)
from .pipette_press import (
    PipettePress,
    add_arguments as add_pipette_press_arguments,
    config_from_args as pipette_press_config_from_args,
)
from .handover import (
    HandoverAssist,
    add_arguments as add_handover_arguments,
    config_from_args as handover_config_from_args,
)
from .grip_hold import (
    AssemblyGripConfig,
    AssemblyLeftGripHold,
    AssemblyRightGripHold,
    BasketLeftGripHold,
    BasketRightGripHold,
    BasketLeftGripConfig,
    LeftGripHold,
    RightGripHold,
    TableLeftGripHold,
    add_arguments as add_grip_hold_arguments,
    assembly_config_from_args,
    config_from_args as grip_hold_config_from_args,
    left_config_from_args as left_grip_hold_config_from_args,
)
from .timing import ActionSchedule
from .water_pouring import (
    WaterPouring,
    add_arguments as add_pour_arguments,
    config_from_args as pour_config_from_args,
)


def parse_options(tokens=()) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-action-fps",
        type=float,
        default=25.0,
        help="training action frequency; use 10 for a 10 Hz model",
    )
    p.add_argument(
        "--model-action-horizon",
        type=int,
        default=50,
        help="model prediction length at load time; use 16 for chunk16",
    )
    p.add_argument("--task", choices=tuple(TASK_PROMPTS), default="water_pouring")
    p.add_argument("--prompt", default=None)
    p.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="command rate in Hz for all tasks (default 10); actual rate depends on hardware and host throughput",
    )
    p.add_argument(
        "--schedule",
        choices=("sequential", "resample", "sample", "median"),
        default="sample",
    )
    p.add_argument(
        "--legacy-steps",
        type=int,
        default=None,
        help="model prefix to execute per inference; default one second of model actions",
    )
    p.add_argument(
        "--sample-start-step",
        type=int,
        default=None,
        help="first model row for sample; default 0 at 10 Hz, 2 at 25 Hz",
    )
    offset_group = p.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--gripper-offset-compensation",
        dest="gripper_offset_compensation",
        action="store_true",
        help="add each gripper's offset to its model input and subtract it from its model output before control",
    )
    offset_group.add_argument(
        "--no-gripper-offset-compensation",
        dest="gripper_offset_compensation",
        action="store_false",
        help="disable gripper offset compensation (the default)",
    )
    p.set_defaults(gripper_offset_compensation=False)
    p.add_argument(
        "--gripper-offset",
        type=float,
        default=0.03,
        help="shared normalized offset; per-side offsets override this value (default: 0.03)",
    )
    p.add_argument(
        "--left-gripper-offset",
        type=float,
        default=None,
        help="left gripper offset in [0, 1]; defaults to --gripper-offset",
    )
    p.add_argument(
        "--right-gripper-offset",
        type=float,
        default=None,
        help="right gripper offset in [0, 1]; defaults to --gripper-offset",
    )
    add_grasp_dip_arguments(p)
    add_bell_press_arguments(p)
    add_handover_arguments(p)
    add_grip_hold_arguments(p)
    add_basket_dip_arguments(p)
    add_basket_release_arguments(p)
    add_table_release_arguments(p)
    add_placement_arguments(p)
    add_assembly_arguments(p)
    add_pipette_press_arguments(p)
    add_relative_arguments(p)
    add_pour_arguments(p)
    add_observation_arguments(p)
    p.add_argument(
        "--model-z-floor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="table_rearrangement: clamp model and final assisted SDK endpoint Z values",
    )
    p.add_argument(
        "--model-min-z-mm",
        type=float,
        default=120.0,
        help="minimum model endpoint Z in each arm base frame (default 120 mm)",
    )
    args = p.parse_args(list(tokens))
    if not np.isfinite(args.model_min_z_mm):
        p.error("--model-min-z-mm must be finite")
    if args.model_z_floor and args.task not in (
        "table_rearrangement",
        "manipulate_pipette",
    ):
        p.error(
            "--model-z-floor requires --task table_rearrangement or manipulate_pipette"
        )
    if args.control_hz != 10:
        p.error("RoboSyn finals require control_hz=10")
    if not np.isfinite(args.model_action_fps) or args.model_action_fps <= 0:
        p.error("--model-action-fps must be finite and positive")
    if args.model_action_horizon <= 0:
        p.error("--model-action-horizon must be positive")
    if args.legacy_steps is None:
        args.legacy_steps = min(
            args.model_action_horizon, max(1, round(args.model_action_fps))
        )
    if args.sample_start_step is None:
        args.sample_start_step = 0 if args.model_action_fps <= args.control_hz else 2
    if args.legacy_steps <= 0 or args.sample_start_step < 0:
        p.error("--legacy-steps must be positive and --sample-start-step non-negative")
    for name in ("gripper_offset", "left_gripper_offset", "right_gripper_offset"):
        value = getattr(args, name)
        if value is None:
            value = args.gripper_offset
            setattr(args, name, value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            p.error(f"--{name.replace('_', '-')} must be finite and in [0, 1]")
    try:
        config_from_args(args)
        assembly_config_from_args(args)
        bell_config_from_args(args)
        handover_config_from_args(args)
        grip_hold_config_from_args(args)
        left_grip_hold_config_from_args(args)
        basket_dip_config_from_args(args)
        basket_release_config_from_args(args)
        table_release_config_from_args(args)
        placement_config_from_args(args)
        assembly_insert_config_from_args(args)
        pipette_press_config_from_args(args)
        relative_config_from_args(args)
        pour_config_from_args(args)
        validate_observation_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return args
