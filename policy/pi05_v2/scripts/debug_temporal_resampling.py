"""Print a native-vs-target action chunk for one LeRobot sample."""

import argparse
from pathlib import Path
import sys

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.training.temporal_resampling import TemporalResampledDataset, make_target_times  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--action-key", default="action")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=None,
        download_videos=False,
    )
    wrapped = TemporalResampledDataset(
        dataset,
        action_horizon=args.action_horizon,
        target_action_fps=args.target_fps,
        action_keys=(args.action_key,),
    )
    item = wrapped[args.index]
    row = dataset.hf_dataset[args.index]
    episode_index = int(np.asarray(row["episode_index"]).reshape(-1)[0])
    episode_end = int(dataset.episode_data_index["to"][episode_index])
    native_count = int(np.ceil((args.action_horizon - 1) / args.target_fps * dataset.meta.fps)) + 1
    native_indices = list(range(args.index, min(args.index + native_count, episode_end)))
    native_rows = dataset.hf_dataset.select(native_indices)
    native_timestamps = np.asarray(
        [float(np.asarray(value).reshape(-1)[0]) for value in native_rows["timestamp"]]
    )
    native_actions = np.stack([np.asarray(value) for value in native_rows[args.action_key]])
    print(f"dataset: {args.repo_id}")
    print(f"native fps: {dataset.meta.fps}")
    print(f"target fps: {args.target_fps}")
    print(f"episode: {episode_index}")
    print(f"current frame index: {args.index}")
    print(f"current timestamp: {float(np.asarray(row['timestamp']).reshape(-1)[0]):.9f}")
    print("native timestamps:", np.array2string(native_timestamps, precision=6))
    print("target timestamps:", np.array2string(make_target_times(args.action_horizon, args.target_fps), precision=4))
    print("native actions shape:", native_actions.shape)
    print("resampled actions shape:", np.asarray(item[args.action_key]).shape)
    print("native first 3 dims:")
    print(native_actions[:, :3])
    print("resampled first 3 dims:")
    print(np.asarray(item[args.action_key])[:, :3])


if __name__ == "__main__":
    main()
