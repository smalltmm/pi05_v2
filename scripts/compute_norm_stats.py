"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _fast_cobotmagic_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    max_frames: int | None = None,
) -> dict[str, normalize.RunningStats]:
    """Compute CobotMagic state/action stats directly from parquet columns.

    The regular LeRobot item path builds one Python sample at a time and performs
    overlapping action-window queries. This path reads only state/action/episode
    columns, vectorizes the action windows per episode, and never touches videos.
    """
    repo_ids = tuple(data_config.repo_ids or ()) or (data_config.repo_id,)
    if data_config.dataset_root is None or any(repo_id is None for repo_id in repo_ids):
        raise ValueError("CobotMagic fast path requires repo_id(s) and dataset_root")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    delta_transform = next(
        (transform for transform in data_config.data_transforms.inputs if transform.__class__.__name__ == "DeltaActions"),
        None,
    )
    delta_mask = None if delta_transform is None else np.asarray(delta_transform.mask, dtype=bool)
    processed = 0

    for repo_id in repo_ids:
        root = pathlib.Path(data_config.dataset_root)
        if data_config.repo_ids:
            root = root / repo_id
        dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root, download_videos=False)
        feature_keys = set(dataset.hf_dataset.column_names)
        state_key = "observation.qpos" if "observation.qpos" in feature_keys else "observation.state"
        columns = dataset.hf_dataset.select_columns([state_key, "action", "episode_index"])
        raw = columns[:]
        states = np.asarray(raw[state_key], dtype=np.float64)
        actions = np.asarray(raw["action"], dtype=np.float64)
        episode_indices = np.asarray(raw["episode_index"])

        boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(episode_indices)]))
        for episode_start, episode_end in zip(starts, ends, strict=True):
            if max_frames is not None and processed >= max_frames:
                break
            episode_limit = episode_end
            if max_frames is not None:
                episode_limit = min(episode_end, episode_start + max_frames - processed)
            episode_states = states[episode_start:episode_limit]
            episode_actions = actions[episode_start:episode_limit]
            stats["state"].update(episode_states)

            # Match LeRobotDataset's boundary behavior: out-of-episode action
            # indices are clamped to the final frame of the episode.
            episode_len = len(episode_actions)
            offsets = np.arange(action_horizon, dtype=np.int64)[None, :]
            for chunk_start in range(0, episode_len, 4096):
                chunk_end = min(chunk_start + 4096, episode_len)
                frame_indices = np.arange(chunk_start, chunk_end, dtype=np.int64)[:, None]
                action_indices = np.minimum(frame_indices + offsets, episode_len - 1)
                action_chunk = episode_actions[action_indices].copy()
                if delta_mask is not None:
                    action_chunk[..., : delta_mask.size] -= episode_states[chunk_start:chunk_end, None, :] * delta_mask
                stats["actions"].update(action_chunk)
            processed += episode_len
        if max_frames is not None and processed >= max_frames:
            break

    return stats


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # Norm statistics only use state/actions. Avoid LeRobotDataset.__getitem__, which
    # eagerly decodes every camera video before those fields are returned.
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        load_images=False,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    is_cobotmagic = (
        data_config.dataset_root is not None
        and (data_config.repo_id or "").startswith("cobotmagic_Sim_")
        # The vectorized legacy path constructs native-fps action windows.
        # Temporal resampling must go through the regular dataset wrapper so
        # statistics are computed in the same 25-Hz action space as training.
        and data_config.target_action_fps is None
    )
    if is_cobotmagic:
        print("Using vectorized parquet-only norm computation (video decoding disabled).")
        stats = _fast_cobotmagic_stats(data_config, config.model.action_horizon, max_frames=max_frames)
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
