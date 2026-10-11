from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import ClassVar, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.temporal_resampling import TemporalResampledDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self.sampling_weights = getattr(dataset, "sampling_weights", None)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class StateActionOnlyDataset(Dataset[dict]):
    """Read LeRobot parquet/state/action data without decoding video frames.

    LeRobotDataset.__getitem__ eagerly decodes every camera in ``meta.video_keys``.
    Norm statistics only need state and actions, so this wrapper reads the underlying
    Hugging Face parquet dataset and reproduces delta-timestamp sampling without the
    video query. Dummy camera tensors keep the normal repack transforms reusable.
    """

    def __init__(self, dataset: lerobot_dataset.LeRobotDataset):
        self._dataset = dataset
        self.meta = dataset.meta
        self.delta_indices = getattr(dataset, "delta_indices", None)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = self._dataset.hf_dataset[index]
        ep_idx = item["episode_index"].item()

        if self.delta_indices is not None:
            query_indices, padding = self._dataset._get_query_indices(index, ep_idx)  # noqa: SLF001
            query_result = self._dataset._query_hf_dataset(query_indices)  # noqa: SLF001
            item = {**item, **padding}
            for key, value in query_result.items():
                item[key] = value

        item = dict(item)
        adapter = getattr(self._dataset, "adapt_item", None)
        if adapter is not None:
            item = adapter(item)
        # RepackTransform expects image source keys, but norm computation never uses
        # their values. Avoid video decoding while preserving the transform shape.
        for key in self.meta.video_keys:
            item.setdefault(key, torch.zeros((1, 1, 3), dtype=torch.uint8))
        return item


class _SelectedRows:
    """Small column-oriented view used by the temporal resampler."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __getitem__(self, key: str):
        return [row[key] for row in self._rows]


class _FilteredHFView:
    """Expose a filtered LeRobot parquet table with remapped episode indices."""

    def __init__(self, hf_dataset, row_indices: np.ndarray, episode_indices: np.ndarray):
        self._hf_dataset = hf_dataset
        self._row_indices = np.asarray(row_indices, dtype=np.int64)
        self._episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.column_names = hf_dataset.column_names

    def __len__(self) -> int:
        return self._row_indices.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = np.arange(len(self))[index]
            return _SelectedRows([self[int(i)] for i in indices])
        if not isinstance(index, int | np.integer):
            raise TypeError(f"Unsupported row index type: {type(index)}")
        logical_index = int(index)
        if logical_index < 0:
            logical_index += len(self)
        row = dict(self._hf_dataset[int(self._row_indices[logical_index])])
        row["episode_index"] = torch.tensor(self._episode_indices[logical_index])
        return row

    def select(self, indices) -> _SelectedRows:
        return _SelectedRows([self[int(index)] for index in indices])


class FilteredLeRobotDataset(Dataset[dict]):
    """Keep only rows belonging to one embodiment while retaining native videos.

    Robotwin stores several robot embodiments in one LeRobot dataset. The
    filtered view maps selected rows to compact episode indices, so action
    resampling cannot cross from a piper episode into another embodiment.
    """

    def __init__(self, dataset: lerobot_dataset.LeRobotDataset, embodiment: str):
        self._dataset = dataset
        self.meta = dataset.meta
        self.delta_timestamps = dataset.delta_timestamps
        if self.delta_timestamps is not None:
            raise ValueError("FilteredLeRobotDataset requires delta_timestamps=None")

        feature_key = "observation.embodiment_id"
        if feature_key not in dataset.hf_dataset.column_names:
            raise ValueError(f"{feature_key!r} is required to filter embodiment={embodiment!r}")

        raw_ids = dataset.hf_dataset[feature_key]
        embodiment_ids = np.stack([np.asarray(value).reshape(-1) for value in raw_ids], axis=0)
        order = (
            dataset.meta.info.get("robotwin_meta", {}).get("embodiment_id_order", [])
            if hasattr(dataset.meta, "info")
            else []
        )
        if embodiment not in order:
            raise ValueError(
                f"Cannot find embodiment {embodiment!r} in metadata order {order!r}"
            )
        embodiment_index = order.index(embodiment)
        selected_mask = np.argmax(embodiment_ids, axis=-1) == embodiment_index
        selected_mask &= np.max(embodiment_ids, axis=-1) > 0.5
        if not np.any(selected_mask):
            raise ValueError(f"No rows found for embodiment={embodiment!r}")

        raw_episode_ids = np.asarray(
            [int(np.asarray(value).reshape(-1)[0]) for value in dataset.hf_dataset["episode_index"]]
        )
        raw_indices = np.flatnonzero(selected_mask).astype(np.int64)
        # Split whenever either the source episode changes or selected rows are
        # not adjacent. This makes the boundary rule safe even for imperfect
        # mixed-embodiment recordings.
        split = np.flatnonzero(
            (raw_episode_ids[raw_indices[1:]] != raw_episode_ids[raw_indices[:-1]])
            | (raw_indices[1:] != raw_indices[:-1] + 1)
        ) + 1
        groups = np.split(raw_indices, split)
        groups = [group for group in groups if group.size > 0]
        logical_indices = np.concatenate(groups)
        logical_episode_indices = np.concatenate(
            [np.full(group.size, episode_index, dtype=np.int64) for episode_index, group in enumerate(groups)]
        )
        episode_from = np.cumsum(np.asarray([0, *[group.size for group in groups[:-1]]], dtype=np.int64))
        episode_to = episode_from + np.asarray([group.size for group in groups], dtype=np.int64)

        self.source_row_indices = logical_indices
        self.hf_dataset = _FilteredHFView(dataset.hf_dataset, logical_indices, logical_episode_indices)
        self.episode_data_index = {"from": episode_from, "to": episode_to}
        self._logical_episode_indices = logical_episode_indices
        self.repo_id = getattr(dataset, "repo_id", None)

    def __len__(self) -> int:
        return int(self.source_row_indices.size)

    def __getitem__(self, index: SupportsIndex) -> dict:
        logical_index = index.__index__()
        if logical_index < 0:
            logical_index += len(self)
        source_index = int(self.source_row_indices[logical_index])
        item = dict(self._dataset[source_index])
        item["episode_index"] = torch.tensor(self._logical_episode_indices[logical_index])
        return item


class FieldAdapterDataset(Dataset[dict]):
    """Adapt state/action layouts to the canonical 14-D two-arm layout."""

    _ADAPTER_INDICES: ClassVar[dict[str, np.ndarray]] = {
        # CobotMagic real ALOHA stores joints plus EE state in a 32-D vector.
        "robosyn_real": np.asarray([*range(7), *range(16, 23)], dtype=np.int64),
        # Robotwin pads each 6-DoF arm with a seventh zero joint.
        "robotwin_piper": np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], dtype=np.int64),
    }

    def __init__(self, dataset: Dataset[dict], adapter: str):
        if adapter not in self._ADAPTER_INDICES:
            raise ValueError(f"Unknown dataset adapter: {adapter}")
        self._dataset = dataset
        self.adapter = adapter
        self.meta = dataset.meta
        self.hf_dataset = dataset.hf_dataset
        self.episode_data_index = dataset.episode_data_index
        self.delta_timestamps = getattr(dataset, "delta_timestamps", None)
        self.repo_id = getattr(dataset, "repo_id", None)
        self.action_adapter = _FeatureIndexAdapter(self._ADAPTER_INDICES[adapter])
        self.feature_name_indices = self._ADAPTER_INDICES[adapter]

    def __len__(self) -> int:
        return len(self._dataset)

    def adapt_item(self, item: dict) -> dict:
        item = dict(item)
        for key in ("observation.state", "observation.qpos", "action", "actions"):
            if key in item:
                value = np.asarray(item[key])
                if value.shape[-1] == self._ADAPTER_INDICES[self.adapter].size:
                    continue
                item[key] = value[..., self._ADAPTER_INDICES[self.adapter]]
        return item

    def __getitem__(self, index: SupportsIndex) -> dict:
        return self.adapt_item(self._dataset[index])


class _FeatureIndexAdapter:
    """Pickle-safe callable used by temporal resampling workers."""

    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __call__(self, values):
        return np.asarray(values)[..., self.indices]


class PromptFromMultiLeRobotTask:
    """Extract prompts from a multi-dataset item."""

    def __init__(self, tasks_by_dataset_index: dict[int, dict[int, str]]):
        self._tasks_by_dataset_index = tasks_by_dataset_index

    def __call__(self, data: dict) -> dict:
        dataset_index = int(data["dataset_index"])
        task_index = int(data["task_index"])
        tasks = self._tasks_by_dataset_index[dataset_index]
        prompt = tasks.get(task_index)
        if prompt is None:
            raise ValueError(f"{task_index=} not found for {dataset_index=}: {tasks}")
        return {**data, "prompt": prompt}


class CanonicalMultiLeRobotDataset(Dataset[dict]):
    """Read multiple local LeRobot datasets and normalize CobotMagic field aliases."""

    _ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "observation.state": ("observation.state", "observation.qpos"),
        "observation.images.cam_high": (
            "observation.images.cam_high",
            "cam_high.color",
            "observation.images.head",
        ),
        "observation.images.cam_left_wrist": (
            "observation.images.cam_left_wrist",
            "cam_left_wrist.color",
            "observation.images.left_wrist",
        ),
        "observation.images.cam_right_wrist": (
            "observation.images.cam_right_wrist",
            "cam_right_wrist.color",
            "observation.images.right_wrist",
        ),
    }

    def __init__(
        self,
        repo_ids: list[str],
        root: str | None,
        delta_timestamps: dict | None,
        *,
        dataset_roots: Sequence[str] = (),
        dataset_filters: Sequence[str | None] = (),
        dataset_adapters: Sequence[str | None] = (),
        dataset_episodes: Sequence[Sequence[int] | None] = (),
        load_images: bool = True,
        action_horizon: int | None = None,
        target_action_fps: float | None = None,
        action_sequence_keys: Sequence[str] = ("actions",),
        action_zoh_indices: Sequence[int] = (),
    ):
        self.repo_ids = repo_ids
        self.root = root
        self._datasets = []
        if dataset_roots and len(dataset_roots) != len(repo_ids):
            raise ValueError("dataset_roots must have one entry per repo_id")
        if dataset_filters and len(dataset_filters) != len(repo_ids):
            raise ValueError("dataset_filters must have one entry per repo_id")
        if dataset_adapters and len(dataset_adapters) != len(repo_ids):
            raise ValueError("dataset_adapters must have one entry per repo_id")
        if dataset_episodes and len(dataset_episodes) != len(repo_ids):
            raise ValueError("dataset_episodes must have one entry per repo_id")
        for dataset_index, repo_id in enumerate(repo_ids):
            dataset_root = (
                dataset_roots[dataset_index]
                if dataset_roots
                else (os.path.join(root, repo_id) if root is not None else None)
            )
            native_dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=(dataset_episodes[dataset_index] if dataset_episodes else None),
                delta_timestamps=delta_timestamps,
                download_videos=load_images,
                video_backend="pyav",
            )
            dataset_filter = dataset_filters[dataset_index] if dataset_filters else None
            if dataset_filter is not None:
                native_dataset = FilteredLeRobotDataset(native_dataset, dataset_filter)
            dataset_adapter = dataset_adapters[dataset_index] if dataset_adapters else None
            if dataset_adapter is not None:
                native_dataset = FieldAdapterDataset(native_dataset, dataset_adapter)
            if target_action_fps is not None:
                if action_horizon is None:
                    raise ValueError("action_horizon is required for temporal resampling")
                observation_dataset = None if load_images else StateActionOnlyDataset(native_dataset)
                dataset = TemporalResampledDataset(
                    native_dataset,
                    action_horizon=action_horizon,
                    target_action_fps=target_action_fps,
                    action_keys=action_sequence_keys,
                    zoh_indices=action_zoh_indices,
                    observation_dataset=observation_dataset,
                )
            else:
                dataset = native_dataset if load_images else StateActionOnlyDataset(native_dataset)
            self._datasets.append(dataset)
        self._cumulative_sizes = np.cumsum([len(dataset) for dataset in self._datasets])

    def set_sampling_group_weights(self, group_weights: Sequence[float]) -> None:
        """Assign per-row weights whose sums match the requested dataset groups."""
        if len(group_weights) != len(self._datasets):
            raise ValueError("sampling weights must have one value per dataset")
        group_weights = np.asarray(group_weights, dtype=np.float64)
        if np.any(group_weights < 0) or not np.isclose(group_weights.sum(), 1.0):
            raise ValueError("sampling weights must be non-negative and sum to 1")
        weights = np.empty(len(self), dtype=np.float64)
        start = 0
        for dataset, group_weight in zip(self._datasets, group_weights, strict=True):
            end = start + len(dataset)
            weights[start:end] = group_weight / max(len(dataset), 1)
            start = end
        self.sampling_weights = torch.from_numpy(weights)

    def __len__(self) -> int:
        return int(self._cumulative_sizes[-1])

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        dataset_index = int(np.searchsorted(self._cumulative_sizes, idx, side="right"))
        start = 0 if dataset_index == 0 else int(self._cumulative_sizes[dataset_index - 1])
        item = dict(self._datasets[dataset_index][idx - start])
        for target, candidates in self._ALIASES.items():
            if target not in item:
                for source in candidates:
                    if source in item:
                        item[target] = item[source]
                        break
        item["dataset_index"] = torch.tensor(dataset_index)
        return item


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    load_images: bool = True,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_ids = tuple(getattr(data_config, "repo_ids", ()) or ())
    dataset_roots = tuple(getattr(data_config, "dataset_roots", ()) or ())
    dataset_filters = tuple(getattr(data_config, "dataset_filters", ()) or ())
    dataset_adapters = tuple(getattr(data_config, "dataset_adapters", ()) or ())
    dataset_episodes = tuple(getattr(data_config, "dataset_episodes", ()) or ())
    dataset_sampling_weights = tuple(getattr(data_config, "dataset_sampling_weights", ()) or ())
    target_action_fps = getattr(data_config, "target_action_fps", None)
    action_zoh_indices = getattr(data_config, "action_zoh_indices", ())
    if repo_ids:
        if data_config.dataset_root is None and not dataset_roots:
            raise ValueError("dataset_root or dataset_roots must be set when repo_ids is used.")
        if target_action_fps is None:
            # Keep the historical behavior byte-for-byte when temporal
            # resampling is disabled. In particular, this intentionally uses
            # the first dataset's native fps for the legacy multi-dataset path.
            first_root = os.path.join(data_config.dataset_root, repo_ids[0])
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0], root=first_root)
            delta_timestamps = {
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            }
        else:
            delta_timestamps = None
        dataset = CanonicalMultiLeRobotDataset(
            list(repo_ids),
            root=data_config.dataset_root,
            delta_timestamps=delta_timestamps,
            dataset_roots=dataset_roots,
            dataset_filters=dataset_filters,
            dataset_adapters=dataset_adapters,
            dataset_episodes=dataset_episodes,
            load_images=load_images,
            action_horizon=action_horizon,
            target_action_fps=target_action_fps,
            action_sequence_keys=data_config.action_sequence_keys,
            action_zoh_indices=action_zoh_indices,
        )
        if dataset_sampling_weights:
            dataset.set_sampling_group_weights(dataset_sampling_weights)
        if data_config.prompt_from_task:
            tasks_by_dataset_index = {i: ds.meta.tasks for i, ds in enumerate(dataset._datasets)}  # noqa: SLF001
            dataset = TransformedDataset(dataset, [PromptFromMultiLeRobotTask(tasks_by_dataset_index)])
        return dataset

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.dataset_root)
    if target_action_fps is None:
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=data_config.dataset_root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
            download_videos=load_images,
        )
        if not load_images:
            dataset = StateActionOnlyDataset(dataset)
    else:
        native_dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=data_config.dataset_root,
            delta_timestamps=None,
            download_videos=load_images,
            video_backend="pyav",
        )
        observation_dataset = None if load_images else StateActionOnlyDataset(native_dataset)
        dataset = TemporalResampledDataset(
            native_dataset,
            action_horizon=action_horizon,
            target_action_fps=target_action_fps,
            action_keys=data_config.action_sequence_keys,
            zoh_indices=action_zoh_indices,
            observation_dataset=observation_dataset,
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    sampling_weights = getattr(dataset, "sampling_weights", None)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if sampling_weights is not None:
        if framework == "pytorch" and torch.distributed.is_initialized():
            sampler = WeightedDistributedSampler(
                sampling_weights,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                seed=seed,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            sampler = torch.utils.data.WeightedRandomSampler(
                sampling_weights,
                num_samples=len(dataset),
                replacement=True,
            )
            local_batch_size = batch_size
    elif framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class WeightedDistributedSampler(torch.utils.data.Sampler[int]):
    """Sample global rows by weight while giving each DDP rank a unique stream."""

    def __init__(self, weights, *, num_replicas: int, rank: int, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.num_samples = (len(self.weights) + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
