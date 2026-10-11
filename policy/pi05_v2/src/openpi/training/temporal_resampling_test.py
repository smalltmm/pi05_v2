import numpy as np

from openpi.training.temporal_resampling import TemporalResampledDataset, resample_trajectory


def _linear(times, slope=2.0, intercept=1.0):
    return (slope * np.asarray(times) + intercept)[:, None].astype(np.float32)


def test_resample_25_to_25_is_native_values():
    source_times = np.arange(12, dtype=np.float64) / 25.0
    target_times = np.arange(10, dtype=np.float64) / 25.0
    values = _linear(source_times)
    np.testing.assert_array_equal(
        resample_trajectory(values, source_times, target_times), values[:10]
    )


def test_resample_30_to_25_linear_signal():
    source_times = np.arange(30, dtype=np.float64) / 30.0
    target_times = np.arange(20, dtype=np.float64) / 25.0
    result = resample_trajectory(_linear(source_times), source_times, target_times)
    np.testing.assert_allclose(result[:, 0], 2 * target_times + 1, rtol=0, atol=1e-6)


def test_resample_10_to_25_linear_signal():
    source_times = np.arange(30, dtype=np.float64) / 10.0
    target_times = np.arange(50, dtype=np.float64) / 25.0
    result = resample_trajectory(_linear(source_times), source_times, target_times)
    np.testing.assert_allclose(result[:, 0], 2 * target_times + 1, rtol=0, atol=1e-6)


def test_gripper_uses_zero_order_hold():
    source_times = np.array([0.0, 0.1])
    values = np.array([[0.0, 0.0], [10.0, 1.0]], dtype=np.float32)
    target_times = np.array([0.0, 0.04, 0.08, 0.10, 0.12])
    result = resample_trajectory(
        values,
        source_times,
        target_times,
        feature_names=["joint", "gripper"],
    )
    np.testing.assert_allclose(result[:, 0], [0.0, 4.0, 8.0, 10.0, 10.0])
    np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0, 1.0, 1.0])


class _FakeSelection:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        return [row[key] for row in self._rows]


class _FakeHF:
    def __init__(self, rows):
        self._rows = rows

    def select(self, indices):
        return _FakeSelection([self._rows[int(index)] for index in indices])


class _FakeMeta:
    fps = 10
    features = {"action": {"names": ["joint"]}}


class _FakeDataset:
    delta_timestamps = None
    meta = _FakeMeta()

    def __init__(self):
        self.rows = [
            {"timestamp": i / 10.0, "episode_index": 0, "action": np.array([float(i)])}
            for i in range(3)
        ] + [
            {"timestamp": i / 10.0, "episode_index": 1, "action": np.array([100.0 + i])}
            for i in range(3, 6)
        ]
        self.hf_dataset = _FakeHF(self.rows)
        self.episode_data_index = {"from": np.array([0, 3]), "to": np.array([3, 6])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])


def test_episode_boundary_does_not_read_next_episode():
    dataset = TemporalResampledDataset(
        _FakeDataset(),
        action_horizon=5,
        target_action_fps=25,
    )
    item = dataset[2]
    # The last target is outside episode 0 and must repeat episode 0's final
    # action, never read episode 1's value (100+).
    np.testing.assert_array_equal(item["action"][:, 0], [2.0] * 5)


def test_native_frequency_wrapper_returns_exact_rows():
    dataset = TemporalResampledDataset(
        _FakeDataset(),
        action_horizon=4,
        target_action_fps=10,
    )
    item = dataset[0]
    np.testing.assert_array_equal(item["action"][:, 0], [0.0, 1.0, 2.0, 2.0])


def test_float32_timestamps_do_not_delay_gripper_on_grid_hits():
    for fps in (10, 30):
        # Nonzero observation time exposes errors hidden by short t=0 tests.
        times = (np.arange(1000, dtype=np.float32) / fps).astype(np.float64)
        anchor = 501
        targets = times[anchor] + np.arange(50) / 25.0
        values = np.arange(1000, dtype=np.float32)[:, None]
        result = resample_trajectory(values, times, targets, zoh_indices=(0,))
        expected = anchor + np.floor(np.arange(50) * fps / 25.0 + 1e-9).astype(int)
        np.testing.assert_array_equal(result[:, 0], expected)
