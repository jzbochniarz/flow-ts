import numpy as np

from flow_ts.data.dataset import TimeSeriesDataset


def test_dataset_indexing(tmp_path):
    # Setup dummy .npz
    dummy_data = np.random.randn(2, 50, 4).astype(np.float32)
    mean = np.mean(dummy_data, axis=(0, 1), keepdims=True)
    std = np.std(dummy_data, axis=(0, 1), keepdims=True)

    path = tmp_path / "dummy.npz"
    np.savez(tmp_path / "dummy.npz", paths=dummy_data, mean=mean, std=std)

    ds = TimeSeriesDataset(str(path), window_len=10, stride=5)

    assert len(ds) == 2 * ((50 - 10) // 5 + 1)
    sample = ds[0]
    assert sample.shape == (10, 4)
