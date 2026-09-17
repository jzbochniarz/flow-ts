from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Windowed, normalized multivariate time series loaded from a cached .npz.
    Used for both synthetic and real data.

    Supports z-score normalization and windowing with a given stride.
    The dataset is expected to be stored in a .npz file with a key "paths"
    containing an array of shape (N, T, C) or (T, C).
    When normalize=True, "mean" and "std" keys must be present in the .npz file.
    """

    def __init__(
        self,
        npz_path: str | Path,
        window_len: int,
        stride: int = 1,
        normalize: bool = True,
    ):
        data = np.load(npz_path)
        raw = data["paths"]  # (N, T, C) or (T, C)
        if raw.ndim == 2:
            raw = raw[None]  # (T, C) -> (1, T, C)

        self.window_len = window_len
        self.stride = stride
        self.n_paths, self.t, self.channels = raw.shape

        if self.t < self.window_len:
            raise ValueError(f"Time dimension ({self.t}) must be >= window_len ({self.window_len})")

        if normalize:
            if "mean" not in data or "std" not in data:
                raise KeyError(
                    f"'{npz_path}' must contain 'mean' and 'std' keys when normalize=True."
                )
            self.mean = data["mean"]
            self.std = data["std"]
            self.raw = (raw - self.mean) / self.std
        else:
            self.mean, self.std = None, None
            self.raw = raw

        self.windows_per_path = (self.t - self.window_len) // self.stride + 1
        self.total_windows = self.n_paths * self.windows_per_path

    def denormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Invert z-score normalization back to original units (e.g. log-returns)."""
        if self.mean is None or self.std is None:
            return x
        mean = np.squeeze(self.mean)
        std = np.squeeze(self.std)

        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return x * std + mean
        return x * std + mean

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        path_idx = idx // self.windows_per_path
        window_idx = idx % self.windows_per_path
        start = window_idx * self.stride
        end = start + self.window_len
        chunk = self.raw[path_idx, start:end]  # (window_len, C)
        return torch.from_numpy(chunk).float()
