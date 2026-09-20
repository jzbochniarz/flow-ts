"""Transform raw Heston price paths into log-returns, with train-derived norm stats baked in."""

from pathlib import Path

import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]  # order matters — train first, stats derived from it


def log_returns(S: np.ndarray) -> np.ndarray:
    """(N_paths, T, C) prices -> (N_paths, T-1, C) log returns."""
    log_S = np.log(S)
    return np.diff(log_S, axis=1)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mean, std = None, None
    for split in SPLITS:
        raw = np.load(RAW_DIR / f"heston_{split}.npz")
        returns = log_returns(raw["paths"])

        if split == "train":
            mean = returns.mean(axis=(0, 1), keepdims=True)
            std = returns.std(axis=(0, 1), keepdims=True) + 1e-6
        assert mean is not None, "train split must be processed first — stats are derived from it"

        np.savez(
            PROCESSED_DIR / f"heston_{split}.npz",
            paths=returns,  # log-returns, NOT normalized
            mean=mean,
            std=std,
        )
        print(f"Processed {split}: {raw['paths'].shape} prices -> {returns.shape} log-returns")


if __name__ == "__main__":
    main()
