from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from flow_ts.data.synthetic_sde import simulate_heston


def sample_asset_params(n_assets: int, ranges: dict, seed: int) -> dict:
    """Sample per-asset Heston params uniformly from configured ranges."""
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(lo, hi, size=n_assets) for name, (lo, hi) in ranges.items()}


def build_correlation_matrix(C: int, corr: dict) -> np.ndarray:
    pp = np.full((C, C), corr["rho_price_price"])
    np.fill_diagonal(pp, 1.0)
    vv = np.full((C, C), corr["rho_vol_vol"])
    np.fill_diagonal(vv, 1.0)
    pv = np.full((C, C), corr["rho_cross_price_vol"])
    np.fill_diagonal(pv, corr["rho_leverage"])
    return np.block([[pp, pv], [pv.T, vv]])


def main(cfg_path: Path = Path("configs/data/heston_synthetic.yaml")):
    cfg = OmegaConf.load(cfg_path)
    C = cfg.n_assets
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    params = sample_asset_params(
        C, OmegaConf.to_container(cfg.asset_param_ranges), cfg.asset_sampling_seed
    )
    S0, v0 = params["S0"], params["v0"]
    kappa, theta, xi = params["kappa"], params["theta"], params["xi"]
    Gamma = build_correlation_matrix(C, cfg.correlation)

    mean, std = None, None
    for split_name, split_cfg in cfg.splits.items():
        S, v = simulate_heston(
            S0=S0,
            v0=v0,
            r=cfg.market.r,
            kappa=kappa,
            theta=theta,
            xi=xi,
            Gamma=Gamma,
            T=cfg.simulation.T,
            N_steps=cfg.simulation.N_steps,
            N_paths=split_cfg.N_paths,
            seed=split_cfg.seed,
        )
        if split_name == "train":
            mean = S.mean(axis=(0, 1), keepdims=True)
            std = S.std(axis=(0, 1), keepdims=True) + 1e-6
        assert mean is not None, (
            "train split must be listed first in config as stats are derived from it"
        )

        np.savez(output_dir / f"heston_{split_name}.npz", paths=S, v=v, mean=mean, std=std)
        print(f"Saved {split_name}: {S.shape}")


if __name__ == "__main__":
    main()
