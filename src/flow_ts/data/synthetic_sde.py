import numpy as np


def simulate_heston(
    S0: np.ndarray,  # (C,) Initial spot prices
    v0: np.ndarray,  # (C,) Initial variances
    r: float,  # Risk-free interest rate
    kappa: np.ndarray,  # (C,) Mean-reversion speeds
    theta: np.ndarray,  # (C,) Long-term variance levels
    xi: np.ndarray,  # (C,) Vol-of-vol
    Gamma: np.ndarray,  # (2C, 2C) Correlation matrix
    T: float,  # Time horizon
    N_steps: int,  # Number of time steps
    N_paths: int,  # Number of Monte Carlo trajectories
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    C = len(S0)
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)

    L = np.linalg.cholesky(Gamma)

    S = np.zeros((N_paths, N_steps + 1, C))
    v = np.zeros((N_paths, N_steps + 1, C))
    S[:, 0, :] = S0
    v[:, 0, :] = v0

    x = np.repeat(np.log(S0)[np.newaxis, :], N_paths, axis=0)
    v_curr = np.repeat(v0[np.newaxis, :], N_paths, axis=0)

    for t in range(1, N_steps + 1):
        Z = rng.standard_normal((N_paths, 2 * C))  # <-- rng, not np.random
        dW = sqrt_dt * (Z @ L.T)
        dW_S = dW[:, :C]
        dW_v = dW[:, C:]

        v_pos = np.maximum(v_curr, 0.0)
        sqrt_v_pos = np.sqrt(v_pos)

        x += (r - 0.5 * v_pos) * dt + sqrt_v_pos * dW_S
        v_curr += kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * dW_v

        S[:, t, :] = np.exp(x)
        v[:, t, :] = v_curr

    return S, v
