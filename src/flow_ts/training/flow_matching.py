import torch
from torch import Tensor
from torch.nn import functional as F


def interpolate(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Endpoints [B, T, C], times [B] -> state and velocity [B, T, C]."""
    if x0.ndim != 3 or x0.shape != x1.shape:
        raise ValueError("Endpoints must have matching shapes [B, T, C].")
    if t.shape != (x1.shape[0],):
        raise ValueError("t must have shape [B].")

    t_view = t[:, None, None]
    x_t = (1 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    return x_t, target_velocity


def sample_flow_batch(x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return x_t [B, T, C], t [B], target_velocity [B, T, C]."""
    x0 = torch.randn_like(x1)
    t = torch.randn(
        x1.shape[0], device=x1.device, dtype=x1.dtype
    ).sigmoid()
    x_t, target_velocity = interpolate(x0, x1, t)
    return x_t, t, target_velocity


def flow_matching_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")
    return F.mse_loss(prediction, target)