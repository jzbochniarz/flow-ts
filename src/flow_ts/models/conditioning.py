import math

import torch
from torch import Tensor, nn


class FourierTimeEmbedder(nn.Module):
    """Fixed sinusoidal time features followed by a trainable MLP."""
    freqs: Tensor

    def __init__(
        self,
        fourier_dim: int,
        d_model: int,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()

        if fourier_dim <= 0 or fourier_dim % 2 != 0:
            raise ValueError("fourier_dim must be a positive even integer.")
        if max_period <= 1:
            raise ValueError("max_period must be greater than 1.")

        half_dim = fourier_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.time_scale = time_scale

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: flow times, shape [...,]

        Returns:
            Time embeddings, shape [..., d_model].
        """
        t = t.to(dtype=self.freqs.dtype).unsqueeze(-1)
        angles = t * self.time_scale * self.freqs
        features = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )
        return self.mlp(features)