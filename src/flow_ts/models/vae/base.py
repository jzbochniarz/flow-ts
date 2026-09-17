from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class VAE(nn.Module, ABC):
    """Base class for the Variational Autoencoder (VAE)."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[B, T, C] -> (mu, logvar), each [B, N, D]."""
        ...

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """[B, N, D] -> reconstruction [B, T, C]."""
        ...

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through the whole VAE.

        Returns:
            recon_x: [B, T, C] - reconstructed input
            mu: [B, N, D] - mean of the latent distribution
            logvar: [B, N, D] - log variance of the latent distribution
            z: [B, N, D] - sampled latent variable
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
