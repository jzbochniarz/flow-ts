import torch
from torch import Tensor, nn

from flow_ts.models.vae.base import VAE


class Encoder(nn.Module):
    """Encoder for the Patch VAE."""

    def __init__(self, input_dim: int, latent_dim: int, patch_len: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_dim = patch_len * input_dim  # dimension of each patch/token

        # Initial linear projection
        self.proj = nn.Linear(self.patch_dim, latent_dim)

        # Local token refinement
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

        # Projection to latent space
        self.to_latent = nn.Linear(latent_dim, 2 * latent_dim)  # Output both mu and logvar

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[B, N, patch_dim] -> (mu, logvar), each [B, N, latent_dim]."""
        h = self.proj(x)  # [B, N, latent_dim]
        h = h + self.mlp(self.norm(h))  # Local token refinement
        h = self.to_latent(h)  # [B, N, 2 * latent_dim]
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)  # Clamp logvar for numerical stability
        return mu, logvar


class Decoder(nn.Module):
    """Decoder for the Patch VAE."""

    def __init__(self, latent_dim: int, output_dim: int, patch_len: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_dim = patch_len * output_dim  # dimension of each patch/token

        # Initial linear projection
        self.proj = nn.Linear(latent_dim, self.patch_dim)

        # Local token refinement
        self.norm = nn.LayerNorm(self.patch_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.patch_dim, 2 * self.patch_dim),
            nn.GELU(),
            nn.Linear(2 * self.patch_dim, self.patch_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """[B, N, latent_dim] -> reconstruction [B, N, patch_dim]."""
        h = self.proj(z)  # [B, N, patch_dim]
        h = h + self.mlp(self.norm(h))  # Local token refinement
        return h


class PatchVAE(VAE):
    """Minimal Patch Variational Autoencoder (VAE) for time series data."""

    def __init__(self, input_channels: int, d_model: int, patch_len: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_dim = patch_len * input_channels  # dimension of each patch/token

        self.encoder = Encoder(input_dim=input_channels, latent_dim=d_model, patch_len=patch_len)
        self.decoder = Decoder(latent_dim=d_model, output_dim=input_channels, patch_len=patch_len)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[B, T, C] -> (mu, logvar), each [B, N, D]."""
        x = self.patchify(x, self.patch_len)  # [B, N, patch_dim]
        mu, logvar = self.encoder(x)  # [B, N, latent_dim]
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """[B, N, latent_dim] -> reconstruction [B, N, patch_dim]."""
        h = self.decoder(z)  # [B, N, patch_dim]
        x = self.depatchify(h, self.patch_len)  # [B, T, C]
        return x

    @staticmethod
    def patchify(x: Tensor, patch_len: int) -> Tensor:
        """[B, T, C] -> [B, n_patches, patch_dim] where patch_dim = patch_len * C."""
        B, T, C = x.shape
        assert T % patch_len == 0, f"Time dimension {T} must be divisible by patch_len {patch_len}"
        n_patches = T // patch_len
        patch_dim = patch_len * C
        x = x.reshape(B, n_patches, patch_dim)
        return x

    @staticmethod
    def depatchify(x: Tensor, patch_len: int) -> Tensor:
        """[B, n_patches, patch_dim] -> [B, T, C] where C = patch_dim // patch_len."""
        B, n_patches, patch_dim = x.shape
        assert patch_dim % patch_len == 0, (
            f"Feature dimension ({patch_dim}) must be divisible by patch_len ({patch_len})"
        )
        C = patch_dim // patch_len  # number of channels
        T = n_patches * patch_len  # total time dimension
        x = x.reshape(B, T, C)
        return x
