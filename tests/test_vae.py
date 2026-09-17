import torch
from torch import nn

from flow_ts.models import VAE


class DummyVAE(VAE):
    def __init__(self, input_dim: int, latent_dim: int, window_len: int, patch_len: int):
        super().__init__()
        self.patch_len = patch_len
        self.window_len = window_len
        assert window_len % patch_len == 0, "window_len must be divisible by patch_len"
        N = window_len // patch_len  # number of patches/tokens
        self.encoder = nn.Sequential(
            nn.Linear(patch_len * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),  # Output both mu and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, patch_len * input_dim)
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T % self.patch_len == 0, "Time dimension must be divisible by patch_len"
        x = x.view(B, T // self.patch_len, self.patch_len * C)
        return x

    def depatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = D // self.patch_len
        x = x.view(B, N * self.patch_len, C)
        return x

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.patchify(x)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon_patched = self.decoder(z)  # (B, N, patch_len * C)
        return self.depatchify(recon_patched)


def test_vae():
    input_dim = 10
    latent_dim = 8
    patch_len = 16
    for batch_size, window_len in [(1, 256), (4, 256), (2, 128)]:
        vae = DummyVAE(
            input_dim=input_dim, latent_dim=latent_dim, window_len=window_len, patch_len=patch_len
        )
        x = torch.randn(batch_size, window_len, input_dim)
        recon_x, mu, logvar, z = vae(x)
        assert recon_x.shape == x.shape
        assert mu.shape == (batch_size, window_len // patch_len, latent_dim)
        assert logvar.shape == mu.shape
        assert z.shape == mu.shape
        assert torch.isfinite(recon_x).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
        assert torch.isfinite(z).all()
