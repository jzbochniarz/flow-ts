import pytest
import torch

from flow_ts.models.vae.patch_vae import PatchVAE


@pytest.mark.parametrize("batch_size,window_len", [(1, 256), (4, 256), (2, 128)])
def test_patch_vae(batch_size: int, window_len: int) -> None:
    input_dim = 10
    d_model = 128
    patch_len = 16

    vae = PatchVAE(input_channels=input_dim, d_model=d_model, patch_len=patch_len)
    x = torch.randn(batch_size, window_len, input_dim)
    n_patches = window_len // patch_len
    patch_dim = patch_len * input_dim

    # 1. Patchify / Depatchify sanity and round-trip
    x_patchified = vae.patchify(x, patch_len)
    assert x_patchified.shape == (batch_size, n_patches, patch_dim)

    x_depatchified = vae.depatchify(x_patchified, patch_len)
    assert torch.equal(x_depatchified, x)

    # 2. Forward pass (train mode)
    vae.train()
    recon_x, mu, logvar, z = vae(x)

    assert recon_x.shape == x.shape
    assert mu.shape == (batch_size, n_patches, d_model)
    assert logvar.shape == mu.shape
    assert z.shape == mu.shape

    assert torch.isfinite(recon_x).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert torch.isfinite(z).all()

    # 3. Backward pass / gradient flow test
    loss = torch.nn.functional.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = loss + 1e-4 * kl_div
    total_loss.backward()

    for name, param in vae.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Zero gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"

    # 4. Eval mode sanity (z must strictly equal mu, deterministic output)
    vae.eval()
    with torch.no_grad():
        recon_eval, mu_eval, logvar_eval, z_eval = vae(x)
        assert torch.equal(z_eval, mu_eval), "In eval mode, z must equal mu"
