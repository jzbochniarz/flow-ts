from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader

from flow_ts.data.dataset import TimeSeriesDataset
from flow_ts.models import PatchVAE


def compute_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def overfit_single_batch(
    model: PatchVAE,
    batch: Tensor,
    optimizer: torch.optim.Optimizer,
    beta: float,
    save_dir: Path,
) -> None:

    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        recon_x, mu, logvar, z = model(batch)

        loss, recon_loss, kl_loss = compute_loss(recon_x, batch, mu, logvar, beta)
        if not torch.isfinite(loss):
            raise ValueError(f"Loss is {loss}, stopping training")

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"{step}: mse={recon_loss.item():.5f}, "
                f"kl={kl_loss.item():.5f}, total={loss.item():.5f}"
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_dir / "last_epoch.pt")


def main(cfg_path: Path = Path("configs/train_vae.yaml")) -> None:
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}.")
    model = PatchVAE(
        input_channels=cfg.model.input_channels,
        d_model=cfg.model.latent_dim,
        patch_len=cfg.model.patch_len,
    )
    model.to(device)
    save_dir = Path(cfg.training.save_dir)
    torch.manual_seed(cfg.training.seed)

    train_dataset = TimeSeriesDataset(
        npz_path=cfg.data.training.path,
        window_len=cfg.data.training.window_len,
        stride=cfg.data.training.stride,
        normalize=cfg.data.training.normalize,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    batch = next(iter(train_loader))
    overfit_single_batch(model, batch, optimizer, cfg.training.beta, save_dir)


if __name__ == "__main__":
    main()
