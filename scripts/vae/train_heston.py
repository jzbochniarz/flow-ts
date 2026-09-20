import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_ts.data.dataset import TimeSeriesDataset
from flow_ts.models import PatchVAE


def parse_config_and_args() -> tuple[DictConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train PatchVAE Encoder.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vae/train_heston.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 step on the first batch to verify the pipeline, then exit.",
    )

    # Add overrides from CLI
    args, overrides = parser.parse_known_args()

    file_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(overrides)
    cfg = OmegaConf.merge(file_cfg, cli_cfg)
    return cfg, args


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


def run_one_epoch(
    model: PatchVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    beta: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    smoke_test: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_recon = total_kl = 0.0
    n_samples = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch.to(device=device, dtype=torch.float32)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if is_train:
                recon_x, mu, logvar, _ = model(x)
            else:
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)
                
            loss, recon_loss, kl_loss = compute_loss(recon_x, x, mu, logvar, beta)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite VAE loss")

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            n_samples += batch_size

            if smoke_test:
                break

    if n_samples == 0:
        raise ValueError("Loader produced no batches")

    return (
        total_loss / n_samples,
        total_recon / n_samples,
        total_kl / n_samples,
    )


def main() -> None:
    cfg, args = parse_config_and_args()
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}.")

    train_ds = TimeSeriesDataset(
        npz_path=cfg.data.train_path,
        window_len=cfg.data.window_len,
        stride=cfg.data.stride,
        normalize=cfg.data.normalize,
    )
    n_paths, T, input_dim = train_ds.raw.shape

    val_ds = TimeSeriesDataset(
        npz_path=cfg.data.val_path,
        window_len=cfg.data.window_len,
        stride=cfg.data.stride,
        normalize=cfg.data.normalize,
    )

    torch.manual_seed(cfg.training.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = PatchVAE(
        input_dim=input_dim,
        latent_dim=cfg.model.latent_dim,
        patch_len=cfg.model.patch_len,
    )
    model.to(device)
    print(
        f"Initialized PatchVAE:\n"
        f"input_dim={input_dim}, latent_dim={cfg.model.latent_dim}, patch_len={cfg.model.patch_len}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    for epoch in tqdm(range(cfg.training.epochs)):
        train_loss, train_recon, train_kl = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=cfg.training.beta,
            device=device,
            smoke_test=args.smoke_test,
        )
        val_loss, val_recon, val_kl = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            beta=cfg.training.beta,
            device=device,
            smoke_test=args.smoke_test,
        )

        print(
            f"\nEpoch {epoch + 1}/{cfg.training.epochs} | "
            f"Train MSE {train_recon:.3e}, KL {train_kl:.3e} | "
            f"Val MSE {val_recon:.3e}, KL {val_kl:.3e}",
            flush=True,
        )

        scheduler.step()

        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "input_dim": input_dim,  # this may differ by dataset, so best to save
        }

        if args.smoke_test:
            torch.save(checkpoint, save_dir / "smoke.pt")
            print("Smoke test complete; saved smoke.pt.", flush=True)
            break

        torch.save(checkpoint, save_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, save_dir / "best.pt")


if __name__ == "__main__":
    main()
