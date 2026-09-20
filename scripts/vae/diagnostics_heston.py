import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from flow_ts.data.dataset import TimeSeriesDataset
from flow_ts.models import PatchVAE


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--out", default="runs/vae_heston")
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    cfg = ckpt["cfg"]

    ds = TimeSeriesDataset(
        npz_path=cfg["data"]["val_path"],
        window_len=cfg["data"]["window_len"],
        stride=cfg["data"]["stride"],
        normalize=cfg["data"]["normalize"],
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = PatchVAE(
        input_dim=ds.raw.shape[-1],
        latent_dim=cfg["model"]["latent_dim"],
        patch_len=cfg["model"]["patch_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    originals, means, samples = [], [], []
    for i, x in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        mu, logvar = model.encode(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)

        originals.append(x.cpu())
        means.append(model.decode(mu).cpu())
        samples.append(model.decode(z).cpu())
        if i + 1 >= args.batches:
            break

    if not originals:
        raise ValueError("Validation loader is empty")

    x = torch.cat(originals)
    reconstructions = {
        "Mean": torch.cat(means),
        "Sample": torch.cat(samples),
    }

    flat_x = x.flatten(0, 1)  # [windows * time, channels]
    std_x = flat_x.std(dim=0)
    corr_x = torch.corrcoef(flat_x.T)
    off_diag = ~torch.eye(x.shape[-1], dtype=torch.bool)

    lines = [
        f"Device: {device} | Windows: {len(x)}",
        f"Zero baseline MSE: {x.square().mean().item():.6e}",
    ]
    correlations = {"Original": corr_x}

    for name, recon in reconstructions.items():
        flat_recon = recon.flatten(0, 1)
        mse = (recon - x).square().mean(dim=(0, 1))
        std_ratio = flat_recon.std(dim=0) / std_x.clamp_min(1e-12)
        corr = torch.corrcoef(flat_recon.T)
        correlations[name] = corr
        corr_error = (corr - corr_x).abs()[off_diag]

        lines.extend([
            f"\n{name} reconstruction MSE: {mse.mean().item():.6e}",
            f"Correlation MAE (off-diagonal): {corr_error.mean().item():.6f}",
            f"Correlation max error: {corr_error.max().item():.6f}",
            "Asset      MSE          Std ratio",
        ])
        for c in range(x.shape[-1]):
            lines.append(
                f"{c:5d}  {mse[c].item():.6e}  {std_ratio[c].item():.4f}"
            )

    report = "\n".join(lines)
    print(report)
    (out_dir / "metrics.txt").write_text(report + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    patch_len = cfg["model"]["patch_len"]

    for c, ax in enumerate(axes[0]):
        if c >= x.shape[-1]:
            ax.axis("off")
            continue
        ax.plot(x[0, :, c].numpy(), label="Original", color="black")
        for name, recon in reconstructions.items():
            ax.plot(recon[0, :, c].numpy(), label=name, alpha=0.75)
        for boundary in range(patch_len, x.shape[1], patch_len):
            ax.axvline(boundary - 0.5, color="gray", alpha=0.2)
        ax.set_title(f"First window, asset {c}")
        ax.set_xlabel("Time")
    axes[0, 0].legend()

    for ax, (name, corr) in zip(axes[1], correlations.items()):
        im = ax.imshow(corr.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{name} correlations")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved diagnostics to {out_dir.resolve()}")


if __name__ == "__main__":
    main()