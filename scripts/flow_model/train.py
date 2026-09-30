import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from typing import cast


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
    cfg = cast(DictConfig, cfg)
    return cfg, args


def compute_loss():
    raise(NotImplementedError)


def run_one_epoch():
    raise(NotImplementedError)

def main() -> None:
    cfg, args = parse_config_and_args()
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}.")

    # Initialize datasets, loaders, model, optimizer etc
    # model.to(device)

    # best_val_loss = float("inf")
    # for epoch in tqdm(range(cfg.training.epochs)):
    #     train_loss, val_loss = ...

    #     is_best = val_loss < best_val_loss
    #     best_val_loss = min(best_val_loss, val_loss)

    #     checkpoint = {
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "scheduler_state_dict": scheduler.state_dict(),
    #         "val_loss": val_loss,
    #         "best_val_loss": best_val_loss,
    #         "cfg": OmegaConf.to_container(cfg, resolve=True),
    #         "input_dim": input_dim,  # this may differ by dataset, so best to save
    #     }

    #     if args.smoke_test:
    #         torch.save(checkpoint, save_dir / "smoke.pt")
    #         print("Smoke test complete; saved smoke.pt.", flush=True)
    #         break

    #     torch.save(checkpoint, save_dir / "last.pt")
    #     if is_best:
    #         torch.save(checkpoint, save_dir / "best.pt")


if __name__ == "__main__":
    main()
