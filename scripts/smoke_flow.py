import argparse

import torch
from torch.utils.data import DataLoader

from flow_ts.data.dataset import TimeSeriesDataset
from flow_ts.models import FlowModel
from flow_ts.models import DiT
from flow_ts.models.conditioning import FourierTimeEmbedder
from flow_ts.training.flow_matching import (
    flow_matching_loss,
    sample_flow_batch,
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/processed/heston_train.npz"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cpu")

    dataset = TimeSeriesDataset(
        npz_path=args.data,
        window_len=64,
        stride=64,
        normalize=True,
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0
    )
    x1 = next(iter(loader)).to(device)

    d_model = 32
    backbone = DiT(
        d_model=d_model,
        depth=2,
        num_heads=4,
        causal=False
    )
    model = FlowModel(
        backbone=backbone,
        time_embedder=FourierTimeEmbedder(
            fourier_dim=32,
            d_model=d_model,
        ),
        patch_len=8,
        d_model=d_model,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i in range(2):
        x_t, t, target = sample_flow_batch(x1)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_t, t, observed_mask=None)

        assert prediction.shape == x1.shape
        loss = flow_matching_loss(prediction, target)
        assert torch.isfinite(loss), "Non-finite loss."

        loss.backward()

        # Check that each main component receives a finite, nonzero gradient.
        components = {
            "input_projection": model.input_projection,
            "time_embedder": model.time_embedder,
            "backbone": model.backbone,
            "output_projection": model.output_projection,
        }

        before = model.output_projection.weight.detach().clone()
        optimizer.step()

        assert not torch.equal(before, model.output_projection.weight), (
            "Optimizer did not update the output projection."
        )
        assert all(torch.isfinite(p).all() for p in model.parameters()), (
            "Non-finite parameter after optimizer step."
        )

        if i > 0:
            for name, module in components.items():
                grads = [
                    p.grad for p in module.parameters() if p.requires_grad
                ]
                assert grads and all(g is not None for g in grads), (
                    f"Missing gradient in {name}."
                )
                assert all(torch.isfinite(g).all() for g in grads), (
                    f"Non-finite gradient in {name}."
                )
                assert any(g.abs().sum() > 0 for g in grads), (
                    f"All gradients are zero in {name}."
                )

    print(f"Device: {device}")
    print(f"Input: {list(x1.shape)} | Output: {list(prediction.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("PASS: finite gradients and successful optimizer update.")


if __name__ == "__main__":
    main()