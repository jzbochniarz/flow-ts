import pytest
import torch
from torch import Tensor, nn

from flow_ts.models import FlowModel
from flow_ts.models.backbone.base import Backbone


class DummyTimeEmbedder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: Tensor) -> Tensor:
        return t.unsqueeze(-1).expand(*t.shape, self.d_model)


class DummyBackbone(Backbone):
    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        assert h.shape == c.shape
        return h + c


@pytest.fixture
def model():
    return FlowModel(
        backbone=DummyBackbone(),
        time_embedder=DummyTimeEmbedder(d_model=16),
        patch_len=4,
        d_model=16,
    )


def test_patch_contents_and_round_trip():
    x = torch.arange(2 * 12 * 3).reshape(2, 12, 3)
    patches = FlowModel.patchify(x, patch_len=4)

    assert patches.shape == (2, 3, 3, 4)

    # Check ordering explicitly, not just invertibility.
    for n in range(3):
        for c in range(3):
            torch.testing.assert_close(
                patches[:, n, c, :],
                x[:, n * 4 : (n + 1) * 4, c],
            )

    torch.testing.assert_close(FlowModel.unpatchify(patches), x)


def test_patchify_preserves_channel_permutation():
    x = torch.randn(2, 12, 3)
    permutation = torch.tensor([2, 0, 1])

    torch.testing.assert_close(
        FlowModel.patchify(x[:, :, permutation], 4),
        FlowModel.patchify(x, 4)[:, :, permutation, :],
    )


def test_condition_uses_time_and_status(model):
    t = torch.tensor([0.25, 0.75])
    mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    mask[:, 0, :] = True

    # Distinct values ensure the status lookup is actually tested.
    with torch.no_grad():
        model.status_embedding.weight[0].fill_(2.0)
        model.status_embedding.weight[1].fill_(5.0)

    c = model._build_condition(t, mask)

    token_times = torch.where(mask, 1.0, t[:, None, None])
    status = torch.where(mask, 5.0, 2.0)
    expected = (token_times + status).unsqueeze(-1).expand_as(c)

    assert c.shape == (2, 3, 3, 16)
    torch.testing.assert_close(c, expected)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_forward_and_backward(model, batch_size):
    x = torch.randn(batch_size, 12, 3)
    t = torch.rand(batch_size)
    mask = torch.zeros(batch_size, 3, 3, dtype=torch.bool)
    mask[:, 0, :] = True

    velocity = model(x, t, mask)

    assert velocity.shape == x.shape
    assert torch.isfinite(velocity).all()

    velocity.square().mean().backward()

    for projection in (model.input_projection, model.output_projection):
        grad = projection.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0


def test_default_mask_means_all_missing(model):
    x = torch.randn(2, 12, 3)
    t = torch.rand(2)
    mask = torch.zeros(2, 3, 3, dtype=torch.bool)

    torch.testing.assert_close(model(x, t), model(x, t, mask))


def test_invalid_inputs_rejected(model):
    with pytest.raises(ValueError):
        model(torch.randn(2, 11, 3), torch.rand(2))

    with pytest.raises(ValueError):
        model(torch.randn(2, 12, 3), torch.rand(2, 1))

    # Wrong shape: a timestep-level mask instead of a patch-level mask.
    with pytest.raises(ValueError):
        model(
            torch.randn(2, 12, 3),
            torch.rand(2),
            torch.zeros(2, 12, 3, dtype=torch.bool),
        )

    # Correct shape, wrong dtype.
    with pytest.raises(ValueError):
        model(
            torch.randn(2, 12, 3),
            torch.rand(2),
            torch.zeros(2, 3, 3),
        )
