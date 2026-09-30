import pytest
import torch

from flow_ts.models.conditioning import FourierTimeEmbedder


@pytest.mark.parametrize("shape", [(4,), (2, 8, 10)])
def test_time_embedding(shape):
    embedder = FourierTimeEmbedder(fourier_dim=32, d_model=64)
    t = torch.linspace(0, 1, steps=shape.numel()).reshape(shape) if isinstance(
        shape, torch.Size
    ) else torch.linspace(0, 1, steps=__import__("math").prod(shape)).reshape(shape)

    out = embedder(t)

    assert out.shape == (*shape, 64)
    assert torch.isfinite(out).all()
    assert not torch.allclose(
        out.reshape(-1, 64)[0],
        out.reshape(-1, 64)[-1],
    )


def test_time_embedding_backward():
    torch.manual_seed(42)
    embedder = FourierTimeEmbedder(fourier_dim=32, d_model=64)

    out = embedder(torch.rand(2, 8, 10))
    out.square().mean().backward()

    for parameter in embedder.mlp.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert embedder.mlp[0].weight.grad.abs().sum() > 0
    assert "freqs" in dict(embedder.named_buffers())
    assert "freqs" not in embedder.state_dict()  # persistent=False