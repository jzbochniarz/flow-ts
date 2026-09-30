import pytest
import torch

from flow_ts.models.backbone.dit import (
    RotaryEmbeddings, SelfAttention, DiTBlock
)


def test_rope_rotation():
    torch.manual_seed(42)
    rope = RotaryEmbeddings(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 12, 8, requires_grad=True)  # [B, H, N, D]

    out = rope(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()

    # Position zero has zero rotation angle.
    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])

    # Each rotation preserves the head vector's Euclidean norm.
    torch.testing.assert_close(
        out.norm(dim=-1),
        x.norm(dim=-1),
    )

    # Nonzero positions should actually change.
    assert not torch.allclose(out[:, :, 1:], x[:, :, 1:])

    # Gradients must propagate through the rotation.
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_rope_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        RotaryEmbeddings(head_dim=7)

    rope = RotaryEmbeddings(head_dim=8, max_seq_len=16)

    with pytest.raises(ValueError):
        rope(torch.randn(2, 3, 17, 8))  # Exceeds cache capacity.

    with pytest.raises(ValueError):
        rope(torch.randn(2, 3, 12, 4))  # Wrong head dimension.



@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_shape_and_gradients(use_rope, causal):
    torch.manual_seed(42)
    attn = SelfAttention(
        d_model=32,
        num_heads=4,
        use_rope=use_rope,
        causal=causal,
    )
    x = torch.randn(2, 8, 32, requires_grad=True) # [B, L, D]

    out = attn(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.square().mean().backward()

    for tensor in (x, *attn.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize("causal", [False, True])
def test_attention_future_dependency(causal):
    torch.manual_seed(42)
    attn = SelfAttention(
        d_model=32,
        num_heads=4,
        use_rope=True,
        causal=causal,
    ).eval()

    x = torch.randn(2, 8, 32)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    with torch.no_grad():
        original_prefix = attn(x)[:, :4]
        changed_prefix = attn(changed)[:, :4]

    if causal:
        torch.testing.assert_close(original_prefix, changed_prefix)
    else:
        assert not torch.allclose(original_prefix, changed_prefix)



def make_block(causal=False):
    return DiTBlock(
        d_model=16,
        num_heads=2,
        mlp_ratio=2.0,
        causal=causal,
    )


def open_gates(block):
    """Activate branches so behavioral tests cannot pass as identity maps."""
    with torch.no_grad():
        # Modulation order: shift, scale, gate for each of three branches.
        bias = block.modulation[-1].bias.reshape(9, 16)
        bias[[2, 5, 8]] = 0.5


def assert_nonzero_finite_gradients(module):
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_block_initial_identity():
    torch.manual_seed(42)
    block = make_block()
    h = torch.randn(2, 4, 3, 16)
    c = torch.randn_like(h)

    out = block(h, c)

    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, h, rtol=0, atol=0)


def test_block_two_optimizer_steps():
    torch.manual_seed(42)
    block = make_block()

    # Disable weight decay to isolate updates caused by gradients.
    optimizer = torch.optim.AdamW(
        block.parameters(), lr=1e-3, weight_decay=0.0
    )

    h = torch.randn(2, 4, 3, 16)
    c = torch.randn_like(h, requires_grad=True)
    target = torch.randn_like(h)

    branches = (
        block.temporal_attn,
        block.channel_attn,
        block.mlp,
    )
    tracked = block.temporal_attn.qkv.weight
    initial_weight = tracked.detach().clone()

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        c.grad = None

        out = block(h, c)
        loss = (out - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()

        assert_nonzero_finite_gradients(block.modulation)

        if step == 0:
            # Closed gates block gradients into branch parameters.
            for branch in branches:
                for parameter in branch.parameters():
                    assert parameter.grad is not None
                    assert torch.count_nonzero(parameter.grad) == 0

            # Gate rows themselves must receive a learning signal.
            gate_grads = block.modulation[-1].weight.grad.reshape(
                9, 16, 16
            )[[2, 5, 8]]
            assert gate_grads.abs().sum() > 0
        else:
            # After one update, gradients reach all three branches.
            for branch in branches:
                assert_nonzero_finite_gradients(branch)

            assert c.grad is not None
            assert torch.isfinite(c.grad).all()
            assert c.grad.abs().sum() > 0

        optimizer.step()

        if step == 0:
            torch.testing.assert_close(
                tracked, initial_weight, rtol=0, atol=0
            )

    assert not torch.equal(tracked, initial_weight)
    assert all(torch.isfinite(p).all() for p in block.parameters())


def test_block_channel_permutation_equivariance():
    torch.manual_seed(42)
    block = make_block().eval()
    open_gates(block)

    h = torch.randn(2, 4, 3, 16)
    c = torch.randn_like(h)
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        expected = block(h, c)[:, :, permutation]
        actual = block(h[:, :, permutation], c[:, :, permutation])

    torch.testing.assert_close(
        actual, expected, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("causal", [False, True])
def test_block_future_dependency(causal):
    torch.manual_seed(42)
    block = make_block(causal=causal).eval()
    open_gates(block)

    h = torch.randn(2, 6, 3, 16)
    c = torch.randn_like(h)

    changed_h = h.clone()
    changed_c = c.clone()
    changed_h[:, 3:] = torch.randn_like(changed_h[:, 3:])
    changed_c[:, 3:] = torch.randn_like(changed_c[:, 3:])

    with torch.no_grad():
        original_prefix = block(h, c)[:, :3]
        changed_prefix = block(changed_h, changed_c)[:, :3]

    if causal:
        torch.testing.assert_close(original_prefix, changed_prefix)
    else:
        assert not torch.allclose(original_prefix, changed_prefix)