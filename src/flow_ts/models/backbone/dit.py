
import torch
from torch import dropout_, nn, Tensor
from torch.nn import functional as F

from .base import Backbone


class RotaryEmbeddings(nn.Module):
    """Apply rotary position embeddings to adjacent feature pairs.

    Expects queries or keys shaped [B, H, N, D], where D = head_dim.
    Positions are consecutive sequence indices starting at zero.
    """

    freqs_cos: Tensor
    freqs_sin: Tensor

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")

        self.head_dim = head_dim

        # One angular frequency per adjacent pair of head features.
        inv_freqs = theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freqs)  # [max_seq_len, D // 2]

        # Fixed caches move with model.to(device). They are reconstructed
        # from configuration, so they need not be saved in state_dict().
        self.register_buffer(
            "freqs_cos", angles.cos(), persistent=False
        )
        self.register_buffer(
            "freqs_sin", angles.sin(), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Rotate queries or keys while preserving their shape and dtype.

        Args:
            x: Floating-point tensor [B, H, N, D], on the same device
                as this module's buffers.

        Returns:
            Rotated tensor [B, H, N, D].
        """
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected [B, H, N, {self.head_dim}], got {list(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError("RoPE input must be floating-point.")

        seq_len = x.shape[-2]
        max_seq_len = self.freqs_cos.shape[0]

        # Reject sequences longer than the fixed cache capacity.
        if seq_len > max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={max_seq_len}."
            )

        # [N, D // 2] broadcasts across batch and head dimensions.
        # Match the input dtype to avoid promoting attention activations.
        cos = self.freqs_cos[:seq_len].to(dtype=x.dtype)
        sin = self.freqs_sin[:seq_len].to(dtype=x.dtype)

        # Rotate adjacent pairs: (x_0, x_1), (x_2, x_3), ...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave each rotated pair to restore [B, H, N, D].
        return torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            use_rope: bool,
            causal: bool = False,
            **kwargs
    ) -> None:
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.rope = (
            RotaryEmbeddings(
                head_dim=self.head_dim,
                **kwargs
            )
            if use_rope
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Token features [B, L, D] where D = d_model.

        Returns:
            Attended features [B, L, D].
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected [B, L, {self.d_model}], got {list(x.shape)}."
            )

        B, L, D = x.shape

        # [B, L, 3 * D] -> [B, L, 3, H, D]
        qkv = self.qkv(x).reshape(
            B, L, 3, self.num_heads, self.head_dim
        )

        # [3, B, H, L, head_dim] and unbind
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0) # each is [B, H, L, head_dim]

        # apply RoPE if specified
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            is_causal=self.causal
        ) # [B, H, L, head_dim]

        # merge the heads back into token features
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.output_projection(out)


class DiTBlock(nn.Module):
    """Temporal attention, channel attention, and MLP with AdaLN-Zero."""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            mlp_ratio: float = 2.0,
            causal: bool = False
    ) -> None:
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("MLP hidden dimension must be positive.")
        self.d_model = d_model

        self.temporal_attn = SelfAttention(
            d_model, num_heads, use_rope=True, causal=causal
        )
        self.channel_attn = SelfAttention(
            d_model, num_heads, use_rope=False, causal=False
        )

        self.norm_temporal = nn.LayerNorm(
            d_model, elementwise_affine=False
        )
        self.norm_channel = nn.LayerNorm(
            d_model, elementwise_affine=False
        )
        self.norm_mlp = nn.LayerNorm(
            d_model, elementwise_affine=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

        # AdaLN-Zero params:
        # Three sets of (scale, shift, gate), each of width d_model
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _temporal_attention(self, h: Tensor) -> Tensor:
        """Attend across tokens/patches, independently for each channel.
        [B, N, C, D] -> [B, N, C, D]
        """
        B, N, C, D = h.shape

        h = h.transpose(1, 2).reshape(B * C, N, D)
        h = self.temporal_attn(h)
        h = h.reshape(B, C, N, D).transpose(1, 2) # [B, N, C, D]
        return h

    def _channel_attention(self, h: Tensor) -> Tensor:
        """Attend across channels, independently for each token."""
        B, N, C, D = h.shape

        h = h.reshape(B * N, C, D)
        h = self.channel_attn(h)
        h = h.reshape(B, N, C, D)
        return h

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: Hidden features [B, N, C, D].
            c: Combined conditioning [B, N, C, D].
        
        Returns:
            Updated hidden features [B, N, C, D]
        """
        if h.ndim != 4 or c.shape != h.shape:
            raise ValueError("h and c must have matching shapes [B, N, C, D].")

        (
            shift_t, scale_t, gate_t,
            shift_c, scale_c, gate_c,
            shift_m, scale_m, gate_m,
        ) = self.modulation(c).chunk(9, dim=-1)

        h = h + gate_t * self._temporal_attention(
            self.modulate(self.norm_temporal(h), shift_t, scale_t)
        )
        h = h + gate_c * self._channel_attention(
            self.modulate(self.norm_channel(h), shift_c, scale_c)
        )
        h = h + gate_m * self.mlp(
            self.modulate(self.norm_mlp(h), shift_m, scale_m)
        )
        return h


class DiT(Backbone):
    """Stack of conditioned temporal/channel attention blocks."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                causal=causal,
            )
            for _ in range(depth)
        ])

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            h: Embedded patches [B, N, C, D].
            c: Combined conditioning [B, N, C, D].

        Returns:
            Hidden features [B, N, C, D].
        """
        for block in self.blocks:
            h = block(h, c)
        return h