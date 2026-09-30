import torch
from torch import Tensor, nn

from flow_ts.models.backbone.base import Backbone


class FlowModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        time_embedder: nn.Module,
        patch_len: int,
        d_model: int,
    ) -> None:
        super().__init__()

        if patch_len <= 0:
            raise ValueError("patch_len must be positive")

        self.backbone = backbone
        self.time_embedder = time_embedder
        self.patch_len = patch_len
        self.d_model = d_model

        self.input_projection = nn.Linear(patch_len, d_model)
        self.output_projection = nn.Linear(d_model, patch_len)

        self.status_embedding = nn.Embedding(2, d_model)
        nn.init.zeros_(self.status_embedding.weight)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        observed_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x_t: Current state [B, T, C]; observed entries are clean.
            t: Flow time [B].
            observed_mask: bool [B, N, C]; True = observed.

        Returns:
            Predicted velocity [B, T, C].
        """
        patches = self.patchify(x_t, self.patch_len)
        B, N, C, _ = patches.shape

        if t.shape != (B,):
            raise ValueError(f"Expected t shape {(B,)}, got {t.shape}")

        if observed_mask is None:
            observed_mask = torch.zeros(B, N, C, dtype=torch.bool, device=x_t.device)
        elif observed_mask.shape != (B, N, C) or observed_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean mask with shape (B, N, C)")

        h = self.input_projection(patches)
        c = self._build_condition(t, observed_mask)

        h = self.backbone(h, c)
        velocity_patches = self.output_projection(h)

        return self.unpatchify(velocity_patches)

    @staticmethod
    def patchify(x: Tensor, patch_len: int) -> Tensor:
        """(B, T, C) -> (B, N, C, P)."""
        if x.ndim != 3:
            raise ValueError("Expected input shape (B, T, C)")

        B, T, C = x.shape
        if T % patch_len != 0:
            raise ValueError(f"Patch length ({patch_len}) must divide window length ({T})")

        N = T // patch_len
        return x.reshape(B, N, patch_len, C).transpose(2, 3)

    @staticmethod
    def unpatchify(x: Tensor) -> Tensor:
        """(B, N, C, P) -> (B, T, C)."""
        if x.ndim != 4:
            raise ValueError("Expected patch shape (B, N, C, P)")

        B, N, C, P = x.shape
        return x.transpose(2, 3).reshape(B, N * P, C)

    def _build_condition(
        self,
        t: Tensor,  # (B,)
        observed_mask: Tensor,  # bool (B, N, C); True if observed
    ) -> Tensor:
        token_times = torch.where(
            observed_mask, torch.ones_like(t[:, None, None]), t[:, None, None]
        )  # (B, N, C)

        return self.time_embedder(token_times) + self.status_embedding(
            observed_mask.long()
        )  # (B, N, C, D)
