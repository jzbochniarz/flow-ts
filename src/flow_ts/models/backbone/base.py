from abc import ABC, abstractmethod

from torch import Tensor, nn


class Backbone(nn.Module, ABC):
    """Transform embedded patches using combined conditioning."""

    @abstractmethod
    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            h: Embedded patches (B, N, C, D).
            c: Combined flow-time and observation-status
                embedding (B, N, C, D).

        Returns:
            Updated hidden features (B, N, C, D).
        """
        ...
