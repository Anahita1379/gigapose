"""Small residual heads applied after the ordinary IST pose estimate."""

from __future__ import annotations

import torch
from torch import nn


class PoseResidualHead(nn.Module):
    """Predict crop-center, log-depth, and optional SO(3) residuals.

    The final layer is deliberately initialized to zero.  A newly constructed
    model therefore reproduces the checkpoint's original pose before learning
    any correction.
    """

    def __init__(self, descriptor_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = 2 * int(descriptor_size) + 3
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.translation = nn.Linear(hidden_dim, 3)
        self.rotation = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)
        nn.init.zeros_(self.rotation.weight)
        nn.init.zeros_(self.rotation.bias)

    def forward(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(descriptor)
        return {
            "translation": self.translation(hidden),
            "rotation": self.rotation(hidden),
        }

