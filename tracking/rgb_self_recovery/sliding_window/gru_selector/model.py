"""Causal GRU that selects one candidate, including GigaPose fallback."""

from __future__ import annotations

import torch
from torch import nn


class GRUCandidateSelector(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        frame_dim: int,
        *,
        candidate_hidden: int = 96,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        statistics: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.candidate_dim = int(candidate_dim)
        self.frame_dim = int(frame_dim)
        self.candidate_hidden = int(candidate_hidden)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)
        self.dropout = float(dropout)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden),
            nn.LayerNorm(candidate_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden, candidate_hidden),
            nn.SiLU(),
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_dim, candidate_hidden // 2),
            nn.LayerNorm(candidate_hidden // 2),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            candidate_hidden + candidate_hidden // 2,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.scorer = nn.Sequential(
            nn.Linear(candidate_hidden + gru_hidden, candidate_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden, 1),
        )
        statistics = statistics or {}
        self.register_buffer(
            "candidate_mean",
            torch.as_tensor(statistics.get("candidate_mean", torch.zeros(candidate_dim))).float(),
        )
        self.register_buffer(
            "candidate_std",
            torch.as_tensor(statistics.get("candidate_std", torch.ones(candidate_dim))).float(),
        )
        self.register_buffer(
            "frame_mean",
            torch.as_tensor(statistics.get("frame_mean", torch.zeros(frame_dim))).float(),
        )
        self.register_buffer(
            "frame_std",
            torch.as_tensor(statistics.get("frame_std", torch.ones(frame_dim))).float(),
        )

    def forward(self, candidate_features, frame_features, candidate_valid, hidden=None):
        candidate = (candidate_features - self.candidate_mean) / self.candidate_std
        frame = (frame_features - self.frame_mean) / self.frame_std
        encoded = self.candidate_encoder(candidate)
        masked = encoded.masked_fill(~candidate_valid[..., None], -1e4)
        pooled = masked.max(dim=2).values
        frame_encoded = self.frame_encoder(frame)
        temporal, hidden = self.gru(torch.cat([pooled, frame_encoded], dim=-1), hidden)
        expanded = temporal[:, :, None, :].expand(
            -1, -1, encoded.shape[2], -1
        )
        logits = self.scorer(torch.cat([encoded, expanded], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid, -1e9)
        return logits, hidden

    def config(self) -> dict[str, int | float]:
        return {
            "candidate_dim": self.candidate_dim,
            "frame_dim": self.frame_dim,
            "candidate_hidden": self.candidate_hidden,
            "gru_hidden": self.gru_hidden,
            "gru_layers": self.gru_layers,
            "dropout": self.dropout,
        }
