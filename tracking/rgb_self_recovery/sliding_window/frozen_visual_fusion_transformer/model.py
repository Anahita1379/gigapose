"""Gated candidate transformer with frozen frame and RGB/mask/CAD embeddings."""

from __future__ import annotations

import torch
from torch import nn

from tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.model import (
    GatedCandidateTransformer,
)


FORMAT = "rgb_self_recovery_frozen_visual_fusion_transformer_v2"


class FrozenVisualFusionTransformer(GatedCandidateTransformer):
    """Fuse cached frozen visual evidence without changing temporal width/depth."""

    def __init__(
        self,
        candidate_dim: int,
        frame_dim: int,
        candidate_count: int,
        *,
        candidate_visual_dim: int,
        frame_visual_dim: int,
        statistics: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        super().__init__(
            candidate_dim,
            frame_dim,
            candidate_count,
            statistics=statistics,
            **kwargs,
        )
        self.candidate_visual_dim = int(candidate_visual_dim)
        self.frame_visual_dim = int(frame_visual_dim)
        if self.candidate_visual_dim <= 0 or self.frame_visual_dim <= 0:
            raise ValueError("Visual dimensions must be positive")
        self.candidate_visual_encoder = nn.Sequential(
            nn.Linear(self.candidate_visual_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.frame_visual_encoder = nn.Sequential(
            nn.Linear(self.frame_visual_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.candidate_visual_gate = nn.Sequential(
            nn.Linear(2 * self.model_dim, self.model_dim), nn.Sigmoid()
        )
        self.frame_visual_gate = nn.Sequential(
            nn.Linear(2 * self.model_dim, self.model_dim), nn.Sigmoid()
        )
        self.candidate_visual_norm = nn.LayerNorm(self.model_dim)
        self.frame_visual_norm = nn.LayerNorm(self.model_dim)
        values = statistics or {}
        self.register_buffer(
            "candidate_visual_mean",
            torch.as_tensor(values.get(
                "candidate_visual_mean", torch.zeros(self.candidate_visual_dim)
            )).float(),
        )
        self.register_buffer(
            "candidate_visual_std",
            torch.as_tensor(values.get(
                "candidate_visual_std", torch.ones(self.candidate_visual_dim)
            )).float(),
        )
        self.register_buffer(
            "frame_visual_mean",
            torch.as_tensor(values.get(
                "frame_visual_mean", torch.zeros(self.frame_visual_dim)
            )).float(),
        )
        self.register_buffer(
            "frame_visual_std",
            torch.as_tensor(values.get(
                "frame_visual_std", torch.ones(self.frame_visual_dim)
            )).float(),
        )

    def config(self):
        return {
            **super().config(),
            "candidate_visual_dim": self.candidate_visual_dim,
            "frame_visual_dim": self.frame_visual_dim,
        }

    def encode_inputs(
        self,
        candidate_features,
        frame_features,
        candidate_visual_features=None,
        frame_visual_features=None,
    ):
        if candidate_visual_features is None or frame_visual_features is None:
            raise ValueError("Frozen visual-fusion model requires visual feature sidecars")
        numeric_candidate, numeric_frame = super().encode_inputs(
            candidate_features, frame_features
        )
        candidate_visual = (
            candidate_visual_features - self.candidate_visual_mean
        ) / self.candidate_visual_std
        frame_visual = (
            frame_visual_features - self.frame_visual_mean
        ) / self.frame_visual_std
        candidate_visual = self.candidate_visual_encoder(candidate_visual)
        frame_visual = self.frame_visual_encoder(frame_visual)
        candidate_gate = self.candidate_visual_gate(
            torch.cat((numeric_candidate, candidate_visual), dim=-1)
        )
        frame_gate = self.frame_visual_gate(
            torch.cat((numeric_frame, frame_visual), dim=-1)
        )
        candidate = self.candidate_visual_norm(
            numeric_candidate + candidate_gate * candidate_visual
        )
        frame = self.frame_visual_norm(numeric_frame + frame_gate * frame_visual)
        return candidate, frame


def load_transformer(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unsupported frozen visual-fusion checkpoint: {path}")
    model = FrozenVisualFusionTransformer(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), payload
