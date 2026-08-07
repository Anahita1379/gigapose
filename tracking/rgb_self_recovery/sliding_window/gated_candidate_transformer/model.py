"""Factorized candidate/frame cross-attention with gated temporal fusion."""

from __future__ import annotations

import torch
from torch import nn


class GatedCandidateTransformer(nn.Module):
    """One shared model for selection, orientation, pose residual, and abstention.

    The model deliberately avoids global self-attention over all L*N candidate
    tokens. A frame query first cross-attends to the N candidates in its frame.
    Separate target queries then cross-attend to the L frame-evidence tokens.
    A learned gate fuses local and temporal evidence.
    """

    def __init__(
        self,
        candidate_dim: int,
        frame_dim: int,
        candidate_count: int,
        *,
        window_length: int = 8,
        model_dim: int = 128,
        heads: int = 4,
        feedforward_dim: int = 256,
        cross_layers: int = 2,
        dropout: float = 0.1,
        attention_future_lag: int = 4,
        use_translation_fixed_lag: bool = True,
        translation_fixed_lag_layers: int = 1,
        translation_position_scale_m: float = 100.0,
        translation_relative_scale_m: float = 10.0,
        maximum_translation_refinement_m: float = 20.0,
        statistics: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if not 0 <= attention_future_lag < window_length:
            raise ValueError(
                "attention_future_lag must be in [0, window_length)"
            )
        self.candidate_dim = int(candidate_dim)
        self.frame_dim = int(frame_dim)
        self.candidate_count = int(candidate_count)
        self.window_length = int(window_length)
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.cross_layers = int(cross_layers)
        self.dropout = float(dropout)
        self.attention_future_lag = int(attention_future_lag)
        self.use_translation_fixed_lag = bool(use_translation_fixed_lag)
        self.translation_fixed_lag_layers = int(translation_fixed_lag_layers)
        self.translation_position_scale_m = float(translation_position_scale_m)
        self.translation_relative_scale_m = float(translation_relative_scale_m)
        self.maximum_translation_refinement_m = float(
            maximum_translation_refinement_m
        )
        if self.translation_fixed_lag_layers < 1:
            raise ValueError("translation_fixed_lag_layers must be positive")
        if min(
            self.translation_position_scale_m,
            self.translation_relative_scale_m,
            self.maximum_translation_refinement_m,
        ) <= 0:
            raise ValueError("Translation fixed-lag scales must be positive")

        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        self.frame_position = nn.Embedding(window_length, model_dim)
        self.candidate_position = nn.Embedding(candidate_count, model_dim)
        self.local_query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.local_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.local_norm = nn.LayerNorm(model_dim)

        self.temporal_query = nn.Linear(model_dim, model_dim)
        self.temporal_attention = nn.ModuleList(
            nn.MultiheadAttention(
                model_dim, heads, dropout=dropout, batch_first=True
            )
            for _ in range(cross_layers)
        )
        self.temporal_norm = nn.ModuleList(
            nn.LayerNorm(model_dim) for _ in range(cross_layers)
        )
        self.temporal_ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(model_dim, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, model_dim),
            )
            for _ in range(cross_layers)
        )
        self.temporal_ffn_norm = nn.ModuleList(
            nn.LayerNorm(model_dim) for _ in range(cross_layers)
        )
        self.fusion_gate = nn.Sequential(nn.Linear(2 * model_dim, model_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(model_dim)

        pair_dim = 2 * model_dim
        self.candidate_scorer = nn.Sequential(
            nn.Linear(pair_dim, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 1)
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(pair_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(pair_dim, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 6)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(pair_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 2)
        )
        self.abstention_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1)
        )
        self.translation_trust_head = nn.Sequential(
            nn.Linear(pair_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1)
        )
        self.rotation_trust_head = nn.Sequential(
            nn.Linear(pair_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1)
        )
        # A second, explicitly translation-focused temporal pass. Its input is
        # the differentiable expected pose proposal (soft candidate selection)
        # plus the fused RGB/CAD/frame evidence. The same lag mask prevents any
        # query from seeing farther into the future than deployment permits.
        self.translation_motion_encoder = nn.Sequential(
            nn.Linear(8, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        self.translation_temporal_attention = nn.ModuleList(
            nn.MultiheadAttention(
                model_dim, heads, dropout=dropout, batch_first=True
            )
            for _ in range(self.translation_fixed_lag_layers)
        )
        self.translation_temporal_norm = nn.ModuleList(
            nn.LayerNorm(model_dim) for _ in range(self.translation_fixed_lag_layers)
        )
        self.translation_temporal_ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(model_dim, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, model_dim),
            )
            for _ in range(self.translation_fixed_lag_layers)
        )
        self.translation_temporal_ffn_norm = nn.ModuleList(
            nn.LayerNorm(model_dim) for _ in range(self.translation_fixed_lag_layers)
        )
        self.translation_fixed_lag_delta_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 3),
        )
        self.translation_fixed_lag_gate_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
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

    def config(self) -> dict[str, int | float]:
        return {
            "candidate_dim": self.candidate_dim,
            "frame_dim": self.frame_dim,
            "candidate_count": self.candidate_count,
            "window_length": self.window_length,
            "model_dim": self.model_dim,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "cross_layers": self.cross_layers,
            "dropout": self.dropout,
            "attention_future_lag": self.attention_future_lag,
            "use_translation_fixed_lag": self.use_translation_fixed_lag,
            "translation_fixed_lag_layers": self.translation_fixed_lag_layers,
            "translation_position_scale_m": self.translation_position_scale_m,
            "translation_relative_scale_m": self.translation_relative_scale_m,
            "maximum_translation_refinement_m": self.maximum_translation_refinement_m,
        }

    @staticmethod
    def temporal_attention_mask(
        frames: int,
        future_lag: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Mask keys beyond each query's permitted fixed-lag horizon.

        A boolean MultiheadAttention mask uses True for disallowed pairs. Query
        position i may attend to key position j exactly when j <= i + lag.
        """
        if not 0 <= future_lag < frames:
            raise ValueError("future_lag must be in [0, frames)")
        positions = torch.arange(frames, device=device)
        return positions[None, :] > positions[:, None] + int(future_lag)

    def encode_inputs(
        self,
        candidate_features: torch.Tensor,
        frame_features: torch.Tensor,
        candidate_visual_features: torch.Tensor | None = None,
        frame_visual_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode numeric evidence; visual subclasses override this hook."""
        del candidate_visual_features, frame_visual_features
        candidate = (candidate_features - self.candidate_mean) / self.candidate_std
        frame = (frame_features - self.frame_mean) / self.frame_std
        return self.candidate_encoder(candidate), self.frame_encoder(frame)

    def forward(
        self,
        candidate_features,
        frame_features,
        candidate_valid,
        frame_valid,
        candidate_visual_features=None,
        frame_visual_features=None,
        candidate_poses=None,
        time_s=None,
    ):
        batch, frames, candidates = candidate_features.shape[:3]
        if frames != self.window_length or candidates != self.candidate_count:
            raise ValueError("Input window/candidate shape differs from model configuration")
        encoded, frame_encoded = self.encode_inputs(
            candidate_features,
            frame_features,
            candidate_visual_features,
            frame_visual_features,
        )
        candidate_positions = self.candidate_position(
            torch.arange(candidates, device=encoded.device)
        )[None, None]
        encoded = encoded + candidate_positions
        frame_positions = self.frame_position(
            torch.arange(frames, device=encoded.device)
        )[None]
        frame_encoded = frame_encoded + frame_positions

        flat_candidates = encoded.reshape(batch * frames, candidates, self.model_dim)
        flat_mask = ~candidate_valid.reshape(batch * frames, candidates)
        query = self.local_query.expand(batch * frames, -1, -1)
        local, local_weights = self.local_attention(
            query, flat_candidates, flat_candidates, key_padding_mask=flat_mask
        )
        local = local[:, 0].reshape(batch, frames, self.model_dim)
        local = self.local_norm(local + frame_encoded)

        temporal = self.temporal_query(local) + frame_positions
        temporal_mask_bool = self.temporal_attention_mask(
            frames, self.attention_future_lag, temporal.device
        )
        temporal_mask = torch.zeros(
            temporal_mask_bool.shape, dtype=temporal.dtype, device=temporal.device
        ).masked_fill(temporal_mask_bool, float("-inf"))
        padding_mask = torch.zeros(
            frame_valid.shape, dtype=temporal.dtype, device=temporal.device
        ).masked_fill(~frame_valid, float("-inf"))
        for attention, norm, ffn, ffn_norm in zip(
            self.temporal_attention,
            self.temporal_norm,
            self.temporal_ffn,
            self.temporal_ffn_norm,
        ):
            update, _ = attention(
                temporal,
                local,
                local,
                attn_mask=temporal_mask,
                key_padding_mask=padding_mask,
            )
            temporal = norm(temporal + update)
            temporal = ffn_norm(temporal + ffn(temporal))
        gate = self.fusion_gate(torch.cat((local, temporal), dim=-1))
        fused = self.fusion_norm(gate * temporal + (1.0 - gate) * local)

        expanded = fused[:, :, None].expand(-1, -1, candidates, -1)
        pair = torch.cat((encoded, expanded), dim=-1)
        logits = self.candidate_scorer(pair).squeeze(-1)
        orientation_logits = self.orientation_head(pair).squeeze(-1)
        residual = self.residual_head(pair)
        log_sigma = self.uncertainty_head(pair).clamp(-5.0, 4.0)
        abstention_logits = self.abstention_head(fused).squeeze(-1)
        translation_trust_logits = self.translation_trust_head(pair).squeeze(-1)
        rotation_trust_logits = self.rotation_trust_head(pair).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid, -1e9)
        orientation_logits = orientation_logits.masked_fill(~candidate_valid, -1e9)
        if self.use_translation_fixed_lag:
            if candidate_poses is None or time_s is None:
                raise ValueError(
                    "candidate_poses and time_s are required for translation fixed lag"
                )
            if candidate_poses.shape[:3] != (batch, frames, candidates):
                raise ValueError("candidate_poses do not align with candidate features")
            # Build key/value motion evidence from frame-local predictions.
            # Using temporally fused predictions here would permit indirect
            # leakage: an earlier query could attend to a later key that had
            # itself already attended beyond the earlier query's lag horizon.
            local_expanded = local[:, :, None].expand(-1, -1, candidates, -1)
            local_pair = torch.cat((encoded, local_expanded), dim=-1)
            local_logits = self.candidate_scorer(local_pair).squeeze(-1)
            local_logits = local_logits.masked_fill(~candidate_valid, -1e9)
            local_residual = self.residual_head(local_pair)
            probability = torch.softmax(local_logits, dim=-1)
            initial_candidate_position = (
                candidate_poses[..., :3, 3] + local_residual[..., :3]
            )
            soft_position = (
                probability[..., None] * initial_candidate_position
            ).sum(dim=2)
            baseline_position = candidate_poses[:, :, 0, :3, 3]
            if time_s.shape != (batch, frames):
                raise ValueError("time_s does not align with frame features")
            interval = (time_s[:, 1:] - time_s[:, :-1]).clamp(0.0, 2.0)
            previous_interval = torch.cat(
                (torch.zeros_like(interval[:, :1]), interval), dim=1
            )
            next_interval = torch.cat(
                (interval, torch.zeros_like(interval[:, :1])), dim=1
            )
            motion_features = torch.cat((
                soft_position / self.translation_position_scale_m,
                (soft_position - baseline_position)
                / self.translation_relative_scale_m,
                previous_interval[..., None],
                next_interval[..., None],
            ), dim=-1)
            motion_encoded = self.translation_motion_encoder(motion_features)
            translation_source = self.fusion_norm(local + motion_encoded)
            translation_temporal = self.fusion_norm(fused + motion_encoded)
            for attention, norm, ffn, ffn_norm in zip(
                self.translation_temporal_attention,
                self.translation_temporal_norm,
                self.translation_temporal_ffn,
                self.translation_temporal_ffn_norm,
            ):
                update, _ = attention(
                    translation_temporal,
                    translation_source,
                    translation_source,
                    attn_mask=temporal_mask,
                    key_padding_mask=padding_mask,
                )
                translation_temporal = norm(translation_temporal + update)
                translation_temporal = ffn_norm(
                    translation_temporal + ffn(translation_temporal)
                )
            translation_fixed_lag_delta = (
                self.maximum_translation_refinement_m
                * torch.tanh(
                    self.translation_fixed_lag_delta_head(translation_temporal)
                )
            )
            translation_fixed_lag_gate = torch.sigmoid(
                self.translation_fixed_lag_gate_head(translation_temporal)
            ).squeeze(-1)
            translation_fixed_lag_delta = torch.where(
                frame_valid[..., None],
                translation_fixed_lag_delta,
                torch.zeros_like(translation_fixed_lag_delta),
            )
            translation_fixed_lag_gate = torch.where(
                frame_valid,
                translation_fixed_lag_gate,
                torch.zeros_like(translation_fixed_lag_gate),
            )
        else:
            soft_position = torch.zeros(
                batch, frames, 3, device=fused.device, dtype=fused.dtype
            )
            translation_fixed_lag_delta = torch.zeros_like(soft_position)
            translation_fixed_lag_gate = torch.zeros(
                batch, frames, device=fused.device, dtype=fused.dtype
            )
        return {
            "candidate_logits": logits,
            "orientation_logits": orientation_logits,
            "residual": residual,
            "log_sigma": log_sigma,
            "abstention_logits": abstention_logits,
            "translation_trust_logits": translation_trust_logits,
            "rotation_trust_logits": rotation_trust_logits,
            "translation_fixed_lag_delta": translation_fixed_lag_delta,
            "translation_fixed_lag_gate": translation_fixed_lag_gate,
            "translation_soft_position": soft_position,
            "fusion_gate": gate,
            "local_candidate_attention": local_weights.reshape(batch, frames, candidates),
        }


def load_transformer(path, device: str = "cpu"):
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "rgb_self_recovery_gated_candidate_transformer_v5":
        raise ValueError(f"Unsupported transformer checkpoint: {path}")
    model = GatedCandidateTransformer(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), payload
