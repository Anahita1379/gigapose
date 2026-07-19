"""Optional learned pose-recovery head and its shared feature definition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import math

from tracking.geometry import apply_direct_residual
from tracking.types import EvaluatedHypothesis, PoseHypothesis, ScoreBreakdown


FEATURE_NAMES = (
    "feature_error",
    "silhouette_error",
    "edge_error",
    "depth_error",
    "bbox_error",
    "motion_error",
    "center_dx_frac",
    "center_dy_frac",
    "log_area_ratio",
    "log_depth_ratio",
    "measurement_score",
)


def breakdown_to_features(
    score: ScoreBreakdown, measurement_score: float
) -> np.ndarray:
    return np.asarray(
        [
            score.feature_error,
            score.silhouette_error,
            score.edge_error,
            score.depth_error,
            score.bbox_error,
            score.motion_error,
            score.center_dx_frac,
            score.center_dy_frac,
            score.log_area_ratio,
            score.log_depth_ratio,
            measurement_score,
        ],
        dtype=np.float32,
    )


class PoseRecoveryHead(nn.Module):
    """Predict SE(3) residual, confidence logit, and scalar pose quality."""

    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.network(features)
        return {
            "rotation_raw": raw[..., :3],
            "translation_raw": raw[..., 3:6],
            "confidence_logit": raw[..., 6],
            "quality": raw[..., 7],
        }


def bounded_vector(raw: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Map an unconstrained vector to an open ball with radius ``max_norm``."""

    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    direction = raw / torch.clamp(norm, min=1e-8)
    magnitude = torch.tanh(norm) * float(max_norm)
    return direction * magnitude


class RecoveryPredictor:
    def __init__(
        self,
        model: PoseRecoveryHead,
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        max_rotation_rad: float,
        max_translation_m: float,
        device: str,
    ):
        self.model = model.to(device).eval()
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-6)
        self.max_rotation_rad = float(max_rotation_rad)
        self.max_translation_m = float(max_translation_m)
        self.device = device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "RecoveryPredictor":
        payload: dict[str, Any] = torch.load(Path(path), map_location=device)
        names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if names != FEATURE_NAMES:
            raise ValueError(
                "Recovery checkpoint feature definition does not match this package."
            )
        model = PoseRecoveryHead(
            input_dim=len(FEATURE_NAMES),
            hidden_dim=int(payload["model_config"]["hidden_dim"]),
        )
        model.load_state_dict(payload["model_state"])
        return cls(
            model,
            feature_mean=np.asarray(payload["feature_mean"]),
            feature_std=np.asarray(payload["feature_std"]),
            max_rotation_rad=float(payload["model_config"]["max_rotation_rad"]),
            max_translation_m=float(payload["model_config"]["max_translation_m"]),
            device=device,
        )

    @torch.no_grad()
    def correct(
        self, evaluated: EvaluatedHypothesis
    ) -> tuple[PoseHypothesis, float, float]:
        features = breakdown_to_features(
            evaluated.score, evaluated.hypothesis.measurement_score
        )
        normalized = (features - self.feature_mean) / self.feature_std
        tensor = torch.as_tensor(normalized, device=self.device).unsqueeze(0)
        output = self.model(tensor)
        rotation = bounded_vector(
            output["rotation_raw"], self.max_rotation_rad
        )[0].cpu().numpy()
        translation = bounded_vector(
            output["translation_raw"], self.max_translation_m
        )[0].cpu().numpy()
        confidence = float(torch.sigmoid(output["confidence_logit"][0]).item())
        predicted_log_quality = float(output["quality"][0].item())
        quality_score = math.exp(-max(predicted_log_quality, 0.0))
        corrected = evaluated.hypothesis.copy(
            source=f"{evaluated.hypothesis.source}|learned_recovery"
        )
        corrected.pose = apply_direct_residual(
            corrected.pose, rotation, translation
        )
        corrected.measurement_score = (
            corrected.measurement_score + confidence + quality_score
        ) / 3.0
        corrected.metadata.update(
            {
                "recovery_confidence": confidence,
                "predicted_log_quality": predicted_log_quality,
            }
        )
        return corrected, confidence, predicted_log_quality
