"""Configuration dataclasses for the isolated tracking pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoreWeights:
    """Positive error weights used by the deterministic candidate scorer."""

    feature: float = 0.15
    silhouette: float = 2.0
    edge: float = 0.75
    depth: float = 0.50
    bbox: float = 0.75
    motion: float = 0.12


@dataclass
class StateConfig:
    """Adaptive-cascade thresholds and relocalization policy."""

    high_confidence: float = 0.67
    low_confidence: float = 0.35
    lost_after_misses: int = 2
    delete_after_misses: int = 12
    safety_interval: int = 15
    min_score_margin: float = 0.03
    max_track_age_without_global: int = 30


@dataclass
class SameFrameRecoveryConfig:
    """Optional escalation from cheap tracking to broad recovery in one frame."""

    enabled: bool = False
    retry_uncertain: bool = True
    retry_lost: bool = True
    force_global: bool = True


@dataclass
class HypothesisConfig:
    """Breadth of local and global pose candidate generation."""

    top_k_relocalization: int = 5
    top_k_uncertain: int = 3
    add_flip_when_uncertain: bool = True
    flip_axis: str = "z"
    center_offsets_px: tuple[float, ...] = (-24.0, 24.0)
    log_depth_offsets: tuple[float, ...] = (-0.12, 0.12)
    translation_offsets_m: tuple[float, ...] = (-0.20, 0.20)
    yaw_offsets_deg: tuple[float, ...] = (-8.0, 8.0)
    deduplicate_translation_m: float = 0.02
    deduplicate_rotation_deg: float = 1.0


@dataclass
class RefinementConfig:
    """Derivative-free local CAD alignment settings."""

    normal_iterations: int = 1
    uncertain_iterations: int = 2
    lost_iterations: int = 3
    normal_beam: int = 1
    uncertain_beam: int = 2
    lost_beam: int = 3
    center_step_px: float = 12.0
    log_depth_step: float = 0.06
    rotation_step_deg: float = 4.0
    step_decay: float = 0.55
    refine_full_rotation: bool = False


@dataclass
class FlowConfig:
    """Sparse Lucas-Kanade optical-flow measurement settings."""

    enabled: bool = True
    max_corners: int = 160
    quality_level: float = 0.01
    min_distance_px: float = 5.0
    window_size: int = 21
    pyramid_levels: int = 3
    min_points: int = 8
    ransac_threshold_px: float = 3.0
    min_inlier_ratio: float = 0.35


@dataclass
class AssociationConfig:
    """Detection-to-track matching weights and gates."""

    iou_weight: float = 0.65
    center_weight: float = 0.35
    max_center_distance_frac: float = 0.30
    min_iou: float = 0.01
    max_cost: float = 0.95
    identity_enabled: bool = False
    use_external_id: bool = False
    external_id_strict: bool = True
    external_id_weight: float = 0.50
    appearance_weight: float = 0.25
    mask_iou_weight: float = 0.15
    appearance_momentum: float = 0.85


@dataclass
class OcclusionConfig:
    """Optional joint rendering of other cars when scoring a target car."""

    enabled: bool = False


@dataclass
class TemporalConfig:
    """Short sliding-window motion proposal settings."""

    window_size: int = 5
    robust_velocity: bool = True


@dataclass
class TrackerConfig:
    """Complete configuration for adaptive pose tracking."""

    score: ScoreWeights = field(default_factory=ScoreWeights)
    state: StateConfig = field(default_factory=StateConfig)
    same_frame_recovery: SameFrameRecoveryConfig = field(
        default_factory=SameFrameRecoveryConfig
    )
    hypotheses: HypothesisConfig = field(default_factory=HypothesisConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    render_scale: float = 0.5
    render_crop_padding_frac: float = 0.75
    render_crop_min_side_px: int = 96
    min_render_pixels: int = 20
    max_edge_distance_px: float = 50.0
    depth_truncation_m: float = 3.0
    motion_translation_scale_m: float = 1.5
    motion_rotation_scale_deg: float = 35.0
    reset_on_scene_change: bool = True

    def validate(self) -> None:
        weights = asdict(self.score)
        if any(float(value) < 0 for value in weights.values()):
            raise ValueError("All score weights must be non-negative.")
        if not 0 <= self.state.low_confidence < self.state.high_confidence <= 1:
            raise ValueError(
                "Require 0 <= low_confidence < high_confidence <= 1."
            )
        if self.state.safety_interval <= 0:
            raise ValueError("safety_interval must be positive.")
        if self.hypotheses.top_k_relocalization <= 0:
            raise ValueError("top_k_relocalization must be positive.")
        if self.hypotheses.flip_axis not in {"x", "y", "z"}:
            raise ValueError("flip_axis must be x, y, or z.")
        if self.min_render_pixels < 0:
            raise ValueError("min_render_pixels must be non-negative.")
        if self.temporal.window_size < 2:
            raise ValueError("temporal.window_size must be at least 2.")
        if not 0 < self.render_scale <= 1:
            raise ValueError("render_scale must be in (0, 1].")
        if self.render_crop_padding_frac < 0:
            raise ValueError("render_crop_padding_frac must be non-negative.")
        if self.render_crop_min_side_px <= 0:
            raise ValueError("render_crop_min_side_px must be positive.")
        if self.association.identity_enabled:
            identity_weights = (
                self.association.external_id_weight,
                self.association.appearance_weight,
                self.association.mask_iou_weight,
            )
            if any(float(value) < 0 for value in identity_weights):
                raise ValueError(
                    "Identity-association weights must be non-negative."
                )
            if not 0 <= self.association.appearance_momentum < 1:
                raise ValueError("appearance_momentum must be in [0, 1).")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrackerConfig":
        known = {
            "score": ScoreWeights,
            "state": StateConfig,
            "same_frame_recovery": SameFrameRecoveryConfig,
            "hypotheses": HypothesisConfig,
            "refinement": RefinementConfig,
            "flow": FlowConfig,
            "association": AssociationConfig,
            "occlusion": OcclusionConfig,
            "temporal": TemporalConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in values.items():
            if key in known:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a JSON object.")
                if key == "hypotheses":
                    value = dict(value)
                    for tuple_key in (
                        "center_offsets_px",
                        "log_depth_offsets",
                        "translation_offsets_m",
                        "yaw_offsets_deg",
                    ):
                        if tuple_key in value:
                            value[tuple_key] = tuple(value[tuple_key])
                kwargs[key] = known[key](**value)
            else:
                kwargs[key] = value
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path | None) -> "TrackerConfig":
        if path is None:
            config = cls()
            config.validate()
            return config
        return cls.from_dict(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
