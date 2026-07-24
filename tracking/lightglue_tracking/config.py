"""Configuration for the isolated LightGlue tracking extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LightGlueTrackingConfig:
    """Runtime policy and geometric gates for ALIKED + LightGlue.

    ``flow_policy`` controls when a LightGlue similarity-transform proposal is
    added.  Association has its own policy because resolving a multi-car
    ambiguity can be valuable even when the matched tracks are currently
    normal.
    """

    enabled: bool = True
    lightglue_root: Path = Path("LightGlue")
    max_num_keypoints: int = 1024
    resize: int = 640
    crop_padding_frac: float = 0.30
    filter_threshold: float = 0.10
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    mixed_precision: bool = False

    ransac_threshold_px: float = 4.0
    min_matches: int = 8
    min_inliers: int = 8
    min_inlier_ratio: float = 0.30
    max_median_reprojection_px: float = 6.0
    minimum_scale: float = 0.50
    maximum_scale: float = 2.00

    association_enabled: bool = True
    association_policy: str = "ambiguous"
    association_weight: float = 0.35
    flow_policy: str = "uncertain_lost"
    lk_fallback: bool = True

    pnp_recovery_enabled: bool = False
    pnp_policy: str = "uncertain_lost"
    pnp_min_points: int = 8
    pnp_min_inliers: int = 6
    pnp_min_inlier_ratio: float = 0.30
    pnp_reprojection_px: float = 6.0
    pnp_iterations: int = 200
    pnp_max_rotation_step_deg: float = 75.0
    pnp_max_translation_step_m: float = 5.0
    template_min_confidence: float = 0.65
    template_normal_only: bool = True

    def validate(self) -> None:
        policies = {"off", "always", "uncertain_lost", "ambiguous"}
        if self.association_policy not in policies:
            raise ValueError(
                f"association_policy must be one of {sorted(policies)}"
            )
        if self.flow_policy not in policies - {"ambiguous"}:
            raise ValueError("flow_policy must be off, always, or uncertain_lost")
        if self.pnp_policy not in policies - {"ambiguous"}:
            raise ValueError("pnp_policy must be off, always, or uncertain_lost")
        if self.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive")
        if self.resize <= 0:
            raise ValueError("resize must be positive")
        if self.crop_padding_frac < 0:
            raise ValueError("crop_padding_frac must be non-negative")
        if not 0 <= self.filter_threshold <= 1:
            raise ValueError("filter_threshold must be in [0, 1]")
        if self.ransac_threshold_px <= 0:
            raise ValueError("ransac_threshold_px must be positive")
        if self.min_matches < 3 or self.min_inliers < 3:
            raise ValueError("similarity estimation requires at least 3 points")
        if not 0 <= self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if self.max_median_reprojection_px <= 0:
            raise ValueError("max_median_reprojection_px must be positive")
        if not 0 < self.minimum_scale <= self.maximum_scale:
            raise ValueError("invalid similarity scale range")
        if self.association_weight < 0:
            raise ValueError("association_weight must be non-negative")
        if self.pnp_min_points < 4 or self.pnp_min_inliers < 4:
            raise ValueError("PnP requires at least 4 points")
        if not 0 <= self.pnp_min_inlier_ratio <= 1:
            raise ValueError("pnp_min_inlier_ratio must be in [0, 1]")
        if self.pnp_reprojection_px <= 0 or self.pnp_iterations <= 0:
            raise ValueError("PnP thresholds and iterations must be positive")
        if not 0 <= self.template_min_confidence <= 1:
            raise ValueError("template_min_confidence must be in [0, 1]")

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["lightglue_root"] = str(self.lightglue_root)
        return values
