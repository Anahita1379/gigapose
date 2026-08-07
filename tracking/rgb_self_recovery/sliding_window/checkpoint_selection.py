"""Validation criteria shared by the isolated CNN and DINO trainers."""

from __future__ import annotations

from collections.abc import Mapping


def checkpoint_selection_values(
    validation_metrics: Mapping[str, float],
    *,
    max_center_px: float,
    max_log_depth: float,
    max_rotation_deg: float,
) -> dict[str, float]:
    """Return lower-is-better checkpoint criteria.

    The joint pose score gives center, depth, and rotation equal influence after
    normalizing them by the correction ranges used by the training targets.
    It intentionally excludes ranking/confidence losses so those terms cannot
    hide a regression in the two geometric validation metrics.
    """

    if max_center_px <= 0 or max_log_depth <= 0 or max_rotation_deg <= 0:
        raise ValueError("Pose-score normalization scales must be positive.")
    center = float(validation_metrics["center_error_crop_px"])
    depth = float(validation_metrics["log_depth_abs_error"])
    rotation = float(validation_metrics["rotation_error_deg"])
    return {
        "loss": float(validation_metrics["loss"]),
        "rotation": rotation,
        "center": center,
        "depth": depth,
        "pose": (
            center / float(max_center_px)
            + depth / float(max_log_depth)
            + rotation / float(max_rotation_deg)
        )
        / 3.0,
    }


CHECKPOINT_NAMES = {
    "loss": "best_loss.ckpt",
    "rotation": "best_rotation.ckpt",
    "center": "best_center.ckpt",
    "depth": "best_depth.ckpt",
    "pose": "best_pose.ckpt",
}
