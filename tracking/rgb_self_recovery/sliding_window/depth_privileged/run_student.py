"""Shared sequence-aware inference entry point for depth-privileged students."""

from __future__ import annotations

from tracking.rgb_self_recovery.sliding_window import runner
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.inference import (
    load_predictor,
)


VARIANTS = {"auxiliary_depth", "teacher_student", "combined"}


def build_parser(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown depth-privileged student variant: {variant}")
    return runner.build_parser(
        description=(
            f"Sequence-aware RGB-only inference for the {variant} "
            "depth-privileged student"
        )
    )


def run_student(variant: str) -> None:
    """Run an RGB-only student; privileged depth is never loaded at inference."""
    args = build_parser(variant).parse_args()
    runner.load_predictor = load_predictor
    runner.run_tracker(
        args,
        report_format=f"rgb_self_recovery_depth_privileged_{variant}_sequence_v1",
    )
