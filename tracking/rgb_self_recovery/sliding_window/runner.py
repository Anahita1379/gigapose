"""Isolated RGB self-recovery runner with a pluggable candidate-window selector."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from tracking.association import (
    assign_prediction_groups,
    associate_tracks,
    detection_appearance_descriptor,
)
from tracking.config import TrackerConfig
from tracking.flow import SparseFlowPoseUpdater
from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.io import (
    PredictionCSVProvider,
    WebDatasetSequence,
    pose_to_csv_row,
    write_tracking_csv,
)
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery import run as base
from tracking.rgb_self_recovery.inference import RecoveryProposal
from tracking.types import Track, TrackMode

from .mask_io import MaskResolver
from .optimizer import SlidingWindowSelector
from .dino_inference import load_predictor
from .sequence import (
    SequenceStamp,
    discontinuity_reason,
    order_sequence_rows,
)


def build_parser(description="RGB self-recovery with five-frame optimization"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mesh", type=Path)
    p.add_argument("--mesh-scale", type=float, default=1.0)
    p.add_argument("--center-mesh", action="store_true")
    p.add_argument("--association-config", type=Path, default=Path("tracking/configs/improved.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scene-id", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--top-k-gigapose", type=int, default=5)
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--max-candidates", type=int, default=48)
    p.add_argument("--refinement-iterations", type=int, default=2)
    p.add_argument("--global-interval", type=int, default=5)
    p.add_argument("--broad-recovery-confidence", type=float, default=.55)
    p.add_argument("--normal-confidence", type=float, default=.65)
    p.add_argument("--lost-confidence", type=float, default=.25)
    p.add_argument("--allow-flip-hypotheses", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--orientation-gates", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--orientation-gate-mode",
        choices=("off", "hard", "soft"),
        help=(
            "Orientation handling. If omitted, --orientation-gates means hard "
            "and --no-orientation-gates means off."
        ),
    )
    p.add_argument("--soft-orientation-previous-weight", type=float, default=0.25)
    p.add_argument("--soft-orientation-rank0-weight", type=float, default=0.05)
    p.add_argument("--soft-orientation-sigma-deg", type=float, default=30.0)
    p.add_argument(
        "--recovery-abstention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain rank-0 GigaPose when a low-confidence recovery makes an unsupported large jump.",
    )
    p.add_argument("--recovery-min-verifier-confidence", type=float, default=0.05)
    p.add_argument("--recovery-large-translation-m", type=float, default=1.0)
    p.add_argument("--recovery-large-rotation-deg", type=float, default=45.0)
    p.add_argument("--normal-max-rotation-step-deg", type=float, default=30)
    p.add_argument("--uncertain-max-rotation-step-deg", type=float, default=60)
    p.add_argument("--max-rank0-rotation-disagreement-deg", type=float, default=90)
    p.add_argument("--rotation-offsets-deg", default="20,45")
    p.add_argument("--yaw-offsets-deg", default="30,90")
    p.add_argument("--log-depth-offsets", default="-0.35,0.35")
    p.add_argument("--center-offsets-px", default="-48,48")
    p.add_argument("--broad-seeds", type=int, default=4)
    p.add_argument("--delete-after-misses", type=int, default=12)
    p.add_argument("--quality-weight", type=float, default=1)
    p.add_argument("--confidence-weight", type=float, default=.5)
    p.add_argument("--silhouette-weight", type=float, default=1.5)
    p.add_argument("--measurement-weight", type=float, default=.05)
    p.add_argument("--history-weight", type=float, default=.2)
    p.add_argument("--use-external-ids", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-every", type=int, default=10)
    p.add_argument("--overlay-axis-length-m", type=float, default=1)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--mask-dir", type=Path)
    p.add_argument(
        "--mask-pattern",
        default="{scene_id:06d}_{im_id:06d}.png",
        help="Relative mask filename; may include detection_id or external_id.",
    )
    p.add_argument(
        "--mask-fallback", choices=("dataset", "bbox", "render"), default="dataset"
    )

    p.add_argument("--window-size", type=int, default=5)
    p.add_argument(
        "--per-frame",
        action="store_true",
        help=(
            "Disable all cross-frame tracking state. Requires --window-size 1; "
            "every frame starts from fresh GigaPose and broad recovery."
        ),
    )
    p.add_argument("--window-candidates", type=int, default=4)
    p.add_argument("--window-fps", type=float, default=20)
    p.add_argument("--window-unary-weight", type=float, default=1)
    p.add_argument("--window-max-relative-speed-mps", type=float, default=80)
    p.add_argument("--window-speed-sigma-mps", type=float, default=10)
    p.add_argument("--window-acceleration-sigma-mps2", type=float, default=20)
    p.add_argument("--window-rotation-sigma-deg", type=float, default=25)
    p.add_argument("--window-angular-acceleration-sigma-deg-s2", type=float, default=180)
    p.add_argument("--window-anchor-translation-sigma-m", type=float, default=.5)
    p.add_argument("--window-anchor-rotation-sigma-deg", type=float, default=8)
    p.add_argument("--window-continuous-refinement", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--sequence-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Order frames by source_run/camera/source_frame and reset temporal "
            "state at sequence boundaries or gaps."
        ),
    )
    p.add_argument(
        "--sequence-max-frame-gap",
        type=int,
        default=1,
        help="Reset when consecutive source_frame values differ by more than this; 0 disables this check.",
    )
    p.add_argument(
        "--sequence-max-time-gap-s",
        type=float,
        default=0.5,
        help="Reset when consecutive timestamps differ by more than this; 0 disables this check.",
    )
    p.add_argument(
        "--window-max-visual-cost-increase",
        type=float,
        default=0.02,
        help="Reject a smoothed endpoint if its rescored visual cost rises by more than this.",
    )
    p.add_argument(
        "--window-max-iou-drop",
        type=float,
        default=0.02,
        help="Reject a smoothed endpoint if its rendered-mask IoU falls by more than this.",
    )
    p.add_argument(
        "--state-iou-confidence-weight",
        type=float,
        default=0.0,
        help=(
            "IoU-derived floor for tracking-state confidence. A value of 0.5 "
            "keeps an aligned low-verifier pose uncertain rather than lost."
        ),
    )
    p.add_argument(
        "--state-iou-min-verifier-confidence",
        type=float,
        default=0.05,
        help="Minimum network confidence required before IoU may raise tracker-state confidence.",
    )
    return p


def _mesh(args):
    return base.resolve_mesh(args)


def _update_appearance(track, frame, detection, config):
    if not config.identity_enabled:
        return
    descriptor = detection_appearance_descriptor(frame.image, detection)
    if descriptor is None:
        return
    if track.appearance_descriptor is None:
        track.appearance_descriptor = descriptor
    else:
        value = config.appearance_momentum * track.appearance_descriptor + (
            1 - config.appearance_momentum
        ) * descriptor
        track.appearance_descriptor = value / max(float(value.sum()), 1e-9)


def _validate(args):
    if args.global_interval <= 0 or args.overlay_every <= 0:
        raise ValueError("intervals must be positive")
    for name in (
        "window_speed_sigma_mps",
        "window_acceleration_sigma_mps2",
        "window_rotation_sigma_deg",
        "window_angular_acceleration_sigma_deg_s2",
        "window_anchor_translation_sigma_m",
        "window_anchor_rotation_sigma_deg",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("window_max_visual_cost_increase", "window_max_iou_drop"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "soft_orientation_previous_weight",
        "soft_orientation_rank0_weight",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.soft_orientation_sigma_deg <= 0:
        raise ValueError("--soft-orientation-sigma-deg must be positive")
    for name in (
        "recovery_large_translation_m",
        "recovery_large_rotation_deg",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "recovery_min_verifier_confidence",
        "state_iou_min_verifier_confidence",
    ):
        if not 0 <= float(getattr(args, name)) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if not 0 <= args.state_iou_confidence_weight <= 1:
        raise ValueError("--state-iou-confidence-weight must be in [0,1]")
    if args.sequence_max_frame_gap < 0:
        raise ValueError("--sequence-max-frame-gap must be non-negative")
    if args.sequence_max_time_gap_s < 0:
        raise ValueError("--sequence-max-time-gap-s must be non-negative")
    if args.per_frame and args.window_size != 1:
        raise ValueError("--per-frame requires --window-size 1")


def _resolved_orientation_mode(args) -> str:
    if args.orientation_gate_mode is not None:
        return str(args.orientation_gate_mode)
    return "hard" if args.orientation_gates else "off"


def _soft_orientation_cost(angle_deg: float, free_deg: float, sigma_deg: float) -> float:
    if not math.isfinite(angle_deg):
        return 0.0
    normalized = max(0.0, angle_deg - free_deg) / sigma_deg
    return float(2.0 * (math.sqrt(1.0 + normalized * normalized) - 1.0))


def _select_with_soft_orientation(ranked, *, track, rank0_pose, args):
    """Rerank softly without forbidding recovery from a rank-0 front/rear flip."""
    previous_pose = None if track is None else np.asarray(track.pose, dtype=float)
    if track is None or track.mode == TrackMode.LOST:
        previous_limit = None
    elif track.mode == TrackMode.NORMAL:
        previous_limit = float(args.normal_max_rotation_step_deg)
    else:
        previous_limit = float(args.uncertain_max_rotation_step_deg)
    rank0_limit = float(args.max_rank0_rotation_disagreement_deg)
    sigma = float(args.soft_orientation_sigma_deg)
    rescored = []
    details = []
    for index, result in enumerate(ranked):
        previous_angle = (
            float("nan")
            if previous_pose is None
            else rotation_error_deg(result.pose, previous_pose)
        )
        rank0_angle = (
            float("nan")
            if rank0_pose is None
            else rotation_error_deg(result.pose, rank0_pose)
        )
        previous_cost = (
            0.0
            if previous_limit is None
            else _soft_orientation_cost(previous_angle, previous_limit, sigma)
        )
        rank0_cost = _soft_orientation_cost(rank0_angle, rank0_limit, sigma)
        penalty = (
            args.soft_orientation_previous_weight * previous_cost
            + args.soft_orientation_rank0_weight * rank0_cost
        )
        rescored.append(
            replace(result, total_error=float(result.total_error + penalty))
        )
        details.append((index, previous_angle, rank0_angle, penalty))
    order = sorted(range(len(rescored)), key=lambda index: rescored[index].total_error)
    ordered = [rescored[index] for index in order]
    selected_index = order[0]
    _, selected_previous, selected_rank0, selected_penalty = details[selected_index]
    diagnostics = {
        "orientation_gate_enabled": 1,
        "orientation_gate_mode": "soft",
        "orientation_gate_triggered": int(selected_index != 0),
        "orientation_gate_fallback": 0,
        "orientation_gate_anchor": "soft_previous_and_rank0",
        "orientation_candidates_rejected": 0,
        "orientation_selected_rank_after_gate": selected_index,
        "rotation_from_previous_before_gate_deg": details[0][1],
        "rotation_from_rank0_before_gate_deg": details[0][2],
        "rotation_from_previous_after_gate_deg": selected_previous,
        "rotation_from_rank0_after_gate_deg": selected_rank0,
        "orientation_soft_penalty": selected_penalty,
        "orientation_soft_previous_weight": args.soft_orientation_previous_weight,
        "orientation_soft_rank0_weight": args.soft_orientation_rank0_weight,
    }
    return ordered[0], ordered, diagnostics


def _state_confidence(result, args) -> tuple[float, float, float]:
    raw = float(
        np.clip(result.confidence * (0.5 + 0.5 * result.silhouette_iou), 0, 1)
    )
    iou_floor = 0.0
    if result.confidence >= args.state_iou_min_verifier_confidence:
        iou_floor = float(
            np.clip(args.state_iou_confidence_weight * result.silhouette_iou, 0, 1)
        )
    return max(raw, iou_floor), raw, iou_floor


def _apply_recovery_abstention(selected, ranked, baseline, track, args):
    """Avoid replacing GigaPose with an unsupported, untrusted large jump."""
    diagnostics = {
        "recovery_abstention_enabled": int(args.recovery_abstention),
        "recovery_abstention_triggered": 0,
        "recovery_abstention_reason": "",
        "recovery_candidate_verifier_confidence": float(selected.confidence),
        "recovery_candidate_from_rank0_translation_m": float("nan"),
        "recovery_candidate_from_rank0_rotation_deg": float("nan"),
        "recovery_temporally_supported": 0,
        "rank0_baseline_verifier_confidence": float("nan"),
        "rank0_baseline_silhouette_iou": float("nan"),
        "rank0_baseline_total_error": float("nan"),
    }
    if baseline is None:
        diagnostics["recovery_abstention_reason"] = "no_rank0_baseline"
        return selected, ranked, diagnostics

    translation = translation_error_m(selected.pose, baseline.pose)
    rotation = rotation_error_deg(selected.pose, baseline.pose)
    diagnostics.update(
        recovery_candidate_from_rank0_translation_m=translation,
        recovery_candidate_from_rank0_rotation_deg=rotation,
        rank0_baseline_verifier_confidence=float(baseline.confidence),
        rank0_baseline_silhouette_iou=float(baseline.silhouette_iou),
        rank0_baseline_total_error=float(baseline.total_error),
    )
    low_confidence = (
        selected.confidence < args.recovery_min_verifier_confidence
    )
    large_jump = (
        translation > args.recovery_large_translation_m
        or rotation > args.recovery_large_rotation_deg
    )
    temporally_supported = False
    if track is not None and track.mode != TrackMode.LOST:
        previous_translation = translation_error_m(selected.pose, track.pose)
        previous_rotation = rotation_error_deg(selected.pose, track.pose)
        rotation_limit = (
            args.normal_max_rotation_step_deg
            if track.mode == TrackMode.NORMAL
            else args.uncertain_max_rotation_step_deg
        )
        translation_limit = max(
            1.0,
            args.window_max_relative_speed_mps / max(args.window_fps, 1e-6),
        )
        temporally_supported = (
            previous_translation <= translation_limit
            and previous_rotation <= rotation_limit
        )
        diagnostics.update(
            recovery_candidate_from_previous_translation_m=previous_translation,
            recovery_candidate_from_previous_rotation_deg=previous_rotation,
            recovery_temporal_translation_limit_m=translation_limit,
            recovery_temporal_rotation_limit_deg=rotation_limit,
        )
    diagnostics["recovery_temporally_supported"] = int(temporally_supported)
    if (
        args.recovery_abstention
        and low_confidence
        and large_jump
        and not temporally_supported
    ):
        diagnostics.update(
            recovery_abstention_triggered=1,
            recovery_abstention_reason="low_confidence_unsupported_large_jump",
        )
        fallback = replace(
            baseline,
            source=f"{baseline.source}|recovery_abstention_rank0",
        )
        return fallback, [fallback], diagnostics
    return selected, ranked, diagnostics


def _visual_cost(result, args) -> float:
    return float(
        args.quality_weight * result.quality
        + args.confidence_weight * (1.0 - result.confidence)
        + args.silhouette_weight * (1.0 - result.silhouette_iou)
    )


def _rescore_window_endpoint(
    predictor,
    renderer,
    frame,
    detection,
    occluder,
    smoothed,
    anchor,
    args,
    diagnostics,
):
    """Accept continuous smoothing only after fresh image/CAD verification."""
    diagnostics.update(
        window_visual_rescored=0,
        window_continuous_accepted=0,
        window_smoothing_translation_m=translation_error_m(
            smoothed.pose, anchor.pose
        ),
        window_smoothing_rotation_deg=rotation_error_deg(
            smoothed.pose, anchor.pose
        ),
        window_visual_cost_before=_visual_cost(anchor, args),
        window_visual_cost_after=_visual_cost(anchor, args),
        window_visual_cost_increase=0.0,
        window_iou_before=float(anchor.silhouette_iou),
        window_iou_after=float(anchor.silhouette_iou),
        window_iou_drop=0.0,
    )
    if not diagnostics.get("window_continuous_refinement"):
        selected = replace(
            anchor,
            source=f"{anchor.source}|window{diagnostics['window_size_used']}",
        )
        return selected

    rescored = predictor.refine_and_score(
        renderer,
        frame,
        detection,
        [RecoveryProposal(smoothed.pose.copy(), smoothed.source, 1.0, 0.0)],
        occluder_mask=occluder,
        iterations=0,
        quality_weight=args.quality_weight,
        confidence_weight=args.confidence_weight,
        silhouette_weight=args.silhouette_weight,
        measurement_weight=0.0,
        prior_weight=0.0,
    )
    diagnostics["window_visual_rescored"] = 1
    if not rescored:
        return replace(anchor, source=f"{smoothed.source}|continuous_rejected_no_score")

    verified = rescored[0]
    before = _visual_cost(anchor, args)
    after = _visual_cost(verified, args)
    cost_increase = after - before
    iou_drop = float(anchor.silhouette_iou - verified.silhouette_iou)
    accepted = (
        cost_increase <= args.window_max_visual_cost_increase
        and iou_drop <= args.window_max_iou_drop
    )
    diagnostics.update(
        window_continuous_accepted=int(accepted),
        window_visual_cost_after=after,
        window_visual_cost_increase=cost_increase,
        window_iou_after=float(verified.silhouette_iou),
        window_iou_drop=iou_drop,
    )
    if not accepted:
        return replace(anchor, source=f"{smoothed.source}|continuous_rejected_visual")

    return replace(
        smoothed,
        confidence=verified.confidence,
        quality=verified.quality,
        silhouette_iou=verified.silhouette_iou,
        total_error=float(anchor.total_error + cost_increase),
        rendered_mask_crop=verified.rendered_mask_crop,
        source=f"{smoothed.source}|continuous_verified",
    )


def run_tracker(args, selector=None, report_format="rgb_self_recovery_sliding_window_v1"):
    _validate(args)
    orientation_mode = _resolved_orientation_mode(args)
    args.orientation_gate_mode_resolved = orientation_mode
    args.orientation_gates = orientation_mode == "hard"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = TrackerConfig.load(args.association_config)
    config.association.use_external_id = args.use_external_ids
    config.association.identity_enabled = True
    source = WebDatasetSequence(
        args.dataset_dir, args.split, load_depth=False, scene_id=args.scene_id
    )
    if args.sequence_aware:
        source.rows = list(order_sequence_rows(source.rows))
    predictions = PredictionCSVProvider(
        args.predictions, args.prediction_translation_unit, "gigapose"
    )
    renderer = CADRenderer(_mesh(args), mesh_scale=args.mesh_scale, center_mesh=args.center_mesh)
    predictor = load_predictor(args.checkpoint, args.device)
    flow = SparseFlowPoseUpdater(config.flow)
    masks = MaskResolver(args.mask_dir, args.mask_pattern, args.mask_fallback)
    selector = selector or SlidingWindowSelector(args)

    tracks, banks = {}, {}
    next_track_id = 0
    current_scene = None
    output_rows, diagnostics = [], []
    states = Counter()
    frames_processed = 0
    previous_sequence_stamp = None
    sequence_reset_reasons = Counter()
    per_frame_reset_count = 0
    started = time.perf_counter()
    try:
        for frame_index, frame in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            sequence_stamp = SequenceStamp.from_frame(frame)
            sequence_reset_reason = None
            if args.sequence_aware:
                sequence_reset_reason = discontinuity_reason(
                    previous_sequence_stamp,
                    sequence_stamp,
                    max_frame_gap=args.sequence_max_frame_gap,
                    max_time_gap_s=args.sequence_max_time_gap_s,
                )
            elif current_scene is not None and frame.scene_id != current_scene:
                sequence_reset_reason = "scene_changed"
            per_frame_reset = bool(args.per_frame and previous_sequence_stamp is not None)
            reset_reason = "per_frame" if per_frame_reset else sequence_reset_reason
            if reset_reason is not None:
                tracks.clear()
                banks.clear()
                if hasattr(selector, "clear"):
                    selector.clear()
                if per_frame_reset:
                    per_frame_reset_count += 1
                else:
                    sequence_reset_reasons[reset_reason] += 1
            sequence_start = int(
                previous_sequence_stamp is None or sequence_reset_reason is not None
            )
            previous_sequence_stamp = sequence_stamp
            current_scene = frame.scene_id
            frames_processed += 1
            mask_statuses = masks.apply_directory(frame)
            active = list(tracks.values())
            matches, unmatched_tracks, unmatched_detections = associate_tracks(
                active, frame.detections, frame.image.shape[:2], config.association, frame.image
            )
            fresh_groups = predictions.groups_for_frame(frame.scene_id, frame.im_id)
            fresh_by_detection = assign_prediction_groups(
                fresh_groups, frame.detections, frame.K, frame.image.shape[:2]
            )
            track_by_detection = {
                detection_index: active[track_index]
                for track_index, detection_index in matches
            }
            process_indices = [index for _, index in matches] + unmatched_detections
            for detection_index in process_indices:
                detection = frame.detections[detection_index]
                track = track_by_detection.get(detection_index)
                track_id = next_track_id if track is None else track.track_id
                fresh = fresh_by_detection.get(detection_index, [])
                mask_source = masks.maybe_render_fallback(
                    frame, detection, fresh, renderer, mask_statuses
                )
                bank = [] if track is None else banks.get(track.track_id, [])
                include_global = (
                    track is None
                    or track.mode != TrackMode.NORMAL
                    or track.age % args.global_interval == 0
                )
                proposals = base.make_proposals(
                    track, bank, fresh, frame, detection, flow, include_global, args
                )
                if not proposals:
                    continue
                occluder = np.zeros_like(detection.mask, dtype=bool)
                for other_index, other in enumerate(frame.detections):
                    if other_index != detection_index:
                        occluder |= np.asarray(other.mask, dtype=bool)
                broad_used = include_global
                if include_global:
                    proposals = base.broad_recovery_proposals(proposals, detection, frame.K, args)
                ranked = predictor.refine_and_score(
                    renderer, frame, detection, proposals,
                    occluder_mask=occluder,
                    iterations=args.refinement_iterations,
                    quality_weight=args.quality_weight,
                    confidence_weight=args.confidence_weight,
                    silhouette_weight=args.silhouette_weight,
                    measurement_weight=args.measurement_weight,
                    prior_weight=args.history_weight,
                )
                preliminary = (
                    float(np.clip(ranked[0].confidence * (.5 + .5 * ranked[0].silhouette_iou), 0, 1))
                    if ranked else 0
                )
                if not broad_used and preliminary < args.broad_recovery_confidence:
                    proposals = base.broad_recovery_proposals(
                        base.make_proposals(track, bank, fresh, frame, detection, flow, True, args),
                        detection, frame.K, args,
                    )
                    ranked = predictor.refine_and_score(
                        renderer, frame, detection, proposals,
                        occluder_mask=occluder,
                        iterations=args.refinement_iterations,
                        quality_weight=args.quality_weight,
                        confidence_weight=args.confidence_weight,
                        silhouette_weight=args.silhouette_weight,
                        measurement_weight=args.measurement_weight,
                        prior_weight=args.history_weight,
                    )
                    broad_used = True
                if not ranked:
                    continue
                rank0_baseline = None
                if fresh:
                    baseline_results = predictor.refine_and_score(
                        renderer,
                        frame,
                        detection,
                        [
                            RecoveryProposal(
                                fresh[0].pose.copy(),
                                "gigapose_rank0_baseline",
                                fresh[0].measurement_score,
                                0.0,
                            )
                        ],
                        occluder_mask=occluder,
                        iterations=0,
                        quality_weight=args.quality_weight,
                        confidence_weight=args.confidence_weight,
                        silhouette_weight=args.silhouette_weight,
                        measurement_weight=args.measurement_weight,
                        prior_weight=0.0,
                    )
                    if baseline_results:
                        rank0_baseline = baseline_results[0]
                rank0_pose = fresh[0].pose if fresh else None
                if orientation_mode == "soft":
                    gated_best, gated, orientation = _select_with_soft_orientation(
                        ranked, track=track, rank0_pose=rank0_pose, args=args
                    )
                else:
                    gated_best, gated, orientation = base.select_with_orientation_gates(
                        ranked, track=track, rank0_pose=rank0_pose, args=args
                    )
                    orientation["orientation_gate_mode"] = orientation_mode
                if orientation.get("orientation_gate_fallback"):
                    fallback_before_rescore = gated_best
                    rescored_fallback = predictor.refine_and_score(
                        renderer,
                        frame,
                        detection,
                        [
                            RecoveryProposal(
                                gated_best.pose.copy(),
                                gated_best.source,
                                gated_best.confidence,
                                gated_best.total_error,
                            )
                        ],
                        occluder_mask=occluder,
                        iterations=0,
                        quality_weight=args.quality_weight,
                        confidence_weight=args.confidence_weight,
                        silhouette_weight=args.silhouette_weight,
                        measurement_weight=args.measurement_weight,
                        prior_weight=args.history_weight,
                    )
                    if rescored_fallback:
                        gated_best = replace(
                            rescored_fallback[0],
                            delta_center_crop_px=(
                                fallback_before_rescore.delta_center_crop_px
                            ),
                            delta_log_depth=fallback_before_rescore.delta_log_depth,
                            delta_rotation_rad=(
                                fallback_before_rescore.delta_rotation_rad
                            ),
                        )
                        gated = [gated_best]
                gated_best, gated, abstention = _apply_recovery_abstention(
                    gated_best,
                    gated,
                    rank0_baseline,
                    track,
                    args,
                )
                if abstention["recovery_abstention_triggered"]:
                    orientation.update(
                        orientation_gate_anchor="recovery_abstention_rank0",
                        rotation_from_previous_after_gate_deg=(
                            float("nan")
                            if track is None
                            else rotation_error_deg(gated_best.pose, track.pose)
                        ),
                        rotation_from_rank0_after_gate_deg=0.0,
                    )
                best, window_ranked, window_diag = selector.select(
                    track_id, frame, gated[: max(args.beam_size, args.window_candidates)]
                )
                anchor = selector.selected_anchor(track_id)
                if anchor is None:
                    raise RuntimeError("Sliding-window selector lost its visual anchor")
                best = _rescore_window_endpoint(
                    predictor,
                    renderer,
                    frame,
                    detection,
                    occluder,
                    best,
                    anchor,
                    args,
                    window_diag,
                )
                window_ranked = [best, *window_ranked[1:]]
                confidence, raw_confidence, iou_confidence_floor = _state_confidence(
                    best, args
                )
                if (
                    orientation_mode == "hard"
                    and orientation.get("orientation_gate_triggered")
                ):
                    confidence = min(confidence, max(args.lost_confidence, args.normal_confidence - 1e-6))
                mode = (
                    TrackMode.LOST
                    if abstention["recovery_abstention_triggered"]
                    else base.mode_from_confidence(confidence, args)
                )
                if track is None:
                    track = Track(
                        track_id=track_id, obj_id=detection.obj_id, pose=best.pose.copy(),
                        previous_pose=None, bbox_xywh=detection.bbox_xywh.copy(),
                        confidence=confidence, mode=mode, source=best.source,
                        scene_id=frame.scene_id, im_id=frame.im_id,
                        external_id=detection.external_id,
                    )
                    next_track_id += 1
                else:
                    track.previous_pose = track.pose.copy()
                    track.pose = best.pose.copy()
                    track.bbox_xywh = detection.bbox_xywh.copy()
                    track.confidence = confidence
                    track.mode = mode
                    track.source = best.source
                    track.im_id = frame.im_id
                    track.age += 1
                    track.missed = 0
                track.previous_gray = frame.gray.copy()
                track.previous_mask = detection.mask.copy()
                track.history.append(track.pose.copy())
                if len(track.history) > 32:
                    del track.history[:-32]
                _update_appearance(track, frame, detection, config.association)
                tracks[track.track_id] = track
                banks[track.track_id] = [
                    RecoveryProposal(item.pose.copy(), item.source, item.confidence, item.total_error)
                    for item in window_ranked[: args.beam_size]
                ]
                states[mode.value] += 1
                output_rows.append(pose_to_csv_row(
                    scene_id=frame.scene_id, im_id=frame.im_id, track_id=track.track_id,
                    obj_id=track.obj_id, confidence=confidence, pose_m=best.pose,
                    mode=mode.value, source=best.source, elapsed_s=0,
                ))
                row = {
                    "scene_id": frame.scene_id, "im_id": frame.im_id,
                    "track_id": track.track_id, "detection_id": detection.detection_id,
                    "mode": mode.value, "source": best.source, "mask_source": mask_source,
                    "confidence": confidence,
                    "state_confidence": confidence,
                    "raw_confidence": raw_confidence,
                    "iou_confidence_floor": iou_confidence_floor,
                    "verifier_confidence": best.confidence,
                    "quality": best.quality, "silhouette_iou": best.silhouette_iou,
                    "total_error": best.total_error, "broad_recovery": int(broad_used),
                    "sequence_run": sequence_stamp.key[0],
                    "sequence_camera": sequence_stamp.key[1],
                    "sequence_source_frame": sequence_stamp.source_frame,
                    "sequence_time_s": sequence_stamp.time_s,
                    "sequence_start": sequence_start,
                    "sequence_reset": int(sequence_reset_reason is not None),
                    "sequence_reset_reason": sequence_reset_reason or "",
                    "temporal_state_reset": int(reset_reason is not None),
                    "temporal_state_reset_reason": reset_reason or "",
                    "candidates": len(ranked),
                    "delta_center_crop_px": float(np.linalg.norm(best.delta_center_crop_px)),
                    "delta_log_depth": best.delta_log_depth,
                    "delta_rotation_deg": math.degrees(float(np.linalg.norm(best.delta_rotation_rad))),
                    "explicit_flip_in_source": int("flip180" in best.source),
                    **orientation,
                    **abstention,
                    **window_diag,
                }
                diagnostics.append(row)
                if args.save_overlays and frame_index % args.overlay_every == 0:
                    base.draw_overlay(
                        frame, detection, track, best, fresh[0].pose if fresh else None,
                        renderer,
                        args.output_dir / "overlays" / f"{frame.scene_id:06d}_{frame.im_id:06d}_t{track.track_id}.jpg",
                        orientation_diagnostics=orientation,
                        axis_length_m=args.overlay_axis_length_m,
                    )
            for track_index in unmatched_tracks:
                track = active[track_index]
                track.missed += 1
                track.mode = TrackMode.LOST
                if track.missed >= args.delete_after_misses:
                    tracks.pop(track.track_id, None)
                    banks.pop(track.track_id, None)
                    selector.remove(track.track_id)
            if frames_processed % 25 == 0:
                print(f"processed {frames_processed}/{len(source)} frames", flush=True)
    finally:
        renderer.close()

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    if diagnostics:
        fields = []
        for row in diagnostics:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (args.output_dir / "candidate_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostics)
    report = {
        "format": report_format,
        "predictions": str(args.predictions), "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint), "mesh": str(_mesh(args)),
        "uses_observed_depth": False, "frames_processed": frames_processed,
        "tracked_instances": len(output_rows), "state_counts": dict(states),
        "window_size": args.window_size,
        "selection_mode": (
            "independent_per_frame"
            if args.per_frame
            else "single_frame_selector_with_tracker_history"
            if args.window_size == 1
            else "sliding_window"
        ),
        "per_frame": args.per_frame,
        "per_frame_reset_count": per_frame_reset_count,
        "sequence_aware": args.sequence_aware,
        "sequence_reset_count": int(sum(sequence_reset_reasons.values())),
        "sequence_reset_reasons": dict(sequence_reset_reasons),
        "mask_dir": str(args.mask_dir) if args.mask_dir else None,
        "mask_fallback": args.mask_fallback, "elapsed_s": time.perf_counter() - started,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    rescored_rows = [
        row for row in diagnostics if int(row.get("window_visual_rescored", 0))
    ]
    report.update(
        recovery_abstention_count=sum(
            int(row.get("recovery_abstention_triggered", 0))
            for row in diagnostics
        ),
        state_iou_confidence_floor_applied_count=sum(
            float(row.get("iou_confidence_floor", 0.0)) > 0.0
            for row in diagnostics
        ),
        window_continuous_rescored_count=len(rescored_rows),
        window_continuous_accepted_count=sum(
            int(row.get("window_continuous_accepted", 0)) for row in rescored_rows
        ),
        window_continuous_rejected_count=sum(
            not int(row.get("window_continuous_accepted", 0))
            for row in rescored_rows
        ),
    )
    if hasattr(selector, "report"):
        report.update(selector.report())
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    args = build_parser().parse_args()
    run_tracker(args)
