"""Standalone RGB-only, render-aware, multi-hypothesis 6D pose tracking."""

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
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from tracking.association import (
    assign_prediction_groups,
    associate_tracks,
    detection_appearance_descriptor,
)
from tracking.config import TrackerConfig
from tracking.flow import SparseFlowPoseUpdater
from tracking.geometry import (
    flipped_pose,
    project_center,
    propagate_pose_window,
    rotate_pose,
    rotation_error_deg,
    shift_projected_center,
    translation_error_m,
)
from tracking.io import (
    PredictionCSVProvider,
    WebDatasetSequence,
    pose_to_csv_row,
    write_tracking_csv,
)
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery.inference import (
    RGBSelfRecoveryPredictor,
    RecoveryProposal,
    RecoveryResult,
)
from tracking.types import Track, TrackMode


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--center-mesh", action="store_true")
    parser.add_argument("--association-config", type=Path, default=Path("tracking/configs/improved.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--top-k-gigapose", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--refinement-iterations", type=int, default=2)
    parser.add_argument("--global-interval", type=int, default=5)
    parser.add_argument("--broad-recovery-confidence", type=float, default=0.55)
    parser.add_argument("--normal-confidence", type=float, default=0.65)
    parser.add_argument("--lost-confidence", type=float, default=0.25)
    parser.add_argument(
        "--allow-flip-hypotheses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow explicit object-Z 180-degree proposals during broad recovery. "
            "Use --no-allow-flip-hypotheses for front/rear-safe tracking."
        ),
    )
    parser.add_argument(
        "--orientation-gates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Filter candidates using previous-pose and current GigaPose-rank-0 "
            "rotation anchors. Disabled by default for backward compatibility."
        ),
    )
    parser.add_argument("--normal-max-rotation-step-deg", type=float, default=30.0)
    parser.add_argument(
        "--uncertain-max-rotation-step-deg", type=float, default=60.0
    )
    parser.add_argument(
        "--max-rank0-rotation-disagreement-deg", type=float, default=90.0
    )
    parser.add_argument("--rotation-offsets-deg", default="20,45")
    parser.add_argument("--yaw-offsets-deg", default="30,90")
    parser.add_argument("--log-depth-offsets", default="-0.35,0.35")
    parser.add_argument("--center-offsets-px", default="-48,48")
    parser.add_argument("--broad-seeds", type=int, default=4)
    parser.add_argument("--delete-after-misses", type=int, default=12)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--silhouette-weight", type=float, default=1.5)
    parser.add_argument("--measurement-weight", type=float, default=0.05)
    parser.add_argument("--history-weight", type=float, default=0.20)
    parser.add_argument("--use-external-ids", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-every", type=int, default=10)
    parser.add_argument("--overlay-axis-length-m", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_mesh(args: argparse.Namespace) -> Path:
    if args.mesh is not None:
        return args.mesh
    candidate = args.dataset_dir / "models" / "obj_000001.ply"
    if candidate.is_file():
        return candidate
    fallback = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Pass --mesh; no default CAD was found.")


def add_unique(
    output: list[RecoveryProposal],
    proposal: RecoveryProposal,
    max_candidates: int,
) -> None:
    if len(output) >= max_candidates or proposal.pose[2, 3] <= 1e-5:
        return
    duplicate = any(
        translation_error_m(proposal.pose, existing.pose) <= 0.015
        and rotation_error_deg(proposal.pose, existing.pose) <= 1.0
        for existing in output
    )
    if not duplicate:
        output.append(proposal)


def broad_recovery_proposals(
    seeds: list[RecoveryProposal],
    detection,
    K: np.ndarray,
    args: argparse.Namespace,
) -> list[RecoveryProposal]:
    output = list(seeds)
    rotation_offsets = parse_float_list(args.rotation_offsets_deg)
    yaw_offsets = parse_float_list(args.yaw_offsets_deg)
    log_depth_offsets = parse_float_list(args.log_depth_offsets)
    center_offsets = parse_float_list(args.center_offsets_px)
    for seed in seeds[: args.broad_seeds]:
        center, valid = project_center(seed.pose, K)
        if valid:
            add_unique(
                output,
                RecoveryProposal(
                    shift_projected_center(
                        seed.pose, K, detection.center - center
                    ),
                    f"{seed.source}|align_detection",
                    seed.measurement_score * 0.95,
                    seed.prior_error,
                ),
                args.max_candidates,
            )
        if args.allow_flip_hypotheses:
            add_unique(
                output,
                RecoveryProposal(
                    flipped_pose(seed.pose, "z"),
                    f"{seed.source}|flip180",
                    seed.measurement_score * 0.85,
                    seed.prior_error,
                ),
                args.max_candidates,
            )
        for offset in log_depth_offsets:
            add_unique(
                output,
                RecoveryProposal(
                    shift_projected_center(
                        seed.pose, K, delta_log_depth=offset
                    ),
                    f"{seed.source}|logz{offset:+.2f}",
                    seed.measurement_score * 0.85,
                    seed.prior_error,
                ),
                args.max_candidates,
            )
        for offset in center_offsets:
            for axis in (0, 1):
                delta = np.zeros(2)
                delta[axis] = offset
                add_unique(
                    output,
                    RecoveryProposal(
                        shift_projected_center(seed.pose, K, delta),
                        f"{seed.source}|uv{axis}{offset:+.0f}",
                        seed.measurement_score * 0.80,
                        seed.prior_error,
                    ),
                    args.max_candidates,
                )
        for degrees in rotation_offsets:
            for axis in (0, 1):
                for sign in (-1.0, 1.0):
                    delta = np.zeros(3)
                    delta[axis] = math.radians(sign * degrees)
                    add_unique(
                        output,
                        RecoveryProposal(
                            rotate_pose(seed.pose, delta, side="left"),
                            f"{seed.source}|R{axis}{sign * degrees:+.0f}",
                            seed.measurement_score * 0.80,
                            seed.prior_error,
                        ),
                        args.max_candidates,
                    )
        for degrees in yaw_offsets:
            for sign in (-1.0, 1.0):
                delta = np.asarray([0.0, 0.0, math.radians(sign * degrees)])
                add_unique(
                    output,
                    RecoveryProposal(
                        rotate_pose(seed.pose, delta, side="left"),
                        f"{seed.source}|yaw{sign * degrees:+.0f}",
                        seed.measurement_score * 0.80,
                        seed.prior_error,
                    ),
                    args.max_candidates,
                )
    return output[: args.max_candidates]


def make_proposals(
    track: Track | None,
    bank: list[RecoveryProposal],
    fresh,
    frame,
    detection,
    flow_updater: SparseFlowPoseUpdater,
    include_global: bool,
    args: argparse.Namespace,
) -> list[RecoveryProposal]:
    output: list[RecoveryProposal] = []
    if track is not None:
        for item in bank:
            add_unique(output, item, args.max_candidates)
        if track.history:
            propagated = propagate_pose_window(
                track.history,
                window_size=5,
                robust=True,
            )
            add_unique(
                output,
                RecoveryProposal(
                    propagated,
                    "constant_velocity",
                    track.confidence,
                    bank[0].prior_error if bank else 0.0,
                ),
                args.max_candidates,
            )
        if track.previous_gray is not None and track.previous_mask is not None:
            flow = flow_updater.estimate(
                track.previous_gray, frame.gray, track.previous_mask
            )
            if flow is not None:
                add_unique(
                    output,
                    RecoveryProposal(
                        flow_updater.update_pose(track.pose, frame.K, flow),
                        "optical_flow",
                        flow.confidence,
                        bank[0].prior_error if bank else 0.0,
                    ),
                    args.max_candidates,
                )
    if include_global or not output:
        for rank, hypothesis in enumerate(fresh[: args.top_k_gigapose]):
            add_unique(
                output,
                RecoveryProposal(
                    hypothesis.pose.copy(),
                    f"gigapose_global_rank_{rank}",
                    hypothesis.measurement_score,
                    0.0,
                ),
                args.max_candidates,
            )
    return output


def update_appearance(track: Track, frame, detection, association_config) -> None:
    if not association_config.identity_enabled:
        return
    descriptor = detection_appearance_descriptor(frame.image, detection)
    if descriptor is None:
        return
    if track.appearance_descriptor is None:
        track.appearance_descriptor = descriptor
        return
    momentum = association_config.appearance_momentum
    updated = momentum * track.appearance_descriptor + (1.0 - momentum) * descriptor
    track.appearance_descriptor = updated / max(float(updated.sum()), 1e-9)


def mode_from_confidence(confidence: float, args: argparse.Namespace) -> TrackMode:
    if confidence >= args.normal_confidence:
        return TrackMode.NORMAL
    if confidence >= args.lost_confidence:
        return TrackMode.UNCERTAIN
    return TrackMode.LOST


def select_with_orientation_gates(
    ranked: Sequence[RecoveryResult],
    *,
    track: Track | None,
    rank0_pose: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[RecoveryResult, list[RecoveryResult], dict[str, Any]]:
    """Select a result without silently changing front/rear orientation.

    A normal/uncertain track is constrained relative to its last accepted pose.
    Whenever a fresh rank-0 GigaPose pose exists, every state is additionally
    constrained to the same front/rear hemisphere. If no scored result passes,
    the best result keeps its refined translation while borrowing the rotation
    from rank 0 (or the previous track as a fallback).
    """

    if not ranked:
        raise ValueError("Orientation gating requires at least one result.")

    previous_pose = None if track is None else np.asarray(track.pose, dtype=float)
    if track is None or track.mode == TrackMode.LOST:
        previous_limit = None
    elif track.mode == TrackMode.NORMAL:
        previous_limit = float(args.normal_max_rotation_step_deg)
    else:
        previous_limit = float(args.uncertain_max_rotation_step_deg)
    rank0_limit = float(args.max_rank0_rotation_disagreement_deg)

    def angles(result: RecoveryResult) -> tuple[float, float]:
        previous = (
            float("nan")
            if previous_pose is None
            else rotation_error_deg(result.pose, previous_pose)
        )
        rank0 = (
            float("nan")
            if rank0_pose is None
            else rotation_error_deg(result.pose, rank0_pose)
        )
        return previous, rank0

    raw_previous, raw_rank0 = angles(ranked[0])
    diagnostics: dict[str, Any] = {
        "orientation_gate_enabled": int(bool(args.orientation_gates)),
        "orientation_gate_triggered": 0,
        "orientation_gate_fallback": 0,
        "orientation_gate_anchor": "",
        "orientation_candidates_rejected": 0,
        "orientation_selected_rank_after_gate": 0,
        "rotation_from_previous_before_gate_deg": raw_previous,
        "rotation_from_rank0_before_gate_deg": raw_rank0,
        "rotation_from_previous_after_gate_deg": raw_previous,
        "rotation_from_rank0_after_gate_deg": raw_rank0,
    }
    if not args.orientation_gates:
        return ranked[0], list(ranked), diagnostics

    eligible: list[tuple[int, RecoveryResult]] = []
    for index, result in enumerate(ranked):
        previous, rank0 = angles(result)
        previous_ok = (
            previous_limit is None
            or not math.isfinite(previous)
            or previous <= previous_limit
        )
        rank0_ok = (
            rank0_pose is None
            or not math.isfinite(rank0)
            or rank0 <= rank0_limit
        )
        if previous_ok and rank0_ok:
            eligible.append((index, result))

    diagnostics["orientation_candidates_rejected"] = len(ranked) - len(eligible)
    if eligible:
        selected_index, selected = eligible[0]
        diagnostics["orientation_gate_triggered"] = int(selected_index != 0)
        diagnostics["orientation_selected_rank_after_gate"] = selected_index
        after_previous, after_rank0 = angles(selected)
        diagnostics["rotation_from_previous_after_gate_deg"] = after_previous
        diagnostics["rotation_from_rank0_after_gate_deg"] = after_rank0
        return selected, [item for _, item in eligible], diagnostics

    anchor_pose = rank0_pose if rank0_pose is not None else previous_pose
    if anchor_pose is None:
        return ranked[0], list(ranked), diagnostics
    anchor_name = "gigapose_rank0" if rank0_pose is not None else "previous_pose"
    fallback_pose = np.asarray(ranked[0].pose, dtype=float).copy()
    fallback_pose[:3, :3] = np.asarray(anchor_pose, dtype=float)[:3, :3]
    fallback = replace(
        ranked[0],
        pose=fallback_pose,
        source=f"{ranked[0].source}|orientation_fallback_{anchor_name}",
    )
    after_previous, after_rank0 = angles(fallback)
    diagnostics.update(
        {
            "orientation_gate_triggered": 1,
            "orientation_gate_fallback": 1,
            "orientation_gate_anchor": anchor_name,
            "rotation_from_previous_after_gate_deg": after_previous,
            "rotation_from_rank0_after_gate_deg": after_rank0,
        }
    )
    return fallback, [fallback], diagnostics


def draw_pose_axes(
    canvas: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    *,
    axis_length_m: float,
    label_prefix: str,
    muted: bool,
) -> None:
    """Project object axes so a silhouette-preserving flip is visible."""

    length = float(axis_length_m)
    if length <= 0:
        return
    object_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.0, 0.0, length],
        ],
        dtype=float,
    )
    pose = np.asarray(pose, dtype=float)
    camera_points = (
        pose[:3, :3] @ object_points.T + pose[:3, 3:4]
    ).T
    if np.any(camera_points[:, 2] <= 1e-6):
        return
    pixels_h = (np.asarray(K, dtype=float) @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    if not np.isfinite(pixels).all():
        return
    origin = tuple(np.round(pixels[0]).astype(int))
    bright_colors = ((255, 55, 55), (55, 255, 80), (70, 130, 255))
    muted_colors = ((170, 80, 80), (80, 170, 95), (90, 120, 170))
    colors = muted_colors if muted else bright_colors
    width = 1 if muted else 2
    for axis_name, endpoint, color in zip("XYZ", pixels[1:], colors):
        end = tuple(np.round(endpoint).astype(int))
        cv2.line(canvas, origin, end, color, width, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label_prefix}{axis_name}",
            end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    # The shared racecar CAD is X-longitudinal. Mark +X explicitly so a
    # front/rear branch change is visible even when the silhouettes overlap.
    plus_x = tuple(np.round(pixels[1]).astype(int))
    cv2.circle(canvas, plus_x, 4 if not muted else 3, colors[0], -1, cv2.LINE_AA)


def draw_overlay(
    frame,
    detection,
    track: Track,
    result: RecoveryResult,
    original_pose: np.ndarray | None,
    renderer: CADRenderer,
    output_path: Path,
    *,
    orientation_diagnostics: dict[str, Any],
    axis_length_m: float,
) -> None:
    canvas = np.asarray(frame.image, dtype=np.uint8).copy()
    if original_pose is not None:
        original_mask, _ = renderer.render(
            original_pose, frame.K, frame.image.shape[:2]
        )
        canvas[original_mask] = (
            0.75 * canvas[original_mask] + 0.25 * np.asarray([255, 50, 50])
        ).astype(np.uint8)
    recovered_mask, _ = renderer.render(
        result.pose, frame.K, frame.image.shape[:2]
    )
    canvas[recovered_mask] = (
        0.60 * canvas[recovered_mask] + 0.40 * np.asarray([30, 230, 80])
    ).astype(np.uint8)
    observed = np.asarray(detection.mask, dtype=np.uint8)
    observed_edge = observed & ~cv2.erode(observed, np.ones((3, 3), np.uint8))
    canvas[observed_edge > 0] = np.asarray([40, 220, 255], dtype=np.uint8)
    x, y, width, height = np.round(detection.bbox_xywh).astype(int)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 220, 40), 2)
    if original_pose is not None:
        draw_pose_axes(
            canvas,
            original_pose,
            frame.K,
            axis_length_m=axis_length_m,
            label_prefix="G",
            muted=True,
        )
    draw_pose_axes(
        canvas,
        result.pose,
        frame.K,
        axis_length_m=axis_length_m,
        label_prefix="T",
        muted=False,
    )
    label = (
        f"T{track.track_id} {track.mode.value} conf={track.confidence:.2f} "
        f"q={result.quality:.2f} IoU={result.silhouette_iou:.2f} {result.source}"
    )
    cv2.putText(
        canvas,
        label,
        (max(0, x), max(18, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (30, 230, 80),
        1,
        cv2.LINE_AA,
    )
    previous_angle = orientation_diagnostics.get(
        "rotation_from_previous_after_gate_deg", float("nan")
    )
    rank0_angle = orientation_diagnostics.get(
        "rotation_from_rank0_after_gate_deg", float("nan")
    )
    gate_status = (
        "fallback"
        if orientation_diagnostics.get("orientation_gate_fallback")
        else (
            "filtered"
            if orientation_diagnostics.get("orientation_gate_triggered")
            else "pass"
        )
    )
    angle_label = (
        f"dRprev={previous_angle:.1f} dRgp0={rank0_angle:.1f} "
        f"gate={gate_status}"
    )
    cv2.putText(
        canvas,
        angle_label,
        (max(0, x), min(canvas.shape[0] - 5, y + height + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (30, 230, 80),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    if args.overlay_every <= 0 or args.global_interval <= 0:
        raise ValueError("Intervals must be positive.")
    if args.normal_max_rotation_step_deg <= 0:
        raise ValueError("--normal-max-rotation-step-deg must be positive.")
    if args.uncertain_max_rotation_step_deg <= 0:
        raise ValueError("--uncertain-max-rotation-step-deg must be positive.")
    if not 0 < args.max_rank0_rotation_disagreement_deg <= 180:
        raise ValueError(
            "--max-rank0-rotation-disagreement-deg must be in (0, 180]."
        )
    if args.overlay_axis_length_m < 0:
        raise ValueError("--overlay-axis-length-m must be non-negative.")
    print(
        "Orientation gates: "
        + (
            "enabled "
            f"(normal_step={args.normal_max_rotation_step_deg:g} deg, "
            f"uncertain_step={args.uncertain_max_rotation_step_deg:g} deg, "
            "rank0_disagreement="
            f"{args.max_rank0_rotation_disagreement_deg:g} deg)"
            if args.orientation_gates
            else "disabled"
        )
    )
    print(
        "Explicit flip hypotheses: "
        + ("enabled" if args.allow_flip_hypotheses else "disabled")
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    association_tracker_config = TrackerConfig.load(args.association_config)
    association_tracker_config.association.use_external_id = args.use_external_ids
    association_tracker_config.association.identity_enabled = True
    source = WebDatasetSequence(
        args.dataset_dir,
        args.split,
        load_depth=False,
        scene_id=args.scene_id,
    )
    predictions = PredictionCSVProvider(
        args.predictions,
        args.prediction_translation_unit,
        "gigapose",
    )
    renderer = CADRenderer(
        resolve_mesh(args),
        mesh_scale=args.mesh_scale,
        center_mesh=args.center_mesh,
    )
    predictor = RGBSelfRecoveryPredictor.load(args.checkpoint, args.device)
    flow_updater = SparseFlowPoseUpdater(association_tracker_config.flow)
    tracks: dict[int, Track] = {}
    banks: dict[int, list[RecoveryProposal]] = {}
    next_track_id = 0
    current_scene = None
    output_rows = []
    diagnostics = []
    state_counts: Counter[str] = Counter()
    orientation_gate_triggers = 0
    orientation_gate_fallbacks = 0
    orientation_candidates_rejected = 0
    started = time.perf_counter()
    frames_processed = 0
    try:
        for frame_index, frame in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if current_scene is not None and frame.scene_id != current_scene:
                tracks.clear()
                banks.clear()
            current_scene = frame.scene_id
            frames_processed += 1
            active = list(tracks.values())
            matches, unmatched_tracks, unmatched_detections = associate_tracks(
                active,
                frame.detections,
                frame.image.shape[:2],
                association_tracker_config.association,
                frame.image,
            )
            fresh_groups = predictions.groups_for_frame(frame.scene_id, frame.im_id)
            fresh_by_detection = assign_prediction_groups(
                fresh_groups,
                frame.detections,
                frame.K,
                frame.image.shape[:2],
            )
            track_by_detection = {
                detection_index: active[track_index]
                for track_index, detection_index in matches
            }
            process_indices = [index for _, index in matches] + unmatched_detections
            for detection_index in process_indices:
                detection = frame.detections[detection_index]
                track = track_by_detection.get(detection_index)
                bank = [] if track is None else banks.get(track.track_id, [])
                fresh = fresh_by_detection.get(detection_index, [])
                include_global = (
                    track is None
                    or track.mode != TrackMode.NORMAL
                    or track.age % args.global_interval == 0
                )
                proposals = make_proposals(
                    track,
                    bank,
                    fresh,
                    frame,
                    detection,
                    flow_updater,
                    include_global,
                    args,
                )
                if not proposals:
                    continue
                occluder_mask = np.zeros_like(detection.mask, dtype=bool)
                for other_index, other_detection in enumerate(frame.detections):
                    if other_index != detection_index:
                        occluder_mask |= np.asarray(
                            other_detection.mask, dtype=bool
                        )
                force_broad = (
                    track is None
                    or track.mode != TrackMode.NORMAL
                    or track.age % args.global_interval == 0
                )
                broad_used = force_broad
                if force_broad:
                    proposals = broad_recovery_proposals(
                        proposals, detection, frame.K, args
                    )
                ranked = predictor.refine_and_score(
                    renderer,
                    frame,
                    detection,
                    proposals,
                    occluder_mask=occluder_mask,
                    iterations=args.refinement_iterations,
                    quality_weight=args.quality_weight,
                    confidence_weight=args.confidence_weight,
                    silhouette_weight=args.silhouette_weight,
                    measurement_weight=args.measurement_weight,
                    prior_weight=args.history_weight,
                )
                preliminary_confidence = (
                    float(
                        np.clip(
                            ranked[0].confidence
                            * (0.5 + 0.5 * ranked[0].silhouette_iou),
                            0.0,
                            1.0,
                        )
                    )
                    if ranked
                    else 0.0
                )
                if (
                    not broad_used
                    and preliminary_confidence
                    < args.broad_recovery_confidence
                ):
                    broad = broad_recovery_proposals(
                        make_proposals(
                            track,
                            bank,
                            fresh,
                            frame,
                            detection,
                            flow_updater,
                            True,
                            args,
                        ),
                        detection,
                        frame.K,
                        args,
                    )
                    ranked = predictor.refine_and_score(
                        renderer,
                        frame,
                        detection,
                        broad,
                        occluder_mask=occluder_mask,
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
                best, gated_ranked, orientation_diagnostics = (
                    select_with_orientation_gates(
                        ranked,
                        track=track,
                        rank0_pose=fresh[0].pose if fresh else None,
                        args=args,
                    )
                )
                if orientation_diagnostics["orientation_gate_fallback"]:
                    fallback_before_rescore = best
                    rescored_fallback = predictor.refine_and_score(
                        renderer,
                        frame,
                        detection,
                        [
                            RecoveryProposal(
                                best.pose.copy(),
                                best.source,
                                best.confidence,
                                best.total_error,
                            )
                        ],
                        occluder_mask=occluder_mask,
                        iterations=0,
                        quality_weight=args.quality_weight,
                        confidence_weight=args.confidence_weight,
                        silhouette_weight=args.silhouette_weight,
                        measurement_weight=args.measurement_weight,
                        prior_weight=args.history_weight,
                    )
                    if rescored_fallback:
                        best = replace(
                            rescored_fallback[0],
                            delta_center_crop_px=(
                                fallback_before_rescore.delta_center_crop_px
                            ),
                            delta_log_depth=(
                                fallback_before_rescore.delta_log_depth
                            ),
                            delta_rotation_rad=(
                                fallback_before_rescore.delta_rotation_rad
                            ),
                        )
                        gated_ranked = [best]
                orientation_gate_triggers += int(
                    orientation_diagnostics["orientation_gate_triggered"]
                )
                orientation_gate_fallbacks += int(
                    orientation_diagnostics["orientation_gate_fallback"]
                )
                orientation_candidates_rejected += int(
                    orientation_diagnostics["orientation_candidates_rejected"]
                )
                confidence = float(
                    np.clip(
                        best.confidence * (0.5 + 0.5 * best.silhouette_iou),
                        0.0,
                        1.0,
                    )
                )
                if orientation_diagnostics["orientation_gate_triggered"]:
                    # A high silhouette score is insufficient evidence for a
                    # front/rear branch change. Force one uncertain frame so a
                    # fresh rank-0 anchor is checked again on the next image.
                    confidence = min(
                        confidence,
                        max(
                            args.lost_confidence,
                            args.normal_confidence - 1e-6,
                        ),
                    )
                mode = mode_from_confidence(confidence, args)
                if track is None:
                    track = Track(
                        track_id=next_track_id,
                        obj_id=detection.obj_id,
                        pose=best.pose.copy(),
                        previous_pose=None,
                        bbox_xywh=detection.bbox_xywh.copy(),
                        confidence=confidence,
                        mode=mode,
                        source=best.source,
                        scene_id=frame.scene_id,
                        im_id=frame.im_id,
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
                update_appearance(
                    track,
                    frame,
                    detection,
                    association_tracker_config.association,
                )
                tracks[track.track_id] = track
                banks[track.track_id] = [
                    RecoveryProposal(
                        item.pose.copy(),
                        item.source,
                        item.confidence,
                        item.total_error,
                    )
                    for item in gated_ranked[: args.beam_size]
                ]
                state_counts[mode.value] += 1
                output_rows.append(
                    pose_to_csv_row(
                        scene_id=frame.scene_id,
                        im_id=frame.im_id,
                        track_id=track.track_id,
                        obj_id=track.obj_id,
                        confidence=confidence,
                        pose_m=best.pose,
                        mode=mode.value,
                        source=best.source,
                        elapsed_s=0.0,
                    )
                )
                diagnostics.append(
                    {
                        "scene_id": frame.scene_id,
                        "im_id": frame.im_id,
                        "track_id": track.track_id,
                        "detection_id": detection.detection_id,
                        "mode": mode.value,
                        "source": best.source,
                        "confidence": confidence,
                        "verifier_confidence": best.confidence,
                        "quality": best.quality,
                        "silhouette_iou": best.silhouette_iou,
                        "total_error": best.total_error,
                        "broad_recovery": int(broad_used),
                        "candidates": len(ranked),
                        "delta_center_crop_px": float(
                            np.linalg.norm(best.delta_center_crop_px)
                        ),
                        "delta_log_depth": best.delta_log_depth,
                        "delta_rotation_deg": math.degrees(
                            float(np.linalg.norm(best.delta_rotation_rad))
                        ),
                        "explicit_flip_in_source": int(
                            "flip180" in best.source
                        ),
                        **orientation_diagnostics,
                    }
                )
                if args.save_overlays and frame_index % args.overlay_every == 0:
                    original_pose = fresh[0].pose if fresh else None
                    draw_overlay(
                        frame,
                        detection,
                        track,
                        best,
                        original_pose,
                        renderer,
                        args.output_dir
                        / "overlays"
                        / f"{frame.scene_id:06d}_{frame.im_id:06d}_t{track.track_id}.jpg",
                        orientation_diagnostics=orientation_diagnostics,
                        axis_length_m=args.overlay_axis_length_m,
                    )

            for track_index in unmatched_tracks:
                track = active[track_index]
                track.missed += 1
                track.mode = TrackMode.LOST
                if track.missed >= args.delete_after_misses:
                    tracks.pop(track.track_id, None)
                    banks.pop(track.track_id, None)
            if frames_processed % 25 == 0:
                print(f"processed {frames_processed}/{len(source)} frames")
    finally:
        renderer.close()

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    if diagnostics:
        with (args.output_dir / "candidate_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    rank0_after_angles = [
        float(row["rotation_from_rank0_after_gate_deg"])
        for row in diagnostics
        if math.isfinite(float(row["rotation_from_rank0_after_gate_deg"]))
    ]
    report = {
        "format": "rgb_render_self_recovery_v1",
        "predictions": str(args.predictions),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mesh": str(resolve_mesh(args)),
        "uses_observed_depth": False,
        "frames_processed": frames_processed,
        "tracked_instances": len(output_rows),
        "state_counts": dict(state_counts),
        "broad_recoveries": int(
            sum(int(row["broad_recovery"]) for row in diagnostics)
        ),
        "orientation_gate_triggers": orientation_gate_triggers,
        "orientation_gate_fallbacks": orientation_gate_fallbacks,
        "orientation_candidates_rejected": orientation_candidates_rejected,
        "explicit_flip_outputs": int(
            sum(int(row["explicit_flip_in_source"]) for row in diagnostics)
        ),
        "post_gate_rank0_rotation_comparisons": len(rank0_after_angles),
        "post_gate_rank0_front_back_violations": int(
            sum(
                angle > args.max_rank0_rotation_disagreement_deg
                for angle in rank0_after_angles
            )
        ),
        "post_gate_rank0_near_180_failures": int(
            sum(angle >= 150.0 for angle in rank0_after_angles)
        ),
        "elapsed_s": time.perf_counter() - started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
