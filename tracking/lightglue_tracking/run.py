"""Run isolated RGB self-recovery tracking with frozen ALIKED + LightGlue."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np
import torch

from tracking.config import TrackerConfig
from tracking.flow import SparseFlowPoseUpdater
from tracking.geometry import propagate_pose_window
from tracking.io import (
    PredictionCSVProvider,
    WebDatasetSequence,
    pose_to_csv_row,
    write_tracking_csv,
)
from tracking.lightglue_tracking.association import associate_tracks
from tracking.lightglue_tracking.config import LightGlueTrackingConfig
from tracking.lightglue_tracking.matcher import (
    ALIKEDLightGlue,
    MatchResult,
    RegionFeatures,
)
from tracking.lightglue_tracking.pnp import (
    PnPResult,
    PnPTemplate,
    build_template,
    recover_pose,
)
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery.inference import (
    RGBSelfRecoveryPredictor,
    RecoveryProposal,
    RecoveryResult,
)
from tracking.rgb_self_recovery.run import (
    add_unique,
    broad_recovery_proposals,
    draw_overlay,
    mode_from_confidence,
    resolve_mesh,
    select_with_orientation_gates,
    update_appearance,
)
from tracking.types import Track, TrackMode


@dataclass
class FeatureKeyframe:
    features: RegionFeatures
    pose: np.ndarray
    scene_id: int
    im_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RGB self-recovery tracking extended with frozen ALIKED + "
            "LightGlue. Existing tracking packages are not modified."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--center-mesh", action="store_true")
    parser.add_argument(
        "--association-config",
        type=Path,
        default=Path("tracking/configs/improved.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-translation-unit", choices=("mm", "m"), default="mm"
    )
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
        default=False,
    )
    parser.add_argument(
        "--orientation-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    lightglue = parser.add_argument_group("ALIKED + LightGlue")
    lightglue.add_argument(
        "--lightglue",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lightglue.add_argument(
        "--lightglue-root", type=Path, default=Path("LightGlue")
    )
    lightglue.add_argument("--lightglue-max-keypoints", type=int, default=1024)
    lightglue.add_argument("--lightglue-resize", type=int, default=640)
    lightglue.add_argument(
        "--lightglue-crop-padding-frac", type=float, default=0.30
    )
    lightglue.add_argument(
        "--lightglue-filter-threshold", type=float, default=0.10
    )
    lightglue.add_argument(
        "--lightglue-depth-confidence", type=float, default=0.95
    )
    lightglue.add_argument(
        "--lightglue-width-confidence", type=float, default=0.99
    )
    lightglue.add_argument("--lightglue-mixed-precision", action="store_true")
    lightglue.add_argument(
        "--lightglue-association",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lightglue.add_argument(
        "--lightglue-association-policy",
        choices=("off", "always", "uncertain_lost", "ambiguous"),
        default="ambiguous",
    )
    lightglue.add_argument(
        "--lightglue-association-weight", type=float, default=0.35
    )
    lightglue.add_argument(
        "--lightglue-flow-policy",
        choices=("off", "always", "uncertain_lost"),
        default="uncertain_lost",
    )
    lightglue.add_argument(
        "--lightglue-lk-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lightglue.add_argument("--lightglue-ransac-threshold-px", type=float, default=4.0)
    lightglue.add_argument("--lightglue-min-matches", type=int, default=8)
    lightglue.add_argument("--lightglue-min-inliers", type=int, default=8)
    lightglue.add_argument(
        "--lightglue-min-inlier-ratio", type=float, default=0.30
    )
    lightglue.add_argument(
        "--lightglue-max-median-reprojection-px", type=float, default=6.0
    )
    lightglue.add_argument(
        "--lightglue-keyframe-interval", type=int, default=10
    )

    pnp = parser.add_argument_group("optional render-backed temporal PnP")
    pnp.add_argument(
        "--lightglue-pnp-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    pnp.add_argument(
        "--lightglue-pnp-policy",
        choices=("off", "always", "uncertain_lost"),
        default="uncertain_lost",
    )
    pnp.add_argument("--lightglue-pnp-min-points", type=int, default=8)
    pnp.add_argument("--lightglue-pnp-min-inliers", type=int, default=6)
    pnp.add_argument(
        "--lightglue-pnp-min-inlier-ratio", type=float, default=0.30
    )
    pnp.add_argument(
        "--lightglue-pnp-reprojection-px", type=float, default=6.0
    )
    pnp.add_argument("--lightglue-pnp-iterations", type=int, default=200)
    pnp.add_argument(
        "--lightglue-pnp-max-rotation-step-deg", type=float, default=75.0
    )
    pnp.add_argument(
        "--lightglue-pnp-max-translation-step-m", type=float, default=5.0
    )
    pnp.add_argument(
        "--lightglue-template-min-confidence", type=float, default=0.65
    )
    return parser.parse_args()


def make_lightglue_config(args: argparse.Namespace) -> LightGlueTrackingConfig:
    config = LightGlueTrackingConfig(
        enabled=bool(args.lightglue),
        lightglue_root=args.lightglue_root,
        max_num_keypoints=args.lightglue_max_keypoints,
        resize=args.lightglue_resize,
        crop_padding_frac=args.lightglue_crop_padding_frac,
        filter_threshold=args.lightglue_filter_threshold,
        depth_confidence=args.lightglue_depth_confidence,
        width_confidence=args.lightglue_width_confidence,
        mixed_precision=args.lightglue_mixed_precision,
        ransac_threshold_px=args.lightglue_ransac_threshold_px,
        min_matches=args.lightglue_min_matches,
        min_inliers=args.lightglue_min_inliers,
        min_inlier_ratio=args.lightglue_min_inlier_ratio,
        max_median_reprojection_px=(
            args.lightglue_max_median_reprojection_px
        ),
        association_enabled=args.lightglue_association,
        association_policy=args.lightglue_association_policy,
        association_weight=args.lightglue_association_weight,
        flow_policy=args.lightglue_flow_policy,
        lk_fallback=args.lightglue_lk_fallback,
        pnp_recovery_enabled=args.lightglue_pnp_recovery,
        pnp_policy=args.lightglue_pnp_policy,
        pnp_min_points=args.lightglue_pnp_min_points,
        pnp_min_inliers=args.lightglue_pnp_min_inliers,
        pnp_min_inlier_ratio=args.lightglue_pnp_min_inlier_ratio,
        pnp_reprojection_px=args.lightglue_pnp_reprojection_px,
        pnp_iterations=args.lightglue_pnp_iterations,
        pnp_max_rotation_step_deg=(
            args.lightglue_pnp_max_rotation_step_deg
        ),
        pnp_max_translation_step_m=(
            args.lightglue_pnp_max_translation_step_m
        ),
        template_min_confidence=args.lightglue_template_min_confidence,
    )
    config.validate()
    return config


def _policy_applies(policy: str, track: Track | None) -> bool:
    if policy == "off" or track is None:
        return False
    if policy == "always":
        return True
    if policy == "uncertain_lost":
        return track.mode != TrackMode.NORMAL
    raise ValueError(f"Unsupported runtime policy: {policy}")


def _frame_needs_features(
    active: Sequence[Track],
    detection_count: int,
    config: LightGlueTrackingConfig,
) -> bool:
    if not config.enabled or not active or detection_count == 0:
        return False
    if config.association_enabled:
        if config.association_policy == "always":
            return True
        if (
            config.association_policy == "ambiguous"
            and len(active) > 1
            and detection_count > 1
        ):
            return True
        if (
            config.association_policy in {"ambiguous", "uncertain_lost"}
            and any(track.mode != TrackMode.NORMAL for track in active)
        ):
            return True
    if config.flow_policy == "always" or config.pnp_policy == "always":
        return True
    return any(
        track.mode != TrackMode.NORMAL
        and (
            config.flow_policy == "uncertain_lost"
            or (
                config.pnp_recovery_enabled
                and config.pnp_policy == "uncertain_lost"
            )
        )
        for track in active
    )


def _extract_detection_features(
    matcher: ALIKEDLightGlue | None,
    frame,
    indices: Sequence[int] | None = None,
) -> list[RegionFeatures | None]:
    output: list[RegionFeatures | None] = [None] * len(frame.detections)
    if matcher is None:
        return output
    selected = range(len(frame.detections)) if indices is None else indices
    for index in selected:
        detection = frame.detections[index]
        output[index] = matcher.extract_region(
            frame.image,
            detection.bbox_xywh,
            detection.mask,
        )
    return output


def _ensure_feature(
    features: list[RegionFeatures | None],
    detection_index: int,
    matcher: ALIKEDLightGlue | None,
    frame,
) -> RegionFeatures | None:
    if features[detection_index] is None and matcher is not None:
        detection = frame.detections[detection_index]
        features[detection_index] = matcher.extract_region(
            frame.image,
            detection.bbox_xywh,
            detection.mask,
        )
    return features[detection_index]


def _match_for_track(
    *,
    track: Track,
    detection_index: int,
    association_pair: MatchResult | None,
    keyframes: dict[int, FeatureKeyframe],
    current: RegionFeatures | None,
    matcher: ALIKEDLightGlue | None,
) -> MatchResult | None:
    if association_pair is not None:
        return association_pair
    keyframe = keyframes.get(track.track_id)
    if keyframe is None or current is None or matcher is None:
        return None
    return matcher.match(keyframe.features, current)


def make_proposals(
    *,
    track: Track | None,
    bank: Sequence[RecoveryProposal],
    fresh,
    frame,
    lightglue_match: MatchResult | None,
    lightglue_keyframe: FeatureKeyframe | None,
    pnp_result: PnPResult | None,
    flow_updater: SparseFlowPoseUpdater,
    include_global: bool,
    args: argparse.Namespace,
    lightglue_config: LightGlueTrackingConfig,
) -> list[RecoveryProposal]:
    output: list[RecoveryProposal] = []
    if track is not None:
        for item in bank:
            add_unique(output, item, args.max_candidates)
        if track.history:
            propagated = propagate_pose_window(
                track.history, window_size=5, robust=True
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
            # Compete explicitly with the angular-velocity hypothesis. Keep
            # its robustly propagated translation but hold the most recently
            # accepted rotation.
            zero_rotation = propagated.copy()
            zero_rotation[:3, :3] = track.pose[:3, :3]
            add_unique(
                output,
                RecoveryProposal(
                    zero_rotation,
                    "constant_translation_zero_rotation",
                    track.confidence,
                    bank[0].prior_error if bank else 0.0,
                ),
                args.max_candidates,
            )
        lightglue_used = False
        if (
            _policy_applies(lightglue_config.flow_policy, track)
            and lightglue_match is not None
            and lightglue_match.similarity is not None
            and lightglue_keyframe is not None
        ):
            similarity = lightglue_match.similarity
            add_unique(
                output,
                RecoveryProposal(
                    flow_updater.update_pose(
                        lightglue_keyframe.pose,
                        frame.K,
                        similarity,
                    ),
                    "aliked_lightglue_similarity",
                    similarity.quality,
                    bank[0].prior_error if bank else 0.0,
                ),
                args.max_candidates,
            )
            lightglue_used = True
        if pnp_result is not None:
            add_unique(output, pnp_result.proposal, args.max_candidates)
        if (
            lightglue_config.lk_fallback
            and not lightglue_used
            and track.previous_gray is not None
            and track.previous_mask is not None
        ):
            flow = flow_updater.estimate(
                track.previous_gray, frame.gray, track.previous_mask
            )
            if flow is not None:
                add_unique(
                    output,
                    RecoveryProposal(
                        flow_updater.update_pose(track.pose, frame.K, flow),
                        "optical_flow_fallback",
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


def _validate_args(args: argparse.Namespace) -> None:
    if args.overlay_every <= 0 or args.global_interval <= 0:
        raise ValueError("Intervals must be positive")
    if args.lightglue_keyframe_interval <= 0:
        raise ValueError("--lightglue-keyframe-interval must be positive")
    if args.normal_max_rotation_step_deg <= 0:
        raise ValueError("--normal-max-rotation-step-deg must be positive")
    if args.uncertain_max_rotation_step_deg <= 0:
        raise ValueError("--uncertain-max-rotation-step-deg must be positive")
    if not 0 < args.max_rank0_rotation_disagreement_deg <= 180:
        raise ValueError(
            "--max-rank0-rotation-disagreement-deg must be in (0, 180]"
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    _validate_args(args)
    lightglue_config = make_lightglue_config(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracker_config = TrackerConfig.load(args.association_config)
    tracker_config.association.use_external_id = args.use_external_ids
    tracker_config.association.identity_enabled = True
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
    matcher = (
        ALIKEDLightGlue(lightglue_config, device=args.device)
        if lightglue_config.enabled
        else None
    )
    flow_updater = SparseFlowPoseUpdater(tracker_config.flow)
    tracks: dict[int, Track] = {}
    banks: dict[int, list[RecoveryProposal]] = {}
    keyframes: dict[int, FeatureKeyframe] = {}
    pnp_templates: dict[int, PnPTemplate] = {}
    next_track_id = 0
    current_scene = None
    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    started = time.perf_counter()
    frames_processed = 0

    try:
        for frame_index, frame in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if current_scene is not None and frame.scene_id != current_scene:
                tracks.clear()
                banks.clear()
                keyframes.clear()
                pnp_templates.clear()
            current_scene = frame.scene_id
            frames_processed += 1
            active = list(tracks.values())
            detection_features = (
                _extract_detection_features(matcher, frame)
                if _frame_needs_features(
                    active, len(frame.detections), lightglue_config
                )
                else [None] * len(frame.detections)
            )
            association = associate_tracks(
                active,
                frame.detections,
                frame.image.shape[:2],
                tracker_config.association,
                lightglue_config,
                matcher,
                {
                    track_id: keyframe.features
                    for track_id, keyframe in keyframes.items()
                },
                detection_features,
                frame.image,
            )
            counters["lightglue_association_pairs"] += (
                association.attempted_lightglue_pairs
            )
            fresh_groups = predictions.groups_for_frame(
                frame.scene_id, frame.im_id
            )
            from tracking.association import assign_prediction_groups

            fresh_by_detection = assign_prediction_groups(
                fresh_groups,
                frame.detections,
                frame.K,
                frame.image.shape[:2],
            )
            track_by_detection = {
                detection_index: active[track_index]
                for track_index, detection_index in association.matches
            }
            active_index_by_track = {
                track.track_id: index for index, track in enumerate(active)
            }
            process_indices = [
                detection_index
                for _, detection_index in association.matches
            ] + association.unmatched_detections

            for detection_index in process_indices:
                detection = frame.detections[detection_index]
                track = track_by_detection.get(detection_index)
                bank = [] if track is None else banks.get(track.track_id, [])
                fresh = fresh_by_detection.get(detection_index, [])
                include_global = bool(
                    track is None
                    or track.mode != TrackMode.NORMAL
                    or track.age % args.global_interval == 0
                )
                pair_result: MatchResult | None = None
                pnp_result: PnPResult | None = None
                current_features = detection_features[detection_index]
                if track is not None:
                    needs_lightglue = bool(
                        _policy_applies(lightglue_config.flow_policy, track)
                        or (
                            lightglue_config.pnp_recovery_enabled
                            and _policy_applies(
                                lightglue_config.pnp_policy, track
                            )
                        )
                    )
                    if needs_lightglue:
                        current_features = _ensure_feature(
                            detection_features,
                            detection_index,
                            matcher,
                            frame,
                        )
                    active_index = active_index_by_track[track.track_id]
                    pair_result = _match_for_track(
                        track=track,
                        detection_index=detection_index,
                        association_pair=association.pair_matches.get(
                            (active_index, detection_index)
                        ),
                        keyframes=keyframes,
                        current=current_features,
                        matcher=matcher,
                    )
                    if pair_result is not None:
                        counters["lightglue_temporal_matches"] += 1
                        if pair_result.similarity is not None:
                            counters["lightglue_valid_similarities"] += 1
                    if (
                        lightglue_config.pnp_recovery_enabled
                        and _policy_applies(
                            lightglue_config.pnp_policy, track
                        )
                        and current_features is not None
                        and track.track_id in pnp_templates
                        and matcher is not None
                    ):
                        pnp_result = recover_pose(
                            template=pnp_templates[track.track_id],
                            current_features=current_features,
                            matcher=matcher,
                            K=frame.K,
                            previous_pose=track.pose,
                            config=lightglue_config,
                        )
                        counters["lightglue_pnp_attempts"] += 1
                        if pnp_result is not None:
                            counters["lightglue_pnp_successes"] += 1

                proposals = make_proposals(
                    track=track,
                    bank=bank,
                    fresh=fresh,
                    frame=frame,
                    lightglue_match=pair_result,
                    lightglue_keyframe=(
                        None if track is None else keyframes.get(track.track_id)
                    ),
                    pnp_result=pnp_result,
                    flow_updater=flow_updater,
                    include_global=include_global,
                    args=args,
                    lightglue_config=lightglue_config,
                )
                if not proposals:
                    continue
                occluder_mask = np.zeros_like(detection.mask, dtype=bool)
                for other_index, other in enumerate(frame.detections):
                    if other_index != detection_index:
                        occluder_mask |= np.asarray(other.mask, dtype=bool)
                broad_used = bool(
                    track is None
                    or track.mode != TrackMode.NORMAL
                    or track.age % args.global_interval == 0
                )
                if broad_used:
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
                            track=track,
                            bank=bank,
                            fresh=fresh,
                            frame=frame,
                            lightglue_match=pair_result,
                            lightglue_keyframe=(
                                None
                                if track is None
                                else keyframes.get(track.track_id)
                            ),
                            pnp_result=pnp_result,
                            flow_updater=flow_updater,
                            include_global=True,
                            args=args,
                            lightglue_config=lightglue_config,
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
                    before_rescore = best
                    rescored = predictor.refine_and_score(
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
                    if rescored:
                        best = replace(
                            rescored[0],
                            delta_center_crop_px=(
                                before_rescore.delta_center_crop_px
                            ),
                            delta_log_depth=before_rescore.delta_log_depth,
                            delta_rotation_rad=(
                                before_rescore.delta_rotation_rad
                            ),
                        )
                        gated_ranked = [best]
                confidence = float(
                    np.clip(
                        best.confidence * (0.5 + 0.5 * best.silhouette_iou),
                        0.0,
                        1.0,
                    )
                )
                if orientation_diagnostics["orientation_gate_triggered"]:
                    confidence = min(
                        confidence,
                        max(
                            args.lost_confidence,
                            args.normal_confidence - 1e-6,
                        ),
                    )
                mode = mode_from_confidence(confidence, args)
                is_new = track is None
                if is_new:
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
                    track, frame, detection, tracker_config.association
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

                refresh_keyframe = bool(
                    lightglue_config.enabled
                    and (
                        is_new
                        or track.mode != TrackMode.NORMAL
                        or track.track_id not in keyframes
                        or track.age % args.lightglue_keyframe_interval == 0
                    )
                )
                if refresh_keyframe:
                    current_features = _ensure_feature(
                        detection_features,
                        detection_index,
                        matcher,
                        frame,
                    )
                    if current_features is not None:
                        keyframes[track.track_id] = FeatureKeyframe(
                            current_features,
                            best.pose.copy(),
                            frame.scene_id,
                            frame.im_id,
                        )
                        if (
                            lightglue_config.pnp_recovery_enabled
                            and confidence
                            >= lightglue_config.template_min_confidence
                            and (
                                not lightglue_config.template_normal_only
                                or mode == TrackMode.NORMAL
                            )
                        ):
                            template = build_template(
                                renderer=renderer,
                                features=current_features,
                                pose_m=best.pose,
                                K=frame.K,
                                scene_id=frame.scene_id,
                                im_id=frame.im_id,
                                minimum_points=(
                                    lightglue_config.pnp_min_points
                                ),
                            )
                            if template is not None:
                                pnp_templates[track.track_id] = template
                                counters["lightglue_pnp_templates"] += 1

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
                similarity = (
                    None if pair_result is None else pair_result.similarity
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
                        "lightglue_matches": (
                            0 if pair_result is None else pair_result.count
                        ),
                        "lightglue_inliers": (
                            0 if similarity is None else similarity.inlier_count
                        ),
                        "lightglue_inlier_ratio": (
                            float("nan")
                            if similarity is None
                            else similarity.inlier_ratio
                        ),
                        "lightglue_reprojection_px": (
                            float("nan")
                            if similarity is None
                            else similarity.median_reprojection_px
                        ),
                        "lightglue_quality": (
                            float("nan")
                            if similarity is None
                            else similarity.quality
                        ),
                        "lightglue_rotation_deg": (
                            float("nan")
                            if similarity is None
                            else math.degrees(similarity.angle_rad)
                        ),
                        "lightglue_pnp_inliers": (
                            0 if pnp_result is None else pnp_result.inlier_count
                        ),
                        "lightglue_pnp_inlier_ratio": (
                            float("nan")
                            if pnp_result is None
                            else pnp_result.inlier_ratio
                        ),
                        "lightglue_pnp_reprojection_px": (
                            float("nan")
                            if pnp_result is None
                            else pnp_result.median_reprojection_px
                        ),
                        "lightglue_pnp_quality": (
                            float("nan")
                            if pnp_result is None
                            else pnp_result.quality
                        ),
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
                if (
                    args.save_overlays
                    and frame_index % args.overlay_every == 0
                ):
                    draw_overlay(
                        frame,
                        detection,
                        track,
                        best,
                        fresh[0].pose if fresh else None,
                        renderer,
                        args.output_dir
                        / "overlays"
                        / (
                            f"{frame.scene_id:06d}_{frame.im_id:06d}"
                            f"_t{track.track_id}.jpg"
                        ),
                        orientation_diagnostics=orientation_diagnostics,
                        axis_length_m=args.overlay_axis_length_m,
                    )

            for track_index in association.unmatched_tracks:
                track = active[track_index]
                track.missed += 1
                track.mode = TrackMode.LOST
                if track.missed >= args.delete_after_misses:
                    tracks.pop(track.track_id, None)
                    banks.pop(track.track_id, None)
                    keyframes.pop(track.track_id, None)
                    pnp_templates.pop(track.track_id, None)
            if frames_processed % 25 == 0:
                print(f"processed {frames_processed}/{len(source)} frames")
    finally:
        renderer.close()

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    if diagnostics:
        with (
            args.output_dir / "candidate_diagnostics.csv"
        ).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(diagnostics[0].keys())
            )
            writer.writeheader()
            writer.writerows(diagnostics)
    elapsed = time.perf_counter() - started
    report = {
        "format": "rgb_render_self_recovery_lightglue_v1",
        "predictions": str(args.predictions),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mesh": str(resolve_mesh(args)),
        "uses_observed_depth": False,
        "frames_processed": frames_processed,
        "tracked_instances": len(output_rows),
        "state_counts": dict(state_counts),
        "lost_rate": (
            state_counts[TrackMode.LOST.value] / max(len(output_rows), 1)
        ),
        "elapsed_s": elapsed,
        "frames_per_second": frames_processed / max(elapsed, 1e-9),
        "lightglue": lightglue_config.as_serializable_dict(),
        "lightglue_counters": dict(counters),
        "lightglue_runtime": (
            matcher.runtime_report() if matcher is not None else {}
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
