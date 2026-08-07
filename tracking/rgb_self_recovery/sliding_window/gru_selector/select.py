"""Replay exported candidates through a trained causal GRU selector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.io import pose_to_csv_row, write_tracking_csv

from .dataset import CandidateBundle
from .train import load_selector
from .temporal_orientation import FixedLagOrientationResolver, TemporalOrientationSelector


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--minimum-recovery-probability", type=float, default=0.0)
    value.add_argument("--minimum-recovery-margin-over-gigapose", type=float, default=0.0)
    value.add_argument("--temporal-orientation", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--orientation-soft-start-deg", type=float, default=45.0)
    value.add_argument("--orientation-hard-limit-deg", type=float, default=90.0)
    value.add_argument("--flip-min-angle-deg", type=float, default=135.0)
    value.add_argument("--rotation-penalty-weight", type=float, default=1.0)
    value.add_argument("--flip-confirmation-frames", type=int, default=2)
    value.add_argument("--stable-orientation-frames", type=int, default=4)
    value.add_argument("--stable-flip-confirmation-frames", type=int, default=5)
    value.add_argument("--flip-min-probability", type=float, default=0.15)
    value.add_argument("--flip-min-margin", type=float, default=0.03)
    value.add_argument("--flip-consistency-deg", type=float, default=45.0)
    value.add_argument("--maximum-angular-prediction-deg", type=float, default=30.0)
    value.add_argument("--orientation-velocity-window", type=int, default=5)
    value.add_argument(
        "--orientation-fixed-lag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Buffer unresolved flip frames and backfill their selected orientations "
            "when the adaptive hysteresis confirms a genuine transition"
        ),
    )
    value.add_argument("--overwrite", action="store_true")
    return value


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.minimum_recovery_probability <= 1:
        raise ValueError("--minimum-recovery-probability must be in [0,1]")
    if args.minimum_recovery_margin_over_gigapose < 0:
        raise ValueError("Recovery margin must be non-negative")
    if args.orientation_fixed_lag and not args.temporal_orientation:
        raise ValueError("--orientation-fixed-lag requires --temporal-orientation")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = CandidateBundle(args.data)
    model, payload = load_selector(args.checkpoint, args.device)
    if payload["candidate_feature_names"] != bundle.manifest["candidate_feature_names"]:
        raise ValueError("GRU checkpoint and candidate feature schemas differ")
    if payload["frame_feature_names"] != bundle.manifest["frame_feature_names"]:
        raise ValueError("GRU checkpoint and frame feature schemas differ")
    if int(payload["saved_candidates"]) != bundle.candidate_count:
        raise ValueError("GRU checkpoint and bundle candidate counts differ")

    arrays = bundle.arrays
    evaluation_fields = {
        "ground_truth_pose",
        "oracle_index",
        "translation_error_m",
        "rotation_error_deg",
    }
    available_evaluation_fields = evaluation_fields.intersection(arrays)
    if available_evaluation_fields and available_evaluation_fields != evaluation_fields:
        missing = sorted(evaluation_fields - available_evaluation_fields)
        raise ValueError(
            "Candidate bundle has an incomplete evaluation schema; missing "
            + ", ".join(missing)
        )
    has_evaluation = available_evaluation_fields == evaluation_fields
    orientation = None
    fixed_lag = None
    if args.temporal_orientation:
        orientation = TemporalOrientationSelector(
            soft_start_deg=args.orientation_soft_start_deg,
            hard_limit_deg=args.orientation_hard_limit_deg,
            flip_min_deg=args.flip_min_angle_deg,
            rotation_penalty_weight=args.rotation_penalty_weight,
            flip_confirmation_frames=args.flip_confirmation_frames,
            stable_orientation_frames=args.stable_orientation_frames,
            stable_flip_confirmation_frames=args.stable_flip_confirmation_frames,
            flip_min_probability=args.flip_min_probability,
            flip_min_margin=args.flip_min_margin,
            flip_consistency_deg=args.flip_consistency_deg,
            maximum_prediction_step_deg=args.maximum_angular_prediction_deg,
            velocity_window=args.orientation_velocity_window,
        )
        if args.orientation_fixed_lag:
            fixed_lag = FixedLagOrientationResolver()
    hidden = None
    previous_segment = None
    output_rows = []
    diagnostics = []
    selected_translation = []
    selected_rotation = []
    baseline_translation = []
    baseline_rotation = []
    oracle_matches = 0
    recoveries = 0
    guarded_fallbacks = 0
    temporal_reranks = 0
    temporal_fallbacks = 0
    confirmed_flips = 0
    fixed_lag_backfilled_frames = 0
    backfill_payloads = []
    with torch.no_grad():
        for index in range(len(bundle)):
            segment = int(arrays["segment_id"][index])
            sequence_reset = previous_segment is None or segment != previous_segment
            if sequence_reset:
                hidden = None
                if orientation is not None:
                    orientation.reset()
                if fixed_lag is not None:
                    fixed_lag.reset()
            previous_segment = segment
            candidate = torch.from_numpy(arrays["features"][index])[None, None].to(args.device)
            frame = torch.from_numpy(arrays["frame_features"][index])[None, None].to(args.device)
            valid = torch.from_numpy(arrays["valid"][index])[None, None].to(args.device)
            logits, hidden = model(candidate, frame, valid, hidden)
            probabilities = logits[0, 0].softmax(dim=-1).cpu().numpy()
            raw_selected = int(np.argmax(probabilities))
            selected = raw_selected
            predicted_rotation = None
            orientation_angles = np.full(len(probabilities), np.nan)
            temporal_reranked = False
            time_value = float(arrays["time_s"][index])
            time_value = time_value if np.isfinite(time_value) else None
            if orientation is not None:
                selected, predicted_rotation, orientation_angles, temporal_reranked = orientation.rerank(
                    arrays["poses"][index], probabilities,
                    arrays["valid"][index], time_value,
                )
                temporal_reranks += int(temporal_reranked)
            selected_before_probability_guard = selected
            probability_guarded_fallback = False
            probability_guard_enabled = (
                args.minimum_recovery_probability > 0
                or args.minimum_recovery_margin_over_gigapose > 0
            )
            if probability_guard_enabled and selected != 0 and (
                probabilities[selected] < args.minimum_recovery_probability
                or probabilities[selected] - probabilities[0]
                < args.minimum_recovery_margin_over_gigapose
            ):
                selected = 0
                guarded_fallbacks += 1
                probability_guarded_fallback = True
            selected_pose = np.asarray(arrays["poses"][index, selected], dtype=float).copy()
            visual_selected_pose = selected_pose.copy()
            orientation_decision = None
            if orientation is not None:
                orientation_decision = orientation.enforce(
                    selected_pose, selected, probabilities, arrays["valid"][index],
                    predicted_rotation, orientation_angles, time_value,
                )
                orientation_decision.temporal_reranked = temporal_reranked
                selected_pose = orientation_decision.pose
                temporal_fallbacks += int(orientation_decision.temporal_fallback)
                confirmed_flips += int(orientation_decision.flip_confirmed)
            resolved_fixed_lag_indices = (
                []
                if fixed_lag is None or orientation_decision is None
                else fixed_lag.update(index, orientation_decision)
            )
            for resolved_index in resolved_fixed_lag_indices:
                payload_to_resolve = backfill_payloads[resolved_index]
                resolved_pose = payload_to_resolve["visual_pose"]
                resolved_selected = payload_to_resolve["selected_index"]
                resolved_source = (
                    payload_to_resolve["source_base"] + "|fixed_lag_flip_backfill"
                )
                output_rows[resolved_index] = pose_to_csv_row(
                    scene_id=payload_to_resolve["scene_id"],
                    im_id=payload_to_resolve["im_id"],
                    track_id=0,
                    obj_id=1,
                    confidence=payload_to_resolve["confidence"],
                    pose_m=resolved_pose,
                    mode="fixed_lag_flip_backfill",
                    source=f"gru_selector|{resolved_source}",
                    elapsed_s=0.0,
                )
                if has_evaluation:
                    selected_rotation[resolved_index] = rotation_error_deg(
                        resolved_pose, payload_to_resolve["ground_truth_pose"]
                    )
                diagnostics[resolved_index]["output_candidate_index"] = resolved_selected
                diagnostics[resolved_index]["temporal_orientation_fallback"] = 0
                diagnostics[resolved_index]["fixed_lag_backfilled"] = 1
                diagnostics[resolved_index]["resolved_flip_state"] = (
                    orientation_decision.flip_state
                )
                diagnostics[resolved_index]["selected_source"] = resolved_source
                if has_evaluation:
                    diagnostics[resolved_index]["selected_rotation_error_deg"] = (
                        selected_rotation[resolved_index]
                    )
                temporal_fallbacks -= 1
                fixed_lag_backfilled_frames += 1
                if has_evaluation:
                    oracle_matches += int(
                        resolved_selected == payload_to_resolve["oracle_index"]
                    )
            recoveries += int(selected != 0)
            oracle = int(arrays["oracle_index"][index]) if has_evaluation else None
            gt_pose = arrays["ground_truth_pose"][index] if has_evaluation else None
            if has_evaluation:
                oracle_matches += int(
                    orientation_decision is not None
                    and not orientation_decision.temporal_fallback
                    and selected == oracle
                    or orientation_decision is None and selected == oracle
                )
                selected_translation.append(translation_error_m(selected_pose, gt_pose))
                selected_rotation.append(rotation_error_deg(selected_pose, gt_pose))
                baseline_translation.append(float(arrays["translation_error_m"][index, 0]))
                baseline_rotation.append(float(arrays["rotation_error_deg"][index, 0]))
            source = str(arrays["sources"][index, selected])
            if orientation_decision is not None:
                source += orientation_decision.source_suffix
            output_rows.append(
                pose_to_csv_row(
                    scene_id=int(arrays["scene_id"][index]),
                    im_id=int(arrays["im_id"][index]),
                    track_id=0,
                    obj_id=1,
                    confidence=float(probabilities[selected]),
                    pose_m=selected_pose,
                    mode=(
                        "temporal_orientation"
                        if orientation_decision is not None and orientation_decision.temporal_fallback
                        else "normal" if selected else "fallback"
                    ),
                    source=f"gru_selector|{source}",
                    elapsed_s=0.0,
                )
            )
            diagnostics.append(
                {
                    "scene_id": int(arrays["scene_id"][index]),
                    "im_id": int(arrays["im_id"][index]),
                    "sequence_run": str(arrays["sequence_run"][index]),
                    "sequence_camera": str(arrays["sequence_camera"][index]),
                    "source_frame": int(arrays["source_frame"][index]),
                    "segment_id": segment,
                    "sequence_reset": int(sequence_reset),
                    "raw_selected_index": raw_selected,
                    "temporal_reranked_index": selected_before_probability_guard,
                    "selected_index": selected,
                    "output_candidate_index": (
                        -1
                        if orientation_decision is not None
                        and orientation_decision.temporal_fallback
                        else selected
                    ),
                    "oracle_index": -1 if oracle is None else oracle,
                    "selected_probability": float(probabilities[selected]),
                    "gigapose_probability": float(probabilities[0]),
                    "guarded_fallback": int(probability_guarded_fallback),
                    "temporal_orientation_enabled": int(orientation is not None),
                    "orientation_fixed_lag_enabled": int(fixed_lag is not None),
                    "temporal_orientation_reranked": int(temporal_reranked),
                    "rotation_from_temporal_prediction_deg": (
                        float("nan") if orientation_decision is None
                        else orientation_decision.rotation_from_prediction_deg
                    ),
                    "temporal_orientation_fallback": (
                        0 if orientation_decision is None else int(orientation_decision.temporal_fallback)
                    ),
                    "flip_like_candidate": (
                        0 if orientation_decision is None else int(orientation_decision.flip_like)
                    ),
                    "flip_transition_confirmed": (
                        0 if orientation_decision is None else int(orientation_decision.flip_confirmed)
                    ),
                    "flip_pending_frames": (
                        0 if orientation_decision is None else orientation_decision.flip_pending_frames
                    ),
                    "flip_state": (
                        0 if orientation_decision is None else orientation_decision.flip_state
                    ),
                    "fixed_lag_backfilled": 0,
                    "resolved_flip_state": (
                        0 if orientation_decision is None else orientation_decision.flip_state
                    ),
                    "orientation_stable_frames": (
                        0 if orientation_decision is None
                        else orientation_decision.orientation_stable_frames
                    ),
                    "orientation_state_locked": (
                        0 if orientation_decision is None
                        else int(orientation_decision.orientation_state_locked)
                    ),
                    "required_flip_confirmation_frames": (
                        0 if orientation_decision is None
                        else orientation_decision.required_flip_confirmation_frames
                    ),
                    "best_nonflip_probability": (
                        float("nan") if orientation_decision is None
                        else orientation_decision.best_nonflip_probability
                    ),
                    "temporal_candidate_names": (
                        "" if orientation_decision is None
                        else "|".join(orientation_decision.temporal_candidate_names)
                    ),
                    "selected_source": source,
                    "selected_translation_error_m": (
                        float("nan") if not has_evaluation else selected_translation[-1]
                    ),
                    "selected_rotation_error_deg": (
                        float("nan") if not has_evaluation else selected_rotation[-1]
                    ),
                    "gigapose_translation_error_m": (
                        float("nan") if not has_evaluation else baseline_translation[-1]
                    ),
                    "gigapose_rotation_error_deg": (
                        float("nan") if not has_evaluation else baseline_rotation[-1]
                    ),
                }
            )
            backfill_payloads.append(
                {
                    "scene_id": int(arrays["scene_id"][index]),
                    "im_id": int(arrays["im_id"][index]),
                    "selected_index": selected,
                    "confidence": float(probabilities[selected]),
                    "visual_pose": visual_selected_pose,
                    "source_base": str(arrays["sources"][index, selected]),
                    "ground_truth_pose": gt_pose,
                    "oracle_index": oracle,
                }
            )

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    with (args.output_dir / "selection_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    report = {
        "format": "rgb_self_recovery_gru_selection_v1",
        "candidate_data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "frame_count": len(bundle),
        "segment_count": len(set(arrays["segment_id"].tolist())),
        "selected_recovery_fraction": recoveries / len(bundle),
        "guarded_fallback_count": guarded_fallbacks,
        "temporal_orientation_rerank_count": temporal_reranks,
        "temporal_orientation_fallback_count": temporal_fallbacks,
        "confirmed_flip_transition_count": confirmed_flips,
        "orientation_fixed_lag_enabled": fixed_lag is not None,
        "fixed_lag_backfilled_frame_count": fixed_lag_backfilled_frames,
        "evaluation_available": has_evaluation,
        "oracle_accuracy": oracle_matches / len(bundle) if has_evaluation else None,
        "gigapose_translation_error_m": _summary(baseline_translation),
        "selected_translation_error_m": _summary(selected_translation),
        "gigapose_rotation_error_deg": _summary(baseline_rotation),
        "selected_rotation_error_deg": _summary(selected_rotation),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
