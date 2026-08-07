"""Export RGB/CAD candidate bundles for deployment without ground truth poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from tracking.association import assign_prediction_groups
from tracking.io import PredictionCSVProvider, WebDatasetSequence
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery import run as base
from tracking.rgb_self_recovery.inference import RecoveryProposal
from tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.common import (
    resolve_predictions,
)
from tracking.rgb_self_recovery.sliding_window.sequence import (
    SequenceStamp,
    discontinuity_reason,
    order_sequence_rows,
)
from tracking.rgb_self_recovery.sliding_window.gru_selector.checkpoints import (
    load_candidate_predictor,
)
from tracking.rgb_self_recovery.sliding_window.gru_selector.features import (
    CANDIDATE_FEATURE_NAMES,
    FRAME_FEATURE_NAMES,
    candidate_features,
    frame_features,
)
from tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates import (
    _mesh,
    _score,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--mesh", type=Path)
    value.add_argument("--mesh-scale", type=float, default=1.0)
    value.add_argument("--center-mesh", action="store_true")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument("--saved-candidates", type=int, default=16)
    value.add_argument("--top-k-gigapose", type=int, default=5)
    value.add_argument("--max-candidates", type=int, default=48)
    value.add_argument("--refinement-iterations", type=int, default=2)
    value.add_argument("--allow-flip-hypotheses", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--rotation-offsets-deg", default="20,45")
    value.add_argument("--yaw-offsets-deg", default="30,90")
    value.add_argument("--log-depth-offsets", default="-0.35,0.35")
    value.add_argument("--center-offsets-px", default="-48,48")
    value.add_argument("--broad-seeds", type=int, default=4)
    value.add_argument("--quality-weight", type=float, default=1.0)
    value.add_argument("--confidence-weight", type=float, default=0.5)
    value.add_argument("--silhouette-weight", type=float, default=1.5)
    value.add_argument("--measurement-weight", type=float, default=0.05)
    value.add_argument("--history-weight", type=float, default=0.0)
    value.add_argument("--sequence-max-frame-gap", type=int, default=1)
    value.add_argument("--sequence-max-time-gap-s", type=float, default=0.5)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--overwrite", action="store_true")
    return value


def _validate(args) -> None:
    if args.saved_candidates < 2:
        raise ValueError("--saved-candidates must include baseline plus recovery")
    if args.max_candidates < args.saved_candidates - 1:
        raise ValueError("--max-candidates is smaller than the requested recovery pool")
    if args.sequence_max_frame_gap < 0 or args.sequence_max_time_gap_s < 0:
        raise ValueError("Sequence gap thresholds must be nonnegative")


def main() -> None:
    args = parser().parse_args()
    _validate(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = WebDatasetSequence(args.dataset_dir, args.split, load_depth=False)
    source.rows = list(order_sequence_rows(source.rows))
    predictions_path = resolve_predictions(args.predictions)
    predictions = PredictionCSVProvider(
        predictions_path, args.prediction_translation_unit, "gigapose"
    )
    predictor = load_candidate_predictor(args.checkpoint, args.device)
    renderer = CADRenderer(
        _mesh(args), mesh_scale=args.mesh_scale, center_mesh=args.center_mesh
    )

    records, skipped = [], []
    previous_saved_stamp = None
    previous_saved_time = None
    segment_id = -1
    started = time.perf_counter()
    try:
        for frame_index, frame in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            key = (frame.scene_id, frame.im_id)
            stamp = SequenceStamp.from_frame(frame)
            if len(frame.detections) != 1:
                skipped.append({
                    "scene_id": key[0], "im_id": key[1],
                    "reason": "requires exactly one target detection/mask",
                })
                continue
            detection = frame.detections[0]
            groups = predictions.groups_for_frame(*key)
            assigned = assign_prediction_groups(
                groups, frame.detections, frame.K, frame.image.shape[:2]
            )
            fresh = assigned.get(0, [])
            if not fresh:
                skipped.append({
                    "scene_id": key[0], "im_id": key[1],
                    "reason": "no assigned GigaPose hypotheses",
                })
                continue

            baseline_pose = fresh[0].pose.copy()
            baseline = RecoveryProposal(
                baseline_pose, "gigapose_rank0_baseline",
                fresh[0].measurement_score, 0.0,
            )
            baseline_scored = _score(
                predictor, renderer, frame, detection, [baseline], args, 0
            )
            if not baseline_scored:
                skipped.append({
                    "scene_id": key[0], "im_id": key[1],
                    "reason": "baseline RGB/CAD scoring failed",
                })
                continue
            seeds = [
                RecoveryProposal(
                    item.pose.copy(), f"gigapose_global_rank_{rank}",
                    item.measurement_score, 0.0,
                )
                for rank, item in enumerate(fresh[: args.top_k_gigapose])
            ]
            proposals = base.broad_recovery_proposals(seeds, detection, frame.K, args)
            recovered = _score(
                predictor, renderer, frame, detection, proposals, args,
                args.refinement_iterations,
            )
            chosen = [baseline_scored[0], *recovered[: args.saved_candidates - 1]]
            valid = np.zeros(args.saved_candidates, dtype=bool)
            valid[: len(chosen)] = True
            poses = np.repeat(
                np.eye(4, dtype=np.float32)[None], args.saved_candidates, axis=0
            )
            features = np.zeros(
                (args.saved_candidates, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
            )
            sources = np.full(args.saved_candidates, "padding", dtype="<U256")
            for candidate_index, result in enumerate(chosen):
                poses[candidate_index] = np.asarray(result.pose, dtype=np.float32)
                features[candidate_index] = candidate_features(
                    result, baseline_pose, baseline=candidate_index == 0
                )
                sources[candidate_index] = result.source

            reason = discontinuity_reason(
                previous_saved_stamp,
                stamp,
                max_frame_gap=args.sequence_max_frame_gap,
                max_time_gap_s=args.sequence_max_time_gap_s,
            )
            if previous_saved_stamp is None or reason is not None:
                segment_id += 1
                delta_time = 0.0
            else:
                delta_time = (
                    0.0 if previous_saved_time is None or stamp.time_s is None
                    else max(stamp.time_s - previous_saved_time, 0.0)
                )
            records.append({
                "scene_id": frame.scene_id,
                "im_id": frame.im_id,
                "sequence_run": stamp.key[0],
                "sequence_camera": stamp.key[1],
                "source_frame": stamp.source_frame,
                "time_s": np.nan if stamp.time_s is None else stamp.time_s,
                "segment_id": segment_id,
                "poses": poses,
                "features": features,
                "frame_features": frame_features(
                    frame, detection, baseline_pose, delta_time
                ),
                "valid": valid,
                "sources": sources,
            })
            previous_saved_stamp = stamp
            previous_saved_time = stamp.time_s
            if len(records) % 25 == 0:
                print(f"exported {len(records)}/{len(source)} frames", flush=True)
    finally:
        renderer.close()

    if not records:
        raise RuntimeError("No runtime candidate frames were exported")
    names = (
        "scene_id", "im_id", "source_frame", "time_s", "segment_id", "poses",
        "features", "frame_features", "valid", "sources",
    )
    arrays = {name: np.asarray([row[name] for row in records]) for name in names}
    arrays["sequence_run"] = np.asarray(
        [row["sequence_run"] for row in records], dtype="<U256"
    )
    arrays["sequence_camera"] = np.asarray(
        [row["sequence_camera"] for row in records], dtype="<U64"
    )
    np.savez_compressed(args.output_dir / "candidates.npz", **arrays)
    report = {
        "format": "rgb_self_recovery_gru_candidates_v1",
        "runtime_without_ground_truth": True,
        "ground_truth_included": False,
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "predictions": str(predictions_path),
        "checkpoint": str(args.checkpoint),
        "mesh": str(_mesh(args)),
        "frame_count": len(records),
        "segment_count": len(set(arrays["segment_id"].tolist())),
        "saved_candidates": args.saved_candidates,
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "baseline_candidate_index": 0,
        "oracle_nonbaseline_fraction": None,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "sequence_max_frame_gap": args.sequence_max_frame_gap,
        "sequence_max_time_gap_s": args.sequence_max_time_gap_s,
        "elapsed_s": time.perf_counter() - started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
