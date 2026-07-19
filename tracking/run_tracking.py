"""Run self-correcting adaptive tracking from precomputed GigaPose top-K poses."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from tracking.config import TrackerConfig
from tracking.io import (
    PredictionCSVProvider,
    WebDatasetSequence,
    merge_prediction_group_sets,
    pose_to_csv_row,
    write_tracking_csv,
)
from tracking.recovery import RecoveryPredictor
from tracking.refinement import CandidateRefiner
from tracking.rendering import CADRenderer
from tracking.scoring import CandidateScorer
from tracking.tracker import AdaptivePoseTracker
from tracking.visualization import save_tracking_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive multi-hypothesis CAD tracking. Precomputed top-K GigaPose "
            "predictions act as global measurements/relocalization proposals."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-predictions",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional pose CSV (for example EPnP). Repeat as needed; "
            "matching instance_id values are merged with the GigaPose top-K group."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument(
        "--center-mesh",
        action="store_true",
        help="Center mesh bounds before rendering; normally leave off for GigaPose CADs.",
    )
    parser.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    parser.add_argument(
        "--auxiliary-translation-unit", choices=("mm", "m"), default="mm"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--recovery-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_mesh(args: argparse.Namespace) -> Path:
    if args.mesh is not None:
        return args.mesh
    candidate = args.dataset_dir / "models" / "obj_000001.ply"
    if candidate.is_file():
        return candidate
    fallback = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Pass --mesh; no default obj_000001.ply was found.")


def _write_diagnostics(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.overlay_every <= 0:
        raise ValueError("--overlay-every must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TrackerConfig.load(args.config)
    config.save(args.output_dir / "resolved_tracker_config.json")
    device = args.device
    if (
        args.recovery_checkpoint is not None
        and device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print("CUDA is unavailable; loading the optional recovery head on CPU.")
        device = "cpu"

    source = WebDatasetSequence(
        args.dataset_dir,
        args.split,
        load_depth=not args.no_depth,
        scene_id=args.scene_id,
    )
    predictions = PredictionCSVProvider(
        args.predictions, args.prediction_translation_unit, "gigapose"
    )
    auxiliary_predictions = [
        PredictionCSVProvider(
            path,
            args.auxiliary_translation_unit,
            f"auxiliary_{index}",
        )
        for index, path in enumerate(args.auxiliary_predictions)
    ]
    recovery = (
        RecoveryPredictor.load(args.recovery_checkpoint, device)
        if args.recovery_checkpoint is not None
        else None
    )
    output_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    states: Counter[str] = Counter()
    frames_processed = 0
    started = time.perf_counter()
    renderer = CADRenderer(
        _resolve_mesh(args),
        mesh_scale=args.mesh_scale,
        center_mesh=args.center_mesh,
    )
    try:
        scorer = CandidateScorer(renderer, config)
        refiner = CandidateRefiner(scorer, config)
        tracker = AdaptivePoseTracker(config, refiner, recovery=recovery)
        total_frames = len(source)
        if args.max_frames is not None:
            total_frames = min(total_frames, args.max_frames)
        for frame_index, frame in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            frames_processed += 1
            groups = merge_prediction_group_sets(
                [
                    predictions.groups_for_frame(frame.scene_id, frame.im_id),
                    *[
                        provider.groups_for_frame(frame.scene_id, frame.im_id)
                        for provider in auxiliary_predictions
                    ],
                ]
            )
            result = tracker.process_frame(frame, groups)
            diagnostic_rows.extend(tracker.diagnostics(result))
            for item in result.instances:
                states[item.track.mode.value] += 1
                output_rows.append(
                    pose_to_csv_row(
                        scene_id=frame.scene_id,
                        im_id=frame.im_id,
                        track_id=item.track.track_id,
                        obj_id=item.track.obj_id,
                        confidence=item.track.confidence,
                        pose_m=item.track.pose,
                        mode=item.track.mode.value,
                        source=item.track.source,
                        elapsed_s=item.elapsed_s,
                    )
                )
            if args.save_overlays and frame_index % args.overlay_every == 0:
                save_tracking_overlay(
                    frame,
                    result,
                    args.output_dir
                    / "overlays"
                    / f"{frame.scene_id:06d}_{frame.im_id:06d}.jpg",
                )
            if (frame_index + 1) % 25 == 0 or frame_index + 1 == total_frames:
                print(f"tracked {frame_index + 1}/{total_frames} frames")
    finally:
        renderer.close()

    output_path = args.output_dir / "tracked_predictions.csv"
    write_tracking_csv(output_path, output_rows)
    _write_diagnostics(args.output_dir / "candidate_diagnostics.csv", diagnostic_rows)
    report = {
        "predictions": str(args.predictions),
        "auxiliary_predictions": [str(path) for path in args.auxiliary_predictions],
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "mesh": str(_resolve_mesh(args)),
        "frames_processed": frames_processed,
        "tracked_instances": len(output_rows),
        "state_counts": dict(states),
        "recovery_checkpoint": (
            str(args.recovery_checkpoint) if args.recovery_checkpoint else None
        ),
        "elapsed_s": time.perf_counter() - started,
        "tracked_predictions": str(output_path),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
