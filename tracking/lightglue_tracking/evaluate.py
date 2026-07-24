"""Compare one or more trackers against BOP/WebDataset ground truth.

In addition to translation and rotation error, this evaluator reports missed
instances, false positives, lost-state rate, approximate identity switches,
and runtime from each tracker's ``run_report.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.evaluate_tracking import load_ground_truth, load_predictions
from tracking.geometry import rotation_error_deg, translation_error_m


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must be NAME=/path/to/tracked_predictions.csv"
        )
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Repeat NAME=/path/to/predictions.csv for each tracker.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def _summary(values: list[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
    }


def _runtime_report(prediction_path: Path) -> dict[str, Any]:
    path = prediction_path.parent / "run_report.json"
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    elapsed = float(report.get("elapsed_s", float("nan")))
    frames = int(report.get("frames_processed", 0))
    return {
        "runtime_report": str(path),
        "elapsed_s": elapsed,
        "frames_processed": frames,
        "frames_per_second": (
            frames / elapsed
            if frames > 0 and np.isfinite(elapsed) and elapsed > 0
            else float("nan")
        ),
    }


def _evaluate_model(
    *,
    name: str,
    path: Path,
    ground_truth: dict[tuple[int, int], list[dict]],
    minimum_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = load_predictions(path, minimum_confidence)
    rows: list[dict[str, Any]] = []
    missed_gt = 0
    false_positive = 0
    total_gt = sum(len(items) for items in ground_truth.values())
    all_predictions = [
        item for items in predictions.values() for item in items
    ]
    lost_predictions = sum(
        str(item.get("mode", "")).lower() == "lost"
        for item in all_predictions
    )
    last_track_by_gt: dict[tuple[int, int, int], int] = {}
    identity_switches = 0
    track_fragments: dict[tuple[int, int, int], set[int]] = {}

    for key in sorted(ground_truth):
        gt_items = ground_truth[key]
        predicted_items = predictions.get(key, [])
        if not gt_items or not predicted_items:
            missed_gt += len(gt_items)
            false_positive += len(predicted_items)
            continue
        cost = np.full((len(gt_items), len(predicted_items)), 1e6)
        pair_values: dict[tuple[int, int], tuple[float, float]] = {}
        for gt_index, gt_item in enumerate(gt_items):
            for prediction_index, prediction in enumerate(predicted_items):
                if gt_item["obj_id"] != prediction["obj_id"]:
                    continue
                translation = translation_error_m(
                    gt_item["pose"], prediction["pose"]
                )
                rotation = rotation_error_deg(
                    gt_item["pose"], prediction["pose"]
                )
                cost[gt_index, prediction_index] = (
                    translation + rotation / 100.0
                )
                pair_values[(gt_index, prediction_index)] = (
                    translation,
                    rotation,
                )
        gt_indices, prediction_indices = linear_sum_assignment(cost)
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        for gt_index, prediction_index in zip(
            gt_indices, prediction_indices
        ):
            gt_index = int(gt_index)
            prediction_index = int(prediction_index)
            if cost[gt_index, prediction_index] >= 1e5:
                continue
            matched_gt.add(gt_index)
            matched_predictions.add(prediction_index)
            translation, rotation = pair_values[
                (gt_index, prediction_index)
            ]
            prediction = predicted_items[prediction_index]
            gt_identity = (
                int(key[0]),
                int(gt_items[gt_index]["obj_id"]),
                gt_index,
            )
            track_id = int(prediction["track_id"])
            previous_track = last_track_by_gt.get(gt_identity)
            if previous_track is not None and previous_track != track_id:
                identity_switches += 1
            last_track_by_gt[gt_identity] = track_id
            track_fragments.setdefault(gt_identity, set()).add(track_id)
            rows.append(
                {
                    "model": name,
                    "scene_id": key[0],
                    "im_id": key[1],
                    "gt_index": gt_index,
                    "track_id": track_id,
                    "confidence": prediction["confidence"],
                    "translation_error_m": translation,
                    "rotation_error_deg": rotation,
                    "mode": prediction["mode"],
                    "source": prediction["source"],
                }
            )
        missed_gt += len(gt_items) - len(matched_gt)
        false_positive += len(predicted_items) - len(matched_predictions)

    summary: dict[str, Any] = {
        "model": name,
        "predictions": str(path),
        **_summary(
            [float(row["translation_error_m"]) for row in rows],
            "translation_error_m",
        ),
        **_summary(
            [float(row["rotation_error_deg"]) for row in rows],
            "rotation_error_deg",
        ),
        "ground_truth_instances": total_gt,
        "matched_instances": len(rows),
        "missed_ground_truth_instances": missed_gt,
        "missed_ground_truth_rate": missed_gt / max(total_gt, 1),
        "false_positive_predictions": false_positive,
        "prediction_rows": len(all_predictions),
        "lost_prediction_rows": lost_predictions,
        "lost_rate": lost_predictions / max(len(all_predictions), 1),
        "identity_switches": identity_switches,
        "gt_trajectories_with_multiple_track_ids": sum(
            len(track_ids) > 1 for track_ids in track_fragments.values()
        ),
        "minimum_confidence": minimum_confidence,
        **_runtime_report(path),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        raise ValueError("Every --model name must be unique")
    ground_truth = load_ground_truth(args.dataset_dir, args.split)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, path in args.model:
        rows, summary = _evaluate_model(
            name=name,
            path=path,
            ground_truth=ground_truth,
            minimum_confidence=args.min_confidence,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{name}: t_med="
            f"{summary.get('translation_error_m_median', float('nan')):.4f}m "
            "R_med="
            f"{summary.get('rotation_error_deg_median', float('nan')):.2f}deg "
            f"IDSW={summary['identity_switches']} "
            f"lost={summary['lost_rate']:.3f}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.output_dir / "per_instance_errors.csv"
    with row_path.open("w", newline="") as handle:
        fieldnames = list(all_rows[0]) if all_rows else [
            "model",
            "scene_id",
            "im_id",
            "gt_index",
            "track_id",
            "confidence",
            "translation_error_m",
            "rotation_error_deg",
            "mode",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_csv = args.output_dir / "model_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "identity_switch_definition": (
            "A GT list index in one scene changes assigned track_id between "
            "successive matched observations. Assetto GT export order is "
            "assumed stable within a scene."
        ),
        "models": summaries,
        "per_instance_csv": str(row_path),
        "summary_csv": str(summary_csv),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
