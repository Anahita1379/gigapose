"""Evaluate tracked pose CSVs against BOP ground truth stored in WebDataset."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.geometry import pose_from_rt, rotation_error_deg, translation_error_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def _shard_path(split_dir: Path, value: Any) -> Path:
    if isinstance(value, str) and value.endswith(".tar"):
        return split_dir / value
    return split_dir / f"shard-{int(value):06d}.tar"


def load_ground_truth(dataset_dir: Path, split: str) -> dict[tuple[int, int], list[dict]]:
    split_dir = dataset_dir / split
    mapping = json.loads((split_dir / "key_to_shard.json").read_text())
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, value in mapping.items():
        by_shard[str(value)].append(key)
    output: dict[tuple[int, int], list[dict]] = {}
    for shard_value, keys in by_shard.items():
        with tarfile.open(_shard_path(split_dir, shard_value)) as tar:
            for key in keys:
                try:
                    handle = tar.extractfile(f"{key}.gt.json")
                except KeyError:
                    continue
                if handle is None:
                    continue
                scene_id, im_id = (int(value) for value in key.split("_"))
                poses = []
                for index, row in enumerate(json.loads(handle.read())):
                    poses.append(
                        {
                            "gt_index": index,
                            "obj_id": int(row["obj_id"]),
                            "pose": pose_from_rt(
                                np.asarray(row["cam_R_m2c"], dtype=float).reshape(3, 3),
                                np.asarray(row["cam_t_m2c"], dtype=float) * 0.001,
                            ),
                        }
                    )
                output[(scene_id, im_id)] = poses
    return output


def load_predictions(path: Path, min_confidence: float) -> dict[tuple[int, int], list[dict]]:
    output: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            confidence = float(row["score"])
            if confidence < min_confidence:
                continue
            pose = pose_from_rt(
                np.fromstring(row["R"], sep=" ").reshape(3, 3),
                np.fromstring(row["t"], sep=" ") * 0.001,
            )
            output[(int(row["scene_id"]), int(row["im_id"]))].append(
                {
                    "track_id": int(row.get("track_id", row.get("instance_id", -1))),
                    "obj_id": int(row["obj_id"]),
                    "confidence": confidence,
                    "pose": pose,
                    "mode": row.get("tracking_mode", ""),
                    "source": row.get("source", ""),
                }
            )
    return output


def _summary(values: list[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    args = parse_args()
    gt = load_ground_truth(args.dataset_dir, args.split)
    predictions = load_predictions(args.predictions, args.min_confidence)
    rows: list[dict] = []
    missed_gt = false_positive = 0
    for key, gt_items in gt.items():
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
                rotation = rotation_error_deg(gt_item["pose"], prediction["pose"])
                cost[gt_index, prediction_index] = translation + rotation / 100.0
                pair_values[(gt_index, prediction_index)] = (translation, rotation)
        gt_indices, prediction_indices = linear_sum_assignment(cost)
        matched_gt, matched_predictions = set(), set()
        for gt_index, prediction_index in zip(gt_indices, prediction_indices):
            if cost[gt_index, prediction_index] >= 1e5:
                continue
            matched_gt.add(int(gt_index))
            matched_predictions.add(int(prediction_index))
            translation, rotation = pair_values[(int(gt_index), int(prediction_index))]
            prediction = predicted_items[int(prediction_index)]
            rows.append(
                {
                    "scene_id": key[0],
                    "im_id": key[1],
                    "gt_index": int(gt_index),
                    "track_id": prediction["track_id"],
                    "confidence": prediction["confidence"],
                    "translation_error_m": translation,
                    "rotation_error_deg": rotation,
                    "mode": prediction["mode"],
                    "source": prediction["source"],
                }
            )
        missed_gt += len(gt_items) - len(matched_gt)
        false_positive += len(predicted_items) - len(matched_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "per_instance_errors.csv"
    with output_csv.open("w", newline="") as handle:
        fieldnames = list(rows[0]) if rows else [
            "scene_id", "im_id", "gt_index", "track_id", "confidence",
            "translation_error_m", "rotation_error_deg", "mode", "source"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        **_summary([row["translation_error_m"] for row in rows], "translation_error_m"),
        **_summary([row["rotation_error_deg"] for row in rows], "rotation_error_deg"),
        "missed_ground_truth_instances": missed_gt,
        "false_positive_predictions": false_positive,
        "minimum_confidence": args.min_confidence,
        "per_instance_csv": str(output_csv),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
