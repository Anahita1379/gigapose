"""Compare GigaPose, numeric, and frozen visual-fusion transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from tracking.io import WebDatasetSequence
from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import CandidateBundle


def parser():
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--data", type=Path, required=True, help="Held-out candidate bundle")
    value.add_argument("--gigapose-predictions", type=Path, required=True)
    value.add_argument("--numeric-output", type=Path, required=True)
    value.add_argument("--visual-output", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--mesh", type=Path)
    value.add_argument("--mesh-scale", type=float, default=1.0)
    value.add_argument("--center-mesh", action="store_true")
    value.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument("--distance-bins-m", default="0,20,40,60,80,100,120,140")
    value.add_argument("--max-overlays", type=int, default=120)
    value.add_argument("--overwrite", action="store_true")
    return value


def _prediction(path: Path) -> Path:
    return path / "tracked_predictions.csv" if path.is_dir() else path


def _diagnostics(path: Path) -> list[dict]:
    path = path / "diagnostics.jsonl" if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _division(numerator: int, denominator: int):
    return None if denominator == 0 else float(numerator / denominator)


def _ece(probability: np.ndarray, target: np.ndarray, bins=10):
    if not len(probability):
        return None
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability >= low) & (
            probability <= high if high == 1 else probability < high
        )
        if selected.any():
            total += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return float(total)


def _trust(rows, component):
    baseline = np.asarray([row[f"baseline_{component}_error_{'m' if component == 'translation' else 'deg'}"] for row in rows])
    proposed = np.asarray([row[f"proposed_{component}_error_{'m' if component == 'translation' else 'deg'}"] for row in rows])
    probability = np.asarray([row[f"{component}_trust_probability"] for row in rows])
    fallback = np.asarray([row[f"{component}_fallback"] for row in rows], dtype=bool)
    trust_target = proposed < baseline
    fallback_target = ~trust_target
    tp = int(np.sum(fallback & fallback_target))
    fp = int(np.sum(fallback & ~fallback_target))
    fn = int(np.sum(~fallback & fallback_target))
    precision, recall = _division(tp, tp + fp), _division(tp, tp + fn)
    return {
        "count": len(rows),
        "trust_positive_fraction": float(trust_target.mean()) if len(rows) else None,
        "brier_score": float(np.mean((probability - trust_target) ** 2)) if len(rows) else None,
        "expected_calibration_error_10bin": _ece(probability, trust_target),
        "trust_accuracy_at_0_5": float(np.mean((probability >= 0.5) == trust_target)) if len(rows) else None,
        "fallback_precision": precision,
        "fallback_recall": recall,
        "fallback_f1": (
            None if precision is None or recall is None or precision + recall == 0
            else float(2 * precision * recall / (precision + recall))
        ),
        "fallback_true_positive": tp,
        "fallback_false_positive": fp,
        "fallback_false_negative": fn,
    }


def _error_stats(rows):
    def stats(key):
        values = np.asarray([row[key] for row in rows], dtype=float)
        return {
            "count": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "rmse": float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None,
        }
    return {
        "translation_error_m": stats("final_translation_error_m"),
        "rotation_error_deg": stats("final_rotation_error_deg"),
    }


def _weather(value: object) -> str:
    text = json.dumps(value).lower()
    for name in ("fog", "rain", "snow"):
        if name in text:
            return name
    if any(name in text for name in ("clear", "sun", "dry")):
        return "clear"
    return "unknown"


def _context(dataset_dir: Path, split: str, bundle: CandidateBundle):
    wanted = {
        (int(scene), int(image)): index
        for index, (scene, image) in enumerate(zip(
            bundle.arrays["scene_id"], bundle.arrays["im_id"]
        ))
    }
    frame_map_path = dataset_dir / "frame_map.json"
    metadata = {}
    if frame_map_path.is_file():
        for row in json.loads(frame_map_path.read_text()):
            metadata[(int(row["scene_id"]), int(row["im_id"]))] = row
    result = {}
    for frame in WebDatasetSequence(dataset_dir, split, load_depth=False):
        key = int(frame.scene_id), int(frame.im_id)
        if key not in wanted:
            continue
        mask_fraction = (
            float(np.mean(frame.detections[0].mask)) if len(frame.detections) == 1 else None
        )
        result[key] = {
            "mask_area_fraction": mask_fraction,
            "weather": _weather(metadata.get(key, frame.metadata)),
        }
    return result


def _subgroups(rows, context):
    enriched = []
    for row in rows:
        value = {**row, **context.get((int(row["scene_id"]), int(row["im_id"])), {})}
        enriched.append(value)
    weather = {
        name: _error_stats([row for row in enriched if row.get("weather") == name])
        for name in ("clear", "fog", "rain", "snow", "unknown")
    }
    mask_bins = []
    edges = (0.0, 0.0005, 0.002, 0.01, 1.01)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = [
            row for row in enriched
            if row.get("mask_area_fraction") is not None
            and low <= row["mask_area_fraction"] < high
        ]
        mask_bins.append({
            "low_fraction": low,
            "high_fraction": high,
            **_error_stats(selected),
        })
    flip_rows = [row for row in enriched if row["baseline_rotation_error_deg"] >= 90.0]
    return {
        "weather": weather,
        "mask_area_bins": mask_bins,
        "front_rear_flip": {
            "baseline_flip_count": len(flip_rows),
            "final_below_90deg_fraction": (
                None if not flip_rows else float(np.mean([
                    row["final_rotation_error_deg"] < 90 for row in flip_rows
                ]))
            ),
            "rotation_improved_fraction": (
                None if not flip_rows else float(np.mean([
                    row["final_rotation_error_deg"] < row["baseline_rotation_error_deg"]
                    for row in flip_rows
                ]))
            ),
        },
    }


def _plot_calibration(all_rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, component in zip(axes, ("translation", "rotation")):
        for name, rows in all_rows.items():
            unit = "m" if component == "translation" else "deg"
            probability = np.asarray([row[f"{component}_trust_probability"] for row in rows])
            target = np.asarray([
                row[f"proposed_{component}_error_{unit}"]
                < row[f"baseline_{component}_error_{unit}"] for row in rows
            ])
            centers, observed = [], []
            for low in np.linspace(0, 0.9, 10):
                selected = (probability >= low) & (probability < low + 0.1 + 1e-8)
                if selected.any():
                    centers.append(float(probability[selected].mean()))
                    observed.append(float(target[selected].mean()))
            axis.plot(centers, observed, marker="o", label=name)
        axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
        axis.set(title=f"{component.title()} trust", xlabel="Predicted trust", ylabel="Observed improvement")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    command = [
        sys.executable, "-m", "tracking.rgb_self_recovery.sliding_window.evaluate_backbones",
        "--dataset-dir", str(args.dataset_dir), "--split", args.split,
        "--model", f"gigapose={_prediction(args.gigapose_predictions)}",
        "--model", f"numeric_transformer={_prediction(args.numeric_output)}",
        "--model", f"frozen_visual_fusion={_prediction(args.visual_output)}",
        "--baseline", "gigapose", "--mesh", str(mesh),
        "--mesh-scale", str(args.mesh_scale),
        "--prediction-translation-unit", args.prediction_translation_unit,
        "--distance-bins-m", args.distance_bins_m,
        "--max-overlays", str(args.max_overlays),
        "--output-dir", str(args.output_dir / "pose_metrics"), "--overwrite",
    ]
    if args.center_mesh:
        command.append("--center-mesh")
    subprocess.run(command, check=True)

    bundle = CandidateBundle(args.data)
    context = _context(args.dataset_dir, args.split, bundle)
    rows = {
        "numeric_transformer": _diagnostics(args.numeric_output),
        "frozen_visual_fusion": _diagnostics(args.visual_output),
    }
    report = {
        "format": "rgb_self_recovery_visual_fusion_evaluation_v1",
        "held_out_required": True,
        "trust_target": "proposal component has lower GT error than original GigaPose component",
        "fallback_positive": "restoring original GigaPose component lowers GT error",
        "mask_metric": "observed instance-mask pixels / full image pixels",
        "models": {},
    }
    for name, values in rows.items():
        report["models"][name] = {
            "translation_trust": _trust(values, "translation"),
            "rotation_trust": _trust(values, "rotation"),
            **_subgroups(values, context),
        }
    _plot_calibration(rows, args.output_dir / "trust_calibration.png")
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
