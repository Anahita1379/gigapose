"""Plot metrics produced by fine_tuning.compare_gigapose_predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_SPECS = [
    ("translation_error_mm", "Translation error", "mm", (0, 50000)),
    ("depth_error_mm", "Depth error", "mm", (0, 50000)),
    ("rotation_error_deg", "Rotation error", "degrees", (0, 180)),
    ("center_error_px", "Projected center error", "px", (0, 1000)),
    ("gt_bbox_center_error_px", "GT bbox center error", "px", (0, 1000)),
    ("add_mm", "ADD", "mm", (0, 50000)),
    ("pred_bbox_iou", "Rendered bbox IoU", "IoU", (0, 1)),
    ("pred_mask_iou", "Rendered mask IoU", "IoU", (0, 1)),
    ("score", "Prediction score", "score", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("fine_tuning/prediction_gt_comparison"),
        help="Directory containing per_instance_metrics.csv.",
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=60)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def methods(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["method"] for row in rows})


def values(rows: list[dict[str, str]], method: str, key: str) -> np.ndarray:
    vals = []
    for row in rows:
        if row["method"] != method:
            continue
        try:
            val = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    return np.asarray(vals, dtype=float)


def plot_histograms(rows: list[dict[str, str]], output_dir: Path, bins: int) -> None:
    method_names = methods(rows)
    for key, title, xlabel, clip_range in PLOT_SPECS:
        series = []
        for method in method_names:
            vals = values(rows, method, key)
            if clip_range is not None:
                vals = vals[(vals >= clip_range[0]) & (vals <= clip_range[1])]
            if vals.size:
                series.append((method, vals))
        if not series:
            continue

        plt.figure(figsize=(9, 5))
        for method, vals in series:
            plt.hist(vals, bins=bins, alpha=0.45, label=f"{method} (n={len(vals)})")
            plt.axvline(np.median(vals), linestyle="--", linewidth=1.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_hist.png", dpi=160)
        plt.close()


def plot_score_vs_error(rows: list[dict[str, str]], output_dir: Path) -> None:
    method_names = methods(rows)
    for error_key, title, ylabel, ylimit in [
        ("translation_error_mm", "Score vs translation error", "translation error (mm)", (0, 50000)),
        ("rotation_error_deg", "Score vs rotation error", "rotation error (deg)", (0, 180)),
        ("center_error_px", "Score vs center error", "center error (px)", (0, 1000)),
    ]:
        plt.figure(figsize=(8, 5))
        any_points = False
        for method in method_names:
            score = values(rows, method, "score")
            err = values(rows, method, error_key)
            n = min(len(score), len(err))
            if n == 0:
                continue
            score = score[:n]
            err = err[:n]
            mask = np.isfinite(score) & np.isfinite(err)
            if ylimit is not None:
                mask &= (err >= ylimit[0]) & (err <= ylimit[1])
            if not mask.any():
                continue
            any_points = True
            plt.scatter(score[mask], err[mask], s=8, alpha=0.35, label=method)
        if not any_points:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("score")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"score_vs_{error_key}.png", dpi=160)
        plt.close()


def plot_recall_curves(rows: list[dict[str, str]], output_dir: Path) -> None:
    method_names = methods(rows)
    for key, title, xlabel, xmax in [
        ("translation_error_mm", "Translation recall", "threshold (mm)", 50000),
        ("rotation_error_deg", "Rotation recall", "threshold (deg)", 180),
        ("center_error_px", "Projected-center recall", "threshold (px)", 1000),
    ]:
        plt.figure(figsize=(8, 5))
        any_curve = False
        thresholds = np.linspace(0, xmax, 200)
        for method in method_names:
            vals = values(rows, method, key)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            any_curve = True
            recall = np.asarray([(vals <= threshold).mean() for threshold in thresholds])
            plt.plot(thresholds, recall, label=f"{method} (n={len(vals)})")
        if not any_curve:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("fraction <= threshold")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_recall.png", dpi=160)
        plt.close()


def plot_paired_improvements(comparison_dir: Path, output_dir: Path, bins: int) -> None:
    paired_csv = comparison_dir / "paired_instance_comparison.csv"
    if not paired_csv.is_file() or paired_csv.stat().st_size == 0:
        return
    rows = load_rows(paired_csv)
    if not rows:
        return

    specs = [
        ("translation_error_mm_improvement", "Translation improvement", "original error - fine-tuned error (mm)"),
        ("depth_error_mm_improvement", "Depth improvement", "original error - fine-tuned error (mm)"),
        ("rotation_error_deg_improvement", "Rotation improvement", "original error - fine-tuned error (deg)"),
        ("center_error_px_improvement", "Projected-center improvement", "original error - fine-tuned error (px)"),
        ("add_mm_improvement", "ADD improvement", "original error - fine-tuned error (mm)"),
        ("score_improvement", "Score improvement", "fine-tuned score - original score"),
    ]
    for key, title, xlabel in specs:
        vals = []
        for row in rows:
            try:
                val = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(val):
                vals.append(val)
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue

        plt.figure(figsize=(9, 5))
        plt.hist(vals, bins=bins, alpha=0.75)
        plt.axvline(0, color="black", linewidth=1.5, label="no change")
        plt.axvline(np.median(vals), color="tab:red", linestyle="--", linewidth=1.5, label=f"median={np.median(vals):.2f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("paired instance count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_hist.png", dpi=160)
        plt.close()

    # Paired scatter: points below y=x mean fine-tuned has lower error.
    for metric, title, label, lim in [
        ("translation_error_mm", "Paired translation error", "translation error (mm)", 50000),
        ("rotation_error_deg", "Paired rotation error", "rotation error (deg)", 180),
        ("center_error_px", "Paired projected-center error", "center error (px)", 1000),
    ]:
        base_key = f"baseline_{metric}"
        tuned_key = f"finetuned_{metric}"
        xy = []
        for row in rows:
            try:
                x = float(row[base_key])
                y = float(row[tuned_key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                xy.append((x, y))
        if not xy:
            continue
        arr = np.asarray(xy)
        mask = (arr[:, 0] >= 0) & (arr[:, 1] >= 0) & (arr[:, 0] <= lim) & (arr[:, 1] <= lim)
        if not mask.any():
            continue
        arr = arr[mask]
        plt.figure(figsize=(6, 6))
        plt.scatter(arr[:, 0], arr[:, 1], s=9, alpha=0.35)
        plt.plot([0, lim], [0, lim], color="black", linewidth=1.2)
        plt.title(title)
        plt.xlabel(f"original {label}")
        plt.ylabel(f"fine-tuned {label}")
        plt.xlim(0, lim)
        plt.ylim(0, lim)
        plt.tight_layout()
        plt.savefig(output_dir / f"paired_{metric}_scatter.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or args.comparison_dir / "per_instance_metrics.csv"
    output_dir = args.output_dir or args.comparison_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    plot_histograms(rows, output_dir, args.bins)
    plot_score_vs_error(rows, output_dir)
    plot_recall_curves(rows, output_dir)
    plot_paired_improvements(args.comparison_dir, output_dir, args.bins)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
