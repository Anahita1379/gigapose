"""Evaluate only translation/rotation errors and plot them vs GT distance.

This is a lighter, pose-focused companion to ``evaluate_gigapose_models.py``.
It evaluates one or more GigaPose prediction CSVs against the prepared dataset
GT and writes:

- per-instance translation/rotation errors;
- overall model summary with mean, variance, std, median, p90;
- camera-by-camera summary;
- distance-bin summary;
- Markdown tables;
- comparison plots for mean/median/variance;
- translation/rotation error vs GT distance plots.

Example:

python -m fine_tuning.evaluate_pose_errors_by_distance \
  --model original=path/to/original.csv \
  --model finetune=path/to/finetune.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir gigaPose_datasets/results/final_results/metrics/pose_distance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning.compare_gigapose_predictions import evaluate_method, load_gt, write_csv


POSE_METRICS = ("translation_error_mm", "rotation_error_deg")


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be name=/path/to/predictions.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Prediction CSV as name=/path/to/file.csv. Repeat for multiple models.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--distance-bin-m",
        type=float,
        default=10.0,
        help="Distance bin width in meters for binned distance plots/tables.",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=None,
        help="Optional max GT distance to include in plots/summaries.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Optional camera_id to include. Repeat for multiple cameras.",
    )
    parser.add_argument("--plot-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--scatter-alpha",
        type=float,
        default=0.30,
        help="Alpha for per-instance scatter plots.",
    )
    return parser.parse_args()


def load_camera_map(dataset_dir: Path) -> dict[tuple[int, int], str]:
    frame_map_path = dataset_dir / "frame_map.json"
    if not frame_map_path.is_file():
        return {}
    rows = json.loads(frame_map_path.read_text())
    return {
        (int(row["scene_id"]), int(row["im_id"])): str(row.get("camera_id", "unknown_camera"))
        for row in rows
    }


def finite_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([finite_float(row.get(key)) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = finite_values(rows, key)
    if vals.size == 0:
        return {
            f"{key}_mean": float("nan"),
            f"{key}_variance": float("nan"),
            f"{key}_std": float("nan"),
            f"{key}_median": float("nan"),
            f"{key}_p90": float("nan"),
        }
    return {
        f"{key}_mean": float(np.mean(vals)),
        f"{key}_variance": float(np.var(vals)),
        f"{key}_std": float(np.std(vals)),
        f"{key}_median": float(np.median(vals)),
        f"{key}_p90": float(np.percentile(vals, 90)),
    }


def summarize_group(name_fields: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(name_fields)
    out["evaluated_instances"] = len(rows)
    for key in POSE_METRICS:
        out.update(metric_stats(rows, key))
    distances = finite_values(rows, "gt_distance_m")
    if distances.size:
        out["gt_distance_m_mean"] = float(np.mean(distances))
        out["gt_distance_m_median"] = float(np.median(distances))
        out["gt_distance_m_min"] = float(np.min(distances))
        out["gt_distance_m_max"] = float(np.max(distances))
    return out


def add_gt_distance_and_camera(
    rows: list[dict[str, Any]],
    gt_by_image: dict[tuple[int, int], list[dict[str, Any]]],
    camera_map: dict[tuple[int, int], str],
) -> None:
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]))
        gt_index = int(row["gt_index"])
        gt = gt_by_image[key][gt_index]
        gt_t = np.asarray(gt["t_mm"], dtype=float).reshape(3)
        row["gt_distance_m"] = float(np.linalg.norm(gt_t) * 0.001)
        row["gt_depth_m"] = float(gt_t[2] * 0.001)
        row["camera_id"] = camera_map.get(key, "unknown_camera")


def filter_rows(rows: list[dict[str, Any]], cameras: set[str] | None, max_distance_m: float | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if cameras is not None and str(row.get("camera_id", "")) not in cameras:
            continue
        dist = finite_float(row.get("gt_distance_m"))
        if max_distance_m is not None and np.isfinite(dist) and dist > max_distance_m:
            continue
        out.append(row)
    return out


def overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    return [
        summarize_group({"method": method}, items)
        for method, items in sorted(grouped.items())
    ]


def camera_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row.get("camera_id", "unknown_camera")))].append(row)
    return [
        summarize_group({"method": method, "camera_id": camera_id}, items)
        for (method, camera_id), items in sorted(grouped.items())
    ]


def distance_bin_label(start: float, end: float) -> str:
    return f"{start:g}-{end:g}m"


def distance_bin_summary(rows: list[dict[str, Any]], bin_width_m: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dist = finite_float(row.get("gt_distance_m"))
        if not np.isfinite(dist):
            continue
        bin_idx = int(math.floor(dist / bin_width_m))
        grouped[(str(row["method"]), str(row.get("camera_id", "unknown_camera")), bin_idx)].append(row)

    out = []
    for (method, camera_id, bin_idx), items in sorted(grouped.items()):
        start = bin_idx * bin_width_m
        end = start + bin_width_m
        summary = summarize_group(
            {
                "method": method,
                "camera_id": camera_id,
                "distance_bin_start_m": start,
                "distance_bin_end_m": end,
                "distance_bin": distance_bin_label(start, end),
            },
            items,
        )
        out.append(summary)
    return out


def format_number(value: Any, digits: int = 2) -> str:
    v = finite_float(value)
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, int]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _, digits in columns:
            value = row.get(key, "")
            if isinstance(value, str):
                cells.append(value)
            else:
                cells.append(format_number(value, digits))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def write_markdown_summary(
    path: Path,
    overall: list[dict[str, Any]],
    camera: list[dict[str, Any]],
    distance_bins: list[dict[str, Any]],
) -> None:
    overall_cols = [
        ("method", "model", 0),
        ("evaluated_instances", "n", 0),
        ("translation_error_mm_mean", "t mean mm", 1),
        ("translation_error_mm_variance", "t variance", 1),
        ("translation_error_mm_std", "t std mm", 1),
        ("translation_error_mm_median", "t median mm", 1),
        ("rotation_error_deg_mean", "R mean deg", 2),
        ("rotation_error_deg_variance", "R variance", 2),
        ("rotation_error_deg_std", "R std deg", 2),
        ("rotation_error_deg_median", "R median deg", 2),
    ]
    camera_cols = [
        ("method", "model", 0),
        ("camera_id", "camera", 0),
        ("evaluated_instances", "n", 0),
        ("translation_error_mm_mean", "t mean mm", 1),
        ("translation_error_mm_variance", "t variance", 1),
        ("translation_error_mm_median", "t median mm", 1),
        ("rotation_error_deg_mean", "R mean deg", 2),
        ("rotation_error_deg_variance", "R variance", 2),
        ("rotation_error_deg_median", "R median deg", 2),
    ]
    distance_cols = [
        ("method", "model", 0),
        ("camera_id", "camera", 0),
        ("distance_bin", "GT distance", 0),
        ("evaluated_instances", "n", 0),
        ("translation_error_mm_median", "t median mm", 1),
        ("translation_error_mm_mean", "t mean mm", 1),
        ("rotation_error_deg_median", "R median deg", 2),
        ("rotation_error_deg_mean", "R mean deg", 2),
    ]
    text = [
        "# Pose error summary",
        "",
        "Lower is better for all metrics.",
        "",
        "## Overall",
        "",
        markdown_table(overall, overall_cols),
        "",
        "## By camera",
        "",
        markdown_table(camera, camera_cols),
        "",
        "## By GT distance bin",
        "",
        markdown_table(distance_bins, distance_cols),
        "",
    ]
    path.write_text("\n".join(text))


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_summary_bars(rows: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    plt = import_matplotlib()
    metrics = [
        ("translation_error_mm", "Translation error (mm)"),
        ("rotation_error_deg", "Rotation error (deg)"),
    ]
    stats = [
        ("mean", "mean"),
        ("median", "median"),
        ("variance", "variance"),
    ]
    methods = [str(row["method"]) for row in rows]
    x = np.arange(len(methods))

    for metric, ylabel in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(max(9, 2.7 * len(methods)), 4))
        for ax, (stat, title) in zip(axes, stats):
            key = f"{metric}_{stat}"
            values = [finite_float(row.get(key)) for row in rows]
            ax.bar(x, values)
            ax.set_title(title)
            ax.set_ylabel(ylabel if stat != "variance" else f"{ylabel} variance")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{ylabel}: model comparison")
        fig.tight_layout()
        fig.savefig(output_dir / f"overall_{metric}_mean_median_variance.{fmt}", dpi=dpi)
        plt.close(fig)


def plot_camera_bars(rows: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    plt = import_matplotlib()
    cameras = sorted({str(row.get("camera_id", "unknown_camera")) for row in rows})
    metrics = [
        ("translation_error_mm_median", "Translation median (mm)"),
        ("rotation_error_deg_median", "Rotation median (deg)"),
        ("translation_error_mm_variance", "Translation variance"),
        ("rotation_error_deg_variance", "Rotation variance"),
    ]
    for metric, ylabel in metrics:
        for camera in cameras:
            items = [row for row in rows if str(row.get("camera_id")) == camera]
            if not items:
                continue
            methods = [str(row["method"]) for row in items]
            values = [finite_float(row.get(metric)) for row in items]
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 4))
            ax.bar(np.arange(len(methods)), values)
            ax.set_title(f"{camera}: {ylabel}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            safe_camera = camera.replace("/", "_").replace(" ", "_")
            fig.savefig(output_dir / f"camera_{safe_camera}_{metric}.{fmt}", dpi=dpi)
            plt.close(fig)


def plot_error_vs_distance(
    instance_rows: list[dict[str, Any]],
    distance_rows: list[dict[str, Any]],
    output_dir: Path,
    fmt: str,
    dpi: int,
    scatter_alpha: float,
) -> None:
    plt = import_matplotlib()
    metrics = [
        ("translation_error_mm", "Translation error (mm)"),
        ("rotation_error_deg", "Rotation error (deg)"),
    ]
    cameras = ["all", *sorted({str(row.get("camera_id", "unknown_camera")) for row in instance_rows})]
    methods = sorted({str(row["method"]) for row in instance_rows})

    for metric, ylabel in metrics:
        for camera in cameras:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in methods:
                rows = [
                    row
                    for row in instance_rows
                    if str(row["method"]) == method
                    and (camera == "all" or str(row.get("camera_id")) == camera)
                ]
                if not rows:
                    continue
                x = np.asarray([finite_float(row.get("gt_distance_m")) for row in rows], dtype=float)
                y = np.asarray([finite_float(row.get(metric)) for row in rows], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[valid], y[valid], s=12, alpha=scatter_alpha, label=f"{method} samples")

                bins = [
                    row
                    for row in distance_rows
                    if str(row["method"]) == method
                    and (camera == "all" or str(row.get("camera_id")) == camera)
                ]
                if bins:
                    bx = np.asarray(
                        [
                            0.5 * (finite_float(row["distance_bin_start_m"]) + finite_float(row["distance_bin_end_m"]))
                            for row in bins
                        ],
                        dtype=float,
                    )
                    by = np.asarray([finite_float(row.get(f"{metric}_median")) for row in bins], dtype=float)
                    valid_bins = np.isfinite(bx) & np.isfinite(by)
                    ax.plot(bx[valid_bins], by[valid_bins], marker="o", linewidth=2, label=f"{method} binned median")
            ax.set_xlabel("GT distance to car (m)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs GT distance ({camera})")
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small")
            fig.tight_layout()
            safe_camera = camera.replace("/", "_").replace(" ", "_")
            fig.savefig(output_dir / f"{metric}_vs_gt_distance_{safe_camera}.{fmt}", dpi=dpi)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_by_image, K_by_image = load_gt(args.dataset_dir, args.split, load_masks=False)
    camera_map = load_camera_map(args.dataset_dir)

    all_rows: list[dict[str, Any]] = []
    for name, prediction_path in args.model:
        rows, _ = evaluate_method(
            name=name,
            prediction_path=prediction_path,
            gt_by_image=gt_by_image,
            K_by_image=K_by_image,
            instance_map={},
            vertices_mm=None,
            renderer=None,
        )
        add_gt_distance_and_camera(rows, gt_by_image, camera_map)
        all_rows.extend(rows)

    cameras = set(args.camera) if args.camera else None
    all_rows = filter_rows(all_rows, cameras, args.max_distance_m)

    overall = overall_summary(all_rows)
    by_camera = camera_summary(all_rows)
    by_distance = distance_bin_summary(all_rows, args.distance_bin_m)

    write_csv(args.output_dir / "pose_instance_errors.csv", all_rows)
    write_csv(args.output_dir / "pose_overall_summary.csv", overall)
    write_csv(args.output_dir / "pose_camera_summary.csv", by_camera)
    write_csv(args.output_dir / "pose_distance_bin_summary.csv", by_distance)
    (args.output_dir / "pose_overall_summary.json").write_text(json.dumps(overall, indent=2))
    (args.output_dir / "pose_camera_summary.json").write_text(json.dumps(by_camera, indent=2))
    (args.output_dir / "pose_distance_bin_summary.json").write_text(json.dumps(by_distance, indent=2))
    write_markdown_summary(args.output_dir / "pose_summary.md", overall, by_camera, by_distance)

    plot_summary_bars(overall, plots_dir, args.plot_format, args.dpi)
    plot_camera_bars(by_camera, plots_dir, args.plot_format, args.dpi)
    plot_error_vs_distance(
        all_rows,
        by_distance,
        plots_dir,
        args.plot_format,
        args.dpi,
        args.scatter_alpha,
    )

    print(f"Evaluated {len(args.model)} model(s), {len(all_rows)} pose instances")
    for row in overall:
        print(
            f"{row['method']}: n={row['evaluated_instances']} "
            f"t_mean={row.get('translation_error_mm_mean', float('nan')):.1f}mm "
            f"t_var={row.get('translation_error_mm_variance', float('nan')):.1f} "
            f"t_med={row.get('translation_error_mm_median', float('nan')):.1f}mm "
            f"R_mean={row.get('rotation_error_deg_mean', float('nan')):.2f}deg "
            f"R_var={row.get('rotation_error_deg_variance', float('nan')):.2f} "
            f"R_med={row.get('rotation_error_deg_median', float('nan')):.2f}deg"
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
