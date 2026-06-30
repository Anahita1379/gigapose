"""Plot multi-model summary metrics from evaluate_gigapose_models outputs.

Example:
    python -m fine_tuning.plot_model_summary \
      --metrics-dir gigaPose_datasets/results/final_results/metrics/all_models

Expected inputs:
    overall_summary.csv
    camera_summary.csv

Outputs are written to:
    <metrics-dir>/plots_summary/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_METRICS = [
    {
        "column": "translation_error_mm_median",
        "label": "Median translation error",
        "ylabel": "mm",
        "filename": "translation_error_mm_median",
        "better": "lower",
    },
    {
        "column": "rotation_error_deg_median",
        "label": "Median rotation error",
        "ylabel": "degrees",
        "filename": "rotation_error_deg_median",
        "better": "lower",
    },
    {
        "column": "center_error_px_median",
        "label": "Median projected-center error",
        "ylabel": "pixels",
        "filename": "center_error_px_median",
        "better": "lower",
    },
    {
        "column": "add_mm_median",
        "label": "Median ADD",
        "ylabel": "mm",
        "filename": "add_mm_median",
        "better": "lower",
    },
    {
        "column": "pred_bbox_iou_median",
        "label": "Median rendered bbox IoU",
        "ylabel": "IoU",
        "filename": "pred_bbox_iou_median",
        "better": "higher",
        "ylim": (0.0, 1.0),
    },
    {
        "column": "pred_mask_iou_median",
        "label": "Median rendered mask IoU",
        "ylabel": "IoU",
        "filename": "pred_mask_iou_median",
        "better": "higher",
        "ylim": (0.0, 1.0),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create overall and camera-by-camera bar charts from "
            "overall_summary.csv and camera_summary.csv."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing overall_summary.csv and camera_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <metrics-dir>/plots_summary.",
    )
    parser.add_argument(
        "--overall-csv",
        type=Path,
        default=None,
        help="Optional explicit path to overall_summary.csv.",
    )
    parser.add_argument(
        "--camera-csv",
        type=Path,
        default=None,
        help="Optional explicit path to camera_summary.csv.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Plot file format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=9.5,
        help="Base figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.2,
        help="Base figure height in inches.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("file", "metric", "method"),
        default="file",
        help=(
            "For overall plots: keep CSV order, sort by metric value, or sort by "
            "method name."
        ),
    )
    parser.add_argument(
        "--camera-order",
        default=None,
        help="Optional comma-separated camera order, e.g. front,rear,stereo_left,stereo_right.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def safe_name(text: str) -> str:
    return (
        text.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def maybe_sort_overall(
    rows: list[dict[str, str]], metric_column: str, sort_by: str
) -> list[dict[str, str]]:
    if sort_by == "metric":
        return sorted(
            rows,
            key=lambda row: (
                not math.isfinite(to_float(row.get(metric_column))),
                to_float(row.get(metric_column)),
            ),
        )
    if sort_by == "method":
        return sorted(rows, key=lambda row: row.get("method", ""))
    return rows


def annotate_bars(ax: plt.Axes, bars, values: list[float], metric: dict[str, object]) -> None:
    finite_values = finite(values)
    if not finite_values:
        return
    span = max(finite_values) - min(finite_values)
    offset = span * 0.015 if span > 0 else max(abs(finite_values[0]) * 0.015, 0.01)
    is_iou = metric.get("ylabel") == "IoU"

    for bar, value in zip(bars, values):
        if not math.isfinite(value):
            continue
        label = f"{value:.3f}" if is_iou else f"{value:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def save_plot(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overall_metric(
    rows: list[dict[str, str]],
    metric: dict[str, object],
    output_dir: Path,
    file_format: str,
    dpi: int,
    fig_size: tuple[float, float],
    sort_by: str,
) -> Path | None:
    column = str(metric["column"])
    if not rows or column not in rows[0]:
        return None

    rows = maybe_sort_overall(rows, column, sort_by)
    methods = [row.get("method", "unknown") for row in rows]
    values = [to_float(row.get(column)) for row in rows]
    if not finite(values):
        return None

    fig, ax = plt.subplots(figsize=fig_size)
    bars = ax.bar(methods, values, color="#4C78A8")
    ax.set_title(f"Overall {metric['label']}")
    ax.set_ylabel(str(metric["ylabel"]))
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=25)
    if "ylim" in metric:
        ax.set_ylim(*metric["ylim"])  # type: ignore[arg-type]
    annotate_bars(ax, bars, values, metric)
    ax.text(
        0.99,
        0.98,
        f"{metric['better']} is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    output_path = output_dir / f"overall_{metric['filename']}.{file_format}"
    save_plot(fig, output_path, dpi)
    return output_path


def ordered_unique(values: list[str], preferred_order: list[str] | None = None) -> list[str]:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    if not preferred_order:
        return seen
    ordered = [value for value in preferred_order if value in seen]
    ordered.extend(value for value in seen if value not in ordered)
    return ordered


def plot_camera_metric(
    rows: list[dict[str, str]],
    metric: dict[str, object],
    output_dir: Path,
    file_format: str,
    dpi: int,
    fig_size: tuple[float, float],
    camera_order: list[str] | None,
) -> Path | None:
    column = str(metric["column"])
    if not rows or column not in rows[0] or "camera_id" not in rows[0]:
        return None

    methods = ordered_unique([row.get("method", "unknown") for row in rows])
    cameras = ordered_unique([row.get("camera_id", "unknown") for row in rows], camera_order)
    if not methods or not cameras:
        return None

    values_by_key = {
        (row.get("camera_id", "unknown"), row.get("method", "unknown")): to_float(row.get(column))
        for row in rows
    }
    if not finite(list(values_by_key.values())):
        return None

    x_positions = list(range(len(cameras)))
    group_width = 0.82
    bar_width = group_width / max(len(methods), 1)

    width = max(fig_size[0], 1.4 * len(cameras) + 1.2 * len(methods))
    fig, ax = plt.subplots(figsize=(width, fig_size[1]))

    for method_index, method in enumerate(methods):
        offsets = [
            x - group_width / 2 + bar_width / 2 + method_index * bar_width
            for x in x_positions
        ]
        values = [values_by_key.get((camera, method), math.nan) for camera in cameras]
        ax.bar(offsets, values, width=bar_width, label=method)

    ax.set_title(f"Camera-by-camera {metric['label']}")
    ax.set_ylabel(str(metric["ylabel"]))
    ax.set_xlabel("Camera")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(cameras, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Model", fontsize=9)
    if "ylim" in metric:
        ax.set_ylim(*metric["ylim"])  # type: ignore[arg-type]
    ax.text(
        0.99,
        0.98,
        f"{metric['better']} is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    output_path = output_dir / f"camera_{metric['filename']}.{file_format}"
    save_plot(fig, output_path, dpi)
    return output_path


def plot_metric_grid(
    rows: list[dict[str, str]],
    output_dir: Path,
    file_format: str,
    dpi: int,
    sort_by: str,
) -> Path | None:
    if not rows:
        return None
    available = [metric for metric in SUMMARY_METRICS if metric["column"] in rows[0]]
    if not available:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes_flat = list(axes.flatten())

    for ax, metric in zip(axes_flat, available):
        column = str(metric["column"])
        metric_rows = maybe_sort_overall(rows, column, sort_by)
        methods = [row.get("method", "unknown") for row in metric_rows]
        values = [to_float(row.get(column)) for row in metric_rows]
        ax.bar(methods, values, color="#4C78A8")
        ax.set_title(str(metric["label"]))
        ax.set_ylabel(str(metric["ylabel"]))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", labelrotation=25)
        if "ylim" in metric:
            ax.set_ylim(*metric["ylim"])  # type: ignore[arg-type]

    for ax in axes_flat[len(available) :]:
        ax.axis("off")

    fig.suptitle("Overall model summary", fontsize=15)
    output_path = output_dir / f"overall_summary_grid.{file_format}"
    save_plot(fig, output_path, dpi)
    return output_path


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir
    output_dir = args.output_dir or metrics_dir / "plots_summary"
    overall_csv = args.overall_csv or metrics_dir / "overall_summary.csv"
    camera_csv = args.camera_csv or metrics_dir / "camera_summary.csv"
    camera_order = (
        [item.strip() for item in args.camera_order.split(",") if item.strip()]
        if args.camera_order
        else None
    )

    overall_rows = read_csv(overall_csv)
    camera_rows = read_csv(camera_csv)

    written: list[Path] = []
    fig_size = (args.fig_width, args.fig_height)

    grid_path = plot_metric_grid(
        overall_rows,
        output_dir,
        args.format,
        args.dpi,
        args.sort_by,
    )
    if grid_path:
        written.append(grid_path)

    for metric in SUMMARY_METRICS:
        overall_path = plot_overall_metric(
            overall_rows,
            metric,
            output_dir,
            args.format,
            args.dpi,
            fig_size,
            args.sort_by,
        )
        if overall_path:
            written.append(overall_path)

        camera_path = plot_camera_metric(
            camera_rows,
            metric,
            output_dir,
            args.format,
            args.dpi,
            fig_size,
            camera_order,
        )
        if camera_path:
            written.append(camera_path)

    print(f"Wrote {len(written)} summary plots to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
