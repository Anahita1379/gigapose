"""Plot extrinsic optimization results.

Typical use:

    python -m fine_tuning.plot_extrinsic_optimization \
      --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata

For metadata-mode optimizations, pass the combined selected samples too if you
want a before/after lidar-camera axes plot:

    python -m fine_tuning.plot_extrinsic_optimization \
      --optimization-dir .../extrinsic_optimization_front_metadata \
      --selected-samples .../combined_front_selected_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-samples",
        type=Path,
        default=None,
        help=(
            "Optional combined/selected_samples.csv. In metadata mode this lets "
            "the script load the original t_lidar_camera_prior and draw original "
            "vs optimized camera axes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <optimization-dir>/plots.",
    )
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([to_float(row, key) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize(vals: np.ndarray) -> dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
    }


def plot_hist(before: list[dict[str, str]], after: list[dict[str, str]], metric: str, output: Path, bins: int) -> None:
    b = values(before, metric)
    a = values(after, metric)
    if b.size == 0 and a.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if b.size:
        ax.hist(b, bins=bins, alpha=0.55, label="before", color="#E45756")
        ax.axvline(np.median(b), color="#E45756", linestyle="--", linewidth=2)
    if a.size:
        ax.hist(a, bins=bins, alpha=0.55, label="after", color="#4C78A8")
        ax.axvline(np.median(a), color="#4C78A8", linestyle="--", linewidth=2)
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output)


def plot_summary_bars(before: list[dict[str, str]], after: list[dict[str, str]], output: Path) -> dict[str, Any]:
    metrics = ["translation_error_mm", "rotation_error_deg"]
    stats = ["mean", "median", "p90"]
    summary: dict[str, Any] = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, metrics):
        b = summarize(values(before, metric))
        a = summarize(values(after, metric))
        summary[f"before_{metric}"] = b
        summary[f"after_{metric}"] = a
        x = np.arange(len(stats))
        width = 0.36
        ax.bar(x - width / 2, [b[s] for s in stats], width, label="before", color="#E45756")
        ax.bar(x + width / 2, [a[s] for s in stats], width, label="after", color="#4C78A8")
        ax.set_xticks(x, stats)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    save(fig, output)
    return summary


def plot_per_sample(before: list[dict[str, str]], after: list[dict[str, str]], metric: str, output: Path) -> None:
    n = min(len(before), len(after))
    if n == 0:
        return
    pairs = []
    for idx in range(n):
        b = to_float(before[idx], metric)
        a = to_float(after[idx], metric)
        if np.isfinite(b) and np.isfinite(a):
            pairs.append((idx, b, a))
    if not pairs:
        return
    pairs.sort(key=lambda item: item[1], reverse=True)
    x = np.arange(len(pairs))
    b = np.asarray([p[1] for p in pairs])
    a = np.asarray([p[2] for p in pairs])
    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.22), 5))
    ax.plot(x, b, marker="o", linewidth=1.4, label="before", color="#E45756")
    ax.plot(x, a, marker="o", linewidth=1.4, label="after", color="#4C78A8")
    ax.set_title(f"Per-sample {metric.replace('_', ' ')}")
    ax.set_xlabel("sample sorted by before-error")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output)


def plot_correction(extrinsics: dict[str, Any], output: Path) -> None:
    t = np.asarray(extrinsics.get("correction_translation_mm", [math.nan] * 3), dtype=float)
    r = np.degrees(np.asarray(extrinsics.get("correction_rotation_rpy_like_vector_rad", [math.nan] * 3), dtype=float))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["tx", "ty", "tz"], t, color="#72B7B2")
    axes[0].set_title("Correction translation")
    axes[0].set_ylabel("mm")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(["rx", "ry", "rz"], r, color="#F58518")
    axes[1].set_title("Correction rotation vector")
    axes[1].set_ylabel("deg")
    axes[1].grid(axis="y", alpha=0.25)
    save(fig, output)


def text_to_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return arr.reshape(4, 4)


def matrix_3x4_or_4x4(value: Any, unit: str = "m") -> np.ndarray:
    flat = np.asarray(value, dtype=float).reshape(-1)
    if flat.size == 16:
        T = flat.reshape(4, 4).copy()
    elif flat.size == 12:
        T = np.eye(4)
        T[:3, :] = flat.reshape(3, 4)
    else:
        raise ValueError(f"Expected 12 or 16 values, got {flat.size}")
    if unit == "m":
        T[:3, 3] *= 1000.0
    return T


def metadata_path_from_label(label_path: Path) -> Path:
    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if not match:
        raise ValueError(f"Cannot infer metadata path from {label_path}")
    timestamp, obj_index = match.groups()
    return label_path.parent.parent / "metadata" / f"sample_{timestamp}_{obj_index}.yaml"


def load_first_prior_from_selected(selected_samples: Path) -> tuple[np.ndarray, str] | None:
    try:
        import yaml
    except ImportError:
        return None
    rows = read_csv(selected_samples)
    for row in rows:
        metadata_path = row.get("sample_metadata_path") or row.get("metadata_path")
        if metadata_path:
            path = Path(metadata_path)
        else:
            label_path = row.get("epnp_label_path")
            if not label_path:
                continue
            path = metadata_path_from_label(Path(label_path))
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "t_lidar_camera_prior" in data:
            return matrix_3x4_or_4x4(data["t_lidar_camera_prior"], "m"), str(path)
    return None


def draw_axes(ax: Any, T: np.ndarray, name: str, colors: tuple[str, str, str], scale: float = 500.0) -> None:
    origin = T[:3, 3]
    axes = T[:3, :3]
    for idx, color in enumerate(colors):
        vec = axes[:, idx] * scale
        ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], color=color, linewidth=2)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], s=45, label=name)


def plot_axes(extrinsics: dict[str, Any], selected_samples: Path | None, output: Path) -> None:
    mode = extrinsics.get("optimization_mode", "")
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    title = ""

    if mode == "sample_metadata_lidar_camera_prior" and selected_samples is not None:
        loaded = load_first_prior_from_selected(selected_samples)
        if loaded is not None:
            before, metadata_path = loaded
            correction = text_to_matrix(extrinsics["T_lidar_camera_correction_left_multiply"])
            after = correction @ before
            title = f"Lidar-camera prior vs optimized\nmetadata: {Path(metadata_path).name}"
    elif extrinsics.get("T_map_cam_initial") is not None and extrinsics.get("T_map_cam_optimized") is not None:
        before = text_to_matrix(extrinsics["T_map_cam_initial"])
        after = text_to_matrix(extrinsics["T_map_cam_optimized"])
        title = "Map-camera initial vs optimized"

    if before is None or after is None:
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_axes(ax, before, "original", ("#E45756", "#F58518", "#FF9DA6"))
    draw_axes(ax, after, "optimized", ("#4C78A8", "#54A24B", "#72B7B2"))
    pts = np.vstack([before[:3, 3], after[:3, 3]])
    center = pts.mean(axis=0)
    radius = max(float(np.linalg.norm(pts[0] - pts[1])) + 800.0, 1200.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.set_title(title)
    ax.legend()
    save(fig, output)


def main() -> None:
    args = parse_args()
    opt_dir = args.optimization_dir
    output_dir = args.output_dir or opt_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    extrinsics_path = opt_dir / "optimized_extrinsics.json"
    report_path = opt_dir / "optimization_report.json"
    before_path = opt_dir / "errors_before_optimization.csv"
    after_path = opt_dir / "errors_after_optimization.csv"

    extrinsics = json.loads(extrinsics_path.read_text())
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    before = read_csv(before_path)
    after = read_csv(after_path)

    plot_hist(before, after, "translation_error_mm", output_dir / "translation_error_mm_hist.png", args.bins)
    plot_hist(before, after, "rotation_error_deg", output_dir / "rotation_error_deg_hist.png", args.bins)
    summary = plot_summary_bars(before, after, output_dir / "before_after_summary_bars.png")
    plot_per_sample(before, after, "translation_error_mm", output_dir / "per_sample_translation_error_mm.png")
    plot_per_sample(before, after, "rotation_error_deg", output_dir / "per_sample_rotation_error_deg.png")
    plot_correction(extrinsics, output_dir / "extrinsic_correction_components.png")
    plot_axes(extrinsics, args.selected_samples, output_dir / "extrinsic_axes_before_after.png")

    plot_report = {
        "optimization_dir": str(opt_dir),
        "plots_dir": str(output_dir),
        "optimizer_report": report,
        "summary_from_error_csvs": summary,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(plot_report, indent=2))
    print(f"Wrote extrinsic optimization plots to {output_dir}")


if __name__ == "__main__":
    main()
