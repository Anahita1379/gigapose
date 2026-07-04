"""Plot GigaPose-vs-EPnPv2 candidate comparison CSVs.

Typical use:

    python -m fine_tuning.plot_epnp_gigapose_comparison \
      --input-csv gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/best_candidate_per_epnp_label.csv

The input CSV should be one of the files written by
``fine_tuning.select_real_label_candidates``:

  - all_candidate_pairs.csv
  - best_candidate_per_epnp_label.csv
  - selected_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help=(
            "CSV from select_real_label_candidates.py, usually "
            "best_candidate_per_epnp_label.csv or selected_samples.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-csv parent>/plots_epnp_gigapose.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--camera-from-path",
        action="store_true",
        help="Group per-camera plots by detecting /front/, /rear/, /stereo_left/ in epnp_label_path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    arr = np.asarray([to_float(row, key) for row in rows], dtype=float)
    return arr[np.isfinite(arr)]


def camera_id(row: dict[str, str]) -> str:
    path = str(row.get("epnp_label_path", ""))
    parts = path.replace("\\", "/").split("/")
    for name in ("front", "rear", "stereo_left", "stereo", "left", "right"):
        if name in parts:
            return name
    return row.get("camera_id") or row.get("camera") or "unknown"


def save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hist(rows: list[dict[str, str]], key: str, title: str, output_path: Path, bins: int) -> None:
    vals = values(rows, key)
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins, color="#4C78A8", alpha=0.85)
    median = float(np.median(vals))
    p90 = float(np.percentile(vals, 90))
    ax.axvline(median, color="#1F77B4", linestyle="--", linewidth=2, label=f"median={median:.2f}")
    ax.axvline(p90, color="#F58518", linestyle=":", linewidth=2, label=f"p90={p90:.2f}")
    ax.set_title(title)
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output_path)


def plot_score_scatter(rows: list[dict[str, str]], y_key: str, output_path: Path) -> None:
    score = values(rows, "score")
    y = values(rows, y_key)
    pairs = []
    for row in rows:
        s = to_float(row, "score")
        err = to_float(row, y_key)
        if np.isfinite(s) and np.isfinite(err):
            pairs.append((s, err))
    if not pairs:
        return
    arr = np.asarray(pairs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(arr[:, 0], arr[:, 1], s=14, alpha=0.45, color="#54A24B")
    ax.set_xlabel("GigaPose score")
    ax.set_ylabel(y_key)
    ax.set_title(f"GigaPose score vs {y_key}")
    ax.grid(alpha=0.25)
    save(fig, output_path)


def plot_error_scatter(rows: list[dict[str, str]], output_path: Path) -> None:
    pairs = []
    for row in rows:
        t = to_float(row, "translation_error_mm")
        r = to_float(row, "rotation_error_deg")
        if np.isfinite(t) and np.isfinite(r):
            pairs.append((t, r, to_float(row, "score")))
    if not pairs:
        return
    arr = np.asarray(pairs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = arr[:, 2] if np.isfinite(arr[:, 2]).any() else None
    scatter = ax.scatter(arr[:, 0], arr[:, 1], c=colors, s=16, alpha=0.55, cmap="viridis")
    if colors is not None:
        fig.colorbar(scatter, ax=ax, label="GigaPose score")
    ax.set_xlabel("translation_error_mm")
    ax.set_ylabel("rotation_error_deg")
    ax.set_title("Translation vs rotation error")
    ax.grid(alpha=0.25)
    save(fig, output_path)


def plot_camera_bars(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[camera_id(row)].append(row)
    cameras = sorted(grouped)
    if len(cameras) <= 1:
        return

    metrics = [
        ("translation_error_mm", "Translation error median by camera"),
        ("rotation_error_deg", "Rotation error median by camera"),
        ("score", "GigaPose score median by camera"),
    ]
    for metric, title in metrics:
        labels = []
        medians = []
        p90s = []
        for cam in cameras:
            vals = values(grouped[cam], metric)
            if vals.size == 0:
                continue
            labels.append(cam)
            medians.append(float(np.median(vals)))
            p90s.append(float(np.percentile(vals, 90)))
        if not labels:
            continue
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5))
        ax.bar(x - 0.18, medians, width=0.36, label="median", color="#4C78A8")
        ax.bar(x + 0.18, p90s, width=0.36, label="p90", color="#F58518")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save(fig, output_dir / f"camera_{metric}.png")


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "group",
        "count",
        "translation_error_mm_mean",
        "translation_error_mm_median",
        "translation_error_mm_p90",
        "rotation_error_deg_mean",
        "rotation_error_deg_median",
        "rotation_error_deg_p90",
        "score_mean",
        "score_median",
    ]

    groups: dict[str, list[dict[str, str]]] = {"overall": rows}
    by_camera: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_camera[camera_id(row)].append(row)
    if len(by_camera) > 1:
        for cam, cam_rows in sorted(by_camera.items()):
            groups[f"camera:{cam}"] = cam_rows

    out = []
    for group, group_rows in groups.items():
        item: dict[str, Any] = {"group": group, "count": len(group_rows)}
        for metric in ("translation_error_mm", "rotation_error_deg", "score"):
            vals = values(group_rows, metric)
            if vals.size:
                item[f"{metric}_mean"] = float(np.mean(vals))
                item[f"{metric}_median"] = float(np.median(vals))
                if metric != "score":
                    item[f"{metric}_p90"] = float(np.percentile(vals, 90))
            else:
                item[f"{metric}_mean"] = math.nan
                item[f"{metric}_median"] = math.nan
                if metric != "score":
                    item[f"{metric}_p90"] = math.nan
        out.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    output_dir = args.output_dir or args.input_csv.parent / "plots_epnp_gigapose"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_hist(
        rows,
        "translation_error_mm",
        "GigaPose vs EPnPv2 translation error",
        output_dir / "translation_error_mm_hist.png",
        args.bins,
    )
    plot_hist(
        rows,
        "rotation_error_deg",
        "GigaPose vs EPnPv2 rotation error",
        output_dir / "rotation_error_deg_hist.png",
        args.bins,
    )
    plot_hist(rows, "score", "GigaPose score", output_dir / "score_hist.png", args.bins)
    plot_score_scatter(rows, "translation_error_mm", output_dir / "score_vs_translation_error_mm.png")
    plot_score_scatter(rows, "rotation_error_deg", output_dir / "score_vs_rotation_error_deg.png")
    plot_error_scatter(rows, output_dir / "translation_vs_rotation_error.png")
    plot_camera_bars(rows, output_dir)
    write_summary(rows, output_dir / "summary.csv")

    print(f"Wrote EPnP/GigaPose comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
