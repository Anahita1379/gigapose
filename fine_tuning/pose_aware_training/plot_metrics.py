"""Plot pose-aware CSV metrics without depending on W&B.

Usage:
    python -m fine_tuning.pose_aware_training.plot_metrics \
      gigaPose_datasets/results/MY_RUN/pose_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.csv_path.with_name("pose_metric_history.png")
    series = defaultdict(lambda: ([], []))
    with args.csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["metric"]
            if not any(
                token in name
                for token in (
                    "translation_error",
                    "rotation_error",
                    "depth_abs_error",
                    "reprojection_error",
                    "loss_log_depth",
                    "loss_reprojection",
                    "loss_instance_log_scale",
                    "loss_scale_consistency",
                    "monitor_scale_",
                    "loss_soft_template",
                )
            ):
                continue
            series[name][0].append(int(row["step"]))
            series[name][1].append(float(row["value"]))

    if not series:
        raise RuntimeError(f"No pose-aware metrics found in {args.csv_path}")
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    groups = (
        ("translation_error", "Translation error"),
        ("rotation_error", "Rotation error"),
        ("depth_abs_error", "Absolute depth error"),
        ("reprojection", "Reprojection metrics/loss"),
        ("scale_abs_log_error", "Instance absolute log-scale error"),
        ("scale_", "Scale diagnostics"),
    )
    for axis, (token, title) in zip(axes.flat, groups):
        for name, (steps, values) in sorted(series.items()):
            if token in name:
                axis.plot(steps, values, label=name, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
