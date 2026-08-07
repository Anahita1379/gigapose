"""Plot learned-filter training history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.run_dir / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    panels = (
        ("loss", "Training objective"),
        ("translation_error_m", "Translation error (m)"),
        ("rotation_error_deg", "Rotation error (deg)"),
        ("mean_pose_gain", "Mean learned pose gain"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for split in ("train", "val"):
            axis.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows], label=split)
        axis.set_title(title); axis.set_xlabel("epoch"); axis.grid(alpha=0.25); axis.legend()
    output = args.run_dir / "training_history.png"
    figure.savefig(output, dpi=160); plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
