"""Plot GRU selector training and validation history."""

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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    history_path = args.run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No epochs in {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    panels = (
        ("loss", "Objective (lower is better)"),
        ("oracle_accuracy", "Oracle candidate accuracy"),
        ("selected_cost", "Selected oracle cost (lower is better)"),
        ("selected_recovery_fraction", "Recovery selection fraction"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for split in ("train", "val"):
            key = f"{split}_{metric}"
            axis.plot(epochs, [float(row[key]) for row in rows], label=split)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    output = args.output or args.run_dir / "training_history.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
