"""Run the current residual GigaPose model on both prepared evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .common import resolve_predictions
from .prepare import DEFAULT_RUNS


DEFAULT_CHECKPOINT = Path(
    "gigaPose_datasets/results/"
    "assettocorsa_ot2block_pose_aware_ist_translation_residual/"
    "checkpoints/best-residual-step010000.ckpt"
)
RESULTS_ROOT = Path("gigaPose_datasets/results")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument(
        "--datasets-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    value.add_argument("--run-prefix", default="gigapose_dev")
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--num-workers", type=int, default=2)
    value.add_argument("--devices", default="0")
    value.add_argument("--dry-run", action="store_true")
    return value


def commands(args: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    result = []
    for dataset_name in DEFAULT_RUNS:
        dataset_dir = args.datasets_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(dataset_dir)
        run_name = f"{args.run_prefix}_{dataset_name}"
        command = [
            sys.executable,
            "-m",
            "fine_tuning.residual_pose_training.infer",
            "--dataset-name",
            dataset_name,
            "--root-dir",
            str(args.datasets_root),
            "--checkpoint",
            str(args.checkpoint),
            "--run-name",
            run_name,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--devices",
            args.devices,
            "--pose-translation-unit",
            "mm",
        ]
        result.append((dataset_name, run_name, command))
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = {}
    for dataset_name, run_name, command in commands(args):
        print(f"[{dataset_name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        result_dir = RESULTS_ROOT / run_name
        predictions = resolve_predictions(result_dir)
        manifest[dataset_name] = {
            "dataset_dir": str(args.datasets_root / dataset_name),
            "run_name": run_name,
            "result_dir": str(result_dir),
            "predictions": str(predictions),
        }
    if not args.dry_run:
        output = RESULTS_ROOT / f"{args.run_prefix}_manifest.json"
        output.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
