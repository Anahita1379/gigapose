"""Run the existing residual GigaPose checkpoint on one prepared GRU dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.common import (
    resolve_predictions,
)


DEFAULT_CHECKPOINT = Path(
    "gigaPose_datasets/results/"
    "assettocorsa_ot2block_pose_aware_ist_translation_residual/"
    "checkpoints/best-residual-step010000.ckpt"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-name", required=True)
    value.add_argument(
        "--datasets-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--run-name")
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--num-workers", type=int, default=2)
    value.add_argument("--devices", default="0")
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    run_name = args.run_name or f"gigapose_{args.dataset_name}"
    command = [
        sys.executable,
        "-m", "fine_tuning.residual_pose_training.infer",
        "--dataset-name", args.dataset_name,
        "--root-dir", str(args.datasets_root),
        "--checkpoint", str(args.checkpoint),
        "--run-name", run_name,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--devices", args.devices,
        "--pose-translation-unit", "mm",
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    result_dir = Path("gigaPose_datasets/results") / run_name
    predictions = resolve_predictions(result_dir)
    report = {
        "format": "rgb_self_recovery_gru_gigapose_run_v1",
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(args.checkpoint),
        "run_name": run_name,
        "result_dir": str(result_dir),
        "predictions": str(predictions),
        "command": command,
    }
    output = result_dir / "gru_gigapose_manifest.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
