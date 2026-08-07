"""Compare raw GigaPose and recovery outputs on one development run."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from .common import resolve_predictions
from .run_suite import MODEL_SPECS, selected_models


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--gigapose-predictions", type=Path, required=True)
    value.add_argument("--runs-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--models", default=",".join(MODEL_SPECS))
    value.add_argument("--max-overlays", type=int, default=120)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--skip-missing", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def command(args: argparse.Namespace) -> list[str]:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    gigapose_predictions = resolve_predictions(args.gigapose_predictions)
    mesh = args.dataset_dir / "models" / "obj_000001.ply"
    result = [
        sys.executable,
        "-m",
        (
            "tracking.rgb_self_recovery.sliding_window.assetto_validation_eval."
            "evaluate_models"
        ),
        "--dataset-dir",
        str(args.dataset_dir),
        "--split",
        "test",
        "--model",
        f"gigapose={gigapose_predictions}",
        "--baseline",
        "gigapose",
        "--mesh",
        str(mesh),
        "--prediction-translation-unit",
        "mm",
        "--distance-bins-m",
        "0,20,40,60,80,100,120",
        "--max-overlays",
        str(args.max_overlays),
        "--output-dir",
        str(args.output_dir),
    ]
    included = 0
    for name in selected_models(args.models):
        prediction = args.runs_root / name / "tracked_predictions.csv"
        if not prediction.is_file():
            if args.skip_missing:
                print(f"Skipping {name}; output is absent: {prediction}")
                continue
            raise FileNotFoundError(prediction)
        result.extend(["--model", f"{name}={prediction}"])
        included += 1
    if not included:
        raise RuntimeError("No recovery outputs are available for comparison")
    if args.max_frames is not None:
        result.extend(["--max-frames", str(args.max_frames)])
    if args.overwrite:
        result.append("--overwrite")
    return result


def main() -> None:
    args = parser().parse_args()
    value = command(args)
    print(shlex.join(value), flush=True)
    if not args.dry_run:
        subprocess.run(value, check=True)


if __name__ == "__main__":
    main()
