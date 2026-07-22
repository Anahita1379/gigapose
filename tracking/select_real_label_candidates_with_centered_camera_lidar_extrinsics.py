"""Select tracked real labels using a centered camera-LiDAR calibration.

This wrapper filters ``tracked_predictions.csv`` by tracking mode/confidence,
then delegates geometric matching to
``fine_tuning.select_real_label_candidates_with_centered_camera_lidar_extrinsics``.
It is separate from the existing tracking selectors so their behavior remains
unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from tracking.select_real_label_candidates import (
    OUTPUT_CSV_NAMES,
    append_tracking_metadata,
    filter_tracking_rows,
    merge_filter_report,
    tracking_metadata_by_filtered_index,
    write_rows,
)


SELECTOR_MODULE = (
    "fine_tuning."
    "select_real_label_candidates_with_centered_camera_lidar_extrinsics"
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Filter tracked predictions, then select EPnP labels using the "
            "explicit centered-object camera-LiDAR calibration. Unrecognized "
            "arguments are forwarded to the fine_tuning selector."
        )
    )
    parser.add_argument("--tracked-predictions", type=Path, required=True)
    parser.add_argument("--optimized-extrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-tracking-modes",
        nargs="+",
        choices=("normal", "uncertain", "lost"),
        default=("normal", "uncertain"),
        help=(
            "Tracking states retained before matching. Use only 'normal' for "
            "strict pseudo-label selection. Lost is excluded by default."
        ),
    )
    parser.add_argument("--min-tracking-confidence", type=float, default=0.0)
    parser.add_argument("--keep-missing-tracking-mode", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if "--gigapose-predictions" in forwarded:
        parser.error(
            "Do not pass --gigapose-predictions; use --tracked-predictions."
        )
    if not math.isfinite(args.min_tracking_confidence):
        parser.error("--min-tracking-confidence must be finite.")
    return args, forwarded


def main(argv: Sequence[str] | None = None) -> None:
    args, forwarded = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_modes = {
        str(value).strip().lower() for value in args.allowed_tracking_modes
    }
    rows, filter_report = filter_tracking_rows(
        args.tracked_predictions,
        allowed_modes=allowed_modes,
        min_confidence=args.min_tracking_confidence,
        keep_missing_mode=args.keep_missing_tracking_mode,
    )
    filtered_path = args.output_dir / "filtered_tracking_predictions.csv"
    write_rows(filtered_path, rows)
    tracking_metadata = tracking_metadata_by_filtered_index(rows)
    command = [
        sys.executable,
        "-m",
        SELECTOR_MODULE,
        "--gigapose-predictions",
        str(filtered_path),
        "--optimized-extrinsics",
        str(args.optimized_extrinsics),
        "--output-dir",
        str(args.output_dir),
        *forwarded,
    ]
    print(json.dumps(filter_report, indent=2))
    print("Running:", shlex.join(command))
    subprocess.run(command, check=True)

    annotated_counts = {
        name: append_tracking_metadata(
            args.output_dir / name, tracking_metadata
        )
        for name in OUTPUT_CSV_NAMES
    }
    merge_filter_report(
        args.output_dir,
        filter_report,
        SELECTOR_MODULE,
        command,
        annotated_counts,
    )
    print(
        "Wrote tracking-filtered centered-calibration labels to "
        f"{args.output_dir / 'selected_samples.csv'}"
    )


if __name__ == "__main__":
    main()
