"""Select EPnP label candidates from tracked pose predictions.

This is a tracking-aware front end for the established fine-tuning selectors.
It removes lost/low-confidence tracking rows before frame-transform estimation,
then delegates the geometric matching to the original selector implementation.
The selected CSV contract therefore remains compatible with the existing
EPnP/GigaPose visualization and extrinsic-optimization tools.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


STANDARD_SELECTOR = "fine_tuning.select_real_label_candidates"
EXTRINSICS_SELECTOR = (
    "fine_tuning.select_real_label_candidates_with_extrinsics"
)
OUTPUT_CSV_NAMES = (
    "all_candidate_pairs.csv",
    "best_candidate_per_epnp_label.csv",
    "selected_samples.csv",
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Filter tracked_predictions.csv by tracking state/confidence, then "
            "run the standard or optimized-extrinsics real-label selector. "
            "All unrecognized options are forwarded to the fine_tuning selector."
        ),
        epilog=(
            "Provide the normal fine_tuning selector arguments such as "
            "--dataset-dir, --epnp-root, --match-key, frame-transform options, "
            "and error thresholds directly. Supplying --optimized-extrinsics "
            "automatically selects the extrinsics-aware implementation."
        ),
    )
    parser.add_argument(
        "--tracked-predictions",
        type=Path,
        required=True,
        help="tracked_predictions.csv produced by either tracking runner.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-tracking-modes",
        nargs="+",
        choices=("normal", "uncertain", "lost"),
        default=("normal", "uncertain"),
        help=(
            "Tracking modes retained before label selection. The default keeps "
            "normal and uncertain rows while always excluding lost rows. For "
            "strict pseudo-labeling, pass only 'normal'."
        ),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.0,
        help=(
            "Filter on the tracked CSV score before geometric selection. This is "
            "separate from the delegated selector's --min-score."
        ),
    )
    parser.add_argument(
        "--optimized-extrinsics",
        type=Path,
        default=None,
        help=(
            "Optional optimized_extrinsics.json (or its directory). When set, "
            "delegate to select_real_label_candidates_with_extrinsics."
        ),
    )
    parser.add_argument(
        "--keep-missing-tracking-mode",
        action="store_true",
        help=(
            "Retain rows whose tracking_mode is empty/missing. Disabled by "
            "default so an incompatible CSV cannot silently bypass lost filtering."
        ),
    )
    args, selector_args = parser.parse_known_args(argv)
    if "--gigapose-predictions" in selector_args:
        parser.error(
            "Do not pass --gigapose-predictions; use --tracked-predictions."
        )
    if not math.isfinite(args.min_tracking_confidence):
        parser.error("--min-tracking-confidence must be finite.")
    return args, selector_args


def filter_tracking_rows(
    path: Path,
    *,
    allowed_modes: set[str],
    min_confidence: float,
    keep_missing_mode: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"scene_id", "im_id", "obj_id", "score", "R", "t"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing prediction columns: {sorted(missing)}"
            )
        if "tracking_mode" not in fieldnames and not keep_missing_mode:
            raise ValueError(
                f"{path} has no tracking_mode column. Refusing to select labels "
                "because lost rows cannot be identified; pass "
                "--keep-missing-tracking-mode only if this is intentional."
            )
        input_rows = list(reader)

    kept: list[dict[str, str]] = []
    mode_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    for original_index, row in enumerate(input_rows):
        mode = str(row.get("tracking_mode", "")).strip().lower()
        mode_counts[mode or "missing"] += 1
        if not mode:
            if not keep_missing_mode:
                rejection_counts["missing_tracking_mode"] += 1
                continue
        elif mode not in allowed_modes:
            rejection_counts[f"tracking_mode:{mode}"] += 1
            continue
        try:
            confidence = float(row.get("score", "nan"))
        except (TypeError, ValueError):
            rejection_counts["invalid_confidence"] += 1
            continue
        if not math.isfinite(confidence):
            rejection_counts["invalid_confidence"] += 1
            continue
        if confidence < min_confidence:
            rejection_counts["below_tracking_confidence"] += 1
            continue
        item = dict(row)
        item["tracking_original_row_index"] = str(original_index)
        kept.append(item)

    report = {
        "tracked_predictions": str(path),
        "input_rows": len(input_rows),
        "input_tracking_mode_counts": dict(mode_counts),
        "allowed_tracking_modes": sorted(allowed_modes),
        "min_tracking_confidence": float(min_confidence),
        "keep_missing_tracking_mode": bool(keep_missing_mode),
        "rejection_counts": dict(rejection_counts),
        "kept_rows": len(kept),
    }
    return kept, report


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Tracking filters removed every prediction row.")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tracking_metadata_by_filtered_index(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, str]]:
    metadata = {}
    for index, row in enumerate(rows):
        metadata[index] = {
            "tracking_mode": str(row.get("tracking_mode", "")),
            "tracking_source": str(row.get("source", "")),
            "tracking_track_id": str(
                row.get("track_id", row.get("instance_id", ""))
            ),
            "tracking_confidence": str(row.get("score", "")),
            "tracking_original_row_index": str(
                row.get("tracking_original_row_index", "")
            ),
        }
    return metadata


def append_tracking_metadata(
    path: Path,
    metadata: dict[int, dict[str, str]],
) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        try:
            index = int(row["prediction_row_index"])
        except (KeyError, TypeError, ValueError):
            continue
        row.update(metadata.get(index, {}))
    if not rows:
        return 0
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def selector_command(
    args: argparse.Namespace,
    selector_args: Sequence[str],
    filtered_predictions: Path,
) -> tuple[str, list[str]]:
    module = (
        EXTRINSICS_SELECTOR
        if args.optimized_extrinsics is not None
        else STANDARD_SELECTOR
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--gigapose-predictions",
        str(filtered_predictions),
        "--output-dir",
        str(args.output_dir),
        *selector_args,
    ]
    if args.optimized_extrinsics is not None:
        command.extend(
            ["--optimized-extrinsics", str(args.optimized_extrinsics)]
        )
    return module, command


def merge_filter_report(
    output_dir: Path,
    filter_report: dict[str, Any],
    selector_module: str,
    command: Sequence[str],
    annotated_counts: dict[str, int],
) -> None:
    path = output_dir / "selection_report.json"
    selector_report = json.loads(path.read_text()) if path.is_file() else {}
    selector_report.update(
        {
            "prediction_source_type": "tracked_predictions",
            "tracking_filter": filter_report,
            "delegated_selector_module": selector_module,
            "delegated_selector_command": list(command),
            "tracking_metadata_rows_annotated": annotated_counts,
        }
    )
    path.write_text(json.dumps(selector_report, indent=2))
    (output_dir / "tracking_filter_report.json").write_text(
        json.dumps(filter_report, indent=2)
    )


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
    metadata = tracking_metadata_by_filtered_index(rows)
    module, command = selector_command(args, forwarded, filtered_path)
    print(json.dumps(filter_report, indent=2))
    print("Running:", shlex.join(command))
    subprocess.run(command, check=True)

    annotated_counts = {
        name: append_tracking_metadata(args.output_dir / name, metadata)
        for name in OUTPUT_CSV_NAMES
    }
    merge_filter_report(
        args.output_dir,
        filter_report,
        module,
        command,
        annotated_counts,
    )
    print(
        f"Wrote tracking-filtered selected labels to "
        f"{args.output_dir / 'selected_samples.csv'}"
    )


if __name__ == "__main__":
    main()
