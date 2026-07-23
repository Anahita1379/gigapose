"""Combine selected absolute pairs for exactly one camera.

Rows from different cameras are rejected.  Duplicate EPnP records are resolved
by retaining the row with the lowest absolute pairing cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from label_selection import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine absolute selected-sample CSVs for one camera."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Expected metadata camera name, e.g. front, rear, or stereo_left.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    combined: list[dict[str, str]] = []
    input_counts: dict[str, int] = {}
    for path in args.input:
        rows = common.read_csv(path)
        input_counts[str(path)] = len(rows)
        for row_number, row in enumerate(rows, start=2):
            mode = row.get("comparison_mode")
            if mode != common.COMPARISON_MODE:
                raise ValueError(
                    f"{path}:{row_number} has comparison_mode={mode!r}; expected "
                    f"{common.COMPARISON_MODE!r}. Pre- and post-calibration rows "
                    "must not be mixed."
                )
            camera = row.get("camera_id", "")
            if camera != args.camera_id:
                raise ValueError(
                    f"{path}:{row_number} belongs to camera {camera!r}, not "
                    f"{args.camera_id!r}. Optimize each camera separately."
                )
            row = dict(row)
            row["combined_source_csv"] = str(path)
            combined.append(row)

    best_by_label: dict[tuple[str, str], dict[str, str]] = {}
    for row in combined:
        identity = (
            row.get("epnp_label_path", ""),
            row.get("epnp_record_index", "0"),
        )
        ranking = (
            float(row["pairing_cost"]),
            float(row["absolute_translation_error_mm"]),
            float(row["absolute_rotation_error_deg"]),
            -float(row.get("score") or 0.0),
        )
        current = best_by_label.get(identity)
        if current is None:
            best_by_label[identity] = row
            continue
        current_ranking = (
            float(current["pairing_cost"]),
            float(current["absolute_translation_error_mm"]),
            float(current["absolute_rotation_error_deg"]),
            -float(current.get("score") or 0.0),
        )
        if ranking < current_ranking:
            best_by_label[identity] = row

    output_rows = sorted(
        best_by_label.values(),
        key=lambda row: (
            row.get("epnp_label_path", ""),
            int(row.get("epnp_record_index") or 0),
        ),
    )
    common.write_csv(args.output, output_rows)
    report = {
        "format": "absolute_selected_samples_combination_v1",
        "camera_id": args.camera_id,
        "comparison_mode": common.COMPARISON_MODE,
        "inputs": input_counts,
        "rows_before_deduplication": len(combined),
        "rows_after_deduplication": len(output_rows),
        "duplicates_removed": len(combined) - len(output_rows),
        "output": str(args.output),
    }
    common.save_json(args.output.with_suffix(".report.json"), report)
    print(
        f"Combined {len(combined)} rows into {len(output_rows)} unique "
        f"{args.camera_id} samples: {args.output}"
    )


if __name__ == "__main__":
    main()
