"""Combine multiple selected_samples.csv files for extrinsic optimization.

Example:

    python -m fine_tuning.combine_selected_samples \
      --input run1/selected_samples.csv \
      --input run2/selected_samples.csv \
      --output combined_selected_samples.csv

The output keeps all columns from all inputs, adds ``source_selected_csv``, and
can optionally deduplicate repeated EPnP labels or repeated image/label pairs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="selected_samples.csv path. Repeat for multiple files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dedupe-by",
        choices=("none", "epnp_label_path", "match_key_epnp", "scene_im_epnp"),
        default="match_key_epnp",
        help=(
            "How to drop duplicate rows. match_key_epnp is a good default when "
            "combining multiple model/camera runs."
        ),
    )
    parser.add_argument(
        "--keep",
        choices=("first", "lowest-error", "highest-score"),
        default="lowest-error",
        help="Which duplicate to keep when --dedupe-by is not none.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def row_error_score(row: dict[str, Any]) -> float:
    t = float_value(row, "translation_error_mm", 1e18)
    r = float_value(row, "rotation_error_deg", 1e18)
    return t / 2000.0 + r / 30.0


def dedupe_key(row: dict[str, Any], mode: str) -> tuple[Any, ...] | None:
    if mode == "none":
        return None
    if mode == "epnp_label_path":
        return (row.get("epnp_label_path", ""), row.get("epnp_record_index", ""))
    if mode == "match_key_epnp":
        return (row.get("match_key", ""), row.get("epnp_label_path", ""), row.get("epnp_record_index", ""))
    if mode == "scene_im_epnp":
        return (
            row.get("scene_id", ""),
            row.get("im_id", ""),
            row.get("epnp_label_path", ""),
            row.get("epnp_record_index", ""),
        )
    raise ValueError(f"Unknown dedupe mode: {mode}")


def prefer(new: dict[str, Any], old: dict[str, Any], mode: str) -> bool:
    if mode == "first":
        return False
    if mode == "highest-score":
        return float_value(new, "score", -1.0) > float_value(old, "score", -1.0)
    if mode == "lowest-error":
        return row_error_score(new) < row_error_score(old)
    raise ValueError(f"Unknown keep mode: {mode}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["source_selected_csv"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows = []
    input_counts = {}
    for path in args.input:
        rows = read_csv(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            row = dict(row)
            row["source_selected_csv"] = str(path)
            all_rows.append(row)

    if args.dedupe_by != "none":
        kept: dict[tuple[Any, ...], dict[str, Any]] = {}
        passthrough = []
        for row in all_rows:
            key = dedupe_key(row, args.dedupe_by)
            if key is None or not any(str(part) for part in key):
                passthrough.append(row)
                continue
            if key not in kept or prefer(row, kept[key], args.keep):
                kept[key] = row
        output_rows = passthrough + list(kept.values())
    else:
        output_rows = all_rows

    output_rows.sort(
        key=lambda row: (
            str(row.get("source_selected_csv", "")),
            str(row.get("match_key", "")),
            str(row.get("epnp_label_path", "")),
        )
    )
    write_csv(args.output, output_rows)

    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    import json

    report = {
        "inputs": input_counts,
        "input_rows_total": len(all_rows),
        "output_rows": len(output_rows),
        "dedupe_by": args.dedupe_by,
        "keep": args.keep,
        "output": str(args.output),
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote combined selected samples to {args.output}")


if __name__ == "__main__":
    main()
