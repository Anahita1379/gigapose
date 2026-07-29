"""Merge trusted selected-frame V1 poses with full-coverage Extended poses."""
from __future__ import annotations

import argparse
from pathlib import Path

from teacher_pipeline.io import write_json, write_rows
from teacher_pipeline.trajectory import load


STATE_FIELDS = (
    "track_s_m",
    "track_s_unwrapped_m",
    "track_d_m",
    "longitudinal_velocity_mps",
    "lateral_velocity_mps",
    "longitudinal_acceleration_mps2",
    "yaw_offset_rad",
)


def _track_token(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value or "")


def _key(row):
    return (
        int(row.get("scene_id", 0)),
        int(row.get("im_id", 0)),
        _track_token(row.get("track_id")),
    )


def merge(v1_rows, extended_rows, v1_confidence, extended_confidence):
    exact = {_key(row): row for row in v1_rows}
    by_frame = {}
    for row in v1_rows:
        by_frame.setdefault(_key(row)[:2], []).append(row)
    output = []
    selected_count = 0
    fallback_count = 0
    for extended in extended_rows:
        row = dict(extended)
        selected = exact.get(_key(row))
        match_mode = "scene_im_track"
        if selected is None:
            candidates = by_frame.get(_key(row)[:2], [])
            if len(candidates) == 1:
                selected = candidates[0]
                match_mode = "unique_scene_im"
                fallback_count += 1
        if selected is not None:
            selected_count += 1
            row["T_map_object_centered_extended"] = row.get(
                "T_map_object_centered_refined"
            )
            row["T_map_object_centered_refined"] = selected[
                "T_map_object_centered_refined"
            ]
            for field in STATE_FIELDS:
                if row.get(field) is not None:
                    row[f"extended_{field}"] = row[field]
                row[field] = None
            for field in (
                "translation_std_m",
                "rotation_std_deg",
                "gigapose_translation_correction_m",
                "gigapose_rotation_correction_deg",
            ):
                if selected.get(field) is not None:
                    row[field] = selected[field]
            row.update(
                hybrid_pose_source="v1_selected_epnp_anchored",
                hybrid_match_mode=match_mode,
                teacher_status="hybrid_v1_selected",
                teacher_confidence=float(
                    selected.get("teacher_confidence", v1_confidence)
                    or v1_confidence
                ),
            )
            row["sources"] = sorted(
                set(row.get("sources", []))
                | set(selected.get("sources", []))
                | {"hybrid", "v1_selected"}
            )
        else:
            row.update(
                hybrid_pose_source="v1_extended_route_fixed",
                hybrid_match_mode=None,
                teacher_status="hybrid_extended_fill",
                teacher_confidence=float(
                    row.get("teacher_confidence", extended_confidence)
                    or extended_confidence
                ),
            )
            row["sources"] = sorted(
                set(row.get("sources", [])) | {"hybrid", "extended_fill"}
            )
        output.append(row)
    return output, selected_count, fallback_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-v1-trajectories", type=Path, required=True)
    parser.add_argument("--extended-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-confidence", type=float, default=0.9)
    parser.add_argument("--extended-confidence", type=float, default=0.5)
    args = parser.parse_args()
    v1_rows = load(args.selected_v1_trajectories)
    extended_rows = load(args.extended_trajectories)
    output, selected_count, fallback_count = merge(
        v1_rows,
        extended_rows,
        args.v1_confidence,
        args.extended_confidence,
    )
    write_rows(args.output, output)
    report = {
        "format": "teacher_v1_hybrid_selected_plus_extended_v1",
        "selected_v1_input_rows": len(v1_rows),
        "extended_input_rows": len(extended_rows),
        "output_rows": len(output),
        "v1_selected_pose_rows": selected_count,
        "extended_fill_pose_rows": len(output) - selected_count,
        "unique_frame_fallback_matches": fallback_count,
        "v1_confidence_default": args.v1_confidence,
        "extended_confidence_default": args.extended_confidence,
    }
    report_path = args.output.with_suffix(".report.json")
    write_json(report_path, report)
    print(
        f"Built {len(output)} hybrid rows: {selected_count} selected V1 + "
        f"{len(output) - selected_count} Extended fill -> {args.output}"
    )


if __name__ == "__main__":
    main()
