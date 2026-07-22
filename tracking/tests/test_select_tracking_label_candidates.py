from __future__ import annotations

import csv
import json

import pytest

from tracking.select_real_label_candidates import (
    EXTRINSICS_SELECTOR,
    STANDARD_SELECTOR,
    append_tracking_metadata,
    filter_tracking_rows,
    parse_args,
    selector_command,
    tracking_metadata_by_filtered_index,
    write_rows,
)


FIELDS = [
    "scene_id",
    "im_id",
    "obj_id",
    "score",
    "R",
    "t",
    "instance_id",
    "track_id",
    "tracking_mode",
    "source",
]


def prediction(mode: str, score: float, track_id: int) -> dict[str, str]:
    return {
        "scene_id": "1",
        "im_id": str(track_id),
        "obj_id": "1",
        "score": str(score),
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "0 0 5000",
        "instance_id": str(track_id),
        "track_id": str(track_id),
        "tracking_mode": mode,
        "source": f"source_{track_id}",
    }


def write_predictions(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_default_filter_skips_lost_and_keeps_normal_uncertain(tmp_path):
    path = tmp_path / "tracked.csv"
    write_predictions(
        path,
        [
            prediction("normal", 0.9, 0),
            prediction("uncertain", 0.5, 1),
            prediction("lost", 0.1, 2),
        ],
    )
    rows, report = filter_tracking_rows(
        path,
        allowed_modes={"normal", "uncertain"},
        min_confidence=0.0,
        keep_missing_mode=False,
    )
    assert [row["tracking_mode"] for row in rows] == ["normal", "uncertain"]
    assert report["rejection_counts"] == {"tracking_mode:lost": 1}
    assert rows[0]["tracking_original_row_index"] == "0"
    assert rows[1]["tracking_original_row_index"] == "1"


def test_strict_normal_and_confidence_filter(tmp_path):
    path = tmp_path / "tracked.csv"
    write_predictions(
        path,
        [prediction("normal", 0.6, 0), prediction("normal", 0.8, 1)],
    )
    rows, report = filter_tracking_rows(
        path,
        allowed_modes={"normal"},
        min_confidence=0.65,
        keep_missing_mode=False,
    )
    assert len(rows) == 1
    assert rows[0]["track_id"] == "1"
    assert report["rejection_counts"]["below_tracking_confidence"] == 1


def test_missing_tracking_mode_is_rejected_safely(tmp_path):
    path = tmp_path / "ordinary.csv"
    fields = [field for field in FIELDS if field != "tracking_mode"]
    row = prediction("normal", 0.9, 0)
    row.pop("tracking_mode")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="no tracking_mode column"):
        filter_tracking_rows(
            path,
            allowed_modes={"normal"},
            min_confidence=0.0,
            keep_missing_mode=False,
        )


def test_selector_dispatches_on_optimized_extrinsics(tmp_path):
    base_args, forwarded = parse_args(
        [
            "--tracked-predictions",
            "tracked.csv",
            "--output-dir",
            str(tmp_path),
            "--dataset-dir",
            "dataset",
        ]
    )
    module, command = selector_command(
        base_args, forwarded, tmp_path / "filtered.csv"
    )
    assert module == STANDARD_SELECTOR
    assert "--dataset-dir" in command
    assert "--optimized-extrinsics" not in command

    ext_args, forwarded = parse_args(
        [
            "--tracked-predictions",
            "tracked.csv",
            "--output-dir",
            str(tmp_path),
            "--optimized-extrinsics",
            "optimized.json",
            "--dataset-dir",
            "dataset",
        ]
    )
    module, command = selector_command(
        ext_args, forwarded, tmp_path / "filtered.csv"
    )
    assert module == EXTRINSICS_SELECTOR
    assert command[-2:] == ["--optimized-extrinsics", "optimized.json"]


def test_selected_outputs_receive_tracking_provenance(tmp_path):
    filtered = [prediction("normal", 0.9, 7)]
    filtered[0]["tracking_original_row_index"] = "12"
    filtered_path = tmp_path / "filtered.csv"
    write_rows(filtered_path, filtered)
    metadata = tracking_metadata_by_filtered_index(filtered)
    selected_path = tmp_path / "selected_samples.csv"
    with selected_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prediction_row_index", "match_key"],
        )
        writer.writeheader()
        writer.writerow({"prediction_row_index": 0, "match_key": "image_1"})
    assert append_tracking_metadata(selected_path, metadata) == 1
    with selected_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["tracking_mode"] == "normal"
    assert row["tracking_track_id"] == "7"
    assert row["tracking_original_row_index"] == "12"
