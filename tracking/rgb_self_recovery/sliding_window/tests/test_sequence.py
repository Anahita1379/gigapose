from types import SimpleNamespace

from tracking.rgb_self_recovery.sliding_window.sequence import (
    SequenceStamp,
    discontinuity_reason,
    order_sequence_rows,
)


def _frame(run="run_a", camera="rear", source_frame=1, timestamp_ms=100, scene_id=2):
    return SimpleNamespace(
        scene_id=scene_id,
        im_id=source_frame,
        metadata={
            "source_run": run,
            "camera_id": camera,
            "source_frame": source_frame,
            "timestamp_ms": timestamp_ms,
        },
    )


def _reason(previous, current):
    return discontinuity_reason(
        SequenceStamp.from_frame(previous),
        SequenceStamp.from_frame(current),
        max_frame_gap=1,
        max_time_gap_s=0.5,
    )


def test_sequence_key_separates_runs_and_cameras():
    assert _reason(_frame(), _frame(run="run_b", source_frame=2, timestamp_ms=200)) == "sequence_changed"
    assert _reason(_frame(), _frame(camera="front", source_frame=2, timestamp_ms=200)) == "sequence_changed"


def test_sequence_detects_frame_and_timestamp_gaps():
    assert _reason(_frame(), _frame(source_frame=3, timestamp_ms=200)) == "source_frame_gap"
    assert _reason(_frame(), _frame(source_frame=2, timestamp_ms=900)) == "timestamp_gap"
    assert _reason(_frame(), _frame(source_frame=2, timestamp_ms=200)) is None


def test_sequence_rows_sort_by_run_camera_and_source_frame():
    rows = [
        {"scene_id": 2, "im_id": 8, "source_run": "b", "camera_id": "rear", "source_frame": 8},
        {"scene_id": 2, "im_id": 2, "source_run": "a", "camera_id": "rear", "source_frame": 2},
        {"scene_id": 2, "im_id": 1, "source_run": "a", "camera_id": "rear", "source_frame": 1},
        {"scene_id": 1, "im_id": 3, "source_run": "a", "camera_id": "front", "source_frame": 3},
    ]
    ordered = order_sequence_rows(rows)
    assert [(row["source_run"], row["camera_id"], row["source_frame"]) for row in ordered] == [
        ("a", "front", 3),
        ("a", "rear", 1),
        ("a", "rear", 2),
        ("b", "rear", 8),
    ]
