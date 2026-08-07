"""Sequence ordering and discontinuity detection for sliding-window tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def metadata_time_s(metadata: Mapping[str, Any]) -> float | None:
    """Return the first available frame timestamp in seconds."""
    for key, scale in (
        ("timestamp_ns", 1e-9),
        ("image_timestamp_ns", 1e-9),
        ("timestamp_us", 1e-6),
        ("timestamp_ms", 1e-3),
        ("sim_time_ms", 1e-3),
        ("time", 1.0),
    ):
        value = _optional_float(metadata.get(key))
        if value is not None:
            return value * scale
    return None


def sequence_key(metadata: Mapping[str, Any], scene_id: int) -> tuple[str, str]:
    """Identify a physical image sequence without conflating cameras or runs."""
    source_run = metadata.get("source_run")
    camera_id = metadata.get("camera_id")
    if source_run not in (None, ""):
        return str(source_run), str(camera_id or "unknown_camera")
    return f"scene:{int(scene_id)}", str(camera_id or "unknown_camera")


def source_frame(metadata: Mapping[str, Any], im_id: int) -> int:
    value = metadata.get("source_frame", metadata.get("frame_id", im_id))
    return int(value)


def sequence_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort frame-map rows by run/camera and physical frame time."""
    scene_id = int(row["scene_id"])
    im_id = int(row["im_id"])
    key = sequence_key(row, scene_id)
    frame = source_frame(row, im_id)
    timestamp = metadata_time_s(row)
    return (*key, frame, float("inf") if timestamp is None else timestamp, scene_id, im_id)


def order_sequence_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=sequence_row_sort_key)


@dataclass(frozen=True)
class SequenceStamp:
    key: tuple[str, str]
    scene_id: int
    source_frame: int
    time_s: float | None

    @classmethod
    def from_frame(cls, frame) -> "SequenceStamp":
        metadata = frame.metadata
        return cls(
            key=sequence_key(metadata, frame.scene_id),
            scene_id=int(frame.scene_id),
            source_frame=source_frame(metadata, frame.im_id),
            time_s=metadata_time_s(metadata),
        )


def discontinuity_reason(
    previous: SequenceStamp | None,
    current: SequenceStamp,
    *,
    max_frame_gap: int,
    max_time_gap_s: float,
) -> str | None:
    """Explain why temporal state must not propagate into ``current``."""
    if previous is None:
        return None
    if current.key != previous.key:
        return "sequence_changed"
    if current.scene_id != previous.scene_id:
        return "scene_changed"
    frame_gap = current.source_frame - previous.source_frame
    if frame_gap <= 0:
        return "nonmonotonic_source_frame"
    if max_frame_gap > 0 and frame_gap > max_frame_gap:
        return "source_frame_gap"
    if current.time_s is not None and previous.time_s is not None:
        time_gap = current.time_s - previous.time_s
        if time_gap <= 0:
            return "nonmonotonic_timestamp"
        if max_time_gap_s > 0 and time_gap > max_time_gap_s:
            return "timestamp_gap"
    return None
