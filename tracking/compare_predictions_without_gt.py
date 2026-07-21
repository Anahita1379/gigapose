"""Compare pose-prediction/tracking CSVs without ground-truth poses.

This evaluator deliberately reports *disagreement* and temporal consistency,
not pose accuracy.  The first (or explicitly selected) model is used as a
reference prediction.  Other models are associated to it independently in
each frame, so standard GigaPose ``MultiHypothesis.csv`` files can be compared
with ``tracked_predictions.csv`` even when their instance identifiers use
different conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.geometry import (
    pose_from_rt,
    rotation_error_deg,
    so3_log,
    translation_error_m,
)


@dataclass
class Prediction:
    model: str
    scene_id: int
    im_id: int
    obj_id: int
    score: float
    pose: np.ndarray
    instance_id: int
    track_id: int | None
    rank: int
    mode: str
    source: str
    row_index: int


METRICS = (
    "translation_change_m",
    "relative_translation_change",
    "rotation_change_deg",
    "lateral_change_m",
    "abs_depth_change_m",
    "abs_distance_change_m",
    "normalized_center_change",
    "projected_center_change_px",
)


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must be name=/path/to/predictions.csv"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pose/tracking CSVs without GT. Reported pose values are "
            "changes relative to a reference prediction, not accuracy errors."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help=(
            "Prediction CSV as name=/path/to/file.csv. Repeat at least twice. "
            "The first model is the default reference."
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference model name. Defaults to the first --model.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset directory. frame_map.json supplies camera names "
            "and, when present, intrinsics for pixel-center disagreement."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Recorded for provenance; no GT is read from this split.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--translation-unit",
        choices=("mm", "m"),
        default="mm",
        help="Translation unit used by all input CSVs (GigaPose default: mm).",
    )
    parser.add_argument("--distance-bin-m", type=float, default=10.0)
    parser.add_argument("--max-distance-m", type=float, default=None)
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Optional camera_id filter. Repeat for multiple cameras.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "Minimum comparison-model score for ordinary summaries. The "
            "confidence sweep always starts from all matched predictions."
        ),
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.35, 0.5, 0.67, 0.8],
    )
    parser.add_argument(
        "--max-normalized-center-distance",
        type=float,
        default=0.35,
        help=("Association gate in normalized pinhole coordinates (x/z,y/z)."),
    )
    parser.add_argument(
        "--max-relative-translation-distance",
        type=float,
        default=0.50,
        help="Association gate: translation difference / mean pose distance.",
    )
    parser.add_argument(
        "--max-association-cost",
        type=float,
        default=1.0,
    )
    parser.add_argument("--plot-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--scatter-alpha", type=float, default=0.20)
    return parser.parse_args()


def _numbers(value: str, count: int, *, field: str, path: Path) -> np.ndarray:
    array = np.fromstring(str(value), sep=" ", dtype=float)
    if array.size != count:
        raise ValueError(
            f"{path}: expected {count} numbers in {field}, got {array.size}"
        )
    return array


def load_predictions(
    path: Path,
    model: str,
    translation_unit: str = "mm",
) -> list[Prediction]:
    """Load one best hypothesis per predicted instance and image."""

    scale = 0.001 if translation_unit == "mm" else 1.0
    grouped: dict[tuple[int, int, int, int], list[Prediction]] = defaultdict(list)
    fallback_next: dict[tuple[int, int], int] = defaultdict(int)
    previous_image: tuple[int, int] | None = None
    previous_group: int | None = None
    previous_rank = -1
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_id", "im_id", "obj_id", "score", "R", "t"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        explicit_rank = "rank" in (reader.fieldnames or [])
        for row_index, row in enumerate(reader):
            scene_id, im_id = int(row["scene_id"]), int(row["im_id"])
            image_key = (scene_id, im_id)
            rank = int(row.get("rank", 0) or 0)
            instance_text = row.get("instance_id", "")
            track_text = row.get("track_id", "")
            track_id = int(track_text) if track_text not in (None, "") else None
            if instance_text not in (None, ""):
                instance_id = int(instance_text)
            elif track_id is not None:
                instance_id = track_id
            else:
                starts_group = (
                    previous_image != image_key
                    or not explicit_rank
                    or rank <= previous_rank
                )
                if starts_group:
                    instance_id = fallback_next[image_key]
                    fallback_next[image_key] += 1
                else:
                    if previous_group is None:
                        raise RuntimeError("Internal hypothesis grouping failure")
                    instance_id = previous_group
            prediction = Prediction(
                model=model,
                scene_id=scene_id,
                im_id=im_id,
                obj_id=int(row["obj_id"]),
                score=float(row["score"]),
                pose=pose_from_rt(
                    _numbers(row["R"], 9, field="R", path=path).reshape(3, 3),
                    _numbers(row["t"], 3, field="t", path=path) * scale,
                ),
                instance_id=instance_id,
                track_id=track_id,
                rank=rank,
                mode=str(row.get("tracking_mode", "")),
                source=str(row.get("source", "")),
                row_index=row_index,
            )
            grouped[(scene_id, im_id, prediction.obj_id, instance_id)].append(
                prediction
            )
            previous_image = image_key
            previous_group = instance_id
            previous_rank = rank

    selected: list[Prediction] = []
    for hypotheses in grouped.values():
        # Explicit rank is authoritative. Score breaks ties and supports files
        # whose hypotheses are not already sorted.
        hypotheses.sort(key=lambda item: (item.rank, -item.score, item.row_index))
        selected.append(hypotheses[0])
    selected.sort(
        key=lambda item: (
            item.scene_id,
            item.im_id,
            item.obj_id,
            item.instance_id,
        )
    )
    return selected


def load_frame_metadata(
    dataset_dir: Path | None,
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], np.ndarray]]:
    if dataset_dir is None:
        return {}, {}
    path = Path(dataset_dir) / "frame_map.json"
    if not path.is_file():
        return {}, {}
    camera_map: dict[tuple[int, int], str] = {}
    intrinsics: dict[tuple[int, int], np.ndarray] = {}
    for row in json.loads(path.read_text()):
        key = (int(row["scene_id"]), int(row["im_id"]))
        camera_map[key] = str(row.get("camera_id", "unknown_camera"))
        values = row.get("cam_K", row.get("K"))
        if values is not None:
            array = np.asarray(values, dtype=float)
            if array.size == 9 and np.isfinite(array).all():
                intrinsics[key] = array.reshape(3, 3)
    return camera_map, intrinsics


def _normalized_center(pose: np.ndarray) -> np.ndarray:
    translation = np.asarray(pose[:3, 3], dtype=float)
    if translation[2] <= 1e-9:
        return np.asarray([np.nan, np.nan])
    return translation[:2] / translation[2]


def _projected_center(pose: np.ndarray, K: np.ndarray | None) -> np.ndarray:
    if K is None:
        return np.asarray([np.nan, np.nan])
    normalized = _normalized_center(pose)
    if not np.isfinite(normalized).all():
        return normalized
    homogeneous = np.asarray([normalized[0], normalized[1], 1.0])
    projected = np.asarray(K, dtype=float) @ homogeneous
    return projected[:2] / projected[2]


def association_cost(
    reference: Prediction,
    candidate: Prediction,
    *,
    max_center_distance: float,
    max_relative_translation: float,
) -> tuple[float, float, float]:
    if reference.obj_id != candidate.obj_id:
        return 1e6, float("inf"), float("inf")
    reference_center = _normalized_center(reference.pose)
    candidate_center = _normalized_center(candidate.pose)
    if not (
        np.isfinite(reference_center).all() and np.isfinite(candidate_center).all()
    ):
        return 1e6, float("inf"), float("inf")
    center_distance = float(np.linalg.norm(reference_center - candidate_center))
    reference_distance = float(np.linalg.norm(reference.pose[:3, 3]))
    candidate_distance = float(np.linalg.norm(candidate.pose[:3, 3]))
    mean_distance = max(0.5 * (reference_distance + candidate_distance), 1e-6)
    relative_translation = (
        translation_error_m(reference.pose, candidate.pose) / mean_distance
    )
    if (
        center_distance > max_center_distance
        or relative_translation > max_relative_translation
    ):
        return 1e6, center_distance, relative_translation
    rotation = rotation_error_deg(reference.pose, candidate.pose)
    cost = (
        0.60 * center_distance / max(max_center_distance, 1e-9)
        + 0.35 * relative_translation / max(max_relative_translation, 1e-9)
        + 0.05 * min(rotation / 180.0, 1.0)
    )
    return float(cost), center_distance, relative_translation


def associate_frame(
    references: list[Prediction],
    candidates: list[Prediction],
    *,
    max_center_distance: float,
    max_relative_translation: float,
    max_cost: float,
) -> tuple[list[tuple[int, int, float, str]], list[int], list[int]]:
    """Associate same-object predictions, preferring shared stable track IDs."""

    matches: list[tuple[int, int, float, str]] = []
    remaining_references = set(range(len(references)))
    remaining_candidates = set(range(len(candidates)))

    reference_track_ids: dict[tuple[int, int], int] = {}
    for index, item in enumerate(references):
        if item.track_id is not None:
            reference_track_ids[(item.obj_id, item.track_id)] = index
    for candidate_index, candidate in enumerate(candidates):
        if candidate.track_id is None:
            continue
        reference_index = reference_track_ids.get(
            (candidate.obj_id, candidate.track_id)
        )
        if (
            reference_index is not None
            and reference_index in remaining_references
            and candidate_index in remaining_candidates
        ):
            matches.append((reference_index, candidate_index, 0.0, "track_id"))
            remaining_references.remove(reference_index)
            remaining_candidates.remove(candidate_index)

    ref_indices = sorted(remaining_references)
    candidate_indices = sorted(remaining_candidates)
    if ref_indices and candidate_indices:
        cost = np.full((len(ref_indices), len(candidate_indices)), 1e6)
        for row, reference_index in enumerate(ref_indices):
            for column, candidate_index in enumerate(candidate_indices):
                cost[row, column] = association_cost(
                    references[reference_index],
                    candidates[candidate_index],
                    max_center_distance=max_center_distance,
                    max_relative_translation=max_relative_translation,
                )[0]
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            value = float(cost[row, column])
            if value > max_cost:
                continue
            reference_index = ref_indices[int(row)]
            candidate_index = candidate_indices[int(column)]
            matches.append((reference_index, candidate_index, value, "pose_hungarian"))
            remaining_references.remove(reference_index)
            remaining_candidates.remove(candidate_index)

    matches.sort(key=lambda item: item[0])
    return matches, sorted(remaining_references), sorted(remaining_candidates)


def _index_by_frame(
    predictions: Iterable[Prediction],
) -> dict[tuple[int, int], list[Prediction]]:
    output: dict[tuple[int, int], list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        output[(prediction.scene_id, prediction.im_id)].append(prediction)
    return output


def compare_models(
    reference_model: str,
    reference_predictions: list[Prediction],
    model: str,
    model_predictions: list[Prediction],
    camera_map: dict[tuple[int, int], str],
    intrinsics: dict[tuple[int, int], np.ndarray],
    *,
    max_center_distance: float,
    max_relative_translation: float,
    max_cost: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_by_frame = _index_by_frame(reference_predictions)
    model_by_frame = _index_by_frame(model_predictions)
    rows: list[dict[str, Any]] = []
    unmatched_reference = unmatched_model = 0
    frame_keys = sorted(set(reference_by_frame) | set(model_by_frame))
    for key in frame_keys:
        references = reference_by_frame.get(key, [])
        candidates = model_by_frame.get(key, [])
        matches, missing_reference, missing_model = associate_frame(
            references,
            candidates,
            max_center_distance=max_center_distance,
            max_relative_translation=max_relative_translation,
            max_cost=max_cost,
        )
        unmatched_reference += len(missing_reference)
        unmatched_model += len(missing_model)
        K = intrinsics.get(key)
        for reference_index, candidate_index, cost, association_method in matches:
            reference = references[reference_index]
            candidate = candidates[candidate_index]
            reference_t = reference.pose[:3, 3]
            candidate_t = candidate.pose[:3, 3]
            delta = candidate_t - reference_t
            reference_distance = float(np.linalg.norm(reference_t))
            candidate_distance = float(np.linalg.norm(candidate_t))
            reference_center = _normalized_center(reference.pose)
            candidate_center = _normalized_center(candidate.pose)
            reference_pixel = _projected_center(reference.pose, K)
            candidate_pixel = _projected_center(candidate.pose, K)
            pixel_change = (
                float(np.linalg.norm(candidate_pixel - reference_pixel))
                if np.isfinite(reference_pixel).all()
                and np.isfinite(candidate_pixel).all()
                else float("nan")
            )
            rows.append(
                {
                    "reference_model": reference_model,
                    "model": model,
                    "scene_id": key[0],
                    "im_id": key[1],
                    "camera_id": camera_map.get(key, "unknown_camera"),
                    "obj_id": candidate.obj_id,
                    "reference_instance_id": reference.instance_id,
                    "model_instance_id": candidate.instance_id,
                    "model_track_id": (
                        candidate.track_id if candidate.track_id is not None else ""
                    ),
                    "association_method": association_method,
                    "association_cost": cost,
                    "reference_score": reference.score,
                    "model_score": candidate.score,
                    "reference_distance_m": reference_distance,
                    "model_distance_m": candidate_distance,
                    "reference_depth_m": float(reference_t[2]),
                    "model_depth_m": float(candidate_t[2]),
                    "translation_change_m": float(np.linalg.norm(delta)),
                    "relative_translation_change": float(
                        np.linalg.norm(delta) / max(reference_distance, 1e-9)
                    ),
                    "translation_change_x_m": float(delta[0]),
                    "translation_change_y_m": float(delta[1]),
                    "translation_change_z_m": float(delta[2]),
                    "lateral_change_m": float(np.linalg.norm(delta[:2])),
                    "depth_change_m": float(delta[2]),
                    "abs_depth_change_m": float(abs(delta[2])),
                    "distance_change_m": candidate_distance - reference_distance,
                    "abs_distance_change_m": float(
                        abs(candidate_distance - reference_distance)
                    ),
                    "log_depth_ratio": float(
                        math.log(
                            max(float(candidate_t[2]), 1e-9)
                            / max(float(reference_t[2]), 1e-9)
                        )
                    ),
                    "rotation_change_deg": rotation_error_deg(
                        candidate.pose, reference.pose
                    ),
                    "normalized_center_change": float(
                        np.linalg.norm(candidate_center - reference_center)
                    ),
                    "projected_center_change_px": pixel_change,
                    "tracking_mode": candidate.mode,
                    "source": candidate.source,
                    "reference_row_index": reference.row_index,
                    "model_row_index": candidate.row_index,
                }
            )
    coverage = {
        "reference_model": reference_model,
        "model": model,
        "reference_predictions": len(reference_predictions),
        "model_predictions": len(model_predictions),
        "matched_predictions": len(rows),
        "unmatched_reference_predictions": unmatched_reference,
        "unmatched_model_predictions": unmatched_model,
        "reference_matched_fraction": (
            len(rows) / len(reference_predictions) if reference_predictions else 0.0
        ),
        "model_matched_fraction": (
            len(rows) / len(model_predictions) if model_predictions else 0.0
        ),
    }
    return rows, coverage


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def metric_stats(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = np.asarray([finite_float(row.get(metric)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    prefix = metric
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_variance": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_variance": float(values.var()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(values.max()),
    }


def summarize(rows: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    output = dict(fields)
    output["matched_predictions"] = len(rows)
    for metric in METRICS:
        output.update(metric_stats(rows, metric))
    output.update(metric_stats(rows, "model_score"))
    return output


def summarize_overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    return [
        summarize(items, {"model": model}) for model, items in sorted(grouped.items())
    ]


def summarize_by_distance(
    rows: list[dict[str, Any]], bin_width_m: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        distance = finite_float(row.get("reference_distance_m"))
        if not np.isfinite(distance):
            continue
        bin_index = int(math.floor(distance / bin_width_m))
        grouped[(str(row["model"]), str(row["camera_id"]), bin_index)].append(row)
    output = []
    for (model, camera_id, bin_index), items in sorted(grouped.items()):
        start = bin_index * bin_width_m
        output.append(
            summarize(
                items,
                {
                    "model": model,
                    "camera_id": camera_id,
                    "distance_bin_start_m": start,
                    "distance_bin_end_m": start + bin_width_m,
                },
            )
        )
    return output


def summarize_by_confidence(
    rows: list[dict[str, Any]], thresholds: list[float]
) -> list[dict[str, Any]]:
    output = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        model_rows = [row for row in rows if str(row["model"]) == model]
        for threshold in thresholds:
            retained = [
                row
                for row in model_rows
                if finite_float(row.get("model_score")) >= threshold
            ]
            output.append(
                summarize(
                    retained,
                    {
                        "model": model,
                        "confidence_threshold": threshold,
                        "total_matched_predictions": len(model_rows),
                        "retained_predictions": len(retained),
                        "retained_fraction": (
                            len(retained) / len(model_rows) if model_rows else 0.0
                        ),
                    },
                )
            )
    return output


def summarize_confidence_by_distance(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    bin_width_m: float,
) -> list[dict[str, Any]]:
    output = []
    for threshold in thresholds:
        retained = [
            row for row in rows if finite_float(row.get("model_score")) >= threshold
        ]
        for summary in summarize_by_distance(retained, bin_width_m):
            summary["confidence_threshold"] = threshold
            output.append(summary)
    return output


def temporal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure frame-normalized motion/smoothness along comparison tracks."""

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track_id = row.get("model_track_id", "")
        if track_id in (None, ""):
            continue
        grouped[(str(row["model"]), int(row["scene_id"]), int(track_id))].append(row)

    output: list[dict[str, Any]] = []
    for (model, scene_id, track_id), items in grouped.items():
        items.sort(key=lambda row: int(row["im_id"]))
        previous_model_velocity: np.ndarray | None = None
        previous_reference_velocity: np.ndarray | None = None
        previous_model_rot_velocity: np.ndarray | None = None
        previous_reference_rot_velocity: np.ndarray | None = None
        for previous, current in zip(items, items[1:]):
            frame_gap = int(current["im_id"]) - int(previous["im_id"])
            if frame_gap <= 0:
                continue
            model_previous = _pose_from_comparison_row(previous, "model")
            model_current = _pose_from_comparison_row(current, "model")
            reference_previous = _pose_from_comparison_row(previous, "reference")
            reference_current = _pose_from_comparison_row(current, "reference")
            model_velocity = (model_current[:3, 3] - model_previous[:3, 3]) / frame_gap
            reference_velocity = (
                reference_current[:3, 3] - reference_previous[:3, 3]
            ) / frame_gap
            model_rot_velocity = (
                so3_log(model_current[:3, :3] @ model_previous[:3, :3].T) / frame_gap
            )
            reference_rot_velocity = (
                so3_log(reference_current[:3, :3] @ reference_previous[:3, :3].T)
                / frame_gap
            )
            temporal = {
                "model": model,
                "scene_id": scene_id,
                "track_id": track_id,
                "im_id_previous": int(previous["im_id"]),
                "im_id": int(current["im_id"]),
                "frame_gap": frame_gap,
                "model_translation_step_m_per_frame": float(
                    np.linalg.norm(model_velocity)
                ),
                "reference_translation_step_m_per_frame": float(
                    np.linalg.norm(reference_velocity)
                ),
                "translation_velocity_disagreement_m_per_frame": float(
                    np.linalg.norm(model_velocity - reference_velocity)
                ),
                "model_rotation_step_deg_per_frame": float(
                    np.degrees(np.linalg.norm(model_rot_velocity))
                ),
                "reference_rotation_step_deg_per_frame": float(
                    np.degrees(np.linalg.norm(reference_rot_velocity))
                ),
                "rotation_velocity_disagreement_deg_per_frame": float(
                    np.degrees(
                        np.linalg.norm(model_rot_velocity - reference_rot_velocity)
                    )
                ),
                "model_translation_acceleration_m_per_frame2": float("nan"),
                "reference_translation_acceleration_m_per_frame2": float("nan"),
                "model_rotation_acceleration_deg_per_frame2": float("nan"),
                "reference_rotation_acceleration_deg_per_frame2": float("nan"),
            }
            if previous_model_velocity is not None:
                temporal["model_translation_acceleration_m_per_frame2"] = float(
                    np.linalg.norm(model_velocity - previous_model_velocity)
                )
                temporal["reference_translation_acceleration_m_per_frame2"] = float(
                    np.linalg.norm(reference_velocity - previous_reference_velocity)
                )
                temporal["model_rotation_acceleration_deg_per_frame2"] = float(
                    np.degrees(
                        np.linalg.norm(model_rot_velocity - previous_model_rot_velocity)
                    )
                )
                temporal["reference_rotation_acceleration_deg_per_frame2"] = float(
                    np.degrees(
                        np.linalg.norm(
                            reference_rot_velocity - previous_reference_rot_velocity
                        )
                    )
                )
            output.append(temporal)
            previous_model_velocity = model_velocity
            previous_reference_velocity = reference_velocity
            previous_model_rot_velocity = model_rot_velocity
            previous_reference_rot_velocity = reference_rot_velocity
    return output


def _pose_from_comparison_row(row: dict[str, Any], prefix: str) -> np.ndarray:
    rotation = np.asarray(row[f"_{prefix}_rotation"], dtype=float).reshape(3, 3)
    translation = np.asarray(row[f"_{prefix}_translation"], dtype=float).reshape(3)
    return pose_from_rt(rotation, translation)


def attach_temporal_pose_data(
    comparison_rows: list[dict[str, Any]],
    prediction_lookup: dict[tuple[str, int], Prediction],
) -> None:
    """Attach private pose arrays used only while calculating temporal metrics."""

    for row in comparison_rows:
        reference = prediction_lookup[
            (str(row["reference_model"]), int(row["reference_row_index"]))
        ]
        model = prediction_lookup[(str(row["model"]), int(row["model_row_index"]))]
        row["_reference_rotation"] = reference.pose[:3, :3].reshape(-1).tolist()
        row["_reference_translation"] = reference.pose[:3, 3].tolist()
        row["_model_rotation"] = model.pose[:3, :3].reshape(-1).tolist()
        row["_model_translation"] = model.pose[:3, 3].tolist()


def temporal_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "model_translation_step_m_per_frame",
        "reference_translation_step_m_per_frame",
        "translation_velocity_disagreement_m_per_frame",
        "model_rotation_step_deg_per_frame",
        "reference_rotation_step_deg_per_frame",
        "rotation_velocity_disagreement_deg_per_frame",
        "model_translation_acceleration_m_per_frame2",
        "reference_translation_acceleration_m_per_frame2",
        "model_rotation_acceleration_deg_per_frame2",
        "reference_rotation_acceleration_deg_per_frame2",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output = []
    for model, items in sorted(grouped.items()):
        summary: dict[str, Any] = {"model": model, "temporal_transitions": len(items)}
        for metric in metrics:
            summary.update(metric_stats(items, metric))
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    private = {key for row in rows for key in row if key.startswith("_")}
    fieldnames = sorted({key for row in rows for key in row if key not in private})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {key: value for key, value in row.items() if key not in private}
                for row in rows
            ]
        )


def _distance_bin_points(
    rows: list[dict[str, Any]], metric: str, bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        distance = finite_float(row.get("reference_distance_m"))
        value = finite_float(row.get(metric))
        if np.isfinite(distance) and np.isfinite(value):
            bins[int(math.floor(distance / bin_width))].append(value)
    indices = sorted(bins)
    x = np.asarray([(index + 0.5) * bin_width for index in indices])
    median = np.asarray([np.median(bins[index]) for index in indices])
    p90 = np.asarray([np.percentile(bins[index], 90) for index in indices])
    return x, median, p90


def create_plots(
    rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
    confidence_distance_rows: list[dict[str, Any]],
    temporal_summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    bin_width: float,
    thresholds: list[float],
    fmt: str,
    dpi: int,
    scatter_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    confidence_plots = output_dir / "plots_vs_confidence"
    plots.mkdir(parents=True, exist_ok=True)
    confidence_plots.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows})

    for metric, ylabel, filename in (
        (
            "translation_change_m",
            "Translation change from reference (m)",
            "translation_change_vs_distance",
        ),
        (
            "rotation_change_deg",
            "Rotation change from reference (deg)",
            "rotation_change_vs_distance",
        ),
        (
            "abs_depth_change_m",
            "Absolute depth change from reference (m)",
            "depth_change_vs_distance",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for model in models:
            model_rows = [row for row in rows if str(row["model"]) == model]
            x_scatter = np.asarray(
                [finite_float(row.get("reference_distance_m")) for row in model_rows]
            )
            y_scatter = np.asarray(
                [finite_float(row.get(metric)) for row in model_rows]
            )
            valid = np.isfinite(x_scatter) & np.isfinite(y_scatter)
            if valid.any():
                ax.scatter(
                    x_scatter[valid],
                    y_scatter[valid],
                    s=9,
                    alpha=scatter_alpha,
                    label=f"{model} instances",
                )
                x, median, p90 = _distance_bin_points(model_rows, metric, bin_width)
                ax.plot(x, median, marker="o", linewidth=2.2, label=f"{model} median")
                ax.plot(x, p90, linestyle="--", linewidth=1.5, label=f"{model} p90")
                plotted = True
        if plotted:
            ax.set_xlabel("Reference-predicted distance (m)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(plots / f"{filename}.{fmt}", dpi=dpi)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for model in models:
        summaries = sorted(
            [row for row in confidence_rows if str(row["model"]) == model],
            key=lambda row: finite_float(row["confidence_threshold"]),
        )
        x = [finite_float(row["confidence_threshold"]) for row in summaries]
        axes[0].plot(
            x,
            [finite_float(row["retained_fraction"]) for row in summaries],
            marker="o",
            label=model,
        )
        axes[1].plot(
            x,
            [finite_float(row["translation_change_m_median"]) for row in summaries],
            marker="o",
            label=model,
        )
        axes[2].plot(
            x,
            [finite_float(row["rotation_change_deg_median"]) for row in summaries],
            marker="o",
            label=model,
        )
    for ax in axes:
        ax.set_xlabel("Minimum model confidence")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Retained fraction of matched predictions")
    axes[1].set_ylabel("Median translation change (m)")
    axes[2].set_ylabel("Median rotation change (deg)")
    if models:
        axes[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(confidence_plots / f"summary_vs_confidence.{fmt}", dpi=dpi)
    plt.close(fig)

    for threshold in thresholds:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
        plotted = False
        for model in models:
            selected = [
                row
                for row in rows
                if str(row["model"]) == model
                and finite_float(row.get("model_score")) >= threshold
            ]
            for ax, metric, label in (
                (axes[0], "translation_change_m", "Translation change (m)"),
                (axes[1], "rotation_change_deg", "Rotation change (deg)"),
            ):
                x, median, _ = _distance_bin_points(selected, metric, bin_width)
                if x.size:
                    ax.plot(x, median, marker="o", linewidth=2, label=model)
                    ax.set_ylabel(f"Median {label}")
                    plotted = True
        if plotted:
            for ax in axes:
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Reference-predicted distance (m)")
            axes[0].legend(fontsize="small")
            fig.suptitle(f"Pose change at model confidence ≥ {threshold:g}")
            fig.tight_layout()
            safe = str(threshold).replace(".", "p")
            fig.savefig(
                confidence_plots / f"pose_change_vs_distance_score_ge_{safe}.{fmt}",
                dpi=dpi,
            )
        plt.close(fig)

    if temporal_summaries:
        labels = [str(row["model"]) for row in temporal_summaries]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(
            x - 0.18,
            [
                finite_float(
                    row.get("reference_translation_acceleration_m_per_frame2_median")
                )
                for row in temporal_summaries
            ],
            width=0.36,
            label="reference",
        )
        axes[0].bar(
            x + 0.18,
            [
                finite_float(
                    row.get("model_translation_acceleration_m_per_frame2_median")
                )
                for row in temporal_summaries
            ],
            width=0.36,
            label="model",
        )
        axes[1].bar(
            x - 0.18,
            [
                finite_float(
                    row.get("reference_rotation_acceleration_deg_per_frame2_median")
                )
                for row in temporal_summaries
            ],
            width=0.36,
            label="reference",
        )
        axes[1].bar(
            x + 0.18,
            [
                finite_float(
                    row.get("model_rotation_acceleration_deg_per_frame2_median")
                )
                for row in temporal_summaries
            ],
            width=0.36,
            label="model",
        )
        axes[0].set_ylabel("Median translation acceleration (m/frame²)")
        axes[1].set_ylabel("Median rotation acceleration (deg/frame²)")
        for ax in axes:
            ax.set_xticks(x, labels, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(plots / f"temporal_smoothness.{fmt}", dpi=dpi)
        plt.close(fig)


def _format(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return "" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    path: Path,
    reference_model: str,
    summaries: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> None:
    lines = [
        "# GT-free tracking comparison",
        "",
        "> These are disagreement and temporal-consistency measurements, not pose-accuracy errors.",
        "",
        f"Reference prediction model: `{reference_model}`.",
        "",
        "## Coverage",
        "",
        "| Model | Matched | Reference coverage | Model coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in coverage:
        lines.append(
            "| {model} | {matched} | {reference}% | {model_fraction}% |".format(
                model=row["model"],
                matched=row["matched_predictions"],
                reference=_format(100 * row["reference_matched_fraction"], 1),
                model_fraction=_format(100 * row["model_matched_fraction"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Pose changes from the reference",
            "",
            "| Model | Translation median (m) | Translation p90 (m) | Rotation median (deg) | Rotation p90 (deg) | Depth median (m) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            "| {model} | {tm} | {tp} | {rm} | {rp} | {dm} |".format(
                model=row["model"],
                tm=_format(row["translation_change_m_median"]),
                tp=_format(row["translation_change_m_p90"]),
                rm=_format(row["rotation_change_deg_median"]),
                rp=_format(row["rotation_change_deg_p90"]),
                dm=_format(row["abs_depth_change_m_median"]),
            )
        )
    lines.extend(
        [
            "",
            "A smaller change means closer agreement with the reference, not necessarily a more accurate pose. Review CAD overlays or an independent label source before deciding which model is better.",
            "",
            "Confidence values from GigaPose and the tracker may have different calibration. Threshold plots are conditional summaries within each model and should not be interpreted as equal-probability operating points.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def validate_args(args: argparse.Namespace) -> tuple[str, list[float]]:
    names = [name for name, _ in args.model]
    if len(args.model) < 2:
        raise ValueError("Provide at least two --model arguments")
    if len(set(names)) != len(names):
        raise ValueError("Every --model name must be unique")
    reference = args.reference or names[0]
    if reference not in names:
        raise ValueError(f"--reference {reference!r} is not one of {names}")
    if args.distance_bin_m <= 0:
        raise ValueError("--distance-bin-m must be positive")
    if args.min_confidence is not None and not np.isfinite(args.min_confidence):
        raise ValueError("--min-confidence must be finite")
    thresholds = sorted(set(float(value) for value in args.confidence_thresholds))
    if not thresholds or not all(np.isfinite(value) for value in thresholds):
        raise ValueError("--confidence-thresholds must be finite")
    if args.max_normalized_center_distance <= 0:
        raise ValueError("--max-normalized-center-distance must be positive")
    if args.max_relative_translation_distance <= 0:
        raise ValueError("--max-relative-translation-distance must be positive")
    return reference, thresholds


def main() -> None:
    args = parse_args()
    reference_model, thresholds = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_map, intrinsics = load_frame_metadata(args.dataset_dir)
    predictions = {
        name: load_predictions(path, name, args.translation_unit)
        for name, path in args.model
    }
    reference_predictions = predictions[reference_model]
    all_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for model, _ in args.model:
        if model == reference_model:
            continue
        rows, model_coverage = compare_models(
            reference_model,
            reference_predictions,
            model,
            predictions[model],
            camera_map,
            intrinsics,
            max_center_distance=args.max_normalized_center_distance,
            max_relative_translation=args.max_relative_translation_distance,
            max_cost=args.max_association_cost,
        )
        all_rows.extend(rows)
        coverage.append(model_coverage)

    cameras = set(args.camera) if args.camera else None
    analysis_rows = []
    for row in all_rows:
        if cameras is not None and str(row["camera_id"]) not in cameras:
            continue
        distance = finite_float(row["reference_distance_m"])
        if args.max_distance_m is not None and distance > args.max_distance_m:
            continue
        analysis_rows.append(row)
    confidence_source_rows = list(analysis_rows)
    if args.min_confidence is not None:
        analysis_rows = [
            row
            for row in analysis_rows
            if finite_float(row["model_score"]) >= args.min_confidence
        ]

    prediction_lookup = {
        (item.model, item.row_index): item
        for values in predictions.values()
        for item in values
    }
    attach_temporal_pose_data(analysis_rows, prediction_lookup)
    temporal = temporal_rows(analysis_rows)
    summaries = summarize_overall(analysis_rows)
    distance_summaries = summarize_by_distance(analysis_rows, args.distance_bin_m)
    confidence_summaries = summarize_by_confidence(confidence_source_rows, thresholds)
    confidence_distance = summarize_confidence_by_distance(
        confidence_source_rows, thresholds, args.distance_bin_m
    )
    temporal_summaries = temporal_summary(temporal)

    write_csv(args.output_dir / "matched_pose_changes.csv", analysis_rows)
    write_csv(args.output_dir / "overall_summary.csv", summaries)
    write_csv(args.output_dir / "coverage.csv", coverage)
    write_csv(args.output_dir / "by_reference_distance.csv", distance_summaries)
    write_csv(args.output_dir / "by_confidence.csv", confidence_summaries)
    write_csv(
        args.output_dir / "by_confidence_and_reference_distance.csv",
        confidence_distance,
    )
    write_csv(args.output_dir / "temporal_transitions.csv", temporal)
    write_csv(args.output_dir / "temporal_summary.csv", temporal_summaries)
    write_report(
        args.output_dir / "REPORT.md",
        reference_model,
        summaries,
        coverage,
    )
    report = {
        "comparison_type": "ground_truth_free",
        "warning": ("Pose changes and temporal consistency are not pose accuracy."),
        "reference_model": reference_model,
        "models": {name: str(path) for name, path in args.model},
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir else None,
        "split": args.split,
        "translation_unit": args.translation_unit,
        "confidence_thresholds": thresholds,
        "minimum_confidence_for_main_summary": args.min_confidence,
        "association": {
            "max_normalized_center_distance": args.max_normalized_center_distance,
            "max_relative_translation_distance": args.max_relative_translation_distance,
            "max_cost": args.max_association_cost,
        },
        "coverage": coverage,
        "overall": summaries,
        "temporal": temporal_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    create_plots(
        analysis_rows,
        confidence_summaries,
        confidence_distance,
        temporal_summaries,
        args.output_dir,
        bin_width=args.distance_bin_m,
        thresholds=thresholds,
        fmt=args.plot_format,
        dpi=args.dpi,
        scatter_alpha=args.scatter_alpha,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
