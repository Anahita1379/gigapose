"""Build canonical teacher observations from existing project data contracts.

The preferred input mode consumes the output of ``label_selection`` together
with the original (or tracked) GigaPose CSV.  Every selected row points to an
EPnP-hybrid JSON and its per-sample metadata YAML.  The legacy generic metadata
CSV mode remains available for small synthetic/custom datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning.optimize_camera_lidar_extrinsics import (
    corrected_lidar_map_z_mm,
    load_metadata_camera,
    load_yaml,
    matrix_3x4_or_4x4_to_transform,
    matrix_from_yaml_key,
    metadata_camera_name,
    metadata_path_from_row,
)
from fine_tuning.select_real_label_candidates import load_json_records

from .geometry import (
    CENTER_RAW_M,
    centered_to_raw_pose,
    pose_error,
    raw_to_centered_pose,
)
from .io import arr, read_rows, write_json, write_rows


def _pose_m(value: Any, unit: str) -> np.ndarray:
    """Use the established loader (which returns mm), then convert to metres."""

    pose = matrix_3x4_or_4x4_to_transform(value, unit)
    output = np.asarray(pose, dtype=float).reshape(4, 4).copy()
    output[:3, 3] *= 0.001
    return output


def _prediction_pose_m(
    row: dict[str, Any], unit: str, origin: str, center_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = arr(row.get("R"), (3, 3))
    translation = arr(row.get("t"), (3,))
    if unit == "mm":
        translation *= 0.001
    elif unit == "auto" and np.nanmedian(np.abs(translation)) >= 1000.0:
        translation *= 0.001
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    if origin == "raw":
        raw = pose
        centered = raw_to_centered_pose(raw, center_raw)
    else:
        centered = pose
        raw = centered_to_raw_pose(centered, center_raw)
    return centered, raw


def _load_frame_map(dataset_dir: Path | None, frame_map: Path | None):
    path = frame_map or (dataset_dir / "frame_map.json" if dataset_dir else None)
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        (int(row["scene_id"]), int(row["im_id"])): row
        for row in data
        if "scene_id" in row and "im_id" in row
    }


def _matching_instance(frame: dict[str, Any], prediction: dict[str, Any]):
    instances = frame.get("instances", [])
    if not isinstance(instances, list) or not instances:
        return {}
    candidate = prediction.get("instance_id", "")
    if candidate not in (None, ""):
        text = str(candidate)
        for instance in instances:
            for key in ("instance_id", "sam_object_id", "object_id", "opp_id"):
                if key in instance and str(instance[key]) == text:
                    return instance
    return instances[0] if len(instances) == 1 else {}


def _record_at(path: Path, index: int) -> dict[str, Any]:
    records = load_json_records(path)
    if index < 0 or index >= len(records):
        raise IndexError(f"{path} has {len(records)} records, not index {index}")
    return records[index]


def _timestamp(row: dict[str, Any], label: dict[str, Any], label_path: Path):
    for source in (label, row):
        for key in ("timestamp_ns", "timestamp", "sim_time_ns", "sim_time_ms"):
            if source.get(key) not in (None, ""):
                return source[key]
    prefix = label_path.stem.rsplit("_", 1)[0]
    return prefix if prefix.isdigit() else row.get("im_id")


def build_from_selected(args: argparse.Namespace) -> tuple[list[dict], dict]:
    predictions = read_rows(args.predictions)
    selected = read_rows(args.selected_samples)
    has_aligned_gigapose = bool(selected) and all(
        row.get("T_gigapose_aligned_epnp_obj") not in (None, "")
        for row in selected
    )
    if args.gigapose_pose_source == "auto":
        resolved_gigapose_pose_source = (
            "aligned_csv" if has_aligned_gigapose else "prediction_csv"
        )
    elif args.gigapose_pose_source == "aligned":
        if not has_aligned_gigapose:
            raise ValueError(
                f"{args.selected_samples} does not provide a non-empty "
                "T_gigapose_aligned_epnp_obj for every selected row"
            )
        resolved_gigapose_pose_source = "aligned_csv"
    else:
        resolved_gigapose_pose_source = "prediction_csv"
    tracked_rows = read_rows(args.tracks) if args.tracks is not None else []
    tracks_by_instance = {
        (
            int(row["scene_id"]),
            int(row["im_id"]),
            str(row.get("instance_id", "")),
        ): row
        for row in tracked_rows
        if row.get("scene_id") not in (None, "")
        and row.get("im_id") not in (None, "")
        and row.get("track_id") not in (None, "")
    }
    tracks_by_frame: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in tracked_rows:
        if row.get("scene_id") in (None, "") or row.get("im_id") in (None, ""):
            continue
        tracks_by_frame.setdefault(
            (int(row["scene_id"]), int(row["im_id"])), []
        ).append(row)
    frames = _load_frame_map(args.dataset_dir, args.frame_map)
    center_raw = np.asarray(args.center_raw, dtype=float)
    observations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for selected_index, selected_row in enumerate(selected):
        try:
            prediction_index = int(selected_row.get("prediction_row_index", -1))
            if prediction_index < 0 or prediction_index >= len(predictions):
                raise IndexError(
                    f"prediction_row_index {prediction_index} is outside "
                    f"the {len(predictions)} prediction rows"
                )
            prediction = predictions[prediction_index]
            for key in ("scene_id", "im_id"):
                selected_value = selected_row.get(key)
                prediction_value = prediction.get(key)
                if (
                    selected_value not in (None, "")
                    and prediction_value not in (None, "")
                    and int(selected_value) != int(prediction_value)
                ):
                    raise ValueError(
                        f"{key} mismatch at prediction_row_index "
                        f"{prediction_index}: selected={selected_value}, "
                        f"prediction={prediction_value}. Pass the same "
                        "prediction CSV used to create selected_samples.csv."
                    )
            label_path = Path(str(selected_row["epnp_label_path"]))
            if args.epnp_root is not None:
                label_path = args.epnp_root / label_path.name
            label_index = int(selected_row.get("epnp_record_index", 0) or 0)
            label = _record_at(label_path, label_index)

            metadata_row = dict(selected_row)
            metadata_row["epnp_label_path"] = str(label_path)
            if args.metadata_root is not None:
                explicit = label.get("metadata_path") or selected_row.get(
                    "sample_metadata_path"
                )
                if explicit:
                    metadata_row["sample_metadata_path"] = str(
                        args.metadata_root / Path(str(explicit)).name
                    )
            elif label.get("metadata_path"):
                metadata_row["sample_metadata_path"] = label["metadata_path"]
            metadata_path = metadata_path_from_row(
                metadata_row, "sample_metadata_path"
            )
            metadata = load_yaml(metadata_path)

            map_lidar = matrix_from_yaml_key(metadata, "t_map_lidar", "m")
            lidar_camera = matrix_from_yaml_key(
                metadata, "t_lidar_camera_prior", "m"
            )
            # Established metadata helpers return millimetres.
            map_lidar[:3, 3] *= 0.001
            lidar_camera[:3, 3] *= 0.001
            map_lidar_raw_metadata = map_lidar.copy()

            corrected_z_mm = corrected_lidar_map_z_mm(label)
            corrected_z_m = (
                float(corrected_z_mm) * 0.001
                if corrected_z_mm is not None
                else None
            )
            z_replacement_m = None
            if args.target_lidar_z_mode == "epnp_corrected":
                if corrected_z_m is None:
                    if not args.allow_missing_corrected_lidar_z:
                        raise ValueError(
                            f"{label_path} has no "
                            "gt_xy_mesh_z_label.ego_z_compensation."
                            "corrected_z_m. Mesh-Z hybrid labels require this "
                            "value; use --allow-missing-corrected-lidar-z only "
                            "for an explicitly reported raw-Z fallback."
                        )
                else:
                    z_replacement_m = float(
                        corrected_z_m - map_lidar[2, 3]
                    )
                    map_lidar[2, 3] = corrected_z_m

            if args.epnp_map_pose_key not in label:
                raise KeyError(args.epnp_map_pose_key)
            map_object_raw = _pose_m(
                label[args.epnp_map_pose_key], args.epnp_map_pose_unit
            )
            map_object_centered = raw_to_centered_pose(
                map_object_raw, center_raw
            )
            camera_object_epnp = None
            if args.epnp_camera_pose_key in label:
                camera_object_epnp = _pose_m(
                    label[args.epnp_camera_pose_key],
                    args.epnp_camera_pose_unit,
                )

            prediction_centered, camera_object_gp_raw = _prediction_pose_m(
                prediction,
                args.prediction_translation_unit,
                args.gigapose_object_origin,
                center_raw,
            )
            if resolved_gigapose_pose_source == "aligned_csv":
                camera_object_gp = _pose_m(
                    selected_row["T_gigapose_aligned_epnp_obj"],
                    args.selected_gigapose_pose_unit,
                )
            else:
                camera_object_gp = prediction_centered
            scene_id = int(selected_row.get("scene_id", prediction["scene_id"]))
            im_id = int(selected_row.get("im_id", prediction["im_id"]))
            frame = frames.get((scene_id, im_id), {})
            instance = _matching_instance(frame, prediction)
            track_id = prediction.get("track_id")
            track_id_source = "prediction.track_id"
            tracked = tracks_by_instance.get(
                (scene_id, im_id, str(prediction.get("instance_id", "")))
            )
            if track_id in (None, "") and tracked is not None:
                track_id = tracked["track_id"]
                track_id_source = "tracks_csv.track_id"
            if track_id in (None, "") and tracked is None:
                candidates = tracks_by_frame.get((scene_id, im_id), [])
                if prediction.get("obj_id") not in (None, ""):
                    same_object = [
                        candidate
                        for candidate in candidates
                        if str(candidate.get("obj_id", ""))
                        == str(prediction["obj_id"])
                    ]
                    candidates = same_object or candidates
                ranked_candidates = []
                for candidate in candidates:
                    try:
                        candidate_centered, _ = _prediction_pose_m(
                            candidate,
                            args.prediction_translation_unit,
                            args.gigapose_object_origin,
                            center_raw,
                        )
                        residual = pose_error(
                            prediction_centered, candidate_centered
                        )
                        gp_t = prediction_centered[:3, 3]
                        candidate_t = candidate_centered[:3, 3]
                        mean_range = max(
                            0.5
                            * (
                                np.linalg.norm(gp_t)
                                + np.linalg.norm(candidate_t)
                            ),
                            1e-6,
                        )
                        relative_translation = float(
                            np.linalg.norm(residual[:3]) / mean_range
                        )
                        if gp_t[2] > 1e-6 and candidate_t[2] > 1e-6:
                            center_distance = float(
                                np.linalg.norm(
                                    gp_t[:2] / gp_t[2]
                                    - candidate_t[:2] / candidate_t[2]
                                )
                            )
                        else:
                            center_distance = float("inf")
                        cost = center_distance + 0.25 * relative_translation
                        ranked_candidates.append(
                            (
                                cost,
                                center_distance,
                                relative_translation,
                                candidate,
                            )
                        )
                    except Exception:
                        continue
                ranked_candidates.sort(key=lambda item: item[0])
                if ranked_candidates:
                    _, center_distance, relative_translation, candidate = (
                        ranked_candidates[0]
                    )
                    if (
                        center_distance
                        <= args.max_track_association_center_distance
                        and relative_translation
                        <= args.max_track_association_relative_translation
                    ):
                        track_id = candidate["track_id"]
                        track_id_source = "tracks_csv.pose_association"
            if track_id in (None, ""):
                for source, source_name in (
                    (label, "epnp_label"),
                    (instance, "frame_map.instance"),
                ):
                    for key in ("track_id", "opponent_id", "opp_id", "vehicle_id"):
                        if source.get(key) not in (None, ""):
                            track_id = source[key]
                            track_id_source = f"{source_name}.{key}"
                            break
                    if track_id not in (None, ""):
                        break
            if track_id in (None, ""):
                track_id = prediction.get("instance_id", selected_row.get("instance_id"))
                track_id_source = "prediction.instance_id_fallback"

            camera = load_metadata_camera(metadata)
            observation = {
                "scene_id": scene_id,
                "im_id": im_id,
                "session_id": frame.get("source_root", metadata_path.parent.parent.parent.name),
                "camera": frame.get("camera_id", metadata_camera_name(metadata_path)),
                "timestamp_ns": _timestamp(selected_row, label, label_path),
                "track_id": track_id,
                "track_id_source": track_id_source,
                "image_path": frame.get("image_path", label.get("image_path")),
                "mask_path": frame.get("mask_path", label.get("mask_path")),
                "bbox_xywh": instance.get("bbox", label.get("bbox_xywh")),
                "prediction_row": prediction_index,
                "selected_sample_row": selected_index,
                "gigapose_score": float(prediction.get("score", selected_row.get("score", 0))),
                "gigapose_pose_source": resolved_gigapose_pose_source,
                "T_camera_object_centered_gigapose": camera_object_gp.tolist(),
                "T_camera_object_raw_gigapose": camera_object_gp_raw.tolist(),
                "T_camera_object_centered_epnp": (
                    camera_object_epnp.tolist()
                    if camera_object_epnp is not None
                    else None
                ),
                "T_map_object_raw_epnp": map_object_raw.tolist(),
                "T_map_object_centered_epnp": map_object_centered.tolist(),
                "T_map_lidar": map_lidar.tolist(),
                "T_map_lidar_raw_metadata": map_lidar_raw_metadata.tolist(),
                "corrected_lidar_map_z_m": corrected_z_m,
                "target_lidar_z_mode": args.target_lidar_z_mode,
                "target_lidar_z_replacement_m": z_replacement_m,
                "T_lidar_camera_initial": lidar_camera.tolist(),
                "K": camera[0].tolist() if camera is not None else None,
                "distortion": camera[1].tolist() if camera is not None else [],
                "distortion_model": camera[2] if camera is not None else "unknown",
                "epnp_label_path": str(label_path),
                "epnp_record_index": label_index,
                "sample_metadata_path": str(metadata_path),
                "object_origin_convention": "centered",
                "center_raw_m": center_raw.tolist(),
                "independent_support": True,
                "teacher_anchor_sources": ["epnp_hybrid", "metadata_ego_pose"],
            }
            observations.append(observation)
        except Exception as exc:
            skipped.append(
                {
                    "selected_sample_row": selected_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if args.strict:
                raise

    z_replacements = [
        float(row["target_lidar_z_replacement_m"])
        for row in observations
        if row.get("target_lidar_z_replacement_m") is not None
    ]
    report = {
        "format": "teacher_observations_epnp_hybrid_v1",
        "prediction_rows": len(predictions),
        "selected_rows": len(selected),
        "observations_written": len(observations),
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
        "gigapose_object_origin": args.gigapose_object_origin,
        "requested_gigapose_pose_source": args.gigapose_pose_source,
        "resolved_gigapose_pose_source": resolved_gigapose_pose_source,
        "selected_csv_has_aligned_gigapose_pose": has_aligned_gigapose,
        "canonical_object_origin": "centered",
        "target_lidar_z_mode": args.target_lidar_z_mode,
        "corrected_lidar_z_count": len(z_replacements),
        "missing_corrected_lidar_z_count": sum(
            row.get("corrected_lidar_map_z_m") is None
            for row in observations
        ),
        "target_lidar_z_replacement_m_median": (
            float(np.median(z_replacements)) if z_replacements else None
        ),
        "target_lidar_z_replacement_m_min": (
            float(np.min(z_replacements)) if z_replacements else None
        ),
        "target_lidar_z_replacement_m_max": (
            float(np.max(z_replacements)) if z_replacements else None
        ),
        "track_rows": len(tracked_rows),
        "instance_id_track_fallbacks": sum(
            row.get("track_id_source") == "prediction.instance_id_fallback"
            for row in observations
        ),
        "pose_associated_track_ids": sum(
            row.get("track_id_source") == "tracks_csv.pose_association"
            for row in observations
        ),
    }
    return observations, report


def build_from_generic(args: argparse.Namespace) -> tuple[list[dict], dict]:
    predictions = read_rows(args.predictions)
    metadata = {
        (str(row.get("scene_id", row.get("session_id"))), str(row.get("im_id", row.get("frame_id", row.get("timestamp_ns"))))): row
        for row in read_rows(args.metadata)
    }
    center_raw = np.asarray(args.center_raw, dtype=float)
    output = []
    for index, prediction in enumerate(predictions):
        key = (str(prediction.get("scene_id", prediction.get("session_id", ""))), str(prediction.get("im_id", prediction.get("frame_id", prediction.get("timestamp_ns", "")))))
        meta = metadata.get(key, {})
        centered, raw = _prediction_pose_m(
            prediction,
            args.prediction_translation_unit,
            args.gigapose_object_origin,
            center_raw,
        )
        record = dict(meta)
        record.update(
            {
                "scene_id": prediction.get("scene_id", meta.get("scene_id", "")),
                "im_id": prediction.get("im_id", meta.get("im_id", "")),
                "timestamp_ns": meta.get("timestamp_ns", prediction.get("timestamp_ns", prediction.get("im_id", ""))),
                "prediction_row": index,
                "track_id": meta.get("track_id", prediction.get("track_id", prediction.get("instance_id", index))),
                "gigapose_score": float(prediction.get("score", 0)),
                "T_camera_object_centered_gigapose": centered.tolist(),
                "T_camera_object_raw_gigapose": raw.tolist(),
                "object_origin_convention": "centered",
                "center_raw_m": center_raw.tolist(),
            }
        )
        output.append(record)
    return output, {
        "format": "teacher_observations_generic_v1",
        "observations_written": len(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selected-samples", type=Path)
    source.add_argument("--metadata", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--frame-map", type=Path)
    parser.add_argument(
        "--tracks",
        type=Path,
        help=(
            "Optional tracked_predictions.csv. Used to recover persistent "
            "track_id by scene_id, im_id, and instance_id when the prediction "
            "CSV used for selection has no track_id column."
        ),
    )
    parser.add_argument(
        "--max-track-association-center-distance", type=float, default=0.35
    )
    parser.add_argument(
        "--max-track-association-relative-translation", type=float, default=0.75
    )
    parser.add_argument("--epnp-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--epnp-map-pose-key", default="T_map_object_raw")
    parser.add_argument("--epnp-camera-pose-key", default="T_camera_object_centered")
    parser.add_argument("--epnp-map-pose-unit", choices=("auto", "m", "mm"), default="m")
    parser.add_argument("--epnp-camera-pose-unit", choices=("auto", "m", "mm"), default="m")
    parser.add_argument(
        "--target-lidar-z-mode",
        choices=("raw", "epnp_corrected"),
        default="epnp_corrected",
        help=(
            "Map-frame Z convention for t_map_lidar. Mesh-Z hybrid labels "
            "require the default epnp_corrected mode."
        ),
    )
    parser.add_argument(
        "--allow-missing-corrected-lidar-z",
        action="store_true",
        help=(
            "Allow an explicitly reported raw-metadata-Z fallback when a "
            "hybrid label has no corrected_z_m."
        ),
    )
    parser.add_argument("--prediction-translation-unit", choices=("auto", "m", "mm"), default="mm")
    parser.add_argument(
        "--gigapose-pose-source",
        choices=("auto", "raw", "aligned"),
        default="auto",
        help=(
            "GigaPose pose used by the teacher. 'auto' follows the existing "
            "optimizer contract: use selected_samples.csv column "
            "T_gigapose_aligned_epnp_obj when present, otherwise load the raw "
            "prediction row."
        ),
    )
    parser.add_argument(
        "--selected-gigapose-pose-unit",
        choices=("auto", "m", "mm"),
        default="mm",
        help="Translation unit of selected CSV GigaPose pose columns.",
    )
    parser.add_argument("--gigapose-object-origin", choices=("raw", "centered"), default="raw")
    parser.add_argument("--center-raw", nargs=3, type=float, default=CENTER_RAW_M.tolist())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selected_samples is not None:
        observations, report = build_from_selected(args)
    else:
        observations, report = build_from_generic(args)
    write_rows(args.output, observations)
    report_path = args.report or args.output.with_suffix(".report.json")
    write_json(report_path, report)
    print(
        f"Wrote {len(observations)} observations to {args.output}; "
        f"report: {report_path}"
    )


if __name__ == "__main__":
    main()
