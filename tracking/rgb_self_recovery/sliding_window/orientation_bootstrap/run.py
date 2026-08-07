"""Correct front/rear polarity without modifying translation or existing pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tracking.geometry import closest_rotation, rotation_error_deg, so3_exp
from tracking.io import pose_to_csv_row, write_tracking_csv

from .optimizer import BootstrapConfig, optimize_polarity_fixed_lag, polarity_rotations


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--predictions", type=Path, required=True,
                       help="One-row-per-frame selector/cascade tracking CSV.")
    value.add_argument("--candidates", type=Path, required=True,
                       help="Existing GRU candidate bundle directory.")
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--epnp-root", type=Path,
                       help="EPnP hybrid-label directory containing <sample>_0.json.")
    value.add_argument("--gigapose-predictions", type=Path,
                       help="Optional MultiHypothesis CSV for a top-k consensus anchor.")
    value.add_argument("--track-map", type=Path,
                       help="Optional aligned teacher track_map.npz. Embedded EPnP raceline tangents are used first.")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument("--window-length", type=int, default=5)
    value.add_argument("--flip-axis", choices=("x", "y", "z"), default="z")
    value.add_argument("--object-forward-axis", choices=("x", "-x", "y", "-y", "z", "-z"), default="x")
    value.add_argument("--epnp-weight", type=float, default=4.0)
    value.add_argument("--map-heading-weight", type=float, default=2.0)
    value.add_argument("--gigapose-weight", type=float, default=1.0)
    value.add_argument("--gigapose-weight-with-epnp", type=float, default=0.25)
    value.add_argument("--visual-weight", type=float, default=0.25)
    value.add_argument("--motion-weight", type=float, default=1.0)
    value.add_argument("--switch-penalty", type=float, default=2.0)
    value.add_argument("--anchor-scale-deg", type=float, default=30.0)
    value.add_argument("--motion-scale-deg", type=float, default=20.0)
    value.add_argument("--visual-rotation-scale-deg", type=float, default=30.0)
    value.add_argument("--visual-translation-scale-m", type=float, default=2.0)
    value.add_argument("--gigapose-consensus-top-k", type=int, default=5)
    value.add_argument("--gigapose-consensus-bandwidth-deg", type=float, default=35.0)
    value.add_argument("--minimum-epnp-label-weight", type=float, default=0.5)
    value.add_argument("--maximum-epnp-center-error-px", type=float, default=25.0)
    value.add_argument("--minimum-epnp-bbox-iou", type=float, default=0.25)
    value.add_argument("--maximum-track-distance-m", type=float, default=20.0)
    value.add_argument("--overwrite", action="store_true")
    return value


def _numbers(text: str, count: int) -> np.ndarray:
    output = np.fromstring(str(text), sep=" ", dtype=float)
    if output.size != count:
        raise ValueError(f"Expected {count} numbers, received {output.size}")
    return output


def _pose_from_row(row: dict[str, str], translation_scale: float) -> np.ndarray:
    output = np.eye(4)
    output[:3, :3] = closest_rotation(_numbers(row["R"], 9).reshape(3, 3))
    output[:3, 3] = _numbers(row["t"], 3) * translation_scale
    return output


def _load_predictions(path: Path, translation_scale: float) -> dict[tuple[int, int], tuple[dict[str, str], np.ndarray]]:
    output: dict[tuple[int, int], tuple[dict[str, str], np.ndarray]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scene_id"]), int(row["im_id"]))
            if key in output:
                raise ValueError(f"{path} has multiple predictions for frame {key}")
            output[key] = (row, _pose_from_row(row, translation_scale))
    return output


def _load_frame_map(dataset_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = dataset_dir / "frame_map.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    rows = json.loads(path.read_text())
    return {(int(row["scene_id"]), int(row["im_id"])): row for row in rows}


def _sample_id(row: dict[str, Any]) -> str:
    metadata = Path(str(row.get("sample_metadata_path", ""))).stem
    if metadata.startswith("sample_"):
        return metadata[len("sample_"):]
    image = Path(str(row.get("frame_file", row.get("image_path", "")))).stem
    return image[len("image_"):] + "_0" if image.startswith("image_") else image + "_0"


def _quality_approved(label: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    checks = [
        (float(label.get("label_weight", 1.0)) >= args.minimum_epnp_label_weight, "label_weight"),
        (float(label.get("center_error_px", math.inf)) <= args.maximum_epnp_center_error_px, "center_error_px"),
        (float(label.get("projected_bbox_iou", 0.0)) >= args.minimum_epnp_bbox_iou, "bbox_iou"),
    ]
    failed = [name for passed, name in checks if not passed]
    pose = np.asarray(label.get("T_camera_object_centered", []), dtype=float)
    if pose.shape not in ((3, 4), (4, 4)) or not np.isfinite(pose).all():
        failed.append("pose")
    return not failed, "approved" if not failed else "rejected:" + "+".join(failed)


def _load_label(epnp_root: Path | None, frame_row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if epnp_root is None:
        return None, "not_requested"
    path = epnp_root / f"{_sample_id(frame_row)}.json"
    if not path.is_file():
        return None, "missing"
    label = json.loads(path.read_text())
    approved, reason = _quality_approved(label, args)
    label["_path"] = str(path)
    label["_approved"] = approved
    return label, reason


def _load_metadata(frame_row: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(str(frame_row.get("sample_metadata_path", "")))
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text())


def _map_camera_rotation(metadata: dict[str, Any] | None) -> np.ndarray | None:
    if not metadata:
        return None
    map_lidar = np.asarray(metadata.get("t_map_lidar", []), dtype=float)
    lidar_camera = np.asarray(metadata.get("t_lidar_camera_prior", []), dtype=float)
    if map_lidar.shape != (4, 4) or lidar_camera.shape != (4, 4):
        return None
    return closest_rotation(map_lidar[:3, :3] @ lidar_camera[:3, :3])


def _embedded_tangent(label: dict[str, Any] | None) -> np.ndarray | None:
    try:
        tangent = label["gt_xy_mesh_z_label"]["attitude"]["raceline"]["tangent_xy"]
    except (KeyError, TypeError):
        return None
    output = np.asarray([float(tangent[0]), float(tangent[1]), 0.0])
    norm = float(np.linalg.norm(output))
    return output / norm if norm > 1e-8 else None


def _forward_axis(text: str) -> np.ndarray:
    sign = -1.0 if text.startswith("-") else 1.0
    axis = text[-1]
    output = np.zeros(3)
    output[{"x": 0, "y": 1, "z": 2}[axis]] = sign
    return output


def _rotation_cost(first: np.ndarray, second: np.ndarray, scale_deg: float) -> float:
    angle = min(rotation_error_deg(first, second), 180.0)
    return min((angle / scale_deg) ** 2, 36.0)


def _gigapose_consensus(path: Path | None, top_k: int, bandwidth_deg: float) -> dict[tuple[int, int], np.ndarray]:
    if path is None:
        return {}
    grouped: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scene_id"]), int(row["im_id"]))
            grouped[key].append((float(row.get("score", 0.0)), closest_rotation(_numbers(row["R"], 9).reshape(3, 3))))
    output = {}
    for key, hypotheses in grouped.items():
        values = sorted(hypotheses, key=lambda item: -item[0])[:top_k]
        weights = np.asarray([max(item[0], 1e-6) for item in values])
        support = []
        for _, candidate in values:
            angles = np.asarray([rotation_error_deg(candidate, rotation) for _, rotation in values])
            support.append(float(np.sum(weights * np.exp(-0.5 * (angles / bandwidth_deg) ** 2))))
        output[key] = values[int(np.argmax(support))][1]
    return output


def _visual_costs(
    target_rotations: np.ndarray,
    target_translation: np.ndarray,
    candidate_poses: np.ndarray,
    candidate_features: np.ndarray,
    candidate_valid: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    names = {name: index for index, name in enumerate(feature_names)}
    error_index = names.get("total_error")
    values = []
    for rotation in target_rotations:
        costs = []
        for index in np.flatnonzero(candidate_valid):
            pose = candidate_poses[index]
            visual = float(candidate_features[index, error_index]) if error_index is not None else 0.0
            angle = rotation_error_deg(rotation, pose[:3, :3]) / args.visual_rotation_scale_deg
            translation = np.linalg.norm(target_translation - pose[:3, 3]) / args.visual_translation_scale_m
            costs.append(visual + angle + translation)
        values.append(min(costs) if costs else 0.0)
    output = np.asarray(values, dtype=float)
    return output - float(np.min(output))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(array), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90))}


def main() -> None:
    args = parser().parse_args()
    positive = [args.anchor_scale_deg, args.motion_scale_deg, args.visual_rotation_scale_deg,
                args.visual_translation_scale_m, args.gigapose_consensus_bandwidth_deg]
    weights = [args.epnp_weight, args.map_heading_weight, args.gigapose_weight,
               args.gigapose_weight_with_epnp, args.visual_weight, args.motion_weight,
               args.switch_penalty]
    if min(positive) <= 0 or min(weights) < 0:
        raise ValueError("Scales must be positive and weights non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.candidates / "manifest.json").read_text())
    bundle = np.load(args.candidates / "candidates.npz", allow_pickle=False)
    frame_map = _load_frame_map(args.dataset_dir)
    scale = 0.001 if args.translation_unit == "mm" else 1.0
    predictions = _load_predictions(args.predictions, scale)
    gp_consensus = _gigapose_consensus(
        args.gigapose_predictions, args.gigapose_consensus_top_k,
        args.gigapose_consensus_bandwidth_deg,
    )
    track_map = None
    if args.track_map is not None:
        from teacher_pipeline.v1_extended.track_map import TrackMap
        track_map = TrackMap(args.track_map)

    flip_vector = np.zeros(3)
    flip_vector[{"x": 0, "y": 1, "z": 2}[args.flip_axis]] = np.pi
    flip_rotation = so3_exp(flip_vector)
    forward_axis = _forward_axis(args.object_forward_axis)
    count = len(bundle["scene_id"])
    alternatives = np.empty((count, 2, 3, 3), dtype=float)
    unary = np.zeros((count, 2), dtype=float)
    records: list[dict[str, Any]] = []
    input_rows: list[dict[str, str]] = []
    input_poses: list[np.ndarray] = []
    epnp_errors_before: list[float] = []
    map_sources = defaultdict(int)

    for index in range(count):
        key = (int(bundle["scene_id"][index]), int(bundle["im_id"][index]))
        if key not in predictions:
            raise KeyError(f"No input prediction for candidate frame {key}")
        if key not in frame_map:
            raise KeyError(f"No frame_map row for {key}")
        row, pose = predictions[key]
        input_rows.append(row)
        input_poses.append(pose)
        alternatives[index] = polarity_rotations(pose[:3, :3], flip_rotation)
        frame_row = frame_map[key]
        label, epnp_reason = _load_label(args.epnp_root, frame_row, args)
        metadata = _load_metadata(frame_row)
        map_camera = _map_camera_rotation(metadata)
        epnp_rotation = None
        if label is not None and label.get("_approved"):
            epnp_rotation = closest_rotation(np.asarray(label["T_camera_object_centered"], dtype=float)[:3, :3])
        gp_rotation = gp_consensus.get(key, closest_rotation(bundle["poses"][index, 0, :3, :3]))

        visual = _visual_costs(
            alternatives[index], pose[:3, 3], bundle["poses"][index],
            bundle["features"][index], bundle["valid"][index],
            list(manifest["candidate_feature_names"]), args,
        )
        unary[index] += args.visual_weight * visual
        gp_weight = args.gigapose_weight_with_epnp if epnp_rotation is not None else args.gigapose_weight
        gp_cost = np.asarray([_rotation_cost(value, gp_rotation, args.anchor_scale_deg) for value in alternatives[index]])
        unary[index] += gp_weight * gp_cost
        epnp_cost = np.zeros(2)
        if epnp_rotation is not None:
            epnp_cost = np.asarray([_rotation_cost(value, epnp_rotation, args.anchor_scale_deg) for value in alternatives[index]])
            unary[index] += args.epnp_weight * epnp_cost
            epnp_errors_before.append(rotation_error_deg(pose, np.block([[epnp_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])))

        tangent = _embedded_tangent(label)
        map_source = "epnp_raceline" if tangent is not None else "none"
        if tangent is None and track_map is not None and map_camera is not None:
            map_position = map_camera @ pose[:3, 3]
            # Translation needs the map-camera offset, not only its rotation.
            if metadata is not None:
                map_lidar = np.asarray(metadata.get("t_map_lidar", []), dtype=float)
                lidar_camera = np.asarray(metadata.get("t_lidar_camera_prior", []), dtype=float)
                if map_lidar.shape == (4, 4) and lidar_camera.shape == (4, 4):
                    map_camera_pose = map_lidar @ lidar_camera
                    point = (map_camera_pose @ np.r_[pose[:3, 3], 1.0])[:3]
                    projection = track_map.project(point)
                    if projection.distance <= args.maximum_track_distance_m:
                        tangent = track_map.sample(projection.s)["tangent"]
                        map_source = "track_map_npz"
        map_cost = np.zeros(2)
        map_dots = [math.nan, math.nan]
        if tangent is not None and map_camera is not None:
            tangent = np.asarray(tangent, dtype=float)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            for state, rotation in enumerate(alternatives[index]):
                direction = map_camera @ rotation @ forward_axis
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    direction /= norm
                    dot = float(np.clip(direction @ tangent, -1.0, 1.0))
                    map_dots[state] = dot
                    angle = math.degrees(math.acos(dot))
                    map_cost[state] = min((angle / args.anchor_scale_deg) ** 2, 36.0)
            unary[index] += args.map_heading_weight * map_cost
        else:
            map_source = "unavailable"
        map_sources[map_source] += 1
        records.append({
            "scene_id": key[0], "im_id": key[1], "segment_id": int(bundle["segment_id"][index]),
            "epnp_status": epnp_reason, "epnp_path": "" if label is None else label.get("_path", ""),
            "epnp_anchor_available": int(epnp_rotation is not None),
            "map_heading_source": map_source, "map_dot_normal": map_dots[0], "map_dot_flipped": map_dots[1],
            "visual_cost_normal": visual[0], "visual_cost_flipped": visual[1],
            "gigapose_cost_normal": gp_cost[0], "gigapose_cost_flipped": gp_cost[1],
            "epnp_cost_normal": epnp_cost[0], "epnp_cost_flipped": epnp_cost[1],
            "map_cost_normal": map_cost[0], "map_cost_flipped": map_cost[1],
            "unary_normal": unary[index, 0], "unary_flipped": unary[index, 1],
        })

    selected_states = np.zeros(count, dtype=np.int64)
    lookahead_costs = np.zeros(count, dtype=float)
    for segment in np.unique(bundle["segment_id"]):
        indices = np.flatnonzero(bundle["segment_id"] == segment)
        states, costs = optimize_polarity_fixed_lag(
            alternatives[indices], unary[indices],
            BootstrapConfig(args.window_length, args.motion_weight, args.motion_scale_deg, args.switch_penalty),
        )
        selected_states[indices] = states
        lookahead_costs[indices] = costs

    output_rows = []
    epnp_errors_after: list[float] = []
    for index, (row, pose) in enumerate(zip(input_rows, input_poses)):
        state = int(selected_states[index])
        corrected = pose.copy()
        corrected[:3, :3] = alternatives[index, state]
        key = (int(bundle["scene_id"][index]), int(bundle["im_id"][index]))
        label, _ = _load_label(args.epnp_root, frame_map[key], args)
        before_epnp = math.nan
        after_epnp = math.nan
        if label is not None and label.get("_approved"):
            reference = np.eye(4)
            reference[:3, :3] = np.asarray(label["T_camera_object_centered"], dtype=float)[:3, :3]
            before_epnp = rotation_error_deg(pose, reference)
            after_epnp = rotation_error_deg(corrected, reference)
            epnp_errors_after.append(after_epnp)
        source = str(row.get("source", "prediction")) + ("|polarity_flip" if state else "|polarity_preserved")
        output_rows.append(pose_to_csv_row(
            scene_id=key[0], im_id=key[1], track_id=int(row.get("track_id", row.get("instance_id", 0)) or 0),
            obj_id=int(row.get("obj_id", 1)), confidence=float(row.get("score", 0.0)), pose_m=corrected,
            mode="orientation_bootstrap", source=source, elapsed_s=float(row.get("time", 0.0) or 0.0),
        ))
        records[index].update({
            "selected_polarity": state, "orientation_changed": state,
            "lookahead_path_cost": lookahead_costs[index],
            "rotation_change_deg": rotation_error_deg(corrected, pose),
            "epnp_error_before_deg": before_epnp, "epnp_error_after_deg": after_epnp,
            "translation_change_m": float(np.linalg.norm(corrected[:3, 3] - pose[:3, 3])),
        })

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    with (args.output_dir / "orientation_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "format": "rgb_self_recovery_orientation_bootstrap_v1",
        "predictions": str(args.predictions), "candidates": str(args.candidates),
        "dataset_dir": str(args.dataset_dir), "epnp_root": None if args.epnp_root is None else str(args.epnp_root),
        "track_map": None if args.track_map is None else str(args.track_map),
        "frame_count": count, "segment_count": int(len(np.unique(bundle["segment_id"]))),
        "window_length": args.window_length, "changed_orientation_count": int(selected_states.sum()),
        "changed_orientation_fraction": float(selected_states.mean()),
        "translation_preservation_max_error_m": float(max(float(record["translation_change_m"]) for record in records)),
        "epnp_rotation_error_before_deg": _summary(epnp_errors_before),
        "epnp_rotation_error_after_deg": _summary(epnp_errors_after),
        "map_heading_source_counts": dict(map_sources),
        "arguments": vars(args),
    }
    report["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["arguments"].items()}
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
