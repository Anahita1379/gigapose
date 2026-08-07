"""Run a GRU selector with absolute EPnP/map/GigaPose orientation anchoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from tracking.geometry import closest_rotation, rotation_error_deg, so3_exp
from tracking.io import pose_to_csv_row, write_tracking_csv
from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import CandidateBundle
from tracking.rgb_self_recovery.sliding_window.gru_selector.train import load_selector

from .optimizer import BootstrapConfig, optimize_candidate_fixed_lag
from .run import (
    _embedded_tangent,
    _forward_axis,
    _gigapose_consensus,
    _load_frame_map,
    _load_label,
    _load_metadata,
    _map_camera_rotation,
    _rotation_cost,
    _summary,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True,
                       help="Existing candidates directory.")
    value.add_argument("--checkpoint", type=Path, required=True,
                       help="Existing GRU selector checkpoint.")
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--mesh", type=Path, required=True,
                       help="The same raw CAD mesh used to export candidates.")
    value.add_argument("--epnp-root", type=Path)
    value.add_argument("--gigapose-predictions", type=Path)
    value.add_argument("--track-map", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--window-length", type=int, default=5)
    value.add_argument("--flip-axis", choices=("x", "y", "z"), default="z")
    value.add_argument("--object-forward-axis", choices=("x", "-x", "y", "-y", "z", "-z"), default="x")
    value.add_argument("--include-synthetic-flips", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--selector-weight", type=float, default=1.0)
    value.add_argument("--visual-weight", type=float, default=0.25)
    value.add_argument("--epnp-rotation-weight", type=float, default=0.0)
    value.add_argument("--epnp-translation-weight", type=float, default=0.0)
    value.add_argument("--map-heading-weight", type=float, default=0.0)
    value.add_argument("--camera-facing-weight", type=float, default=50.0)
    value.add_argument(
        "--camera-facing-rules",
        default="front:away,stereo_left:away,rear:toward",
        help="Comma-separated camera:toward|away rules.",
    )
    value.add_argument("--gigapose-weight", type=float, default=1.0)
    value.add_argument("--gigapose-weight-with-epnp", type=float, default=0.25)
    value.add_argument("--motion-weight", type=float, default=1.0)
    value.add_argument("--switch-penalty", type=float, default=2.0)
    value.add_argument("--synthetic-flip-penalty", type=float, default=0.5)
    value.add_argument("--anchor-scale-deg", type=float, default=30.0)
    value.add_argument("--epnp-translation-scale-m", type=float, default=2.0)
    value.add_argument("--motion-scale-deg", type=float, default=20.0)
    value.add_argument("--gigapose-consensus-top-k", type=int, default=5)
    value.add_argument("--gigapose-consensus-bandwidth-deg", type=float, default=35.0)
    value.add_argument("--minimum-epnp-label-weight", type=float, default=0.5)
    value.add_argument("--maximum-epnp-center-error-px", type=float, default=25.0)
    value.add_argument("--minimum-epnp-bbox-iou", type=float, default=0.25)
    value.add_argument("--maximum-track-distance-m", type=float, default=20.0)
    value.add_argument("--overwrite", action="store_true")
    return value


def _center_preserving_flip(pose: np.ndarray, flip: np.ndarray, center_raw: np.ndarray) -> np.ndarray:
    output = np.asarray(pose, dtype=float).copy()
    center_camera = output[:3, 3] + output[:3, :3] @ center_raw
    output[:3, :3] = closest_rotation(output[:3, :3] @ flip)
    output[:3, 3] = center_camera - output[:3, :3] @ center_raw
    return output


def _centered_translation(pose: np.ndarray, center_raw: np.ndarray) -> np.ndarray:
    return np.asarray(pose[:3, 3], dtype=float) + pose[:3, :3] @ center_raw


def _candidate_states(
    poses: np.ndarray,
    valid: np.ndarray,
    flip: np.ndarray,
    center_raw: np.ndarray,
    include_flips: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(poses)
    if not include_flips:
        return poses.copy(), valid.copy(), np.arange(count), np.zeros(count, dtype=np.int64)
    flipped = np.stack([_center_preserving_flip(pose, flip, center_raw) for pose in poses])
    return (
        np.concatenate([poses, flipped]),
        np.concatenate([valid, valid]),
        np.tile(np.arange(count), 2),
        np.repeat(np.arange(2), count),
    )


def _map_pose(metadata: dict[str, Any] | None) -> np.ndarray | None:
    if not metadata:
        return None
    map_lidar = np.asarray(metadata.get("t_map_lidar", []), dtype=float)
    lidar_camera = np.asarray(metadata.get("t_lidar_camera_prior", []), dtype=float)
    if map_lidar.shape != (4, 4) or lidar_camera.shape != (4, 4):
        return None
    return map_lidar @ lidar_camera


def _camera_facing_rules(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        try:
            camera, mode = (value.strip().lower() for value in item.split(":", 1))
        except ValueError as error:
            raise ValueError(f"Invalid camera-facing rule {item!r}") from error
        if mode not in {"toward", "away"}:
            raise ValueError(f"Camera-facing mode must be toward or away, got {mode!r}")
        output[camera] = mode
    if not output:
        raise ValueError("At least one camera-facing rule is required")
    return output


def _camera_facing_dot(
    pose: np.ndarray, center_raw: np.ndarray, forward_axis: np.ndarray
) -> float:
    center_camera = _centered_translation(pose, center_raw)
    norm = float(np.linalg.norm(center_camera))
    if norm <= 1e-8:
        return math.nan
    viewing_ray = center_camera / norm
    forward_camera = pose[:3, :3] @ forward_axis
    forward_camera /= max(float(np.linalg.norm(forward_camera)), 1e-9)
    return float(np.clip(forward_camera @ viewing_ray, -1.0, 1.0))


def main() -> None:
    args = parser().parse_args()
    scales = [args.anchor_scale_deg, args.epnp_translation_scale_m,
              args.motion_scale_deg, args.gigapose_consensus_bandwidth_deg]
    weights = [args.selector_weight, args.visual_weight, args.epnp_rotation_weight,
               args.epnp_translation_weight, args.map_heading_weight,
               args.camera_facing_weight,
               args.gigapose_weight, args.gigapose_weight_with_epnp,
               args.motion_weight, args.switch_penalty, args.synthetic_flip_penalty]
    if min(scales) <= 0 or min(weights) < 0:
        raise ValueError("Scales must be positive and weights non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = CandidateBundle(args.data)
    arrays = bundle.arrays
    model, payload = load_selector(args.checkpoint, args.device)
    if payload["candidate_feature_names"] != bundle.manifest["candidate_feature_names"]:
        raise ValueError("GRU checkpoint and candidate feature schemas differ")
    if payload["frame_feature_names"] != bundle.manifest["frame_feature_names"]:
        raise ValueError("GRU checkpoint and frame feature schemas differ")
    if int(payload["saved_candidates"]) != bundle.candidate_count:
        raise ValueError("GRU checkpoint and candidate counts differ")
    frame_map = _load_frame_map(args.dataset_dir)
    gp_consensus = _gigapose_consensus(
        args.gigapose_predictions, args.gigapose_consensus_top_k,
        args.gigapose_consensus_bandwidth_deg,
    )
    mesh = trimesh.load(args.mesh, process=False)
    center_raw = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    flip_vector = np.zeros(3)
    flip_vector[{"x": 0, "y": 1, "z": 2}[args.flip_axis]] = np.pi
    flip = so3_exp(flip_vector)
    forward_axis = _forward_axis(args.object_forward_axis)
    camera_rules = _camera_facing_rules(args.camera_facing_rules)
    track_map = None
    if args.track_map is not None:
        from teacher_pipeline.v1_extended.track_map import TrackMap
        track_map = TrackMap(args.track_map)

    frame_count = len(bundle)
    candidate_count = bundle.candidate_count
    state_count = candidate_count * (2 if args.include_synthetic_flips else 1)
    probabilities = np.zeros((frame_count, candidate_count), dtype=float)
    hidden = None
    previous_segment = None
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(frame_count):
            segment = int(arrays["segment_id"][index])
            if previous_segment is None or segment != previous_segment:
                hidden = None
            previous_segment = segment
            features = torch.from_numpy(arrays["features"][index])[None, None].to(args.device)
            frame = torch.from_numpy(arrays["frame_features"][index])[None, None].to(args.device)
            valid = torch.from_numpy(arrays["valid"][index])[None, None].to(args.device)
            logits, hidden = model(features, frame, valid, hidden)
            probabilities[index] = logits[0, 0].softmax(dim=-1).cpu().numpy()

    state_poses = np.empty((frame_count, state_count, 4, 4), dtype=float)
    state_valid = np.zeros((frame_count, state_count), dtype=bool)
    unary = np.full((frame_count, state_count), np.inf, dtype=float)
    state_candidate = None
    state_polarity = None
    context: list[dict[str, Any]] = []
    feature_names = list(bundle.manifest["candidate_feature_names"])
    total_error_index = feature_names.index("total_error")
    epnp_approved_count = 0
    map_source_counts: dict[str, int] = {}

    for index in range(frame_count):
        key = (int(arrays["scene_id"][index]), int(arrays["im_id"][index]))
        frame_row = frame_map[key]
        label, epnp_status = _load_label(args.epnp_root, frame_row, args)
        metadata = _load_metadata(frame_row)
        map_camera = _map_camera_rotation(metadata)
        map_camera_pose = _map_pose(metadata)
        poses, valid, candidate_index, polarity = _candidate_states(
            np.asarray(arrays["poses"][index], dtype=float),
            np.asarray(arrays["valid"][index], dtype=bool),
            flip, center_raw, args.include_synthetic_flips,
        )
        state_poses[index] = poses
        state_valid[index] = valid
        state_candidate = candidate_index
        state_polarity = polarity

        probability_cost = -np.log(np.clip(probabilities[index, candidate_index], 1e-12, 1.0))
        probability_cost -= float(np.min(probability_cost[valid]))
        visual_cost = np.asarray(arrays["features"][index, candidate_index, total_error_index], dtype=float)
        visual_cost -= float(np.min(visual_cost[valid]))
        costs = args.selector_weight * probability_cost + args.visual_weight * visual_cost
        costs += args.synthetic_flip_penalty * polarity

        camera = str(arrays["sequence_camera"][index]).strip().lower()
        facing_rule = camera_rules.get(camera)
        facing_dots = np.full(state_count, np.nan)
        if facing_rule is not None and args.camera_facing_weight > 0:
            expected_sign = 1.0 if facing_rule == "away" else -1.0
            for state in np.flatnonzero(valid):
                facing_dots[state] = _camera_facing_dot(
                    poses[state], center_raw, forward_axis
                )
                if expected_sign * facing_dots[state] < 0.0:
                    costs[state] += args.camera_facing_weight

        epnp_pose = None
        if label is not None and label.get("_approved"):
            epnp_pose = np.eye(4)
            epnp_pose[:3, :] = np.asarray(label["T_camera_object_centered"], dtype=float)[:3, :]
            epnp_approved_count += 1
            for state in np.flatnonzero(valid):
                costs[state] += args.epnp_rotation_weight * _rotation_cost(
                    poses[state], epnp_pose, args.anchor_scale_deg
                )
                delta = np.linalg.norm(_centered_translation(poses[state], center_raw) - epnp_pose[:3, 3])
                costs[state] += args.epnp_translation_weight * min(
                    (delta / args.epnp_translation_scale_m) ** 2, 36.0
                )

        gp_rotation = gp_consensus.get(key, np.asarray(arrays["poses"][index, 0, :3, :3], dtype=float))
        epnp_cost_active = (
            epnp_pose is not None
            and (args.epnp_rotation_weight > 0 or args.epnp_translation_weight > 0)
        )
        gp_weight = (
            args.gigapose_weight_with_epnp
            if epnp_cost_active else args.gigapose_weight
        )
        for state in np.flatnonzero(valid):
            costs[state] += gp_weight * _rotation_cost(
                poses[state], gp_rotation, args.anchor_scale_deg
            )

        tangent = _embedded_tangent(label)
        map_source = "epnp_raceline" if tangent is not None else "none"
        if tangent is None and track_map is not None and map_camera_pose is not None:
            raw_selected = int(np.argmax(np.where(arrays["valid"][index], probabilities[index], -np.inf)))
            center_camera = _centered_translation(arrays["poses"][index, raw_selected], center_raw)
            point = (map_camera_pose @ np.r_[center_camera, 1.0])[:3]
            projection = track_map.project(point)
            if projection.distance <= args.maximum_track_distance_m:
                tangent = track_map.sample(projection.s)["tangent"]
                map_source = "track_map_npz"
        if tangent is not None and map_camera is not None:
            tangent = np.asarray(tangent, dtype=float)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            for state in np.flatnonzero(valid):
                direction = map_camera @ poses[state, :3, :3] @ forward_axis
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    dot = float(np.clip((direction / norm) @ tangent, -1.0, 1.0))
                    angle = math.degrees(math.acos(dot))
                    costs[state] += args.map_heading_weight * min(
                        (angle / args.anchor_scale_deg) ** 2, 36.0
                    )
        else:
            map_source = "unavailable"
        map_source_counts[map_source] = map_source_counts.get(map_source, 0) + 1
        unary[index] = np.where(valid, costs, np.inf)
        context.append({
            "key": key, "epnp_pose": epnp_pose, "epnp_status": epnp_status,
            "map_source": map_source,
            "camera": camera, "camera_facing_rule": facing_rule,
            "camera_facing_dots": facing_dots,
            "raw_selected": int(np.argmax(np.where(arrays["valid"][index], probabilities[index], -np.inf))),
        })

    assert state_candidate is not None and state_polarity is not None
    selected_states = np.zeros(frame_count, dtype=np.int64)
    path_costs = np.zeros(frame_count, dtype=float)
    for segment in np.unique(arrays["segment_id"]):
        indices = np.flatnonzero(arrays["segment_id"] == segment)
        states, costs = optimize_candidate_fixed_lag(
            state_poses[indices], unary[indices], state_valid[indices],
            state_polarity,
            BootstrapConfig(
                args.window_length, args.motion_weight,
                args.motion_scale_deg, args.switch_penalty,
            ),
        )
        selected_states[indices] = states
        path_costs[indices] = costs

    rows = []
    diagnostics = []
    before_rotation, after_rotation = [], []
    before_translation, after_translation = [], []
    changed_candidate = 0
    synthetic_flips = 0
    for index, state in enumerate(selected_states):
        candidate = int(state_candidate[state])
        polarity = int(state_polarity[state])
        raw = int(context[index]["raw_selected"])
        pose = state_poses[index, state]
        key = context[index]["key"]
        changed_candidate += int(candidate != raw)
        synthetic_flips += polarity
        source = str(arrays["sources"][index, candidate])
        if polarity:
            source += "|center_preserving_flip180"
        rows.append(pose_to_csv_row(
            scene_id=key[0], im_id=key[1], track_id=0, obj_id=1,
            confidence=float(probabilities[index, candidate]), pose_m=pose,
            mode="anchored_fixed_lag_selector",
            source=f"gru_selector_anchored|{source}", elapsed_s=0.0,
        ))
        epnp_pose = context[index]["epnp_pose"]
        raw_pose = np.asarray(arrays["poses"][index, raw], dtype=float)
        br = ar = bt = at = math.nan
        if epnp_pose is not None:
            br = rotation_error_deg(raw_pose, epnp_pose)
            ar = rotation_error_deg(pose, epnp_pose)
            bt = float(np.linalg.norm(_centered_translation(raw_pose, center_raw) - epnp_pose[:3, 3]))
            at = float(np.linalg.norm(_centered_translation(pose, center_raw) - epnp_pose[:3, 3]))
            before_rotation.append(br); after_rotation.append(ar)
            before_translation.append(bt); after_translation.append(at)
        diagnostics.append({
            "scene_id": key[0], "im_id": key[1],
            "segment_id": int(arrays["segment_id"][index]),
            "raw_selected_candidate": raw, "anchored_selected_candidate": candidate,
            "selected_polarity": polarity,
            "raw_probability": float(probabilities[index, raw]),
            "selected_probability": float(probabilities[index, candidate]),
            "selected_source": source, "lookahead_path_cost": float(path_costs[index]),
            "epnp_status": context[index]["epnp_status"],
            "map_heading_source": context[index]["map_source"],
            "camera": context[index]["camera"],
            "camera_facing_rule": context[index]["camera_facing_rule"],
            "camera_facing_dot": float(context[index]["camera_facing_dots"][state]),
            "epnp_rotation_before_deg": br, "epnp_rotation_after_deg": ar,
            "epnp_translation_before_m": bt, "epnp_translation_after_m": at,
        })

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", rows)
    with (args.output_dir / "selection_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader(); writer.writerows(diagnostics)
    report = {
        "format": "rgb_self_recovery_anchored_selector_v1",
        "frame_count": frame_count,
        "segment_count": int(len(np.unique(arrays["segment_id"]))),
        "candidate_count": candidate_count, "state_count": state_count,
        "window_length": args.window_length,
        "mesh_center_raw_m": center_raw.tolist(),
        "changed_candidate_count": changed_candidate,
        "changed_candidate_fraction": changed_candidate / frame_count,
        "synthetic_center_preserving_flip_count": synthetic_flips,
        "quality_approved_epnp_count": epnp_approved_count,
        "epnp_pose_cost_enabled": bool(
            args.epnp_rotation_weight > 0 or args.epnp_translation_weight > 0
        ),
        "map_heading_cost_enabled": bool(args.map_heading_weight > 0),
        "camera_facing_cost_enabled": bool(args.camera_facing_weight > 0),
        "camera_facing_rule_evaluated_count": int(sum(
            np.isfinite(row["camera_facing_dot"]) for row in diagnostics
        )),
        "camera_facing_rule_violation_count": int(sum(
            np.isfinite(row["camera_facing_dot"])
            and (
                row["camera_facing_dot"] < 0
                if row["camera_facing_rule"] == "away"
                else row["camera_facing_dot"] > 0
            )
            for row in diagnostics
            if row["camera_facing_rule"] in {"away", "toward"}
        )),
        "map_heading_source_counts": map_source_counts,
        "epnp_rotation_before_deg": _summary(before_rotation),
        "epnp_rotation_after_deg": _summary(after_rotation),
        "epnp_translation_before_m": _summary(before_translation),
        "epnp_translation_after_m": _summary(after_translation),
        "elapsed_s": time.perf_counter() - started,
        "score_semantics": "GRU probability of the selected source candidate; not pose accuracy",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
