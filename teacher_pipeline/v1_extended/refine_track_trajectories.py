"""Offline physical trajectory optimization in racetrack coordinates.

The optimized state per observation is [s, d, v_s, v_d, a_s, delta_yaw].
All observations in a tracklet are solved jointly, so both past and future
frames constrain weak measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from teacher_pipeline.geometry import as_pose, raw_to_centered_pose, so3_log
from teacher_pipeline.io import write_json, write_rows
from teacher_pipeline.trajectory import load

from .track_map import TrackMap, unwrap_progress


def _rotation_mean(values):
    u, _, vt = np.linalg.svd(np.mean(values, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def _times(rows, fps):
    values = np.asarray(
        [float(row.get("timestamp_ns", row.get("im_id", i))) for i, row in enumerate(rows)]
    )
    differences = np.diff(values)
    positive = differences[differences > 0]
    if len(positive) and np.median(positive) > 1e6:
        values = (values - values[0]) * 1e-9
    elif len(positive) and np.median(positive) > 1e3:
        values = (values - values[0]) * 1e-3
    else:
        values = np.arange(len(rows), dtype=float) / fps
    return values


def _split_tracklet(rows, max_gap_s):
    rows = sorted(
        rows, key=lambda row: float(row.get("timestamp_ns", row.get("im_id", 0)))
    )
    if max_gap_s <= 0 or len(rows) < 2:
        return [rows]
    segments = [[rows[0]]]
    for row in rows[1:]:
        previous = segments[-1][-1]
        current_time = float(row.get("timestamp_ns", 0) or 0)
        previous_time = float(previous.get("timestamp_ns", 0) or 0)
        gap = current_time - previous_time
        if abs(gap) > 1e6:
            gap *= 1e-9
        elif abs(gap) > 1e3:
            gap *= 1e-3
        if gap <= 0 or gap > max_gap_s:
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def _camera_pose(row, source):
    if source == "raw":
        return raw_to_centered_pose(as_pose(row["T_camera_object_raw_gigapose"]))
    if source == "tracked" and row.get("T_camera_object_centered_tracked") is not None:
        return as_pose(row["T_camera_object_centered_tracked"])
    return as_pose(row["T_camera_object_centered_gigapose"])


def _map_measurements(rows, extrinsic, source):
    output = []
    for row in rows:
        fixed = extrinsic if extrinsic is not None else as_pose(row["T_lidar_camera_initial"])
        output.append(as_pose(row["T_map_lidar"]) @ fixed @ _camera_pose(row, source))
    return output


def _quality_sigmas(row, pose, args):
    distance = max(float(np.linalg.norm(pose[:3, 3])), 1.0)
    box = row.get("bbox_xywh")
    height = float(box[3]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0
    score = float(row.get("gigapose_score", 0.5) or 0.5)
    image_quality = np.clip(height / args.reference_bbox_height_px, 0.15, 1.5)
    quality = np.clip(score * image_quality, 0.08, 1.0)
    return (
        (args.sigma_s_base_m + args.sigma_s_per_range * distance) / quality,
        (args.sigma_d_base_m + args.sigma_d_per_range * distance) / quality,
        np.radians((args.sigma_yaw_base_deg + args.sigma_yaw_per_range_deg * distance) / quality),
    )


def _lidar_map_centroid(row):
    value = row.get("lidar_centroid_map")
    if value is not None:
        return np.asarray(value, dtype=float).reshape(3)
    value = row.get("lidar_centroid")
    if value is None:
        return None
    point = np.r_[np.asarray(value, dtype=float).reshape(3), 1.0]
    return (as_pose(row["T_map_lidar"]) @ point)[:3]


def _jacobian_sparsity(rows, lidar_active):
    """Declare the local block structure used by finite differences.

    Per-frame observation terms touch one six-value state. Motion terms touch
    only two adjacent states. Supplying this pattern prevents SciPy from
    perturbing every one of the O(N) variables separately for every Jacobian.
    """
    frame_count = len(rows)
    residual_count = sum(5 + (3 if active else 0) for active in lidar_active)
    residual_count += 6 * max(frame_count - 1, 0)
    residual_count += max(frame_count - 2, 0)
    pattern = lil_matrix((residual_count, 6 * frame_count), dtype=np.int8)
    cursor = 0
    for index, active in enumerate(lidar_active):
        count = 5 + (3 if active else 0)
        pattern[cursor : cursor + count, 6 * index : 6 * (index + 1)] = 1
        cursor += count
    for index in range(frame_count - 1):
        count = 6 + (1 if index + 1 < frame_count - 1 else 0)
        pattern[cursor : cursor + count, 6 * index : 6 * (index + 2)] = 1
        cursor += count
    return pattern.tocsr()


def _soft_l1_measurement_residual(value, scale):
    """Encode a soft-L1 measurement cost for a linear least-squares solver."""
    value = float(value)
    if scale <= 0:
        return value
    normalized = value / scale
    transformed = np.sqrt(2.0 * (np.sqrt(1.0 + normalized ** 2) - 1.0))
    return float(np.copysign(scale * transformed, value))


def optimize_track(rows, track, extrinsic, args):
    rows = sorted(rows, key=lambda row: float(row.get("timestamp_ns", row.get("im_id", 0))))
    measurements = _map_measurements(rows, extrinsic, args.pose_source)
    times = _times(rows, args.fps)
    dt = np.maximum(np.diff(times), 1.0 / (5.0 * args.fps))
    projections = []
    reference = None
    for pose in measurements:
        projection = track.project(pose[:3, 3], reference_s=reference)
        projections.append(projection)
        reference = projection.s
    s_measurement = unwrap_progress(np.asarray([p.s for p in projections]), track.length)
    d_measurement = np.asarray([p.d for p in projections])
    frames = []
    relative_rotations = []
    for projection, pose in zip(projections, measurements):
        sample = track.sample(projection.s)
        frame = np.column_stack([sample["tangent"], sample["lateral"], sample["normal"]])
        frames.append(frame)
        relative_rotations.append(frame.T @ pose[:3, :3])
    alignment = _rotation_mean(np.stack(relative_rotations))
    yaw_measurement = np.asarray([
        np.arctan2((frame.T @ pose[:3, :3] @ alignment.T)[1, 0],
                   (frame.T @ pose[:3, :3] @ alignment.T)[0, 0])
        for frame, pose in zip(frames, measurements)
    ])
    vs = np.gradient(s_measurement, times) if len(rows) > 1 else np.zeros(1)
    vd = np.gradient(d_measurement, times) if len(rows) > 1 else np.zeros(1)
    acceleration = np.gradient(vs, times) if len(rows) > 2 else np.zeros(len(rows))
    initial = np.column_stack([s_measurement, d_measurement, vs, vd, acceleration, yaw_measurement])
    sigmas = [_quality_sigmas(row, pose, args) for row, pose in zip(rows, measurements)]
    lidar = [_lidar_map_centroid(row) for row in rows]
    lidar_active = []
    for row, centroid in zip(rows, lidar):
        count = int(float(row.get("lidar_point_count", 0) or 0))
        range_m = float(row.get("lidar_range_m", 0) or 0)
        lidar_active.append(
            centroid is not None
            and count >= args.min_lidar_points
            and (range_m <= 0 or range_m <= args.lidar_reliable_range_m)
        )

    def residual(flat):
        state = flat.reshape(-1, 6)
        values = []
        for i, row in enumerate(rows):
            sigma_s, sigma_d, sigma_yaw = sigmas[i]
            measurement_values = [
                (state[i, 0] - s_measurement[i]) / sigma_s,
                (state[i, 1] - d_measurement[i]) / sigma_d,
                np.arctan2(np.sin(state[i, 5] - yaw_measurement[i]),
                           np.cos(state[i, 5] - yaw_measurement[i])) / sigma_yaw,
            ]
            values.extend(
                _soft_l1_measurement_residual(
                    value, getattr(args, "measurement_robust_scale", 3.0)
                )
                for value in measurement_values
            )
            sample = track.sample(state[i, 0])
            values.extend([
                max(0.0, state[i, 1] - sample["right_width"]) / args.sigma_boundary_m,
                max(0.0, -sample["left_width"] - state[i, 1]) / args.sigma_boundary_m,
            ])
            if lidar_active[i]:
                predicted = track.pose(state[i, 0], state[i, 1], state[i, 5], alignment)[:3, 3]
                values.extend(((predicted - lidar[i]) / args.sigma_lidar_m).tolist())
        for i, delta_t in enumerate(dt):
            current, following = state[i], state[i + 1]
            values.extend([
                (following[0] - current[0] - current[2] * delta_t - 0.5 * current[4] * delta_t ** 2) / args.sigma_motion_s_m,
                (following[2] - current[2] - current[4] * delta_t) / args.sigma_velocity_mps,
                (following[1] - current[1] - current[3] * delta_t) / args.sigma_motion_d_m,
                (following[3] - current[3]) / args.sigma_lateral_velocity_mps,
                np.arctan2(np.sin(following[5] - current[5]), np.cos(following[5] - current[5])) / np.radians(args.sigma_yaw_rate_deg),
                current[4] / args.sigma_acceleration_mps2,
            ])
            if i + 1 < len(dt):
                values.append((following[4] - current[4]) / args.sigma_jerk_mps3)
        return np.asarray(values)

    result = least_squares(
        residual,
        initial.reshape(-1),
        loss="linear",
        max_nfev=args.max_iterations,
        jac_sparsity=_jacobian_sparsity(rows, lidar_active),
        tr_solver="lsmr",
    )
    state = result.x.reshape(-1, 6)
    for i, row in enumerate(rows):
        pose = track.pose(state[i, 0], state[i, 1], state[i, 5], alignment)
        initial_pose = measurements[i]
        row.update({
            "T_map_object_centered_refined": pose.tolist(),
            "T_camera_object_centered_physical_input": _camera_pose(
                row, args.pose_source
            ).tolist(),
            "track_s_m": float(track.wrap_s(state[i, 0])), "track_s_unwrapped_m": float(state[i, 0]),
            "track_d_m": float(state[i, 1]), "longitudinal_velocity_mps": float(state[i, 2]),
            "lateral_velocity_mps": float(state[i, 3]), "longitudinal_acceleration_mps2": float(state[i, 4]),
            "yaw_offset_rad": float(state[i, 5]), "teacher_status": "physical_track_optimized",
            "teacher_pose_source": args.pose_source,
            "gigapose_translation_correction_m": (pose[:3, 3] - initial_pose[:3, 3]).tolist(),
            "gigapose_rotation_correction_deg": float(np.degrees(np.linalg.norm(so3_log(pose[:3, :3] @ initial_pose[:3, :3].T)))),
            "sources": sorted(set(row.get("sources", []) + ["gigapose", "track_map", "temporal"] + (["lidar"] if lidar[i] is not None else []))),
        })
    return rows, {"track_id": rows[0].get("track_id"), "count": len(rows), "success": bool(result.success),
                  "cost": float(result.cost), "nfev": int(result.nfev)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", type=Path, required=True); p.add_argument("--track-map", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path)
    p.add_argument("--extrinsics", type=Path); p.add_argument("--pose-source", choices=("raw", "teacher-input", "tracked"), default="raw")
    p.add_argument("--fps", type=float, default=20.0); p.add_argument("--max-iterations", type=int, default=250)
    p.add_argument(
        "--max-temporal-gap-s",
        type=float,
        default=2.0,
        help="Split a reused/interrupted track ID across larger timestamp gaps",
    )
    p.add_argument("--reference-bbox-height-px", type=float, default=100.0)
    p.add_argument("--sigma-s-base-m", type=float, default=.35); p.add_argument("--sigma-s-per-range", type=float, default=.025)
    p.add_argument("--sigma-d-base-m", type=float, default=.20); p.add_argument("--sigma-d-per-range", type=float, default=.008)
    p.add_argument("--sigma-yaw-base-deg", type=float, default=2.0); p.add_argument("--sigma-yaw-per-range-deg", type=float, default=.04)
    p.add_argument("--sigma-motion-s-m", type=float, default=.25); p.add_argument("--sigma-motion-d-m", type=float, default=.12)
    p.add_argument("--sigma-velocity-mps", type=float, default=2.0); p.add_argument("--sigma-lateral-velocity-mps", type=float, default=1.0)
    p.add_argument("--sigma-acceleration-mps2", type=float, default=8.0); p.add_argument("--sigma-jerk-mps3", type=float, default=12.0)
    p.add_argument("--sigma-yaw-rate-deg", type=float, default=8.0); p.add_argument("--sigma-boundary-m", type=float, default=.15)
    p.add_argument("--min-lidar-points", type=int, default=8); p.add_argument("--lidar-reliable-range-m", type=float, default=60.0); p.add_argument("--sigma-lidar-m", type=float, default=.35)
    p.add_argument(
        "--measurement-robust-scale",
        type=float,
        default=3.0,
        help="Soft-L1 transition in normalized residual units for GigaPose terms only",
    )
    return p.parse_args()


def main():
    args = parse_args(); rows = load(args.trajectories); track = TrackMap(args.track_map); extrinsic = None
    if args.extrinsics:
        extrinsic = as_pose(json.loads(args.extrinsics.read_text())["T_lidar_camera_optimized"])
    grouped = {}
    for row in rows: grouped.setdefault(str(row.get("track_id")), []).append(row)
    output, reports = [], []
    segments = [
        (track_id, segment_index, segment)
        for track_id, values in grouped.items()
        for segment_index, segment in enumerate(
            _split_tracklet(values, args.max_temporal_gap_s)
        )
    ]
    for track_id, segment_index, values in segments:
        print(
            f"Optimizing track_id={track_id}, segment={segment_index} "
            f"({len(values)} frames, {6 * len(values)} variables)...",
            flush=True,
        )
        refined, report = optimize_track(values, track, extrinsic, args)
        for row in refined:
            row["teacher_track_segment_index"] = segment_index
        report["segment_index"] = segment_index
        output.extend(refined); reports.append(report)
        print(
            f"Finished track_id={report['track_id']}, segment={segment_index}: "
            f"success={report['success']}, nfev={report['nfev']}, "
            f"cost={report['cost']:.3f}",
            flush=True,
        )
    output.sort(key=lambda row: (str(row.get("track_id")), float(row.get("timestamp_ns", row.get("im_id", 0)))))
    write_rows(args.output, output); report_path = args.report or args.output.with_suffix(".report.json")
    write_json(report_path, {"format": "teacher_v1_extended_physical", "state": ["s", "d", "v_s", "v_d", "a_s", "delta_yaw"], "track_count": len(grouped), "segment_count": len(reports), "rows": len(output), "pose_source": args.pose_source, "tracks": reports})
    print(f"Optimized {len(output)} observations across {len(grouped)} track IDs / {len(reports)} continuous segments -> {args.output}")


if __name__ == "__main__": main()
