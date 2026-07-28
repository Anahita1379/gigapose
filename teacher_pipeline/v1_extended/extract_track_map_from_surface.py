"""Extract an approximate centerline from an ASCII track_scene.ply surface."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from teacher_pipeline.geometry import as_pose
from teacher_pipeline.trajectory import load
from .prepare_track_map import prepare


def read_ascii_ply(path: Path):
    with path.open() as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            header.append(line.strip())
            if line.strip() == "end_header": break
        if "format ascii 1.0" not in header:
            raise ValueError("Surface extractor currently requires ASCII PLY")
        vertex_count = int(next(line.split()[-1] for line in header if line.startswith("element vertex")))
        face_count = int(next(line.split()[-1] for line in header if line.startswith("element face")))
        vertices = np.asarray([[float(value) for value in handle.readline().split()[:3]] for _ in range(vertex_count)])
        faces = []
        for _ in range(face_count):
            values = handle.readline().split(); count = int(values[0])
            indices = [int(value) for value in values[1 : 1 + count]]
            if count == 3: faces.append(indices)
            elif count > 3:
                faces.extend([[indices[0], indices[i], indices[i + 1]] for i in range(1, count - 1)])
    return vertices, np.asarray(faces, dtype=np.int64)


def _largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if not count: raise ValueError("Track raster produced no connected surface")
    sizes = np.bincount(labels.reshape(-1)); sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _cycle_core(skeleton):
    core = skeleton.copy()
    kernel = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
    while True:
        degree = ndimage.convolve(core.astype(np.uint8), kernel, mode="constant")
        endpoints = core & (degree <= 1)
        if not endpoints.any(): return core
        core[endpoints] = False
        if not core.any(): raise ValueError("Surface skeleton has no closed-loop core")


def _candidate_cycles(core):
    import networkx as nx

    coordinates = {tuple(value) for value in np.argwhere(core)}
    graph = nx.Graph()
    graph.add_nodes_from(coordinates)
    directions = [
        (a, b)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        if (a, b) != (0, 0)
    ]
    for row, column in coordinates:
        for dr, dc in directions:
            neighbor = (row + dr, column + dc)
            if neighbor in coordinates and neighbor > (row, column):
                graph.add_edge((row, column), neighbor)
    cycles = [
        np.asarray(cycle, dtype=int)
        for cycle in nx.cycle_basis(graph)
        if len(cycle) >= 0.1 * len(coordinates)
    ]
    return sorted(cycles, key=len, reverse=True)


def _ordered_cycle(core):
    cycles = _candidate_cycles(core)
    if cycles:
        return cycles[0]

    coordinates = {tuple(value) for value in np.argwhere(core)}
    directions = [
        (a, b)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        if (a, b) != (0, 0)
    ]

    start = min(coordinates); output = [start]; previous = None; current = start
    for _ in range(len(coordinates) * 2):
        candidates = [(current[0] + dr, current[1] + dc) for dr, dc in directions]
        candidates = [value for value in candidates if value in coordinates and value != previous]
        if not candidates: break
        unvisited = [value for value in candidates if value not in set(output)]
        if unvisited: candidates = unvisited
        if previous is None: following = candidates[0]
        else:
            incoming = np.asarray(current) - np.asarray(previous)
            following = max(candidates, key=lambda value: float(incoming @ (np.asarray(value) - np.asarray(current))))
        if following == start: break
        previous, current = current, following; output.append(current)
    if len(output) < 0.6 * len(coordinates):
        raise ValueError(f"Could order only {len(output)}/{len(coordinates)} centerline pixels; inspect the diagnostic raster")
    return np.asarray(output, dtype=int)


def _cycle_length_pixels(cycle):
    closed = np.vstack([cycle, cycle[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _select_guided_cycle(cycles, guide_pixels, min_length_ratio):
    if not cycles:
        raise ValueError("Surface skeleton contains no credible long cycle")
    lengths = np.asarray([_cycle_length_pixels(cycle) for cycle in cycles])
    eligible = [
        (index, cycle)
        for index, cycle in enumerate(cycles)
        if lengths[index] >= min_length_ratio * lengths.max()
    ]
    diagnostics = []
    for index, cycle in eligible:
        distances, _ = cKDTree(cycle.astype(float)).query(guide_pixels)
        diagnostics.append(
            {
                "candidate_index": int(index),
                "pixel_count": int(len(cycle)),
                "length_px": float(lengths[index]),
                "guide_distance_px_median": float(np.median(distances)),
                "guide_distance_px_p90": float(np.percentile(distances, 90)),
                "score": float(np.median(distances) + 0.35 * np.percentile(distances, 90)),
            }
        )
    chosen = min(diagnostics, key=lambda row: row["score"])
    return cycles[chosen["candidate_index"]], diagnostics, chosen


def _apply_planar_transform(xy, parameters):
    theta, tx, ty = np.asarray(parameters, dtype=float)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return np.asarray(xy, dtype=float) @ rotation.T + np.asarray([tx, ty])


def _invert_planar_transform(xy, parameters):
    theta, tx, ty = np.asarray(parameters, dtype=float)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return (np.asarray(xy, dtype=float) - np.asarray([tx, ty])) @ rotation


def _cycle_map_xy(cycle, minimum, resolution, axis_order, translation_xy):
    ac_xz = minimum + cycle[:, ::-1] * resolution
    if axis_order == "z-negx-y":
        map_xy = np.column_stack([ac_xz[:, 1], -ac_xz[:, 0]])
    else:
        map_xy = ac_xz
    return map_xy + np.asarray(translation_xy)


def _fit_planar_transform(candidate_xy, guide_xy, max_yaw_deg):
    """Robustly register a fixed loop to map-frame guide points with SE(2)."""
    candidate_xy = np.asarray(candidate_xy, dtype=float)
    guide_xy = np.asarray(guide_xy, dtype=float)
    tree = cKDTree(candidate_xy)
    center_delta = np.median(guide_xy, axis=0) - np.median(candidate_xy, axis=0)
    span = max(
        100.0,
        0.65
        * max(
            np.linalg.norm(np.ptp(candidate_xy, axis=0)),
            np.linalg.norm(np.ptp(guide_xy, axis=0)),
        ),
    )

    def objective(parameters):
        local_guide = _invert_planar_transform(guide_xy, parameters)
        distances, _ = tree.query(local_guide)
        # Median handles occasional bad odometry poses; p90 prevents a fit that
        # explains only one small portion of a complete driven lap.
        return float(np.median(distances) + 0.35 * np.percentile(distances, 90))

    yaw_limit = np.deg2rad(max_yaw_deg)
    bounds = [
        (-yaw_limit, yaw_limit),
        (center_delta[0] - span, center_delta[0] + span),
        (center_delta[1] - span, center_delta[1] + span),
    ]
    global_result = differential_evolution(
        objective,
        bounds,
        seed=0,
        popsize=10,
        maxiter=35,
        tol=1e-4,
        polish=False,
        workers=1,
    )
    result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 300, "xtol": 1e-5, "ftol": 1e-5},
    )
    parameters = result.x if result.fun <= global_result.fun else global_result.x
    local_guide = _invert_planar_transform(guide_xy, parameters)
    distances, _ = tree.query(local_guide)
    return parameters, distances


def _select_aligned_guided_cycle(
    cycles,
    guide_map_xy,
    minimum,
    resolution,
    axis_order,
    translation_xy,
    min_length_ratio,
    max_yaw_deg,
):
    if not cycles:
        raise ValueError("Surface skeleton contains no credible long cycle")
    lengths = np.asarray([_cycle_length_pixels(cycle) for cycle in cycles])
    eligible = [
        (index, cycle)
        for index, cycle in enumerate(cycles)
        if lengths[index] >= min_length_ratio * lengths.max()
    ]
    diagnostics = []
    transforms = {}
    for index, cycle in eligible:
        candidate_xy = _cycle_map_xy(
            cycle, minimum, resolution, axis_order, translation_xy
        )
        parameters, distances = _fit_planar_transform(
            candidate_xy, guide_map_xy, max_yaw_deg
        )
        score = float(
            np.median(distances) + 0.35 * np.percentile(distances, 90)
        )
        diagnostics.append(
            {
                "candidate_index": int(index),
                "pixel_count": int(len(cycle)),
                "length_px": float(lengths[index]),
                "guide_distance_m_median": float(np.median(distances)),
                "guide_distance_m_p90": float(np.percentile(distances, 90)),
                "score": score,
                "auto_planar_yaw_deg": float(np.rad2deg(parameters[0])),
                "auto_planar_translation_m": parameters[1:].tolist(),
            }
        )
        transforms[index] = parameters
    chosen = min(diagnostics, key=lambda row: row["score"])
    index = chosen["candidate_index"]
    return cycles[index], diagnostics, chosen, transforms[index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface-ply", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic-png", type=Path); p.add_argument("--resolution-m", type=float, default=.35)
    p.add_argument("--morphology-radius-px", type=int, default=2); p.add_argument("--resample-spacing-m", type=float, default=1.0)
    p.add_argument(
        "--axis-order",
        choices=("xzy", "z-negx-y", "xyz"),
        default="xzy",
        help=(
            "AC-to-map axis convention. For the 2026-05-05 recording, "
            "z-negx-y means map=(AC_Z,-AC_X,AC_Y)."
        ),
    )
    p.add_argument("--translation", nargs=3, type=float, default=(0, 0, 0))
    p.add_argument("--guide-observations", type=Path, help="Full observations containing recorded ego T_map_lidar poses. Long loop candidates are selected against this driven path, avoiding pit-lane branches.")
    p.add_argument("--guide-pose-field", default="T_map_lidar")
    p.add_argument(
        "--vertical-alignment",
        choices=("epnp", "none"),
        default="epnp",
        help=(
            "With guide observations, estimate the PLY-to-map vertical offset "
            "from T_map_object_centered_epnp anchors (default), or retain only "
            "the explicit --translation Z with 'none'."
        ),
    )
    p.add_argument("--candidate-min-length-ratio", type=float, default=.75)
    p.add_argument("--max-guide-distance-m", type=float, default=20.0)
    p.add_argument(
        "--max-auto-yaw-deg",
        type=float,
        default=30.0,
        help="Maximum absolute yaw used for guided automatic planar registration",
    )
    args = p.parse_args(); vertices, faces = read_ascii_ply(args.surface_ply)
    horizontal = vertices[:, [0, 2]]; minimum = horizontal.min(axis=0) - 2.0; maximum = horizontal.max(axis=0) + 2.0
    width, height = np.ceil((maximum - minimum) / args.resolution_m).astype(int) + 1
    if width * height > 150_000_000: raise ValueError("Track raster is too large; increase --resolution-m")
    pixels = np.rint((horizontal - minimum) / args.resolution_m).astype(np.int32)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    polygons = pixels[faces]
    for start in range(0, len(polygons), 10000): cv2.fillPoly(occupancy, polygons[start : start + 10000], 1)
    radius = args.morphology_radius_px
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
    occupancy = _largest_component(occupancy.astype(bool)); distance = ndimage.distance_transform_edt(occupancy)
    core = _cycle_core(_largest_component(skeletonize(occupancy)))
    cycles = _candidate_cycles(core)
    guide_pixels = None
    guide_rows = []
    candidate_diagnostics = []
    chosen_diagnostic = None
    planar_transform = np.asarray([0.0, 0.0, 0.0])
    if args.guide_observations:
        guide_rows = sorted(
            load(args.guide_observations),
            key=lambda row: (
                int(row.get("scene_id", 0)),
                int(row.get("im_id", 0)),
            ),
        )
        guide_map = []
        seen = set()
        for row in guide_rows:
            value = row.get(args.guide_pose_field)
            if value is None:
                continue
            token = (row.get("scene_id"), row.get("im_id"))
            if token in seen:
                continue
            seen.add(token)
            guide_map.append(as_pose(value)[:3, 3])
        if len(guide_map) < 3:
            raise ValueError("--guide-observations has fewer than 3 unique ego poses")
        guide_map = np.asarray(guide_map)
        ordered, candidate_diagnostics, chosen_diagnostic, planar_transform = (
            _select_aligned_guided_cycle(
                cycles,
                guide_map[:, :2],
                minimum,
                args.resolution_m,
                args.axis_order,
                np.asarray(args.translation)[:2],
                args.candidate_min_length_ratio,
                args.max_auto_yaw_deg,
            )
        )
        guide_median = chosen_diagnostic["guide_distance_m_median"]
        if guide_median > args.max_guide_distance_m:
            raise ValueError(
                f"Best loop is still {guide_median:.2f} m median from the ego path; "
                "track and metadata map frames are likely misaligned"
            )
    else:
        ordered = cycles[0] if cycles else _ordered_cycle(core)
    world_horizontal = minimum + ordered[:, ::-1] * args.resolution_m
    tree = cKDTree(horizontal); _, nearest = tree.query(world_horizontal)
    vertical = vertices[nearest, 1]
    if args.axis_order == "xzy":
        center = np.column_stack(
            [world_horizontal[:, 0], world_horizontal[:, 1], vertical]
        )
    elif args.axis_order == "z-negx-y":
        center = np.column_stack(
            [world_horizontal[:, 1], -world_horizontal[:, 0], vertical]
        )
    else:
        center = np.column_stack(
            [world_horizontal[:, 0], vertical, world_horizontal[:, 1]]
        )
    center += np.asarray(args.translation)
    if args.guide_observations:
        center[:, :2] = _apply_planar_transform(
            center[:, :2], planar_transform
        )
    vertical_alignment = {
        "mode": args.vertical_alignment,
        "anchor_count": 0,
        "inlier_count": 0,
        "automatic_offset_m": 0.0,
    }
    if args.guide_observations and args.vertical_alignment == "epnp":
        epnp_xyz = []
        for row in guide_rows:
            value = row.get("T_map_object_centered_epnp")
            if value is not None:
                epnp_xyz.append(as_pose(value)[:3, 3])
        vertical_alignment["anchor_count"] = len(epnp_xyz)
        if len(epnp_xyz) < 3:
            raise ValueError(
                "--vertical-alignment epnp requires at least 3 "
                "T_map_object_centered_epnp anchors; use "
                "--vertical-alignment none and --translation X Y Z if the "
                "vertical offset is known manually"
            )
        epnp_xyz = np.asarray(epnp_xyz)
        horizontal_distances, center_indices = cKDTree(center[:, :2]).query(
            epnp_xyz[:, :2]
        )
        inliers = horizontal_distances <= args.max_guide_distance_m
        vertical_alignment["inlier_count"] = int(inliers.sum())
        if inliers.sum() < 3:
            raise ValueError(
                "Fewer than 3 EPnP anchors are horizontally close to the "
                "selected track loop; cannot estimate its vertical map offset"
            )
        vertical_residuals = (
            epnp_xyz[inliers, 2] - center[center_indices[inliers], 2]
        )
        automatic_offset = float(np.median(vertical_residuals))
        center[:, 2] += automatic_offset
        remaining = vertical_residuals - automatic_offset
        vertical_alignment.update(
            {
                "automatic_offset_m": automatic_offset,
                "residual_abs_m_median": float(np.median(np.abs(remaining))),
                "residual_abs_m_p90": float(
                    np.percentile(np.abs(remaining), 90)
                ),
            }
        )
    segment = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1); cumulative = np.r_[0.0, np.cumsum(segment)]
    targets = np.arange(0.0, cumulative[-1], args.resample_spacing_m)
    extended = np.vstack([center, center[:1]])
    center = np.column_stack([np.interp(targets, cumulative, extended[:, axis]) for axis in range(3)])
    widths_px = distance[ordered[:, 0], ordered[:, 1]]
    widths = np.interp(targets, cumulative, np.r_[widths_px, widths_px[:1]]) * args.resolution_m
    output = prepare(center, widths, widths, closed=True); args.output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, **output)
    diagnostic = args.diagnostic_png or args.output.with_suffix(".png")
    view = np.dstack([occupancy * 90, occupancy * 90, occupancy * 90]).astype(np.uint8)
    view[ordered[:, 0], ordered[:, 1]] = (40, 40, 255)
    if args.guide_observations:
        local_guide = _invert_planar_transform(guide_map[:, :2], planar_transform)
        local_guide -= np.asarray(args.translation)[:2]
        if args.axis_order == "z-negx-y":
            guide_horizontal = np.column_stack(
                [-local_guide[:, 1], local_guide[:, 0]]
            )
        else:
            guide_horizontal = local_guide
        guide_xy = (guide_horizontal - minimum) / args.resolution_m
        guide_pixels = guide_xy[:, ::-1]
        guide_int = np.rint(guide_pixels[:, ::-1]).astype(np.int32)
        cv2.polylines(view, [guide_int], False, (255, 255, 0), 2)
    cv2.imwrite(str(diagnostic), view)
    report = {
        "surface_ply": str(args.surface_ply),
        "axis_order": args.axis_order,
        "translation": list(args.translation),
        "vertical_alignment": vertical_alignment,
        "auto_planar_alignment": {
            "yaw_deg": float(np.rad2deg(planar_transform[0])),
            "translation_m": planar_transform[1:].tolist(),
            "max_yaw_deg": args.max_auto_yaw_deg,
        },
        "guided": args.guide_observations is not None,
        "guide_observations": None if args.guide_observations is None else str(args.guide_observations),
        "long_cycle_candidate_count": len(cycles),
        "long_cycle_candidates": [
            {
                "candidate_index": index,
                "pixel_count": int(len(cycle)),
                "length_px": _cycle_length_pixels(cycle),
            }
            for index, cycle in enumerate(cycles)
        ],
        "candidate_min_length_ratio": args.candidate_min_length_ratio,
        "chosen_candidate": chosen_diagnostic,
        "candidate_diagnostics": candidate_diagnostics,
        "track_length_m": float(output["track_length_m"]),
    }
    args.output.with_suffix(".report.json").write_text(__import__("json").dumps(report, indent=2))
    print(f"Extracted {len(center)} samples, length={float(output['track_length_m']):.1f} m -> {args.output}")
    print(f"Inspect centerline diagnostic before use: {diagnostic}")


if __name__ == "__main__": main()
