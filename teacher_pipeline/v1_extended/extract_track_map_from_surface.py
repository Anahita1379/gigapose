"""Extract an approximate centerline from an ASCII track_scene.ply surface."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

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


def _ordered_cycle(core):
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
    cycles = nx.cycle_basis(graph)
    if cycles:
        # cycle_basis returns each cycle in traversal order. The drivable loop
        # is overwhelmingly longer than pixel-scale junction cycles.
        cycle = max(cycles, key=len)
        if len(cycle) >= 0.1 * len(coordinates):
            return np.asarray(cycle, dtype=int)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface-ply", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic-png", type=Path); p.add_argument("--resolution-m", type=float, default=.35)
    p.add_argument("--morphology-radius-px", type=int, default=2); p.add_argument("--resample-spacing-m", type=float, default=1.0)
    p.add_argument("--axis-order", choices=("xzy", "xyz"), default="xzy", help="AC track PLY is Y-up; xzy maps it to a Z-up map frame")
    p.add_argument("--translation", nargs=3, type=float, default=(0, 0, 0))
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
    core = _cycle_core(_largest_component(skeletonize(occupancy))); ordered = _ordered_cycle(core)
    world_horizontal = minimum + ordered[:, ::-1] * args.resolution_m
    tree = cKDTree(horizontal); _, nearest = tree.query(world_horizontal)
    vertical = vertices[nearest, 1]
    if args.axis_order == "xzy": center = np.column_stack([world_horizontal[:, 0], world_horizontal[:, 1], vertical])
    else: center = np.column_stack([world_horizontal[:, 0], vertical, world_horizontal[:, 1]])
    center += np.asarray(args.translation)
    segment = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1); cumulative = np.r_[0.0, np.cumsum(segment)]
    targets = np.arange(0.0, cumulative[-1], args.resample_spacing_m)
    extended = np.vstack([center, center[:1]])
    center = np.column_stack([np.interp(targets, cumulative, extended[:, axis]) for axis in range(3)])
    widths_px = distance[ordered[:, 0], ordered[:, 1]]
    widths = np.interp(targets, cumulative, np.r_[widths_px, widths_px[:1]]) * args.resolution_m
    output = prepare(center, widths, widths, closed=True); args.output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, **output)
    diagnostic = args.diagnostic_png or args.output.with_suffix(".png")
    view = np.dstack([occupancy * 90, occupancy * 90, occupancy * 90]).astype(np.uint8); view[ordered[:, 0], ordered[:, 1]] = (40, 40, 255); cv2.imwrite(str(diagnostic), view)
    print(f"Extracted {len(center)} samples, length={float(output['track_length_m']):.1f} m -> {args.output}")
    print(f"Inspect centerline diagnostic before use: {diagnostic}")


if __name__ == "__main__": main()
