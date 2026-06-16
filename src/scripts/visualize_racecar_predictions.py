"""
Visualize racecar pose predictions by projecting the CAD bounding box.

This is a quick qualitative check for GigaPose outputs. It reads the BOP-style
prediction CSV written by test.py, loads the matching source frame from the
rosbag extraction, crops the middle RGB camera view, and draws:

  - the projected 3D CAD bounding box in green
  - the predicted object coordinate axes in red/green/blue
  - the predicted object origin as a yellow dot

If the pose is reasonable, the green box should roughly sit on top of the car.
This does not prove metric accuracy, but it is the fastest way to see whether
the prediction is in the right part of the image with a plausible orientation.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "gigaPose_datasets/results/large_racecar_test/predictions/"
            "large-pbrreal-rgb-mmodel_racecar-test_racecar_test.csv"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/anahita/Dataset/rosbag_extracted_300m"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gigaPose_datasets/results/large_racecar_test/overlays"),
    )
    parser.add_argument(
        "--frame-map",
        type=Path,
        default=Path("gigaPose_datasets/datasets/racecar/frame_map.json"),
        help=(
            "JSON map from GigaPose im_id values to original source frame ids. "
            "Written by prepare_racecar_dataset.py."
        ),
    )
    parser.add_argument(
        "--frame-ids",
        default=None,
        help=(
            "Comma-separated source frame ids used to build the dataset, e.g. "
            "005020,005025,005030. Overrides --frame-map."
        ),
    )
    parser.add_argument("--split-file", default="val_image.txt")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Visualize every Nth prediction after sorting by image id.",
    )
    return parser.parse_args()


def frame_stem_from_line(line):
    match = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", line.strip())
    if not match:
        raise ValueError(f"Could not parse frame id from line: {line}")
    return match.group(1)


def load_frame_stems(source_root, split_file):
    split_path = source_root / split_file
    lines = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    return [frame_stem_from_line(line) for line in lines]


def normalize_frame_id(frame_id):
    digits = "".join(ch for ch in frame_id.strip() if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame id from: {frame_id}")
    return digits.zfill(6)


def load_im_id_to_frame_id(args):
    if args.frame_ids:
        stems = [
            normalize_frame_id(part)
            for part in args.frame_ids.split(",")
            if part.strip()
        ]
        return {im_id: stem for im_id, stem in enumerate(stems)}

    if args.frame_map.exists():
        import json

        rows = json.loads(args.frame_map.read_text())
        return {int(row["im_id"]): row["source_frame_id"] for row in rows}

    stems = load_frame_stems(args.source_root, args.split_file)
    return {im_id: stem for im_id, stem in enumerate(stems)}


def crop_middle_rgb(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    third = width // 3
    return np.array(image.crop((third, 0, 2 * third, height)))


def maybe_adjust_intrinsics(K, full_rgb_width, crop_width):
    K = K.astype(float).copy()
    # Intrinsics in this dataset are already for the 768-wide middle image. If a
    # future export stores triple-wide intrinsics, this shifts cx into the crop.
    if K[0, 2] >= crop_width and full_rgb_width == crop_width * 3:
        K[0, 2] -= crop_width
    return K


def load_prediction_rows(path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "scene_id": int(row["scene_id"]),
                    "im_id": int(row["im_id"]),
                    "obj_id": int(row["obj_id"]),
                    "score": float(row["score"]),
                    "R": np.fromstring(row["R"], sep=" ").reshape(3, 3),
                    "t": np.fromstring(row["t"], sep=" ").reshape(3, 1),
                }
            )
    return sorted(rows, key=lambda x: (x["scene_id"], x["im_id"]))


def mesh_bbox_corners(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    x0, y0, z0 = bounds[0]
    x1, y1, z1 = bounds[1]
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def project_points(points_obj, R, t, K):
    points_cam = (R @ points_obj.T + t).T
    valid = points_cam[:, 2] > 1e-6
    projected = (K @ points_cam.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected, valid


def draw_line_if_valid(image, points, valid, i, j, color, thickness=2):
    if valid[i] and valid[j]:
        p0 = tuple(np.round(points[i]).astype(int))
        p1 = tuple(np.round(points[j]).astype(int))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def draw_bbox(image, points, valid):
    # Bottom face, top face, and vertical edges of the CAD axis-aligned bbox.
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        draw_line_if_valid(image, points, valid, i, j, (0, 255, 0), thickness=2)


def draw_axes(image, R, t, K, axis_length):
    axes = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ]
    )
    points, valid = project_points(axes, R, t, K)
    colors = [(255, 0, 0), (0, 255, 0), (0, 80, 255)]
    for idx, color in zip([1, 2, 3], colors):
        draw_line_if_valid(image, points, valid, 0, idx, color, thickness=3)
    if valid[0]:
        center = tuple(np.round(points[0]).astype(int))
        cv2.circle(image, center, 4, (0, 255, 255), -1, cv2.LINE_AA)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    im_id_to_frame_id = load_im_id_to_frame_id(args)
    predictions = load_prediction_rows(args.predictions)
    corners = mesh_bbox_corners(args.mesh)
    axis_length = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)) * 0.2)

    selected = predictions[:: args.every]
    if args.max_images is not None:
        selected = selected[: args.max_images]

    for pred in selected:
        if pred["im_id"] not in im_id_to_frame_id:
            raise ValueError(
                f"No source frame id found for prediction im_id={pred['im_id']}. "
                "Pass --frame-ids or --frame-map matching the prepared dataset."
            )
        stem = im_id_to_frame_id[pred["im_id"]]
        rgb_path = args.source_root / "images" / f"{stem}.png"
        K_path = args.source_root / "intrinsics" / f"{stem}.npy"

        rgb_full = Image.open(rgb_path)
        image = crop_middle_rgb(rgb_path)
        K = maybe_adjust_intrinsics(np.load(K_path), rgb_full.size[0], image.shape[1])

        overlay = image.copy()
        bbox_2d, valid = project_points(corners, pred["R"], pred["t"], K)
        draw_bbox(overlay, bbox_2d, valid)
        draw_axes(overlay, pred["R"], pred["t"], K, axis_length)

        label = f"im_id={pred['im_id']} frame={stem} score={pred['score']:.3f}"
        cv2.putText(
            overlay,
            label,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Save side-by-side: original crop on the left, overlay on the right.
        side_by_side = np.concatenate([image, overlay], axis=1)
        save_path = args.output_dir / f"{pred['im_id']:06d}_{stem}_overlay.png"
        Image.fromarray(side_by_side).save(save_path)

    print(f"Saved {len(selected)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
