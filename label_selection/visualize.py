"""Visualize native, known-center-converted, and EPnP poses.

Colors:

* blue: GigaPose native/raw-CAD pose with the raw CAD;
* magenta: the same GigaPose pose converted by the known CAD center, rendered
  with the centered CAD (it should overlap blue exactly);
* green: EPnP centered pose (original or extrinsics-corrected target);
* yellow: detector/GSAM bounding box when available.

Nothing in this visualizer estimates a GigaPose-to-EPnP alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw

from fine_tuning import visualize_epnp_gigapose_comparison as image_io
from fine_tuning.visualize_epnp_gigapose_comparison_extrinsics import (
    load_metadata_camera,
    project_points,
)
from label_selection import common


RAW_COLOR = (40, 130, 255)
KNOWN_CENTER_COLOR = (255, 80, 230)
EPNP_COLOR = (20, 235, 90)
BBOX_COLOR = (255, 205, 0)
EDGES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize absolute GigaPose/EPnP comparisons."
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--draw-mask-bbox", action="store_true")
    parser.add_argument(
        "--projection-model",
        choices=("metadata", "pinhole"),
        default="metadata",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _project(
    corners: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    model: str,
    projection_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    return project_points(corners, T, K, D, model, projection_model)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    corners: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    model: str,
    projection_model: str,
    color: tuple[int, int, int],
    width: int,
) -> None:
    uv, valid = _project(corners, T, K, D, model, projection_model)
    for first, second in EDGES:
        if bool(valid[first]) and bool(valid[second]):
            draw.line(
                [tuple(uv[first].round()), tuple(uv[second].round())],
                fill=color,
                width=width,
            )


def _metadata_camera_or_shard(
    row: dict[str, str],
    dataset_dir: Path,
    split: str,
    scene_id: int,
    im_id: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    metadata_value = row.get("sample_metadata_path", "")
    metadata_path = Path(metadata_value) if metadata_value else None
    if metadata_path is not None and metadata_path.is_file():
        return load_metadata_camera(metadata_path)
    K = image_io.load_camera_K(dataset_dir, split, scene_id, im_id)
    return K, np.empty((0,), dtype=float), "pinhole"


def _project_center(
    T: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    model: str,
    projection_model: str,
) -> np.ndarray | None:
    uv, valid = _project(
        np.zeros((1, 3), dtype=float),
        T,
        K,
        D,
        model,
        projection_model,
    )
    return uv[0] if bool(valid[0]) else None


def _choose_bbox(
    instances: list[dict[str, Any]],
    T_target: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    model: str,
    projection_model: str,
) -> tuple[list[float] | None, int | None]:
    center = _project_center(T_target, K, D, model, projection_model)
    if center is None:
        return None, None
    candidates: list[tuple[float, int, list[float]]] = []
    for index, instance in enumerate(instances):
        bbox = instance.get("bbox")
        if not bbox:
            continue
        box = list(map(float, bbox))
        box_center = np.asarray(
            [box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=float
        )
        candidates.append((float(np.linalg.norm(center - box_center)), index, box))
    if not candidates:
        return None, None
    _, index, box = min(candidates)
    return box, index


def _draw_legend(draw: ImageDraw.ImageDraw, row: dict[str, str]) -> None:
    lines = (
        ("GigaPose raw CAD/native pose", RAW_COLOR),
        ("GigaPose known-center conversion (must overlap raw)", KNOWN_CENTER_COLOR),
        ("EPnP centered target", EPNP_COLOR),
        (
            "absolute error: "
            f"{float(row['absolute_translation_error_mm']):.0f} mm, "
            f"{float(row['absolute_rotation_error_deg']):.1f} deg",
            (255, 255, 255),
        ),
    )
    y = 8
    for text, color in lines:
        draw.rectangle((8, y, 760, y + 20), fill=(0, 0, 0))
        draw.text((12, y + 3), text, fill=color)
        y += 22


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace overlays"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_raw_mm = image_io.read_ply_vertices(mesh_path)
    if vertices_raw_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    raw_corners = image_io.bbox_corners_from_vertices(vertices_raw_mm)

    frame_map = image_io.load_frame_map(args.dataset_dir)
    rows = common.read_csv(args.candidate_csv)
    if rows:
        invalid_modes = sorted(
            {
                row.get("comparison_mode", "")
                for row in rows
                if row.get("comparison_mode")
                not in (common.COMPARISON_MODE, common.POST_CALIBRATION_MODE)
            }
        )
        if invalid_modes:
            raise ValueError(
                "This visualizer accepts only the new absolute pipeline; found "
                f"comparison modes {invalid_modes}"
            )
    rows.sort(
        key=lambda row: (
            int(row["scene_id"]),
            int(row["im_id"]),
            int(row.get("epnp_record_index") or 0),
        )
    )
    rows = rows[:: max(args.every, 1)]
    if args.max_images is not None:
        rows = rows[: args.max_images]

    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        frame_info = frame_map.get((scene_id, im_id), {})
        image = image_io.load_image(
            args.dataset_dir, args.split, scene_id, im_id, frame_info
        )
        K, D, model = _metadata_camera_or_shard(
            row, args.dataset_dir, args.split, scene_id, im_id
        )
        C = common.text_to_matrix(row["T_object_raw_object_centered"])
        center_mm = C[:3, 3]
        centered_corners = raw_corners - center_mm.reshape(1, 3)
        T_raw = common.text_to_matrix(row["T_gigapose_native_raw_cad"])
        T_centered = common.text_to_matrix(
            row["T_gigapose_known_center_aligned"]
        )
        T_epnp = common.text_to_matrix(
            row["T_epnp_camera_object_centered_target"]
        )

        uv_raw, valid_raw = _project(
            raw_corners, T_raw, K, D, model, args.projection_model
        )
        uv_centered, valid_centered = _project(
            centered_corners,
            T_centered,
            K,
            D,
            model,
            args.projection_model,
        )
        valid = valid_raw & valid_centered
        consistency_px = (
            float(np.max(np.linalg.norm(uv_raw[valid] - uv_centered[valid], axis=1)))
            if np.any(valid)
            else float("nan")
        )
        if np.isfinite(consistency_px) and consistency_px > 1e-5:
            raise RuntimeError(
                "Known center conversion changed the physical GigaPose projection "
                f"by {consistency_px:.6g}px for {scene_id}_{im_id}"
            )

        draw = ImageDraw.Draw(image)
        _draw_box(
            draw,
            raw_corners,
            T_raw,
            K,
            D,
            model,
            args.projection_model,
            RAW_COLOR,
            5,
        )
        _draw_box(
            draw,
            centered_corners,
            T_centered,
            K,
            D,
            model,
            args.projection_model,
            KNOWN_CENTER_COLOR,
            2,
        )
        _draw_box(
            draw,
            centered_corners,
            T_epnp,
            K,
            D,
            model,
            args.projection_model,
            EPNP_COLOR,
            3,
        )

        bbox_index = None
        if args.draw_mask_bbox:
            bbox, bbox_index = _choose_bbox(
                frame_info.get("instances") or [],
                T_epnp,
                K,
                D,
                model,
                args.projection_model,
            )
            if bbox is not None:
                x, y, width, height = bbox
                draw.rectangle(
                    (x, y, x + width, y + height), outline=BBOX_COLOR, width=2
                )
                draw.text(
                    (x, max(0, y - 14)),
                    f"detection bbox #{bbox_index}",
                    fill=BBOX_COLOR,
                )
        _draw_legend(draw, row)

        output_path = args.output_dir / (
            f"{scene_id:06d}_{im_id:06d}_"
            f"epnp{int(row.get('epnp_record_index') or 0):02d}.jpg"
        )
        image.save(output_path, quality=94)
        diagnostics.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "epnp_label_path": row.get("epnp_label_path", ""),
                "absolute_translation_error_mm": row.get(
                    "absolute_translation_error_mm", ""
                ),
                "absolute_rotation_error_deg": row.get(
                    "absolute_rotation_error_deg", ""
                ),
                "raw_vs_known_center_projection_max_error_px": consistency_px,
                "bbox_index": "" if bbox_index is None else bbox_index,
                "overlay": str(output_path),
            }
        )

    common.write_csv(args.output_dir / "visualization_index.csv", diagnostics)
    finite_consistency = [
        float(row["raw_vs_known_center_projection_max_error_px"])
        for row in diagnostics
        if np.isfinite(float(row["raw_vs_known_center_projection_max_error_px"]))
    ]
    report = {
        "format": "absolute_pose_visualization_v1",
        "candidate_csv": str(args.candidate_csv),
        "dataset_dir": str(args.dataset_dir),
        "mesh": str(mesh_path),
        "overlays_written": len(diagnostics),
        "empirical_prediction_label_alignment_used": False,
        "raw_vs_known_center_projection_max_error_px": (
            max(finite_consistency) if finite_consistency else None
        ),
        "colors": {
            "blue": "GigaPose native/raw-CAD pose with raw CAD",
            "magenta": "same GigaPose pose after known center conversion",
            "green": "EPnP centered target",
            "yellow": "detector/GSAM bbox",
        },
    }
    common.save_json(args.output_dir / "visualization_report.json", report)
    print(
        f"Wrote {len(diagnostics)} absolute-comparison overlays to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
