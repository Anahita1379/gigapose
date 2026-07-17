"""Visualize extrinsics-corrected EPnP and GigaPose poses without frame mixing.

This is the extrinsics-aware counterpart of
``fine_tuning.visualize_epnp_gigapose_comparison``. The original visualizer is
left unchanged.

The candidate selector compares poses after an optional frame alignment:

    right alignment: T_gigapose_aligned = T_gigapose_raw @ X

For a right-side alignment, directly rendering the original GigaPose CAD with
``T_gigapose_aligned`` is incorrect because that pose expects points in the
EPnP object frame. This visualizer instead renders both boxes in the native
GigaPose CAD frame:

    GigaPose: T_gigapose_raw
    EPnP:     T_epnp_corrected @ inv(X)

The transform is recovered exactly from each CSV row's raw and aligned
GigaPose poses, so the normal command does not need another transform file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw

from fine_tuning import visualize_epnp_gigapose_comparison as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="selected_samples.csv from the extrinsics-aware candidate selector.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "translation_error", "rotation_error", "score"),
        default="image",
    )
    parser.add_argument("--draw-mask-bbox", action="store_true")
    parser.add_argument(
        "--bbox-match-mode",
        choices=("instance_id", "nearest_projected_center"),
        default="nearest_projected_center",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-translation-error-mm", type=float, default=None)
    parser.add_argument("--max-rotation-error-deg", type=float, default=None)
    parser.add_argument(
        "--frame-transform-side",
        choices=("right", "left"),
        default="right",
        help=(
            "Alignment convention used by the selector. Your current candidate "
            "runs use 'right'. The transform is recovered from each CSV row."
        ),
    )
    return parser.parse_args()


def aligned_pose_for_row(row: dict[str, str]) -> np.ndarray:
    value = row.get("T_gigapose_aligned_epnp_obj")
    if not value:
        raise ValueError(
            "Candidate row is missing T_gigapose_aligned_epnp_obj."
        )
    return base.text_to_matrix(value)


def native_cad_poses(
    row: dict[str, str],
    frame_transform_side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return raw GigaPose, corrected EPnP, alignment X, and consistency error."""

    raw_value = row.get("T_gigapose_cam_obj")
    epnp_value = row.get("T_epnp_obj")
    if not raw_value or not epnp_value:
        raise ValueError(
            "Candidate row must contain T_gigapose_cam_obj and T_epnp_obj."
        )

    T_raw = base.text_to_matrix(raw_value)
    T_aligned = aligned_pose_for_row(row)
    T_epnp = base.text_to_matrix(epnp_value)

    if frame_transform_side == "right":
        X = np.linalg.inv(T_raw) @ T_aligned
        T_epnp_native = T_epnp @ np.linalg.inv(X)
        reconstructed = T_raw @ X
        T_gigapose_native = T_raw
    elif frame_transform_side == "left":
        X = T_aligned @ np.linalg.inv(T_raw)
        # A left transform changes the camera-frame convention rather than the
        # object/CAD convention. Render the aligned prediction and EPnP pose.
        T_epnp_native = T_epnp
        reconstructed = X @ T_raw
        T_gigapose_native = T_aligned
    else:
        raise ValueError(
            f"Unknown frame-transform side: {frame_transform_side!r}"
        )

    consistency = float(np.max(np.abs(reconstructed - T_aligned)))
    return T_gigapose_native, T_epnp_native, X, consistency


def keep_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not base.keep_row(row, args):
        return False
    return bool(row.get("T_gigapose_cam_obj"))


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be positive")
    if args.max_images is not None and args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = base.read_ply_vertices(mesh_path)
    if vertices_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    corners_mm = base.bbox_corners_from_vertices(vertices_mm)

    frame_map = base.load_frame_map(args.dataset_dir)
    rows = base.read_rows(args.candidate_csv)
    input_row_count = len(rows)
    rows = [row for row in rows if keep_row(row, args)]
    rows.sort(key=lambda row: base.row_sort_key(row, args.sort_by))
    rows = rows[:: args.every]
    if args.max_images is not None:
        rows = rows[: args.max_images]

    index_rows: list[dict[str, Any]] = []
    for row in rows:
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        frame_info = frame_map.get((scene_id, im_id), {})
        image = base.load_image(
            args.dataset_dir, args.split, scene_id, im_id, frame_info
        )
        K = base.load_camera_K(args.dataset_dir, args.split, scene_id, im_id)
        T_gigapose, T_epnp, X, consistency = native_cad_poses(
            row, args.frame_transform_side
        )
        if consistency > 1e-5:
            raise ValueError(
                f"Frame-transform reconstruction failed for "
                f"{scene_id:06d}_{im_id:06d}: max error={consistency:.6g}"
            )

        draw = ImageDraw.Draw(image)
        base.draw_projected_box(
            draw,
            corners_mm,
            T_epnp,
            K,
            base.EPNP_COLOR,
            "EPnPv2 corrected (native GigaPose CAD frame)",
            0,
        )
        base.draw_projected_box(
            draw,
            corners_mm,
            T_gigapose,
            K,
            base.GIGAPOSE_COLOR,
            (
                f"GigaPose raw CAD: aligned t="
                f"{float(row['translation_error_mm']):.0f}mm "
                f"R={float(row['rotation_error_deg']):.1f}deg "
                f"score={float(row['score']):.3f}"
            ),
            26,
        )

        bbox_idx = None
        bbox_dist = None
        if args.draw_mask_bbox:
            instances = frame_info.get("instances") or []
            bbox, bbox_idx, bbox_dist = base.choose_detection_bbox(
                instances,
                row,
                T_gigapose,
                T_epnp,
                K,
                args.bbox_match_mode,
            )
            if bbox:
                label = "GSAM/detection bbox"
                if bbox_idx is not None:
                    label += f" #{bbox_idx}"
                if bbox_dist is not None:
                    label += f" d={bbox_dist:.0f}px"
                base.draw_bbox(draw, bbox, label)

        output_path = args.output_dir / (
            f"{scene_id:06d}_{im_id:06d}_epnp"
            f"{int(row.get('epnp_record_index') or 0):02d}.jpg"
        )
        image.save(output_path, quality=94)
        index_rows.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "match_key": row.get("match_key", ""),
                "score": row.get("score", ""),
                "aligned_translation_error_mm": row.get(
                    "translation_error_mm", ""
                ),
                "aligned_rotation_error_deg": row.get(
                    "rotation_error_deg", ""
                ),
                "frame_transform_side": args.frame_transform_side,
                "frame_transform_translation_norm_mm": float(
                    np.linalg.norm(X[:3, 3])
                ),
                "frame_transform_reconstruction_max_abs": consistency,
                "epnp_label_path": row.get("epnp_label_path", ""),
                "bbox_match_mode": (
                    args.bbox_match_mode if args.draw_mask_bbox else ""
                ),
                "bbox_match_index": "" if bbox_idx is None else bbox_idx,
                "bbox_match_distance_px": (
                    "" if bbox_dist is None else f"{bbox_dist:.6g}"
                ),
                "output": str(output_path),
            }
        )

    index_path = args.output_dir / "visual_overlay_index.csv"
    fieldnames = [
        "scene_id",
        "im_id",
        "match_key",
        "score",
        "aligned_translation_error_mm",
        "aligned_rotation_error_deg",
        "frame_transform_side",
        "frame_transform_translation_norm_mm",
        "frame_transform_reconstruction_max_abs",
        "epnp_label_path",
        "bbox_match_mode",
        "bbox_match_index",
        "bbox_match_distance_px",
        "output",
    ]
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(
        f"Wrote {len(index_rows)} native-CAD EPnPv2/GigaPose overlays to "
        f"{args.output_dir} ({len(rows)} visualized after filters from "
        f"{input_row_count} input rows)"
    )


if __name__ == "__main__":
    main()
