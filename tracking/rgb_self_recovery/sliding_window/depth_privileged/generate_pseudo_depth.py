"""Render metric object pseudo-depth from Assetto GT transforms and CAD."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from Assetto_data_prep.common import cad_to_camera_pose, validate_model_row
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery.sliding_window import assetto_export


DEFAULT_BINS = ",".join((
    "distance0_20", "distance20_40", "distance40_60",
    "distance60_80", "distance80_100", "distance100_120",
))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--assetto-root", type=Path, required=True)
    value.add_argument("--mesh", type=Path, required=True)
    value.add_argument("--mesh-scale", type=float, default=1.0)
    value.add_argument("--cad-axis-convention", choices=("x-forward-z-up", "csp"), default="x-forward-z-up")
    value.add_argument("--fit-aabb", choices=("none", "uniform", "nonuniform"), default="nonuniform")
    value.add_argument("--distance-bins", default=DEFAULT_BINS)
    value.add_argument("--output-folder", default="depth_info")
    value.add_argument(
        "--output-root",
        type=Path,
        help="Optional separate root for tests; default writes inside each source bin.",
    )
    value.add_argument("--minimum-overlap-pixels", type=int, default=8)
    value.add_argument("--preview-count-per-bin", type=int, default=3)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--overwrite", action="store_true")
    return value


def encode_sparse_depth(depth_m: np.ndarray, valid: np.ndarray, *, center_depth_m: float) -> bytes:
    valid = np.asarray(valid, dtype=bool)
    ys, xs = np.where(valid)
    if not xs.size:
        raise ValueError("Cannot encode empty pseudo-depth")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop_valid = valid[y0:y1, x0:x1]
    crop_depth = np.where(crop_valid, np.asarray(depth_m, np.float32)[y0:y1, x0:x1], 0.0)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        depth_m=crop_depth.astype(np.float32), valid=crop_valid.astype(np.bool_),
        bbox_xyxy=np.asarray([x0, y0, x1, y1], np.int32),
        image_shape=np.asarray(valid.shape, np.int32),
        center_depth_m=np.asarray(center_depth_m, np.float32), unit=np.asarray("m"),
        format=np.asarray("assetto_cad_object_pseudo_depth_v1"),
    )
    return buffer.getvalue()


def decode_packed_mask(path: Path, instance_id: int, shape: tuple[int, int]):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f"Could not decode mask {path}")
    if value.shape[:2] != shape:
        raise ValueError(f"Mask shape {value.shape[:2]} != camera shape {shape}: {path}")
    rgb = value[..., ::-1].astype(np.uint32)
    packed = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
    return packed == int(instance_id)


def output_bin_dir(item, output_root: Path | None) -> Path:
    if output_root is None:
        return item["bin_dir"]
    return output_root / str(item["bin_name"])


def output_path(item, folder: str, output_root: Path | None) -> Path:
    image = Path(str(item["manifest"]["image"]))
    return output_bin_dir(item, output_root) / folder / image.parent.name / f"{image.stem}.npz"


def preview_path(item, folder: str, output_root: Path | None) -> Path:
    image = Path(str(item["manifest"]["image"]))
    return output_bin_dir(item, output_root) / f"{folder}_previews" / image.parent.name / f"{image.stem}.jpg"


def draw_preview(image_path: Path, depth_m: np.ndarray, valid: np.ndarray, output: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    values = depth_m[valid]
    low, high = float(values.min()), float(values.max())
    normalized = np.zeros(depth_m.shape, np.uint8)
    normalized[valid] = np.clip(255 * (values - low) / max(high - low, 1e-6), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    canvas = image.copy()
    canvas[valid] = (0.35 * image[valid] + 0.65 * color[valid]).astype(np.uint8)
    cv2.putText(canvas, f"pseudo-depth {low:.2f}-{high:.2f} m", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parser().parse_args()
    if not args.assetto_root.is_dir():
        raise FileNotFoundError(args.assetto_root)
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)
    if args.minimum_overlap_pixels < 1 or args.preview_count_per_bin < 0:
        raise ValueError("Overlap must be positive and preview count non-negative")
    index = assetto_export.load_index(args.assetto_root)
    index, selected_bins = assetto_export.select_bins(index, args.distance_bins)
    alignment = assetto_export.make_alignment(args, index)
    rows_by_bin = {name: [] for name in selected_bins}
    previews = {name: 0 for name in selected_bins}
    counters = {name: {"selected": 0, "written": 0, "existing": 0, "failed_overlap": 0} for name in selected_bins}
    started, processed = time.perf_counter(), 0
    with tempfile.TemporaryDirectory(prefix="assetto_pseudo_depth_") as directory:
        aligned_mesh = Path(directory) / "aligned_model.ply"
        alignment.mesh.export(aligned_mesh)
        renderer = CADRenderer(aligned_mesh)
        try:
            for item in index:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                processed += 1
                bin_name = str(item["bin_name"])
                counters[bin_name]["selected"] += 1
                path = output_path(item, args.output_folder, args.output_root)
                if path.is_file() and not args.overwrite:
                    counters[bin_name]["existing"] += 1
                    continue
                transform, camera, box = item["transform"], item["camera"], item["box"]
                validate_model_row(alignment, transform)
                pose = cad_to_camera_pose(transform, alignment.local_center)
                height, width = int(camera["image_height"]), int(camera["image_width"])
                K = np.asarray([[float(camera["fx"]), 0, float(camera["cx"])], [0, float(camera["fy"]), float(camera["cy"])], [0, 0, 1]])
                rendered_mask, rendered_depth = renderer.render(pose, K, (height, width))
                mask_path = item["bin_dir"] / str(item["manifest"]["masks"])
                observed = decode_packed_mask(mask_path, int(box["instance_id"]), (height, width))
                rendered_mask = np.asarray(rendered_mask, bool)
                valid = observed & rendered_mask & np.isfinite(rendered_depth) & (rendered_depth > 0)
                overlap, observed_pixels, rendered_pixels = int(valid.sum()), int(observed.sum()), int(rendered_mask.sum())
                union = int((observed | rendered_mask).sum())
                iou, coverage = overlap / max(union, 1), overlap / max(observed_pixels, 1)
                row = {
                    "source_run": item["key"][0], "frame": item["key"][1], "camera_id": item["key"][2],
                    "depth_info": "", "status": "failed_overlap", "center_depth_m": float(pose[2, 3]),
                    "minimum_depth_m": "", "maximum_depth_m": "", "valid_pixels": overlap,
                    "observed_mask_pixels": observed_pixels, "rendered_pixels": rendered_pixels,
                    "render_observed_iou": iou, "observed_coverage": coverage,
                }
                if overlap < args.minimum_overlap_pixels:
                    counters[bin_name]["failed_overlap"] += 1
                    rows_by_bin[bin_name].append(row)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encode_sparse_depth(rendered_depth, valid, center_depth_m=float(pose[2, 3])))
                counters[bin_name]["written"] += 1
                row.update(depth_info=str(path.relative_to(output_bin_dir(item, args.output_root))), status="written", minimum_depth_m=float(rendered_depth[valid].min()), maximum_depth_m=float(rendered_depth[valid].max()))
                rows_by_bin[bin_name].append(row)
                if previews[bin_name] < args.preview_count_per_bin:
                    draw_preview(item["bin_dir"] / str(item["manifest"]["image"]), rendered_depth, valid, preview_path(item, args.output_folder, args.output_root))
                    previews[bin_name] += 1
                if processed % 100 == 0:
                    print(f"processed {processed}/{len(index)} frames", flush=True)
        finally:
            renderer.close()
    fields = ["source_run", "frame", "camera_id", "depth_info", "status", "center_depth_m", "minimum_depth_m", "maximum_depth_m", "valid_pixels", "observed_mask_pixels", "rendered_pixels", "render_observed_iou", "observed_coverage"]
    for bin_name, rows in rows_by_bin.items():
        bin_output = (args.assetto_root if args.output_root is None else args.output_root) / bin_name
        bin_output.mkdir(parents=True, exist_ok=True)
        with (bin_output / f"{args.output_folder}_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "format": "assetto_cad_object_pseudo_depth_report_v1", "assetto_root": str(args.assetto_root),
        "mesh": str(args.mesh), "translation_unit": "m", "storage": "compressed sparse float32 bounding-box NPZ",
        "validity": "intersection of packed instance mask and CAD render", "selected_bins": selected_bins,
        "processed": processed, "bins": counters, "elapsed_s": time.perf_counter() - started,
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    report_root = args.assetto_root if args.output_root is None else args.output_root
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{args.output_folder}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
