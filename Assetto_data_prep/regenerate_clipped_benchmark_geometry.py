#!/usr/bin/env python3
"""Regenerate pose-rendered masks and tight boxes after image clipping.

The source export is never modified. Images are linked/copied into a new
distance-bin root, masks are rendered from T_camera_opponent_visual, and both
transforms.csv and bboxes_3d.csv receive mask-tight inclusive pixel bounds.
Per-run principal-point offsets make post-crop intrinsics explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from Assetto_data_prep.common import (
    build_aligned_mesh,
    cad_to_camera_pose,
    encode_instance_ids,
    validate_model_row,
)
from fine_tuning.ac_geometry import InstanceRenderer


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    value.add_argument("--cad-scale", type=float, default=1.0)
    value.add_argument(
        "--cad-axis-convention", choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    value.add_argument(
        "--fit-aabb", choices=("none", "uniform", "nonuniform"),
        default="nonuniform",
    )
    value.add_argument(
        "--principal-point-offset", action="append", default=[], metavar="RUN=DX,DY",
        help="Add this post-crop pixel offset to cx,cy; repeat for multiple runs.",
    )
    value.add_argument(
        "--image-mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    value.add_argument("--minimum-mask-pixels", type=int, default=1)
    value.add_argument("--max-frames", type=int, help="Diagnostic partial generation")
    value.add_argument("--overwrite", action="store_true")
    return value


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict[str, str]) -> tuple[str, int, str]:
    frame = row.get("frame", row.get("source_frame"))
    if frame is None:
        raise KeyError("Manifest/CSV row has neither frame nor source_frame")
    return row["source_run"], int(frame), row["camera_id"]


def _manifest_image_path(row: dict[str, str]) -> Path:
    """Resolve the relative RGB path in either supported manifest schema."""
    if row.get("image"):
        return Path(row["image"])
    if row.get("output_filename"):
        return Path("images") / row["camera_id"] / row["output_filename"]
    raise KeyError("Manifest row has neither image nor output_filename")


def _manifest_mask_path(row: dict[str, str]) -> Path:
    """Resolve the relative packed-mask path in either manifest schema."""
    if row.get("masks"):
        return Path(row["masks"])
    if row.get("output_filename"):
        return (
            Path("masks") / row["camera_id"]
            / Path(row["output_filename"]).with_suffix(".png").name
        )
    raise KeyError("Manifest row has neither masks nor output_filename")


def _offsets(values: list[str]) -> dict[str, tuple[float, float]]:
    output = {}
    for value in values:
        try:
            run, numbers = value.split("=", 1)
            dx, dy = (float(item) for item in numbers.split(","))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid --principal-point-offset {value!r}; use RUN=DX,DY"
            ) from exc
        if not run or run in output:
            raise ValueError(f"Empty or duplicate correction run: {run!r}")
        output[run] = (dx, dy)
    return output


def _link_or_copy(source: Path, output: Path, mode: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, output)
    elif mode == "symlink":
        output.symlink_to(source.resolve())
    else:
        os.link(source, output)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _visualized(instance_ids: np.ndarray) -> np.ndarray:
    output = np.zeros((*instance_ids.shape, 4), dtype=np.uint8)
    output[..., 3] = 255
    foreground = instance_ids > 0
    output[foreground, :3] = np.asarray([0, 160, 255], dtype=np.uint8)
    return output


def _update_calibrations(bin_source: Path, bin_output: Path, offsets) -> None:
    source = bin_source / "calibration"
    if not source.is_dir():
        return
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Calibration updates require PyYAML") from exc
    target = bin_output / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        run = path.stem
        dx, dy = offsets.get(run, (0.0, 0.0))
        for camera in payload.get("cameras", {}).values():
            intrinsics = camera.get("intrinsics", {})
            if not intrinsics:
                continue
            intrinsics["cx"] = float(intrinsics["cx"]) + dx
            intrinsics["cy"] = float(intrinsics["cy"]) + dy
            intrinsics["K"] = [
                float(intrinsics["fx"]), 0.0, float(intrinsics["cx"]),
                0.0, float(intrinsics["fy"]), float(intrinsics["cy"]),
                0.0, 0.0, 1.0,
            ]
        (target / path.name).write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    args = parser().parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if source == output:
        raise ValueError("Source and output roots must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    offsets = _offsets(args.principal_point_offset)

    bins = sorted(
        path for path in source.iterdir()
        if path.is_dir() and (path / "source_manifest.csv").is_file()
    )
    # One car geometry is present. Use the first transform row to align the CAD.
    reference = None
    for bin_dir in bins:
        transform_path = bin_dir / "csv" / "transforms.csv"
        if transform_path.is_file():
            _, values = _rows(transform_path)
            if values:
                reference = values[0]
                break
    if reference is None:
        raise ValueError("No transform rows found")
    alignment = build_aligned_mesh(
        args.cad_path, reference, args.cad_scale,
        args.cad_axis_convention, args.fit_aabb,
    )
    renderer = InstanceRenderer(alignment.mesh)
    processed = empty = below_minimum = 0
    per_run: dict[str, dict[str, int]] = {}
    try:
        for bin_source in bins:
            bin_output = output / bin_source.name
            manifest_fields, manifests = _rows(bin_source / "source_manifest.csv")
            _write_rows(bin_output / "source_manifest.csv", manifest_fields, manifests)
            if not manifests:
                continue
            camera_fields, cameras = _rows(bin_source / "csv" / "camera_frames.csv")
            transform_fields, transforms = _rows(bin_source / "csv" / "transforms.csv")
            box_fields, boxes = _rows(bin_source / "csv" / "bboxes_3d.csv")
            cameras_by_key = {_key(row): row for row in cameras}
            transforms_by_key = {_key(row): row for row in transforms}
            boxes_by_key = {_key(row): row for row in boxes}
            manifest_by_key = {_key(row): row for row in manifests}
            if not (
                set(cameras_by_key) == set(transforms_by_key)
                == set(boxes_by_key) == set(manifest_by_key)
            ):
                raise ValueError(f"CSV/manifest keys differ in {bin_source.name}")

            for camera in cameras:
                dx, dy = offsets.get(camera["source_run"], (0.0, 0.0))
                camera["cx"] = f"{float(camera['cx']) + dx:.9f}"
                camera["cy"] = f"{float(camera['cy']) + dy:.9f}"
            for box in boxes:
                dx, dy = offsets.get(box["source_run"], (0.0, 0.0))
                box["cx"] = f"{float(box['cx']) + dx:.9f}"
                box["cy"] = f"{float(box['cy']) + dy:.9f}"

            for key in sorted(manifest_by_key):
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                manifest = manifest_by_key[key]
                camera = cameras_by_key[key]
                transform = transforms_by_key[key]
                box = boxes_by_key[key]
                validate_model_row(alignment, transform)
                image_relative = _manifest_image_path(manifest)
                image_source = bin_source / image_relative
                image_output = bin_output / image_relative
                if not image_source.is_file():
                    raise FileNotFoundError(image_source)
                _link_or_copy(image_source, image_output, args.image_mode)
                with Image.open(image_source) as image:
                    width, height = image.size
                if (width, height) != (
                    int(camera["image_width"]), int(camera["image_height"])
                ):
                    raise ValueError(f"Image/camera size differs for {key}")
                K = np.asarray([
                    [float(camera["fx"]), 0.0, float(camera["cx"])],
                    [0.0, float(camera["fy"]), float(camera["cy"])],
                    [0.0, 0.0, 1.0],
                ])
                pose = cad_to_camera_pose(transform, alignment.local_center)
                segmentation, depth = renderer.render([pose], K, width=width, height=height)
                mask = np.logical_and(segmentation == 1, depth > 0)
                bounds = _bbox(mask)
                run_report = per_run.setdefault(key[0], {"frames": 0, "empty": 0, "below_minimum": 0})
                run_report["frames"] += 1
                if bounds is None:
                    empty += 1
                    run_report["empty"] += 1
                    bounds = (0, 0, -1, -1)
                pixels = int(mask.sum())
                if pixels < args.minimum_mask_pixels:
                    below_minimum += 1
                    run_report["below_minimum"] += 1
                x0, y0, x1, y1 = bounds
                for row in (transform, box):
                    row.update({
                        "xmin": str(x0), "ymin": str(y0),
                        "xmax": str(x1), "ymax": str(y1),
                        "truncated": str(int(
                            bounds is not None and (
                                x0 <= 0 or y0 <= 0 or x1 >= width - 1 or y1 >= height - 1
                            )
                        )),
                    })
                instance_id = int(box["instance_id"])
                ids = np.zeros((height, width), dtype=np.uint32)
                ids[mask] = instance_id
                mask_relative = _manifest_mask_path(manifest)
                mask_output = bin_output / mask_relative
                visual_output = (
                    bin_output / "masks_visualized" / key[2] / mask_relative.name
                )
                mask_output.parent.mkdir(parents=True, exist_ok=True)
                visual_output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(encode_instance_ids(ids)).save(mask_output)
                Image.fromarray(_visualized(ids)).save(visual_output)
                processed += 1
                if processed % 250 == 0:
                    print(f"regenerated {processed} frames", flush=True)

            _write_rows(bin_output / "csv" / "camera_frames.csv", camera_fields, cameras)
            _write_rows(bin_output / "csv" / "transforms.csv", transform_fields, transforms)
            _write_rows(bin_output / "csv" / "bboxes_3d.csv", box_fields, boxes)
            opponent_source = bin_source / "csv" / "opponents.csv"
            if opponent_source.is_file():
                shutil.copy2(opponent_source, bin_output / "csv" / "opponents.csv")
            _update_calibrations(bin_source, bin_output, offsets)
    finally:
        renderer.close()

    report = {
        "format": "assetto_pose_regenerated_clipped_geometry_v1",
        "source_root": str(source),
        "output_root": str(output),
        "cad_path": str(args.cad_path.resolve()),
        "cad_scale": args.cad_scale,
        "cad_axis_convention": args.cad_axis_convention,
        "fit_aabb": args.fit_aabb,
        "image_mode": args.image_mode,
        "principal_point_offsets": {
            run: {"dx_px": dx, "dy_px": dy} for run, (dx, dy) in offsets.items()
        },
        "processed_frames": processed,
        "empty_rendered_masks": empty,
        "below_minimum_mask_pixels": below_minimum,
        "per_run": per_run,
        "mask_encoding": "instance_id = R + 256*G + 65536*B",
        "bbox_convention": "inclusive xmin,ymin,xmax,ymax derived from rendered mask",
        "ground_truth_pose_derived": True,
        "limitations": [
            "Pose rendering does not model occlusion by static track geometry.",
            "These oracle masks are suitable for GT supervision and geometry validation.",
        ],
    }
    (output / "regeneration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
