import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import trimesh
import webdataset as wds
import yaml
from bop_toolkit_lib import pycoco_utils
from PIL import Image

"""
Build a GigaPose-compatible test dataset from ARCL camera metadata YAML files.

This loader is intentionally separate from the existing racecar loaders. It
reads files like:

  <sequence>/stereo_left/metadata/sample_3967128010822_0.yaml

and writes the WebDataset/BOP-style layout consumed by this repository:

  <output-root>/<dataset-name>/models/obj_000001.ply
  <output-root>/<dataset-name>/models/models_info.json
  <output-root>/<dataset-name>/test/shard-000000.tar
  <output-root>/<dataset-name>/test/key_to_shard.json
  <output-root>/<dataset-name>/test_targets_bop19.json
  <output-root>/cnos-fastsam/cnos-fastsam_racecar-test.json

By default the dataset name is "racecar" so the existing
src/utils/dataset.py mapping can find the generated detection file without any
code changes.
"""


DEFAULT_METADATA_DIR = Path(
    "/media/hdd2/ARCL_multicar_bags/camera_dataset/"
    "2026-05-18-v1-v4/stereo_left/metadata"
)
DEFAULT_CAD_PATH = Path("/home/anahita/CAD_car/racecar0_highres.ply")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an ARCL metadata sequence for GigaPose."
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD_PATH)
    parser.add_argument(
        "--output-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument("--dataset-name", default="racecar")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional image directory. If omitted, paths are read from YAML or guessed.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Optional depth directory. Missing depth is replaced by a zero image.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional mask directory. Missing masks fall back to bbox/full image.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Optional fixed bbox as x,y,w,h, used when YAML has no bbox or mask.",
    )
    parser.add_argument(
        "--detection-file-name",
        default="cnos-fastsam_racecar-test.json",
        help="Keep this default when --dataset-name racecar is used by test.py.",
    )
    parser.add_argument(
        "--fail-on-missing-image",
        action="store_true",
        help="Raise an error instead of skipping metadata with unresolved RGB paths.",
    )
    return parser.parse_args()


def load_yaml(path):
    data = yaml.safe_load(path.read_text())
    return data if data is not None else {}


def iter_nested_items(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from iter_nested_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_nested_items(value)


def value_at_path(data, dotted_path):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def matrix_from_value(value):
    if value is None:
        return None
    if isinstance(value, dict):
        K = intrinsics_from_dict(value)
        if K is not None:
            return K
        for key in ("K", "k", "cam_K", "data", "matrix", "camera_matrix"):
            if key in value:
                K = matrix_from_value(value[key])
                if K is not None:
                    return K
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3, 3):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size == 12:
        return flat.reshape(3, 4)[:, :3]
    return None


def get_first_number(data, keys):
    for key in keys:
        if key in data:
            value = as_number(data[key])
            if value is not None:
                return value
    return None


def get_nested_number(data, outer_keys, inner_keys):
    for outer_key in outer_keys:
        value = data.get(outer_key)
        if not isinstance(value, dict):
            continue
        number = get_first_number(value, inner_keys)
        if number is not None:
            return number
    return None


def intrinsics_from_dict(data):
    fx = get_first_number(
        data,
        (
            "fx",
            "focal_x",
            "focal_length_x",
            "focal_length_x_px",
            "focal_length_px_x",
        ),
    )
    fy = get_first_number(
        data,
        (
            "fy",
            "focal_y",
            "focal_length_y",
            "focal_length_y_px",
            "focal_length_px_y",
        ),
    )
    cx = get_first_number(
        data,
        (
            "cx",
            "principal_x",
            "principal_point_x",
            "principal_point_x_px",
            "ppx",
        ),
    )
    cy = get_first_number(
        data,
        (
            "cy",
            "principal_y",
            "principal_point_y",
            "principal_point_y_px",
            "ppy",
        ),
    )

    if fx is None:
        fx = get_nested_number(data, ("focal", "focal_length"), ("x", "u", "0"))
    if fy is None:
        fy = get_nested_number(data, ("focal", "focal_length"), ("y", "v", "1"))
    if cx is None:
        cx = get_nested_number(
            data, ("principal", "principal_point", "center"), ("x", "u", "0")
        )
    if cy is None:
        cy = get_nested_number(
            data, ("principal", "principal_point", "center"), ("y", "v", "1")
        )

    if any(v is None for v in (fx, fy, cx, cy)):
        return None
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def extract_intrinsics(metadata):
    direct_paths = (
        "K",
        "cam_K",
        "camera_matrix",
        "intrinsics",
        "camera_intrinsics",
        "camera.intrinsics",
        "camera.K",
        "camera_info.K",
        "camera_info.P",
        "projection_matrix",
    )
    for path in direct_paths:
        K = matrix_from_value(value_at_path(metadata, path))
        if K is not None:
            return K

    for _, value in iter_nested_items(metadata):
        if isinstance(value, dict):
            K = intrinsics_from_dict(value)
            if K is not None:
                return K

    for key, value in iter_nested_items(metadata):
        if str(key).lower() in {"k", "cam_k", "camera_matrix", "intrinsics"}:
            K = matrix_from_value(value)
            if K is not None:
                return K

    raise ValueError("Could not find camera intrinsics in metadata YAML.")


def resolve_path(raw_path, metadata_path):
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    sequence_root = metadata_path.parent.parent
    candidates = [
        metadata_path.parent / path,
        sequence_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return sequence_root / path


def extract_path_from_metadata(metadata, metadata_path, wanted):
    wanted_keys = {
        "image": {
            "image",
            "image_path",
            "image_file",
            "rgb",
            "rgb_path",
            "rgb_file",
            "color",
            "color_path",
            "filename",
            "file_name",
        },
        "depth": {"depth", "depth_path", "depth_file"},
        "mask": {"mask", "mask_path", "mask_file", "segmentation_path", "roi_path"},
    }[wanted]
    blocked_tokens = {
        "image": ("depth", "mask", "segmentation", "metadata", "calib"),
        "depth": ("mask", "segmentation", "metadata", "calib"),
        "mask": ("depth", "metadata", "calib"),
    }[wanted]

    for key, value in iter_nested_items(metadata):
        key_l = str(key).lower()
        if key_l not in wanted_keys or not isinstance(value, str):
            continue
        value_l = value.lower()
        if any(token in value_l for token in blocked_tokens):
            continue
        path = resolve_path(value, metadata_path)
        if path is not None:
            return path
    return None


def candidate_dirs(base_dir, wanted):
    names_by_kind = {
        "image": ("rgb", "image", "images", "color", "left", "data"),
        "depth": ("depth", "depths"),
        "mask": ("mask", "masks", "segmentation", "segmentations"),
    }
    dirs = [base_dir]
    dirs.extend(base_dir / name for name in names_by_kind[wanted])
    dirs.extend(base_dir.parent / name for name in names_by_kind[wanted])
    return dirs


def guess_sidecar_path(metadata_path, explicit_dir, wanted):
    stem = metadata_path.stem
    base_dir = explicit_dir if explicit_dir is not None else metadata_path.parent.parent
    extensions = {
        "image": (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        "depth": (".png", ".tif", ".tiff", ".npy"),
        "mask": (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"),
    }[wanted]
    for directory in candidate_dirs(base_dir, wanted):
        for ext in extensions:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    # Some capture stacks store metadata as sample_<timestamp>_0.yaml and RGB as
    # <timestamp>.png. Try any long numeric token before giving up.
    tokens = re.findall(r"\d{6,}", stem)
    for token in tokens:
        for directory in candidate_dirs(base_dir, wanted):
            for ext in extensions:
                candidate = directory / f"{token}{ext}"
                if candidate.exists():
                    return candidate
    return None


def find_data_path(metadata, metadata_path, explicit_dir, wanted):
    path = extract_path_from_metadata(metadata, metadata_path, wanted)
    if path is not None and path.exists():
        return path
    return guess_sidecar_path(metadata_path, explicit_dir, wanted)


def read_image(path):
    return Image.open(path).convert("RGB")


def read_depth(path, image_size):
    if path is None:
        width, height = image_size
        return np.zeros((height, width), dtype=np.uint16)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.uint16)
    if depth.shape != (image_size[1], image_size[0]):
        depth = np.array(Image.fromarray(depth).resize(image_size, Image.NEAREST))
    return depth


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        height, width = mask.shape
        return [0, 0, width, height]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def mask_from_bbox(bbox, image_size):
    width, height = image_size
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(0, min(width, x + max(1, w)))
    y1 = max(0, min(height, y + max(1, h)))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def clip_bbox(bbox, image_size):
    width, height = image_size
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(0, min(width, x + max(1, w)))
    y1 = max(0, min(height, y + max(1, h)))
    return x0, y0, x1, y1


def bbox_from_xywh_dict(data):
    values = [as_number(data.get(name)) for name in ("x", "y", "width", "height")]
    if any(value is None for value in values):
        return None
    return values


def extract_bbox(metadata):
    for key in ("yolo_bbox", "crop_transform"):
        value = metadata.get(key)
        if isinstance(value, dict):
            bbox = bbox_from_xywh_dict(value)
            if bbox is not None:
                return bbox, key

    bbox_keys = {"bbox", "box", "bbox_obj", "bbox_visib", "bbox_est", "bounding_box"}
    for key, value in iter_nested_items(metadata):
        if str(key).lower() not in bbox_keys:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size >= 4:
            return arr[:4].tolist(), str(key)
    return None, None


def read_mask(path, image_size, placement_bbox=None):
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.array(Image.open(path).convert("L"))
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape == (image_size[1], image_size[0]):
        return (mask > 0).astype(np.uint8)

    if placement_bbox is not None:
        x0, y0, x1, y1 = clip_bbox(placement_bbox, image_size)
        target_w = max(1, x1 - x0)
        target_h = max(1, y1 - y0)
        resized = Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST)
        full_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = (np.array(resized) > 0).astype(np.uint8)
        return full_mask

    if mask.shape != (image_size[1], image_size[0]):
        mask = np.array(Image.fromarray(mask).resize(image_size, Image.NEAREST))
    return (mask > 0).astype(np.uint8)


def parse_fixed_bbox(text):
    if text is None:
        return None
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be exactly x,y,w,h")
    return parts


def make_detection_mask(metadata, mask_path, image_size, fixed_bbox):
    bbox, bbox_source = extract_bbox(metadata)
    if bbox is None:
        bbox = fixed_bbox
        bbox_source = "fixed_bbox"
    if bbox is not None:
        bbox_mask = mask_from_bbox(bbox, image_size)
        roi_mask = read_mask(mask_path, image_size, placement_bbox=bbox)
        if roi_mask is not None:
            combined = np.logical_and(bbox_mask > 0, roi_mask > 0).astype(np.uint8)
            if combined.sum() > 0:
                return combined, f"{bbox_source}+roi_path"
        return bbox_mask, bbox_source

    mask = read_mask(mask_path, image_size)
    if mask is not None:
        return mask, "roi_path"

    width, height = image_size
    return np.ones((height, width), dtype=np.uint8), "full_image_fallback"


def mesh_info(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    return {
        "diameter": float(np.linalg.norm(size)),
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def write_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_key_to_shard_map(wds_dir):
    key_to_shard = {}
    for shard_path in sorted(wds_dir.glob("shard-*.tar")):
        shard_id = int(re.findall(r"\d+", shard_path.name)[0])
        with tarfile.open(shard_path, mode="r") as tar:
            keys = {name.split(".")[0] for name in tar.getnames()}
        for key in keys:
            key_to_shard[key] = shard_id
    return key_to_shard


def write_models(cad_path, model_dir, object_id):
    model_dir.mkdir(parents=True, exist_ok=True)
    cad_ply = model_dir / f"obj_{object_id:06d}.ply"
    cad_obj = model_dir / f"obj_{object_id:06d}.obj"
    cad_ply.write_bytes(cad_path.read_bytes())
    mesh = trimesh.load(cad_ply, force="mesh")
    mesh.export(cad_obj)
    (model_dir / "models_info.json").write_text(
        json.dumps({str(object_id): mesh_info(cad_ply)}, indent=2)
    )


def main():
    args = parse_args()
    fixed_bbox = parse_fixed_bbox(args.bbox)
    metadata_paths = sorted(args.metadata_dir.glob("*.yaml"))
    if args.max_images is not None:
        metadata_paths = metadata_paths[: args.max_images]
    if not metadata_paths:
        raise FileNotFoundError(f"No YAML files found in {args.metadata_dir}")

    dataset_dir = args.output_root / args.dataset_name
    test_dir = dataset_dir / "test"
    cnos_dir = args.output_root / "cnos-fastsam"
    test_dir.mkdir(parents=True, exist_ok=True)
    cnos_dir.mkdir(parents=True, exist_ok=True)
    write_models(args.cad_path, dataset_dir / "models", args.object_id)

    targets = []
    detections = []
    frame_map = []
    skipped = []

    shard_writer = wds.ShardWriter(
        pattern=str(test_dir / "shard-%06d.tar"),
        maxcount=1000,
        encoder=False,
    )

    try:
        im_id = 0
        for metadata_path in metadata_paths:
            metadata = load_yaml(metadata_path)
            image_path = find_data_path(
                metadata, metadata_path, args.image_dir, "image"
            )
            if image_path is None or not image_path.exists():
                message = f"Could not resolve RGB image for {metadata_path}"
                if args.fail_on_missing_image:
                    raise FileNotFoundError(message)
                skipped.append(message)
                continue

            depth_path = find_data_path(metadata, metadata_path, args.depth_dir, "depth")
            mask_path = find_data_path(metadata, metadata_path, args.mask_dir, "mask")

            rgb = read_image(image_path)
            depth = read_depth(depth_path, rgb.size)
            K = extract_intrinsics(metadata)
            mask, mask_source = make_detection_mask(
                metadata, mask_path, rgb.size, fixed_bbox
            )
            bbox = bbox_from_mask(mask)
            key = f"{args.scene_id:06d}_{im_id:06d}"

            depth_image = Image.fromarray(depth.astype(np.uint16))
            camera = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0}
            shard_writer.write(
                {
                    "__key__": key,
                    "rgb.png": write_png_bytes(rgb),
                    "depth.png": write_png_bytes(depth_image),
                    "camera.json": json.dumps(camera).encode(),
                }
            )

            targets.append(
                {
                    "scene_id": args.scene_id,
                    "im_id": im_id,
                    "obj_id": args.object_id,
                    "inst_count": 1,
                }
            )
            detections.append(
                {
                    "scene_id": args.scene_id,
                    "image_id": im_id,
                    "category_id": args.object_id,
                    "score": 1.0,
                    "bbox": bbox,
                    "segmentation": pycoco_utils.binary_mask_to_rle(mask),
                    "time": 0.0,
                }
            )
            frame_map.append(
                {
                    "scene_id": args.scene_id,
                    "im_id": im_id,
                    "metadata_path": str(metadata_path),
                    "image_path": str(image_path),
                    "depth_path": str(depth_path) if depth_path else None,
                    "mask_path": str(mask_path) if mask_path else None,
                    "mask_source": mask_source,
                    "bbox": bbox,
                }
            )
            im_id += 1
    finally:
        shard_writer.close()

    if not targets:
        raise RuntimeError("No frames were written. Check image paths/directories.")

    (test_dir / "key_to_shard.json").write_text(
        json.dumps(make_key_to_shard_map(test_dir), indent=2)
    )
    (dataset_dir / "test_targets_bop19.json").write_text(
        json.dumps(targets, indent=2)
    )
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    (cnos_dir / args.detection_file_name).write_text(json.dumps(detections, indent=2))

    print(f"Prepared {len(targets)} frames in {dataset_dir}")
    print(f"CAD: {dataset_dir / 'models' / f'obj_{args.object_id:06d}.ply'}")
    print(f"WebDataset: {test_dir}")
    print(f"Targets: {dataset_dir / 'test_targets_bop19.json'}")
    print(f"Detections: {cnos_dir / args.detection_file_name}")
    if skipped:
        print(f"Skipped {len(skipped)} metadata files without resolvable images.")


if __name__ == "__main__":
    main()
