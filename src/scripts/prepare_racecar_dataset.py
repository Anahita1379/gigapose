import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import trimesh
import webdataset as wds
from bop_toolkit_lib import pycoco_utils
from PIL import Image

"""
Build a minimal GigaPose/BOP-style test dataset from the rosbag extraction.

Source data:
  - images/: triple-wide RGB frames, with left/middle/right camera views side by side.
  - depth/: single-view depth images matching the middle RGB view.
  - intrinsics/: per-frame camera intrinsics for the middle view.
  - CAD_car/: the racecar mesh.

Output data:
  - <output-root>/racecar/models/: BOP-style CAD files and models_info.json.
  - <output-root>/racecar/test/: WebDataset shard consumed by GigaPoseTestSet.
  - <output-root>/racecar/test_targets_bop19.json: one target object per frame.
  - <output-root>/cnos-fastsam/cnos-fastsam_racecar-test.json: placeholder detections.

The placeholder detections are only a bootstrap. Because we do not have real
boxes/segmentations yet, they use either the full image or a rough depth mask.
Pose quality will depend on replacing these with a better detector or masks.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/anahita/Dataset/rosbag_extracted_300m"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("gigaPose_datasets/datasets"),
    )
    parser.add_argument("--dataset-name", default="racecar")
    parser.add_argument("--split-file", default="val_image.txt")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument(
        "--frame-ids",
        default=None,
        help="Comma-separated source frame ids/stems to keep, e.g. 005020,005025.",
    )
    parser.add_argument(
        "--frame-ids-file",
        type=Path,
        default=None,
        help="Text file of source frame ids/stems to keep, one per line.",
    )
    parser.add_argument("--cad-name", default="racecar0_highres.ply")
    parser.add_argument(
        "--mask-mode",
        choices=("full", "depth"),
        default="full",
        help="How to make placeholder detections when no boxes/masks are available.",
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        default=None,
        help=(
            "Optional binary/bright-foreground mask to use for every selected frame. "
            "The mask is resized to the cropped RGB size."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Grayscale threshold used with --mask-file.",
    )
    return parser.parse_args()


def frame_stem_from_line(line):
    """Extract frame ids like 000123 from split-file paths."""
    match = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", line.strip())
    if not match:
        raise ValueError(f"Could not parse frame id from line: {line}")
    return match.group(1)


def load_frame_stems(source_root, split_file, max_images):
    split_path = source_root / split_file
    lines = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    stems = [frame_stem_from_line(line) for line in lines]
    if max_images is not None:
        stems = stems[:max_images]
    return stems


def normalize_frame_id(frame_id):
    digits = "".join(ch for ch in frame_id.strip() if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame id from: {frame_id}")
    return digits.zfill(6)


def load_requested_frame_ids(frame_ids, frame_ids_file):
    requested = []
    if frame_ids:
        requested.extend(part for part in frame_ids.split(",") if part.strip())
    if frame_ids_file is not None:
        requested.extend(
            line.strip()
            for line in frame_ids_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not requested:
        return None
    return [normalize_frame_id(frame_id) for frame_id in requested]


def frame_files_exist(source_root, frame_id):
    return all(
        (
            (source_root / "images" / f"{frame_id}.png").exists(),
            (source_root / "depth" / f"{frame_id}.png").exists(),
            (source_root / "intrinsics" / f"{frame_id}.npy").exists(),
        )
    )


def select_frame_stems(stems, requested_frame_ids, source_root):
    if requested_frame_ids is None:
        return stems
    stem_set = set(stems)
    missing = [
        frame_id
        for frame_id in requested_frame_ids
        if frame_id not in stem_set and not frame_files_exist(source_root, frame_id)
    ]
    if missing:
        raise ValueError(
            "Requested frame ids are not present in the split file or source files: "
            + ", ".join(missing)
        )
    return requested_frame_ids


def crop_middle_rgb(image_path):
    """The rosbag RGB files contain three camera views; GigaPose gets the middle one."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    third = width // 3
    return image.crop((third, 0, 2 * third, height))


def load_depth(depth_path):
    depth = Image.open(depth_path)
    if depth.mode not in ("I;16", "I"):
        depth = depth.convert("I")
    return np.array(depth)


def maybe_adjust_intrinsics(K, full_rgb_width, crop_width):
    K = K.astype(float).copy()
    # Some exports store intrinsics for the triple-wide image; this dataset's
    # files usually already store middle-camera intrinsics. If cx is already
    # inside the 768-wide cropped image, leave K alone.
    if K[0, 2] >= crop_width and full_rgb_width == crop_width * 3:
        K[0, 2] -= crop_width
    return K


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        height, width = mask.shape
        return [0, 0, width, height]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def make_placeholder_mask(depth, mode):
    """Create a detection mask when no real segmentation annotations exist."""
    height, width = depth.shape
    if mode == "depth":
        mask = (depth > 0) & (depth < 65535)
        if mask.sum() > 0:
            return mask.astype(np.uint8)
    return np.ones((height, width), dtype=np.uint8)


def load_mask_file(mask_path, target_size, threshold):
    mask = Image.open(mask_path).convert("L")
    width, height = mask.size
    # If a mask is saved for the original triple-wide RGB frame, use the middle
    # camera region to match crop_middle_rgb.
    if width / height > 4.0:
        third = width // 3
        mask = mask.crop((third, 0, 2 * third, height))
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return (np.array(mask) > threshold).astype(np.uint8)


def mesh_info(mesh_path):
    """Create the subset of BOP models_info.json fields this repo reads."""
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    diameter = float(np.linalg.norm(size))
    return {
        "diameter": diameter,
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def make_key_to_shard_map(wds_dir):
    """Map image keys to shard ids; WebSceneDataset requires this index file."""
    key_to_shard = {}
    for shard_path in wds_dir.glob("shard-*.tar"):
        shard_id = int(re.findall(r"\d+", shard_path.name)[0])
        names = tarfile.TarFile(shard_path).getnames()
        keys = {name.split(".")[0] for name in names}
        for key in keys:
            key_to_shard[key] = shard_id
    return key_to_shard


def main():
    args = parse_args()
    source_root = args.source_root
    dataset_dir = args.output_root / args.dataset_name
    model_dir = dataset_dir / "models"
    test_dir = dataset_dir / "test"
    detections_dir = args.output_root / "cnos-fastsam"

    model_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    # GigaPose expects CAD names like obj_000001.*. Panda3D can render OBJ
    # reliably, so keep the original PLY and export an OBJ copy for templates.
    cad_src = source_root / "CAD_car" / args.cad_name
    cad_ply = model_dir / "obj_000001.ply"
    cad_obj = model_dir / "obj_000001.obj"
    cad_ply.write_bytes(cad_src.read_bytes())
    mesh = trimesh.load(cad_ply, force="mesh")
    mesh.export(cad_obj)
    (model_dir / "models_info.json").write_text(
        json.dumps({"1": mesh_info(cad_ply)}, indent=2)
    )

    requested_frame_ids = load_requested_frame_ids(args.frame_ids, args.frame_ids_file)
    stems = load_frame_stems(
        source_root,
        args.split_file,
        None if requested_frame_ids is not None else args.max_images,
    )
    stems = select_frame_stems(stems, requested_frame_ids, source_root)
    shard_writer = wds.ShardWriter(
        pattern=str(test_dir / "shard-%06d.tar"),
        maxcount=1000,
        encoder=False,
    )

    targets = []
    detections = []
    frame_map = []
    for im_id, stem in enumerate(stems):
        rgb_path = source_root / "images" / f"{stem}.png"
        depth_path = source_root / "depth" / f"{stem}.png"
        K_path = source_root / "intrinsics" / f"{stem}.npy"

        rgb_full = Image.open(rgb_path)
        rgb = crop_middle_rgb(rgb_path)
        depth = load_depth(depth_path)
        if depth.shape != (rgb.height, rgb.width):
            depth = np.array(Image.fromarray(depth).resize(rgb.size, Image.NEAREST))

        # Store camera data in the imagewise/WebDataset format expected by
        # src.custom_megapose.web_scene_dataset.load_scene_ds_obs.
        K = maybe_adjust_intrinsics(np.load(K_path), rgb_full.size[0], rgb.width)
        camera = {
            "cam_K": K.reshape(-1).tolist(),
            "depth_scale": 1.0,
        }

        if args.mask_file is not None:
            mask = load_mask_file(args.mask_file, rgb.size, args.mask_threshold)
        else:
            mask = make_placeholder_mask(depth, args.mask_mode)
        bbox = bbox_from_mask(mask)
        key = f"000001_{im_id:06d}"

        # WebDataset writes raw bytes into the tar shard, so use in-memory PNGs
        # instead of creating temporary cropped files beside the source dataset.
        rgb_buffer = io.BytesIO()
        depth_buffer = io.BytesIO()
        rgb.save(rgb_buffer, format="PNG")
        Image.fromarray(depth.astype(np.uint16)).save(depth_buffer, format="PNG")
        shard_writer.write(
            {
                "__key__": key,
                "rgb.png": rgb_buffer.getvalue(),
                "depth.png": depth_buffer.getvalue(),
                "camera.json": json.dumps(camera).encode(),
            }
        )

        # test_targets_bop19.json tells GigaPose which object to estimate in
        # each frame. The CNOS-style detection file gives its initial crop/mask.
        targets.append(
            {"scene_id": 1, "im_id": im_id, "obj_id": 1, "inst_count": 1}
        )
        detections.append(
            {
                "scene_id": 1,
                "image_id": im_id,
                "category_id": 1,
                "score": 1.0,
                "bbox": bbox,
                "segmentation": pycoco_utils.binary_mask_to_rle(mask),
                "time": 0.0,
            }
        )
        frame_map.append({"scene_id": 1, "im_id": im_id, "source_frame_id": stem})

    shard_writer.close()
    (test_dir / "key_to_shard.json").write_text(json.dumps(make_key_to_shard_map(test_dir)))
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
    (detections_dir / "cnos-fastsam_racecar-test.json").write_text(
        json.dumps(detections, indent=2)
    )
    print(f"Prepared {len(stems)} frames in {dataset_dir}")


if __name__ == "__main__":
    main()
