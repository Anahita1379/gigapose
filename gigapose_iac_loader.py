from pathlib import Path
from PIL import Image
import numpy as np
import json
import shutil
import trimesh
import cv2


ROOT = Path("/home/anahita/Dataset/rosbag_extracted_300m")
SPLIT_FILE = Path("/home/anahita/self-supervised-depth-completion/splits/iac_train.txt")

OUT_ROOT = Path("/mnt/ssd2tb/gigapose/datasets/racecar")
CAD_FILE = ROOT / "CAD_car/racecar0_highres.ply"
MASK_FILE = ROOT / "mask/mask_005020.png"
START_ID = 5020          # change to 12849 if needed
MAX_FRAMES = 100         # set None for all frames

OBJ_ID = 1

# If you know approximate car center in cropped-middle image coordinates
U_CENTER = 384
V_CENTER = 160
BOX_W = 250
BOX_H = 120

def bbox_from_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(target_size, resample=Image.NEAREST)
    mask_np = np.array(mask)

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def load_rgb_middle(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    third = w // 3
    img = img.crop((third, 0, 2 * third, h))

    return img, w


def adjust_K_for_middle_crop(K, full_w):
    K = K.copy()
    third = full_w // 3
    K[0, 2] -= third
    return K


def load_K(path):
    return np.load(path).astype(np.float32)


def get_frame_id(img_rel):
    stem = Path(img_rel).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits)


def make_bbox(u, v, box_w, box_h, W, H):
    x = int(max(0, u - box_w // 2))
    y = int(max(0, v - box_h // 2))
    x2 = int(min(W - 1, u + box_w // 2))
    y2 = int(min(H - 1, v + box_h // 2))

    return [x, y, x2 - x, y2 - y]


def main():
    rgb_dir = OUT_ROOT / "test/000001/rgb"
    model_dir = OUT_ROOT / "models"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy CAD model into BOP-style name
    dst_model = model_dir / "obj_000001.ply"
    if not dst_model.exists():
        shutil.copy(CAD_FILE, dst_model)

    # Create models_info.json
    mesh = trimesh.load(CAD_FILE)
    diameter = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    models_info = {
        str(OBJ_ID): {
            "diameter": diameter,
            "min_x": float(mesh.bounds[0, 0]),
            "min_y": float(mesh.bounds[0, 1]),
            "min_z": float(mesh.bounds[0, 2]),
            "size_x": float(mesh.bounds[1, 0] - mesh.bounds[0, 0]),
            "size_y": float(mesh.bounds[1, 1] - mesh.bounds[0, 1]),
            "size_z": float(mesh.bounds[1, 2] - mesh.bounds[0, 2]),
        }
    }

    with open(model_dir / "models_info.json", "w") as f:
        json.dump(models_info, f, indent=2)

    # Read split file
    with open(SPLIT_FILE) as f:
        samples = [line.strip().split() for line in f if line.strip()]

    selected = []
    for sample in samples:
        img_rel, depth_rel, K_rel = sample
        frame_id = get_frame_id(img_rel)

        if frame_id >= START_ID:
            selected.append(sample)

        if MAX_FRAMES is not None and len(selected) >= MAX_FRAMES:
            break

    scene_camera = {}
    scene_gt_info = {}
    test_targets = []
    detections = []

    for out_i, sample in enumerate(selected):
        img_rel, depth_rel, K_rel = sample

        image_path = ROOT / img_rel
        K_path = ROOT / K_rel

        img, full_w = load_rgb_middle(image_path)
        W, H = img.size

        K = load_K(K_path)
        K = adjust_K_for_middle_crop(K, full_w)

        out_name = f"{out_i:06d}.png"
        img.save(rgb_dir / out_name)

        cam_K = K.reshape(-1).tolist()

        scene_camera[str(out_i)] = {
            "cam_K": cam_K,
            "depth_scale": 1.0,
        }

        # bbox = make_bbox(
        #     U_CENTER,
        #     V_CENTER,
        #     BOX_W,
        #     BOX_H,
        #     W,
        #     H,
        # )
        
        if out_i == 0:
            bbox = bbox_from_mask(MASK_FILE, target_size=(W, H))
        else:
            bbox = last_bbox

        last_bbox = bbox

        scene_gt_info[str(out_i)] = [
            {
                "bbox_obj": bbox,
                "bbox_visib": bbox,
                "visib_fract": 1.0,
            }
        ]

        test_targets.append(
            {
                "scene_id": 1,
                "im_id": out_i,
                "obj_id": OBJ_ID,
                "inst_count": 1,
            }
        )

        detections.append(
            {
                "scene_id": 1,
                "im_id": out_i,
                "obj_id": OBJ_ID,
                "score": 1.0,
                "bbox_est": bbox,
            }
        )

    scene_dir = OUT_ROOT / "test/000001"

    with open(scene_dir / "scene_camera.json", "w") as f:
        json.dump(scene_camera, f, indent=2)

    with open(scene_dir / "scene_gt_info.json", "w") as f:
        json.dump(scene_gt_info, f, indent=2)

    with open(OUT_ROOT / "test_targets_bop19.json", "w") as f:
        json.dump(test_targets, f, indent=2)

    # Useful generic detection file
    det_dir = OUT_ROOT / "default_detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    with open(det_dir / "detections.json", "w") as f:
        json.dump(detections, f, indent=2)

    print("Done.")
    print("Dataset written to:", OUT_ROOT)
    print("Frames:", len(selected))
    print("CAD:", dst_model)
    print("RGB dir:", rgb_dir)
    print("scene_camera:", scene_dir / "scene_camera.json")
    print("detections:", det_dir / "detections.json")


if __name__ == "__main__":
    main()