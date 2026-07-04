"""Select real-world GigaPose label candidates close to EPnPv2 labels.

This is a template/runnable scaffold for the real-world labeling pipeline:

1. Read GigaPose BOP/MultiHypothesis prediction CSV.
2. Read actual pose label files from ``EPnPv2_labels``.
3. Estimate or load a constant frame transform between GigaPose and EPnPv2.
4. Match predictions to EPnPv2 labels by image/frame key.
5. Save the close-enough candidates for later extrinsic optimization.

The EPnPv2 label parser is intentionally flexible. It supports CSV/JSON/NPZ/NPY
files when they contain one of these pose forms:

- a 4x4 matrix field such as ``T``, ``pose``, ``T_map_object``
- columns/keys ``R`` + ``t``
- quaternion + translation, using common names like ``qx,qy,qz,qw,x,y,z``

If your EPnPv2 files use different names, pass ``--epnp-key-field`` and/or add
the names in ``pose_from_mapping`` below.

For the Assetto/ARCL folders where labels are named like
``3900034190304_0.json`` and GigaPose frame_map image stems are named like
``image_3900034190304``, use:

``--epnp-strip-trailing-instance-id --epnp-key-prefix image_ --match-key image_stem``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATES = ("2026-05-05", "2026-05-18", "2026-05-26")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/media/hdd2/ARCL_multicar_bags/camera_dataset"),
        help="Root containing real-world session folders.",
    )
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Date prefix to include. Repeatable. Defaults to the requested May 2026 dates.",
    )
    parser.add_argument(
        "--gigapose-predictions",
        type=Path,
        required=True,
        help="GigaPose prediction CSV or MultiHypothesis.csv.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional prepared GigaPose dataset dir containing frame_map.json.",
    )
    parser.add_argument(
        "--frame-map",
        type=Path,
        default=None,
        help="Optional explicit frame_map.json path.",
    )
    parser.add_argument(
        "--epnp-root",
        type=Path,
        default=None,
        help=(
            "Folder containing EPnPv2 label files. If omitted, the script searches "
            "for EPnPv2_labels under matching source sessions."
        ),
    )
    parser.add_argument(
        "--epnp-glob",
        default="**/*",
        help="Glob below EPnPv2 label folders. Keep broad unless the folder is huge.",
    )
    parser.add_argument(
        "--epnp-key-field",
        default=None,
        help=(
            "Optional field/column to use as match key, e.g. filename, image_path, "
            "frame_id, timestamp, sim_time_ms."
        ),
    )
    parser.add_argument(
        "--epnp-key-prefix",
        default="",
        help=(
            "Optional prefix to add to EPnPv2 match keys after normalization. "
            "Useful when EPnP files are named 3900..._0.json but frame_map uses image_3900..."
        ),
    )
    parser.add_argument(
        "--epnp-strip-trailing-instance-id",
        action="store_true",
        help=(
            "Strip a trailing _<integer> from EPnPv2 keys derived from label filenames, "
            "e.g. 3900034190304_0 -> 3900034190304."
        ),
    )
    parser.add_argument(
        "--match-key",
        choices=("auto", "scene_im", "image_stem", "image_name", "frame_id", "sim_time_ms"),
        default="auto",
        help="How to match GigaPose rows to EPnPv2 labels.",
    )
    parser.add_argument(
        "--frame-transform-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON with a 4x4 frame transform. If omitted, the script "
            "estimates one from easy one-to-one matches."
        ),
    )
    parser.add_argument(
        "--frame-transform-side",
        choices=("left", "right"),
        default="left",
        help=(
            "How to apply the GigaPose-to-EPnP alignment. "
            "'left' assumes T_epnp ~= X @ T_gigapose (camera-frame correction). "
            "'right' assumes T_epnp ~= T_gigapose @ X (object/CAD-frame correction). "
            "For EPnPv2 T_camera_object_centered labels, 'right' is usually the better diagnostic."
        ),
    )
    parser.add_argument(
        "--epnp-translation-unit",
        choices=("auto", "m", "mm"),
        default="auto",
        help="Unit of EPnPv2 translation labels.",
    )
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-translation-error-mm", type=float, default=2000.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument(
        "--frame-transform-refine-iterations",
        type=int,
        default=0,
        help=(
            "Optionally re-estimate the frame/object transform using only inlier "
            "one-to-one pairs after an initial estimate. This is useful when some "
            "GigaPose/EPnP pairs are wrong or 180-degree flipped."
        ),
    )
    parser.add_argument(
        "--frame-transform-inlier-translation-mm",
        type=float,
        default=None,
        help=(
            "Translation threshold for transform-refinement inliers. Defaults to "
            "--max-translation-error-mm when refinement is enabled."
        ),
    )
    parser.add_argument(
        "--frame-transform-inlier-rotation-deg",
        type=float,
        default=None,
        help=(
            "Rotation threshold for transform-refinement inliers. Defaults to "
            "--max-rotation-error-deg when refinement is enabled."
        ),
    )
    parser.add_argument(
        "--max-candidates-per-key",
        type=int,
        default=20,
        help="Limit combinatorics when an image has many predictions/labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gigaPose_datasets/results/real_world_label_candidates"),
    )
    return parser.parse_args()


def matrix_to_text(T: np.ndarray) -> str:
    return " ".join(f"{v:.12g}" for v in np.asarray(T, dtype=float).reshape(-1))


def text_to_matrix(value: str) -> np.ndarray:
    values = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if values.size != 16:
        raise ValueError(f"Expected 16 numbers for a 4x4 matrix, got {values.size}")
    return values.reshape(4, 4)


def make_transform(R: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t_mm, dtype=float).reshape(3)
    return T


def matrix_3x4_or_4x4_to_transform(value: Any) -> np.ndarray:
    arr = text_to_matrix(value) if isinstance(value, str) else np.asarray(value, dtype=float)
    flat = arr.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4).copy()
    if flat.size == 12:
        T = np.eye(4, dtype=float)
        T[:3, :] = flat.reshape(3, 4)
        return T
    raise ValueError(f"Expected 12 or 16 values for a 3x4/4x4 pose matrix, got {flat.size}")


def rotation_error_deg(pred_R: np.ndarray, gt_R: np.ndarray) -> float:
    delta = pred_R @ gt_R.T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            item = {
                "row_index": row_index,
                "scene_id": int(row["scene_id"]),
                "im_id": int(row["im_id"]),
                "obj_id": int(row["obj_id"]),
                "score": float(row["score"]),
                "R": np.fromstring(row["R"], sep=" ", dtype=float).reshape(3, 3),
                "t": np.fromstring(row["t"], sep=" ", dtype=float).reshape(3),
            }
            if "instance_id" in row and row["instance_id"] != "":
                item["instance_id"] = int(row["instance_id"])
            rows.append(item)
    return rows


def infer_prediction_translation_scale(rows: list[dict[str, Any]]) -> float:
    z_values = np.asarray([row["t"][2] for row in rows if np.isfinite(row["t"][2])])
    if z_values.size == 0:
        return 1.0
    median_z = float(np.median(np.abs(z_values)))
    return 1000.0 if median_z < 1000.0 else 1.0


def collapse_top_predictions_per_detection(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not predictions or "instance_id" not in predictions[0]:
        return predictions

    top_by_detection: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in predictions:
        key = (row["scene_id"], row["im_id"], row["instance_id"])
        if key not in top_by_detection or row["score"] > top_by_detection[key]["score"]:
            top_by_detection[key] = row
    return list(top_by_detection.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_numbers(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    return np.fromstring(str(value).strip().strip("[]").replace(",", " "), sep=" ")


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    return q / max(np.linalg.norm(q), 1e-12)


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        return np.eye(3)
    quats = []
    for R in rotations:
        q = rotation_to_quat_xyzw(R)
        if quats and float(np.dot(quats[0], q)) < 0:
            q = -q
        quats.append(q)
    M = np.zeros((4, 4), dtype=float)
    for q in quats:
        M += np.outer(q, q)
    _, vecs = np.linalg.eigh(M)
    return quat_xyzw_to_R(vecs[:, -1])


def normalize_translation(t: np.ndarray, unit: str) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(3)
    if unit == "m":
        return t * 1000.0
    if unit == "mm":
        return t
    # Auto heuristic: real car translations below ~1000 are very likely meters.
    return t * 1000.0 if np.nanmedian(np.abs(t)) < 1000.0 else t


def pose_from_mapping(row: dict[str, Any], translation_unit: str) -> np.ndarray | None:
    matrix_keys = [
        "T",
        "pose",
        "matrix",
        "transform",
        "T_map_object",
        "T_map_obj",
        "T_camera_object",
        "T_camera_object_centered",
        "T_cam_obj",
    ]
    for key in matrix_keys:
        if key in row and row[key] not in ("", None):
            try:
                T = matrix_3x4_or_4x4_to_transform(row[key])
                T[:3, 3] = normalize_translation(T[:3, 3], translation_unit)
                return T
            except Exception:
                pass

    if "R" in row and "t" in row:
        R = split_numbers(row["R"]).reshape(3, 3)
        t = normalize_translation(split_numbers(row["t"])[:3], translation_unit)
        return make_transform(R, t)

    scalar_sets = [
        ("r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "x", "y", "z"),
        ("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22", "tx", "ty", "tz"),
    ]
    for keys in scalar_sets:
        if all(key in row for key in keys):
            vals = [float(row[key]) for key in keys]
            R = np.asarray(vals[:9], dtype=float).reshape(3, 3)
            t = normalize_translation(np.asarray(vals[9:12]), translation_unit)
            return make_transform(R, t)

    quat_sets = [
        ("qx", "qy", "qz", "qw", "x", "y", "z"),
        ("qx", "qy", "qz", "qw", "tx", "ty", "tz"),
        ("x", "y", "z", "qx", "qy", "qz", "qw"),
        ("tx", "ty", "tz", "qx", "qy", "qz", "qw"),
    ]
    for keys in quat_sets:
        if all(key in row for key in keys):
            vals = [float(row[key]) for key in keys]
            if keys[0] in ("qx",):
                q = np.asarray(vals[:4], dtype=float)
                t = np.asarray(vals[4:7], dtype=float)
            else:
                t = np.asarray(vals[:3], dtype=float)
                q = np.asarray(vals[3:7], dtype=float)
            return make_transform(quat_xyzw_to_R(q), normalize_translation(t, translation_unit))

    return None


def discover_sessions(source_root: Path, dates: tuple[str, ...]) -> list[Path]:
    if not source_root.exists():
        return []
    sessions = []
    for child in sorted(source_root.iterdir()):
        if child.is_dir() and any(child.name.startswith(date) for date in dates):
            sessions.append(child)
    return sessions


def discover_epnp_roots(source_root: Path, dates: tuple[str, ...], explicit_root: Path | None) -> list[Path]:
    if explicit_root is not None:
        return [explicit_root]
    roots = []
    for session in discover_sessions(source_root, dates):
        roots.extend(sorted(path for path in session.rglob("EPnPv2_labels") if path.is_dir()))
    return roots


def row_key_from_mapping(row: dict[str, Any], key_field: str | None) -> str | None:
    if key_field and key_field in row and row[key_field] not in ("", None):
        return normalize_key(row[key_field])
    for key in ("filename", "file_name", "image", "image_path", "rgb_path", "frame", "frame_id", "sim_time_ms", "timestamp"):
        if key in row and row[key] not in ("", None):
            return normalize_key(row[key])
    return None


def normalize_key(value: Any) -> str:
    text = str(value).strip()
    if "/" in text or "\\" in text:
        path = Path(text)
        return path.stem
    return Path(text).stem if "." in Path(text).name else text


def adjust_epnp_key(key: str, key_prefix: str, strip_trailing_instance_id: bool) -> str:
    if strip_trailing_instance_id:
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            key = parts[0]
    if key_prefix and not key.startswith(key_prefix):
        key = f"{key_prefix}{key}"
    return key


def load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("labels", "objects", "instances", "poses", "annotations"):
            if isinstance(data.get(key), list):
                return [item for item in data[key] if isinstance(item, dict)]
        return [data]
    return []


def load_np_records(path: Path) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = list(data.keys())
        if "poses" in keys:
            poses = np.asarray(data["poses"])
        elif "T" in keys:
            poses = np.asarray(data["T"])
        else:
            return []
        records = []
        for idx, pose in enumerate(poses.reshape(-1, 4, 4)):
            records.append({"T": pose, "frame_id": idx})
        return records
    arr = np.asarray(data)
    if arr.shape[-2:] == (4, 4):
        return [{"T": pose, "frame_id": idx} for idx, pose in enumerate(arr.reshape(-1, 4, 4))]
    return []


def load_epnp_labels(
    roots: list[Path],
    glob_pattern: str,
    key_field: str | None,
    translation_unit: str,
    key_prefix: str = "",
    strip_trailing_instance_id: bool = False,
) -> list[dict[str, Any]]:
    labels = []
    for root in roots:
        for path in sorted(root.glob(glob_pattern)):
            if not path.is_file():
                continue
            records: list[dict[str, Any]]
            suffix = path.suffix.lower()
            try:
                if suffix == ".csv":
                    with path.open(newline="") as f:
                        records = list(csv.DictReader(f))
                elif suffix in (".json", ".jsn"):
                    records = load_json_records(path)
                elif suffix in (".npz", ".npy"):
                    records = load_np_records(path)
                else:
                    continue
            except Exception as exc:
                print(f"Skipping unreadable EPnPv2 label file {path}: {exc}")
                continue

            for record_index, record in enumerate(records):
                T = pose_from_mapping(record, translation_unit)
                if T is None:
                    continue
                key = row_key_from_mapping(record, key_field)
                if key is None:
                    key = normalize_key(path.stem)
                key = adjust_epnp_key(key, key_prefix, strip_trailing_instance_id)
                labels.append(
                    {
                        "epnp_label_path": str(path),
                        "epnp_record_index": record_index,
                        "match_key": key,
                        "T_epnp": T,
                    }
                )
    return labels


def load_frame_map(path: Path | None, dataset_dir: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    frame_map_path = path or (dataset_dir / "frame_map.json" if dataset_dir else None)
    if not frame_map_path or not frame_map_path.is_file():
        return {}
    rows = json.loads(frame_map_path.read_text())
    return {(int(row["scene_id"]), int(row["im_id"])): row for row in rows}


def prediction_key(pred: dict[str, Any], frame_map: dict[tuple[int, int], dict[str, Any]], mode: str) -> str:
    scene_im = f"{int(pred['scene_id']):06d}_{int(pred['im_id']):06d}"
    if mode == "scene_im":
        return scene_im
    info = frame_map.get((int(pred["scene_id"]), int(pred["im_id"])), {})
    if mode in ("auto", "image_stem", "image_name"):
        for key in ("image_path", "rgb_path", "file_name", "filename", "image"):
            if key in info:
                value = Path(str(info[key]))
                return value.name if mode == "image_name" else value.stem
        for instance in info.get("instances", []):
            for key in ("image_path", "rgb_path", "file_name", "filename", "image"):
                if key in instance:
                    value = Path(str(instance[key]))
                    return value.name if mode == "image_name" else value.stem
    if mode == "frame_id":
        for key in ("frame_id", "frame", "im_id"):
            if key in info:
                return str(info[key])
        return str(pred["im_id"])
    if mode == "sim_time_ms":
        for key in ("sim_time_ms", "timestamp", "time"):
            if key in info:
                return str(info[key])
    return scene_im


def load_gigapose_predictions(
    path: Path,
    frame_map: dict[tuple[int, int], dict[str, Any]],
    match_key_mode: str,
) -> list[dict[str, Any]]:
    rows = collapse_top_predictions_per_detection(load_prediction_rows(path))
    t_scale = infer_prediction_translation_scale(rows)
    output = []
    for row in rows:
        item = dict(row)
        item["t_mm"] = np.asarray(row["t"], dtype=float).reshape(3) * t_scale
        item["T_gigapose"] = make_transform(row["R"], item["t_mm"])
        item["match_key"] = prediction_key(row, frame_map, match_key_mode)
        output.append(item)
    return output


def estimate_frame_transform(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    side: str,
) -> np.ndarray:
    """Estimate X such that T_epnp ≈ X @ T_gigapose or T_gigapose @ X."""
    if not pairs:
        return np.eye(4)
    if side == "left":
        transforms = [T_epnp @ np.linalg.inv(T_giga) for T_giga, T_epnp in pairs]
    elif side == "right":
        transforms = [np.linalg.inv(T_giga) @ T_epnp for T_giga, T_epnp in pairs]
    else:
        raise ValueError(f"Unknown frame transform side: {side}")
    X = np.eye(4, dtype=float)
    X[:3, :3] = average_rotations([T[:3, :3] for T in transforms])
    X[:3, 3] = np.median(np.asarray([T[:3, 3] for T in transforms]), axis=0)
    return X


def apply_frame_transform(T_gigapose: np.ndarray, X: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return X @ T_gigapose
    if side == "right":
        return T_gigapose @ X
    raise ValueError(f"Unknown frame transform side: {side}")


def transform_pair_errors(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    side: str,
) -> list[tuple[float, float]]:
    errors = []
    for T_giga, T_epnp in pairs:
        T_aligned = apply_frame_transform(T_giga, X, side)
        errors.append(
            (
                float(np.linalg.norm(T_aligned[:3, 3] - T_epnp[:3, 3])),
                rotation_error_deg(T_aligned[:3, :3], T_epnp[:3, :3]),
            )
        )
    return errors


def refine_frame_transform(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    side: str,
    initial_X: np.ndarray,
    iterations: int,
    max_translation_error_mm: float,
    max_rotation_error_deg: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Iteratively re-estimate X from inlier one-to-one pairs."""
    X = initial_X
    history = []
    if iterations <= 0 or not pairs:
        return X, history

    current_pairs = pairs
    for iteration in range(iterations):
        errors = transform_pair_errors(pairs, X, side)
        inlier_indices = [
            idx
            for idx, (t_err, r_err) in enumerate(errors)
            if t_err <= max_translation_error_mm and r_err <= max_rotation_error_deg
        ]
        history.append(
            {
                "iteration": iteration,
                "input_pairs": len(pairs),
                "inlier_pairs": len(inlier_indices),
                "inlier_translation_threshold_mm": max_translation_error_mm,
                "inlier_rotation_threshold_deg": max_rotation_error_deg,
            }
        )
        if len(inlier_indices) < 3:
            break
        refined_pairs = [pairs[idx] for idx in inlier_indices]
        if len(refined_pairs) == len(current_pairs):
            current_pairs = refined_pairs
            X = estimate_frame_transform(current_pairs, side)
            break
        current_pairs = refined_pairs
        X = estimate_frame_transform(current_pairs, side)
    return X, history


def initial_one_to_one_pairs(
    preds_by_key: dict[str, list[dict[str, Any]]],
    epnp_by_key: dict[str, list[dict[str, Any]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for key in sorted(set(preds_by_key) & set(epnp_by_key)):
        preds = preds_by_key[key]
        labels = epnp_by_key[key]
        if len(preds) == 1 and len(labels) == 1:
            pairs.append((preds[0]["T_gigapose"], labels[0]["T_epnp"]))
    return pairs


def load_frame_transform(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    for key in ("T_epnp_gigapose", "T_target_source", "matrix", "T"):
        if key in data:
            return np.asarray(data[key], dtype=float).reshape(4, 4)
    raise ValueError(f"{path} does not contain a 4x4 transform")


def save_frame_transform(path: Path, X: np.ndarray, estimated_from_pairs: int, side: str) -> None:
    if side == "left":
        description = "Frame transform X such that T_epnp ≈ X @ T_gigapose."
        key = "T_epnp_gigapose_left"
    elif side == "right":
        description = "Object/CAD-frame transform X such that T_epnp ≈ T_gigapose @ X."
        key = "T_epnp_gigapose_right"
    else:
        raise ValueError(f"Unknown frame transform side: {side}")
    path.write_text(
        json.dumps(
            {
                "description": description,
                "frame_transform_side": side,
                key: np.asarray(X).reshape(4, 4).tolist(),
                "T_epnp_gigapose": np.asarray(X).reshape(4, 4).tolist(),
                "estimated_from_pairs": estimated_from_pairs,
                "translation_unit": "mm",
            },
            indent=2,
        )
    )


def make_candidate_rows(
    preds_by_key: dict[str, list[dict[str, Any]]],
    epnp_by_key: dict[str, list[dict[str, Any]]],
    X_epnp_gigapose: np.ndarray,
    max_candidates_per_key: int,
    frame_transform_side: str,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(preds_by_key) & set(epnp_by_key)):
        preds = sorted(preds_by_key[key], key=lambda row: row.get("score", 0.0), reverse=True)[
            :max_candidates_per_key
        ]
        labels = epnp_by_key[key][:max_candidates_per_key]
        for pred in preds:
            T_pred_aligned = apply_frame_transform(
                pred["T_gigapose"], X_epnp_gigapose, frame_transform_side
            )
            for label in labels:
                T_epnp = label["T_epnp"]
                rows.append(
                    {
                        "match_key": key,
                        "scene_id": pred["scene_id"],
                        "im_id": pred["im_id"],
                        "obj_id": pred["obj_id"],
                        "instance_id": pred.get("instance_id", ""),
                        "prediction_row_index": pred["row_index"],
                        "score": pred["score"],
                        "epnp_label_path": label["epnp_label_path"],
                        "epnp_record_index": label["epnp_record_index"],
                        "translation_error_mm": float(
                            np.linalg.norm(T_pred_aligned[:3, 3] - T_epnp[:3, 3])
                        ),
                        "rotation_error_deg": rotation_error_deg(
                            T_pred_aligned[:3, :3], T_epnp[:3, :3]
                        ),
                        "T_gigapose_cam_obj": matrix_to_text(pred["T_gigapose"]),
                        "T_gigapose_aligned_epnp_obj": matrix_to_text(T_pred_aligned),
                        "T_epnp_obj": matrix_to_text(T_epnp),
                    }
                )
    rows.sort(key=lambda row: (row["match_key"], row["translation_error_mm"], row["rotation_error_deg"]))
    return rows


def best_rows_per_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["match_key"]), str(row["epnp_label_path"]), int(row["epnp_record_index"]))
        if key not in best:
            best[key] = row
            continue
        old = best[key]
        if (
            float(row["translation_error_mm"]),
            float(row["rotation_error_deg"]),
            -float(row["score"]),
        ) < (
            float(old["translation_error_mm"]),
            float(old["rotation_error_deg"]),
            -float(old["score"]),
        ):
            best[key] = row
    return sorted(best.values(), key=lambda row: (row["match_key"], row["epnp_record_index"]))


def summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str, items: list[dict[str, Any]]) -> dict[str, float]:
        vals = np.asarray([float(row[key]) for row in items], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {f"{key}_mean": float("nan"), f"{key}_median": float("nan"), f"{key}_p90": float("nan")}
        return {
            f"{key}_mean": float(np.mean(vals)),
            f"{key}_median": float(np.median(vals)),
            f"{key}_p90": float(np.percentile(vals, 90)),
        }

    out: dict[str, Any] = {
        "candidate_pairs": len(rows),
        "selected_samples": len(selected),
    }
    out.update(stats("translation_error_mm", selected))
    out.update(stats("rotation_error_deg", selected))
    return out


def main() -> None:
    args = parse_args()
    dates = tuple(args.date or DEFAULT_DATES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_map = load_frame_map(args.frame_map, args.dataset_dir)
    predictions = load_gigapose_predictions(args.gigapose_predictions, frame_map, args.match_key)
    predictions = [row for row in predictions if float(row.get("score", 0.0)) >= args.min_score]

    epnp_roots = discover_epnp_roots(args.source_root, dates, args.epnp_root)
    labels = load_epnp_labels(
        epnp_roots,
        args.epnp_glob,
        args.epnp_key_field,
        args.epnp_translation_unit,
        args.epnp_key_prefix,
        args.epnp_strip_trailing_instance_id,
    )
    if not labels:
        raise ValueError(
            "No EPnPv2 pose labels were loaded. Check --epnp-root, --epnp-glob, "
            "--epnp-key-field, and the pose field names."
        )

    preds_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    epnp_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        preds_by_key[str(row["match_key"])].append(row)
    for row in labels:
        epnp_by_key[str(row["match_key"])].append(row)

    if args.frame_transform_json:
        X = load_frame_transform(args.frame_transform_json)
        estimated_pairs = 0
        refinement_history = []
    else:
        one_to_one = initial_one_to_one_pairs(preds_by_key, epnp_by_key)
        X = estimate_frame_transform(one_to_one, args.frame_transform_side)
        estimated_pairs = len(one_to_one)
        inlier_t = (
            args.frame_transform_inlier_translation_mm
            if args.frame_transform_inlier_translation_mm is not None
            else args.max_translation_error_mm
        )
        inlier_r = (
            args.frame_transform_inlier_rotation_deg
            if args.frame_transform_inlier_rotation_deg is not None
            else args.max_rotation_error_deg
        )
        X, refinement_history = refine_frame_transform(
            one_to_one,
            args.frame_transform_side,
            X,
            args.frame_transform_refine_iterations,
            inlier_t,
            inlier_r,
        )
    save_frame_transform(
        args.output_dir / "frame_transform_gigapose_to_epnp.json",
        X,
        estimated_pairs,
        args.frame_transform_side,
    )

    all_candidates = make_candidate_rows(
        preds_by_key,
        epnp_by_key,
        X,
        args.max_candidates_per_key,
        args.frame_transform_side,
    )
    best_candidates = best_rows_per_label(all_candidates)
    selected = [
        row
        for row in best_candidates
        if float(row["translation_error_mm"]) <= args.max_translation_error_mm
        and float(row["rotation_error_deg"]) <= args.max_rotation_error_deg
    ]

    write_csv(args.output_dir / "all_candidate_pairs.csv", all_candidates)
    write_csv(args.output_dir / "best_candidate_per_epnp_label.csv", best_candidates)
    write_csv(args.output_dir / "selected_samples.csv", selected)

    report = {
        "source_root": str(args.source_root),
        "dates": dates,
        "epnp_roots": [str(path) for path in epnp_roots],
        "gigapose_predictions": str(args.gigapose_predictions),
        "frame_transform_side": args.frame_transform_side,
        "frame_transform_estimated_from_one_to_one_pairs": estimated_pairs,
        "frame_transform_refine_iterations": args.frame_transform_refine_iterations,
        "frame_transform_refinement_history": refinement_history,
        "min_score": args.min_score,
        "max_translation_error_mm": args.max_translation_error_mm,
        "max_rotation_error_deg": args.max_rotation_error_deg,
        "loaded_predictions_after_score_filter": len(predictions),
        "loaded_epnp_labels": len(labels),
        **summarize(all_candidates, selected),
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote selected samples to {args.output_dir / 'selected_samples.csv'}")


if __name__ == "__main__":
    main()
