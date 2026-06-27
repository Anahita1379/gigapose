"""Numerically compare two GigaPose prediction CSVs against prepared Assetto GT.

The prepared Assetto inference/training WebDataset already stores camera-frame
ground-truth poses in BOP style:

  - ``gt.json``: ``cam_R_m2c`` and ``cam_t_m2c`` in millimeters
  - ``camera.json``: camera intrinsics

This script reads that GT directly from the shards, evaluates two prediction
CSVs, and writes per-instance plus summary metrics.  It handles both regular
GigaPose CSVs and MultiHypothesis CSVs with a global ``instance_id`` column.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--finetuned-predictions", type=Path, required=True)
    parser.add_argument(
        "--baseline-name",
        default="baseline",
        help="Name to use in output tables for --baseline-predictions.",
    )
    parser.add_argument(
        "--finetuned-name",
        default="finetuned",
        help="Name to use in output tables for --finetuned-predictions.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa_inference"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--max-mesh-points", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/prediction_gt_comparison"),
    )
    return parser.parse_args()


def rotation_error_deg(pred_R: np.ndarray, gt_R: np.ndarray) -> float:
    delta = pred_R @ gt_R.T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def project_origin(t_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    uvw = K @ t_mm.reshape(3)
    return uvw[:2] / max(uvw[2], 1e-9)


def bbox_xywh_center(bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=float)


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(
        0.0, min(ay2, by2) - max(ay, by)
    )
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else float("nan")


def shard_path_for_key(split_dir: Path, key: str) -> Path:
    mapping = json.loads((split_dir / "key_to_shard.json").read_text())
    if key not in mapping:
        raise KeyError(f"{key} missing from {split_dir / 'key_to_shard.json'}")
    return split_dir / f"shard-{int(mapping[key]):06d}.tar"


def load_gt(dataset_dir: Path, split: str) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[tuple[int, int], np.ndarray]]:
    split_dir = dataset_dir / split
    key_to_shard = json.loads((split_dir / "key_to_shard.json").read_text())
    shard_to_keys: dict[int, list[str]] = defaultdict(list)
    for key, shard_id in key_to_shard.items():
        shard_to_keys[int(shard_id)].append(key)

    gt_by_image: dict[tuple[int, int], list[dict[str, Any]]] = {}
    K_by_image: dict[tuple[int, int], np.ndarray] = {}
    for shard_id, keys in shard_to_keys.items():
        shard_path = split_dir / f"shard-{shard_id:06d}.tar"
        with tarfile.open(shard_path) as tar:
            for key in keys:
                scene_id, im_id = (int(part) for part in key.split("_"))
                gt = json.loads(tar.extractfile(f"{key}.gt.json").read())
                gt_info = json.loads(tar.extractfile(f"{key}.gt_info.json").read())
                camera = json.loads(tar.extractfile(f"{key}.camera.json").read())
                items = []
                for local_idx, (pose, info) in enumerate(zip(gt, gt_info)):
                    items.append(
                        {
                            "gt_index": local_idx,
                            "obj_id": int(pose["obj_id"]),
                            "R": np.asarray(pose["cam_R_m2c"], dtype=float).reshape(3, 3),
                            "t_mm": np.asarray(pose["cam_t_m2c"], dtype=float).reshape(3),
                            "bbox_visib": list(map(float, info["bbox_visib"])),
                            "bbox_obj": list(map(float, info["bbox_obj"])),
                        }
                    )
                gt_by_image[(scene_id, im_id)] = items
                K_by_image[(scene_id, im_id)] = np.asarray(
                    camera["cam_K"], dtype=float
                ).reshape(3, 3)
    return gt_by_image, K_by_image


def global_instance_map(dataset_dir: Path) -> dict[int, tuple[int, int, int]]:
    """Map MultiHypothesis global instance_id -> (scene_id, im_id, gt_index)."""
    frame_map_path = dataset_dir / "frame_map.json"
    if not frame_map_path.is_file():
        return {}
    rows = json.loads(frame_map_path.read_text())
    mapping = {}
    instance_id = 0
    for row in rows:
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        for gt_index, _ in enumerate(row.get("instances", [])):
            mapping[instance_id] = (scene_id, im_id, gt_index)
            instance_id += 1
    return mapping


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
    """Return multiplier to convert prediction translations to millimeters."""
    z_values = np.asarray([row["t"][2] for row in rows if np.isfinite(row["t"][2])])
    if z_values.size == 0:
        return 1.0
    median_z = float(np.median(np.abs(z_values)))
    # BOP CSVs should be millimeters, but some local scripts/checkpoints emit
    # meters.  Assetto car depths are usually ~4-100 m, so median z below 1000
    # is a strong signal that the CSV is in meters and needs mm conversion.
    return 1000.0 if median_z < 1000.0 else 1.0


def collapse_top_predictions_per_detection(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one hypothesis per detection/instance in each image.

    MultiHypothesis CSVs contain several pose hypotheses per detection.  The
    ``instance_id`` values are useful for collapsing hypotheses, but they are
    not guaranteed to be the same ordering as the prepared Assetto GT instances.
    Pairing to GT is therefore done per image after this collapse.
    """
    if not predictions or "instance_id" not in predictions[0]:
        return predictions

    top_by_detection: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in predictions:
        key = (row["scene_id"], row["im_id"], row["instance_id"])
        if key not in top_by_detection or row["score"] > top_by_detection[key]["score"]:
            top_by_detection[key] = row
    return list(top_by_detection.values())


def read_ply_vertices(path: Path, max_points: int) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        return None
    header_end += len(b"end_header\n")
    header = data[:header_end].decode("latin1")
    vertex_count = None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            break
    if vertex_count is None:
        return None
    if "format binary_little_endian" in header:
        vertices = np.frombuffer(
            data[header_end : header_end + vertex_count * 12],
            dtype="<f4",
        ).reshape(vertex_count, 3)
    elif "format ascii" in header:
        rows = data[header_end:].decode("latin1").splitlines()[:vertex_count]
        vertices = np.asarray([[float(v) for v in row.split()[:3]] for row in rows])
    else:
        return None
    vertices = np.asarray(vertices, dtype=float)
    if np.linalg.norm(vertices.ptp(axis=0)) < 100.0:
        vertices = vertices * 1000.0
    if max_points > 0 and len(vertices) > max_points:
        idx = np.linspace(0, len(vertices) - 1, max_points).astype(int)
        vertices = vertices[idx]
    return vertices


def add_error_mm(pred: dict[str, Any], gt: dict[str, Any], vertices_mm: np.ndarray | None) -> float:
    if vertices_mm is None:
        return float("nan")
    pred_points = (pred["R"] @ vertices_mm.T).T + pred["t_mm"]
    gt_points = (gt["R"] @ vertices_mm.T).T + gt["t_mm"]
    return float(np.linalg.norm(pred_points - gt_points, axis=1).mean())


def assign_predictions_without_instance_ids(
    predictions: list[dict[str, Any]],
    gt_items: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not predictions or not gt_items:
        return []
    # Regular non-MultiHypothesis GigaPose CSVs usually have one row per
    # detection, but if extra hypotheses exist, keep a small high-score pool.
    pool_size = min(len(predictions), max(len(gt_items), len(gt_items) * 4))
    candidates = sorted(predictions, key=lambda item: -item["score"])[:pool_size]
    n_gt = len(gt_items)
    n_pred = len(candidates)

    # Exact brute force is fine for the common <=4-object case.
    if n_gt <= 6 and n_pred <= 12:
        best_cost = float("inf")
        best_pairs: list[tuple[int, int]] = []
        for pred_indices in itertools.permutations(range(n_pred), min(n_gt, n_pred)):
            cost = 0.0
            pairs = []
            for gt_index, pred_index in enumerate(pred_indices):
                cost += np.linalg.norm(
                    candidates[pred_index]["t_mm"] - gt_items[gt_index]["t_mm"]
                )
                pairs.append((pred_index, gt_index))
            if cost < best_cost:
                best_cost = cost
                best_pairs = pairs
        return [(candidates[p], gt_items[g]) for p, g in best_pairs]

    # Greedy fallback for dense hypothesis files without instance ids.
    remaining_pred = set(range(n_pred))
    remaining_gt = set(range(n_gt))
    pairs = []
    while remaining_pred and remaining_gt:
        best = min(
            (
                (
                    np.linalg.norm(candidates[p]["t_mm"] - gt_items[g]["t_mm"]),
                    p,
                    g,
                )
                for p in remaining_pred
                for g in remaining_gt
            ),
            key=lambda item: item[0],
        )
        _, p, g = best
        pairs.append((candidates[p], gt_items[g]))
        remaining_pred.remove(p)
        remaining_gt.remove(g)
    return pairs


def evaluate_method(
    name: str,
    prediction_path: Path,
    gt_by_image: dict[tuple[int, int], list[dict[str, Any]]],
    K_by_image: dict[tuple[int, int], np.ndarray],
    instance_map: dict[int, tuple[int, int, int]],
    vertices_mm: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_prediction_rows(prediction_path)
    t_scale = infer_prediction_translation_scale(rows)
    for row in rows:
        row["t_mm"] = row["t"] * t_scale

    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    predictions_by_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in collapse_top_predictions_per_detection(rows):
        predictions_by_image[(row["scene_id"], row["im_id"])].append(row)
    for image_key, gt_items in gt_by_image.items():
        evaluated.extend(
            assign_predictions_without_instance_ids(
                predictions_by_image.get(image_key, []), gt_items
            )
        )

    metrics = []
    for pred, gt in evaluated:
        image_key = (int(gt.get("scene_id", pred["scene_id"])), int(gt.get("im_id", pred["im_id"])))
        # gt dicts loaded above do not carry the image key; use pred key for K.
        image_key = (pred["scene_id"], pred["im_id"])
        K = K_by_image[image_key]
        pred_center = project_origin(pred["t_mm"], K)
        gt_center = project_origin(gt["t_mm"], K)
        metrics.append(
            {
                "method": name,
                "scene_id": pred["scene_id"],
                "im_id": pred["im_id"],
                "instance_id": pred.get("instance_id", ""),
                "gt_index": gt["gt_index"],
                "score": pred["score"],
                "translation_error_mm": float(np.linalg.norm(pred["t_mm"] - gt["t_mm"])),
                "depth_error_mm": float(abs(pred["t_mm"][2] - gt["t_mm"][2])),
                "rotation_error_deg": rotation_error_deg(pred["R"], gt["R"]),
                "center_error_px": float(np.linalg.norm(pred_center - gt_center)),
                "gt_bbox_center_error_px": float(
                    np.linalg.norm(pred_center - bbox_xywh_center(gt["bbox_visib"]))
                ),
                "add_mm": add_error_mm(pred, gt, vertices_mm),
                "pred_z_mm": float(pred["t_mm"][2]),
                "gt_z_mm": float(gt["t_mm"][2]),
                "translation_scale_to_mm": t_scale,
            }
        )

    summary = summarize(name, metrics, len(rows), t_scale)
    return metrics, summary


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize(name: str, rows: list[dict[str, Any]], prediction_rows: int, t_scale: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": name,
        "prediction_rows": prediction_rows,
        "evaluated_instances": len(rows),
        "translation_scale_to_mm": t_scale,
    }
    metric_keys = [
        "score",
        "translation_error_mm",
        "depth_error_mm",
        "rotation_error_deg",
        "center_error_px",
        "gt_bbox_center_error_px",
        "add_mm",
    ]
    for key in metric_keys:
        values = finite_values(rows, key)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p90"] = float(np.percentile(values, 90))
    if rows:
        for threshold in (50, 100, 250, 500, 1000):
            values = finite_values(rows, "translation_error_mm")
            summary[f"translation_recall_{threshold}mm"] = float((values <= threshold).mean())
        for threshold in (5, 10, 20, 45, 90):
            values = finite_values(rows, "rotation_error_deg")
            summary[f"rotation_recall_{threshold}deg"] = float((values <= threshold).mean())
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_paired_rows(
    rows: list[dict[str, Any]], baseline_name: str, finetuned_name: str
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]), int(row["gt_index"]))
        by_key[key][row["method"]] = row

    metric_keys = [
        "score",
        "translation_error_mm",
        "depth_error_mm",
        "rotation_error_deg",
        "center_error_px",
        "gt_bbox_center_error_px",
        "add_mm",
    ]
    paired = []
    for (scene_id, im_id, gt_index), methods in sorted(by_key.items()):
        if baseline_name not in methods or finetuned_name not in methods:
            continue
        base = methods[baseline_name]
        tuned = methods[finetuned_name]
        out: dict[str, Any] = {
            "scene_id": scene_id,
            "im_id": im_id,
            "gt_index": gt_index,
            "baseline_method": baseline_name,
            "finetuned_method": finetuned_name,
            "baseline_instance_id": base.get("instance_id", ""),
            "finetuned_instance_id": tuned.get("instance_id", ""),
        }
        for metric in metric_keys:
            base_value = float(base[metric])
            tuned_value = float(tuned[metric])
            out[f"baseline_{metric}"] = base_value
            out[f"finetuned_{metric}"] = tuned_value
            # For error metrics, positive improvement means fine-tuned is better.
            # For score, positive improvement means fine-tuned score is higher.
            if metric == "score":
                out[f"{metric}_improvement"] = tuned_value - base_value
            else:
                out[f"{metric}_improvement"] = base_value - tuned_value
        paired.append(out)
    return paired


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"paired_instances": len(rows)}
    if not rows:
        return summary
    for metric in [
        "translation_error_mm",
        "depth_error_mm",
        "rotation_error_deg",
        "center_error_px",
        "gt_bbox_center_error_px",
        "add_mm",
        "score",
    ]:
        key = f"{metric}_improvement"
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[f"{metric}_improvement_mean"] = float(values.mean())
        summary[f"{metric}_improvement_median"] = float(np.median(values))
        summary[f"{metric}_finetuned_better_fraction"] = float((values > 0).mean())
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_image, K_by_image = load_gt(args.dataset_dir, args.split)
    instance_map = global_instance_map(args.dataset_dir)
    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = read_ply_vertices(mesh_path, args.max_mesh_points)

    all_rows = []
    summaries = []
    for name, path in (
        (args.baseline_name, args.baseline_predictions),
        (args.finetuned_name, args.finetuned_predictions),
    ):
        rows, summary = evaluate_method(
            name=name,
            prediction_path=path,
            gt_by_image=gt_by_image,
            K_by_image=K_by_image,
            instance_map=instance_map,
            vertices_mm=vertices_mm,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(args.output_dir / "per_instance_metrics.csv", all_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summaries)
    paired_rows = make_paired_rows(all_rows, args.baseline_name, args.finetuned_name)
    write_csv(args.output_dir / "paired_instance_comparison.csv", paired_rows)
    paired = paired_summary(paired_rows)
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2))
    (args.output_dir / "paired_summary.json").write_text(json.dumps(paired, indent=2))

    print(f"Loaded GT for {len(gt_by_image)} images / {sum(len(v) for v in gt_by_image.values())} instances")
    for summary in summaries:
        print(
            f"{summary['method']}: evaluated={summary['evaluated_instances']} "
            f"t_med={summary.get('translation_error_mm_median', float('nan')):.1f}mm "
            f"R_med={summary.get('rotation_error_deg_median', float('nan')):.2f}deg "
            f"center_med={summary.get('center_error_px_median', float('nan')):.1f}px "
            f"t_scale={summary['translation_scale_to_mm']:g}"
        )
    print(
        f"paired: n={paired.get('paired_instances', 0)} "
        f"translation_improvement_med={paired.get('translation_error_mm_improvement_median', float('nan')):.1f}mm "
        f"rotation_improvement_med={paired.get('rotation_error_deg_improvement_median', float('nan')):.2f}deg "
        f"center_improvement_med={paired.get('center_error_px_improvement_median', float('nan')):.1f}px"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
