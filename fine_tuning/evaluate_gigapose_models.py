"""Evaluate and compare any number of GigaPose prediction CSVs.

Example:

python -m fine_tuning.evaluate_gigapose_models \
  --model original=path/to/originalMultiHypothesis.csv \
  --model finetune=path/to/finetuneMultiHypothesis.csv \
  --model finetune2=path/to/finetune2MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --rendered-iou \
  --output-dir fine_tuning/metrics/all_models
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning.compare_gigapose_predictions import (
    evaluate_method,
    load_gt,
    load_render_mesh,
    read_ply_vertices,
    summarize,
    write_csv,
)
from fine_tuning.ac_geometry import InstanceRenderer


ERROR_METRICS = [
    "translation_error_mm",
    "depth_error_mm",
    "rotation_error_deg",
    "center_error_px",
    "gt_bbox_center_error_px",
    "add_mm",
]
HIGHER_IS_BETTER = [
    "score",
    "pred_bbox_iou",
    "pred_mask_iou",
]
ALL_METRICS = ["score", *ERROR_METRICS, "pred_bbox_iou", "pred_mask_iou"]


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must be formatted as name=/path/to/predictions.csv"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Repeat as name=/path/to/predictions.csv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa_benchmark"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--max-mesh-points", type=int, default=5000)
    parser.add_argument(
        "--rendered-iou",
        action="store_true",
        help="Render predicted masks and report bbox/mask IoU. Slower but useful.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/metrics/all_models"),
    )
    return parser.parse_args()


def load_camera_map(dataset_dir: Path) -> dict[tuple[int, int], str]:
    frame_map_path = dataset_dir / "frame_map.json"
    if not frame_map_path.is_file():
        return {}
    rows = json.loads(frame_map_path.read_text())
    return {
        (int(row["scene_id"]), int(row["im_id"])): str(row["camera_id"])
        for row in rows
    }


def add_camera_ids(
    rows: list[dict[str, Any]],
    camera_map: dict[tuple[int, int], str],
) -> None:
    for row in rows:
        row["camera_id"] = camera_map.get(
            (int(row["scene_id"]), int(row["im_id"])),
            "unknown_camera",
        )


def summarize_by_camera(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row.get("camera_id", "unknown_camera")))].append(row)

    summaries = []
    for (method, camera_id), items in sorted(grouped.items()):
        summary = summarize(method, items, prediction_rows=len(items), t_scale=float("nan"))
        summary["camera_id"] = camera_id
        summaries.append(summary)
    return summaries


def finite_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def make_pairwise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_instance: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]), int(row["gt_index"]))
        by_instance[key][str(row["method"])] = row

    output = []
    for (scene_id, im_id, gt_index), methods in sorted(by_instance.items()):
        names = sorted(methods)
        for idx_a, name_a in enumerate(names):
            for name_b in names[idx_a + 1 :]:
                a = methods[name_a]
                b = methods[name_b]
                item: dict[str, Any] = {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "gt_index": gt_index,
                    "camera_id": a.get("camera_id", b.get("camera_id", "")),
                    "model_a": name_a,
                    "model_b": name_b,
                }
                for metric in ALL_METRICS:
                    av = finite_float(a.get(metric))
                    bv = finite_float(b.get(metric))
                    item[f"{name_a}_{metric}"] = av
                    item[f"{name_b}_{metric}"] = bv
                    if metric in HIGHER_IS_BETTER:
                        item[f"{metric}_winner"] = (
                            name_a if av > bv else name_b if bv > av else "tie"
                        )
                        item[f"{name_b}_minus_{name_a}_{metric}"] = bv - av
                    else:
                        item[f"{metric}_winner"] = (
                            name_a if av < bv else name_b if bv < av else "tie"
                        )
                        item[f"{name_a}_minus_{name_b}_{metric}"] = av - bv
                output.append(item)
    return output


def make_best_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_instance: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]), int(row["gt_index"]))
        by_instance[key].append(row)

    output = []
    for (scene_id, im_id, gt_index), items in sorted(by_instance.items()):
        item: dict[str, Any] = {
            "scene_id": scene_id,
            "im_id": im_id,
            "gt_index": gt_index,
            "camera_id": items[0].get("camera_id", ""),
            "model_count": len(items),
        }
        for metric in ALL_METRICS:
            candidates = [
                (str(row["method"]), finite_float(row.get(metric)))
                for row in items
            ]
            candidates = [(name, val) for name, val in candidates if np.isfinite(val)]
            if not candidates:
                continue
            if metric in HIGHER_IS_BETTER:
                best_name, best_value = max(candidates, key=lambda pair: pair[1])
            else:
                best_name, best_value = min(candidates, key=lambda pair: pair[1])
            item[f"best_{metric}_model"] = best_name
            item[f"best_{metric}"] = best_value
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.model) < 2:
        raise ValueError("Pass at least two --model entries to compare models.")

    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = read_ply_vertices(mesh_path, args.max_mesh_points)
    gt_by_image, K_by_image = load_gt(
        args.dataset_dir, args.split, load_masks=args.rendered_iou
    )
    camera_map = load_camera_map(args.dataset_dir)
    renderer = InstanceRenderer(load_render_mesh(mesh_path)) if args.rendered_iou else None

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for name, prediction_path in args.model:
            rows, summary = evaluate_method(
                name=name,
                prediction_path=prediction_path,
                gt_by_image=gt_by_image,
                K_by_image=K_by_image,
                instance_map={},
                vertices_mm=vertices_mm,
                renderer=renderer,
            )
            add_camera_ids(rows, camera_map)
            all_rows.extend(rows)
            summaries.append(summary)
    finally:
        if renderer is not None:
            renderer.close()

    camera_summaries = summarize_by_camera(all_rows)
    pairwise_rows = make_pairwise_rows(all_rows)
    best_rows = make_best_model_rows(all_rows)

    write_csv(args.output_dir / "all_instance_metrics.csv", all_rows)
    write_csv(args.output_dir / "overall_summary.csv", summaries)
    write_csv(args.output_dir / "camera_summary.csv", camera_summaries)
    write_csv(args.output_dir / "pairwise_instance_comparison.csv", pairwise_rows)
    write_csv(args.output_dir / "best_model_per_instance.csv", best_rows)
    (args.output_dir / "overall_summary.json").write_text(json.dumps(summaries, indent=2))
    (args.output_dir / "camera_summary.json").write_text(json.dumps(camera_summaries, indent=2))

    print(
        f"Evaluated {len(args.model)} models on {len(gt_by_image)} images / "
        f"{sum(len(v) for v in gt_by_image.values())} GT instances"
    )
    for summary in summaries:
        print(
            f"{summary['method']}: n={summary['evaluated_instances']} "
            f"t_med={summary.get('translation_error_mm_median', float('nan')):.1f}mm "
            f"R_med={summary.get('rotation_error_deg_median', float('nan')):.2f}deg "
            f"center_med={summary.get('center_error_px_median', float('nan')):.1f}px "
            f"bboxIoU_med={summary.get('pred_bbox_iou_median', float('nan')):.3f} "
            f"maskIoU_med={summary.get('pred_mask_iou_median', float('nan')):.3f}"
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
