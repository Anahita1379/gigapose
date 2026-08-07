"""Paired CNN/DINO pose evaluation with distance plots and CAD overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.geometry import pose_from_rt, rotation_error_deg, translation_error_m
from tracking.rendering import CADRenderer


COLORS = [
    (235, 65, 55),
    (50, 110, 240),
    (245, 165, 35),
    (170, 65, 210),
    (20, 190, 190),
]
GT_COLOR = (40, 210, 90)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--model", action="append", required=True,
        help="Repeat NAME=tracked_predictions.csv (a result directory is also accepted).",
    )
    p.add_argument("--baseline", required=True, help="NAME from --model")
    p.add_argument("--mesh", type=Path)
    p.add_argument("--mesh-scale", type=float, default=1.0)
    p.add_argument("--center-mesh", action="store_true")
    p.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    p.add_argument("--distance-bins-m", default="0,20,40,60,80,100,120")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--max-overlays", type=int, default=120)
    p.add_argument(
        "--overlay-frame", action="append", default=[], metavar="SCENE_ID:IM_ID",
        help="Render a specific frame. Repeat to select multiple overlays.",
    )
    p.add_argument("--overlay-opacity", type=float, default=0.30)
    p.add_argument("--overwrite", action="store_true")
    return p


def _models(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model must use NAME=PATH: {value}")
        name, path_text = value.split("=", 1)
        name, path = name.strip(), Path(path_text)
        if not name or name in result:
            raise ValueError(f"Empty or duplicate model name: {name!r}")
        if path.is_dir():
            path = path / "tracked_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def _pose(row, unit):
    rotation = np.fromstring(row["R"], sep=" ", dtype=float)
    translation = np.fromstring(row["t"], sep=" ", dtype=float)
    if rotation.size != 9 or translation.size != 3:
        raise ValueError("Prediction R/t must contain 9/3 values")
    if unit == "mm":
        translation *= 0.001
    return pose_from_rt(rotation.reshape(3, 3), translation)


def load_predictions(path, unit):
    """Keep the highest-confidence row for each frame (the one-car protocol)."""
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = int(row["scene_id"]), int(row["im_id"])
            score = float(row.get("score", 0.0))
            if key not in result or score > result[key]["score"]:
                result[key] = {
                    "pose": _pose(row, unit),
                    "score": score,
                    "track_id": row.get("track_id", row.get("instance_id", "")),
                    "source": row.get("source", ""),
                }
    return result


def _gt_pose(value):
    return pose_from_rt(
        np.asarray(value["cam_R_m2c"], dtype=float).reshape(3, 3),
        np.asarray(value["cam_t_m2c"], dtype=float) * 0.001,
    )


def add_error_m(prediction_pose, ground_truth_pose, mesh_vertices_m):
    """Average distance of corresponding CAD vertices (ADD), in metres."""
    vertices = np.asarray(mesh_vertices_m, dtype=np.float64)
    predicted = vertices @ prediction_pose[:3, :3].T + prediction_pose[:3, 3]
    truth = vertices @ ground_truth_pose[:3, :3].T + ground_truth_pose[:3, 3]
    return float(np.linalg.norm(predicted - truth, axis=1).mean())


def _mesh_vertices(path, mesh_scale, center_mesh):
    try:
        import trimesh
    except ImportError:
        trimesh = None
    if trimesh is not None:
        mesh = trimesh.load(Path(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Could not load a non-empty mesh from {path}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    else:
        # ADD only needs vertices. Keep evaluation usable in lightweight
        # environments where the optional trimesh package is unavailable.
        type_map = {
            "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
            "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
            "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        with Path(path).open("rb") as handle:
            header_lines = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Incomplete PLY header: {path}")
                decoded = line.decode("ascii").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break
            if not header_lines or header_lines[0] != "ply":
                raise ValueError(f"Not a PLY mesh: {path}")
            format_line = next(
                (line for line in header_lines if line.startswith("format ")), None
            )
            vertex_line = next(
                (line for line in header_lines if line.startswith("element vertex ")), None
            )
            if format_line is None or vertex_line is None:
                raise ValueError(f"PLY lacks format or vertex count: {path}")
            vertex_count = int(vertex_line.split()[2])
            start = header_lines.index(vertex_line) + 1
            properties = []
            for line in header_lines[start:]:
                if line.startswith("element ") or line == "end_header":
                    break
                fields = line.split()
                if len(fields) != 3 or fields[0] != "property" or fields[1] not in type_map:
                    raise ValueError(f"Unsupported PLY vertex property: {line}")
                properties.append((fields[2], type_map[fields[1]]))
            names = [name for name, _dtype in properties]
            if not {"x", "y", "z"}.issubset(names):
                raise ValueError(f"PLY has no XYZ vertices: {path}")
            if "binary_little_endian" in format_line:
                records = np.fromfile(handle, dtype=np.dtype(properties), count=vertex_count)
                vertices = np.column_stack(
                    [records[axis] for axis in ("x", "y", "z")]
                ).astype(np.float64)
            elif "ascii" in format_line:
                rows = [handle.readline().split() for _ in range(vertex_count)]
                xyz = [names.index(axis) for axis in ("x", "y", "z")]
                vertices = np.asarray(
                    [[float(row[index]) for index in xyz] for row in rows],
                    dtype=np.float64,
                )
            else:
                raise ValueError(f"Unsupported PLY format: {format_line}")
    vertices *= float(mesh_scale)
    if center_mesh:
        vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return vertices


def collect_records(
    dataset_dir, split, predictions, max_frames=None, mesh_vertices_m=None
):
    records, skipped = [], []
    for index, (key, image, K, _depth, gt, _extra) in enumerate(
        iter_gt_frames(dataset_dir, split, load_depth=False)
    ):
        if max_frames is not None and index >= max_frames:
            break
        scene_id, im_id = (int(value) for value in key.split("_"))
        if len(gt) != 1:
            skipped.append({
                "scene_id": scene_id,
                "im_id": im_id,
                "reason": f"expected one GT instance, found {len(gt)}",
            })
            continue
        truth = _gt_pose(gt[0])
        row = {
            "scene_id": scene_id,
            "im_id": im_id,
            "key": key,
            "image": image,
            "K": K,
            "gt_pose": truth,
            "distance_m": float(truth[2, 3]),
            "models": {},
        }
        for name, values in predictions.items():
            prediction = values.get((scene_id, im_id))
            if prediction is None:
                continue
            pose = prediction["pose"]
            row["models"][name] = {
                **prediction,
                "translation_error_m": translation_error_m(pose, truth),
                "rotation_error_deg": rotation_error_deg(pose, truth),
                **({
                    "add_m": add_error_m(pose, truth, mesh_vertices_m)
                } if mesh_vertices_m is not None else {}),
            }
        records.append(row)
    return records, skipped


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "count": 0, "mean": None, "rmse": None,
            "median": None, "p90": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize(records, names, bins):
    output = {}
    for name in names:
        rows = [row["models"][name] for row in records if name in row["models"]]
        result = {
            "prediction_count": len(rows),
            "translation_error_m": _stats(
                [row["translation_error_m"] for row in rows]
            ),
            "rotation_error_deg": _stats(
                [row["rotation_error_deg"] for row in rows]
            ),
            "add_m": _stats([row["add_m"] for row in rows if "add_m" in row]),
            "recall_translation_le_1m": (
                None if not rows else float(np.mean([
                    row["translation_error_m"] <= 1.0 for row in rows
                ]))
            ),
            "recall_rotation_le_10deg": (
                None if not rows else float(np.mean([
                    row["rotation_error_deg"] <= 10.0 for row in rows
                ]))
            ),
            "distance_bins": [],
        }
        for low, high in zip(bins[:-1], bins[1:]):
            selected = [
                row["models"][name] for row in records
                if name in row["models"] and low <= row["distance_m"] < high
            ]
            result["distance_bins"].append({
                "low_m": low,
                "high_m": high,
                "translation_error_m": _stats(
                    [row["translation_error_m"] for row in selected]
                ),
                "rotation_error_deg": _stats(
                    [row["rotation_error_deg"] for row in selected]
                ),
                "add_m": _stats([
                    row["add_m"] for row in selected if "add_m" in row
                ]),
            })
        output[name] = result
    return output


def paired_comparisons(records, baseline, names):
    output = {}
    for name in names:
        if name == baseline:
            continue
        pairs = [
            (row["models"][baseline], row["models"][name])
            for row in records
            if baseline in row["models"] and name in row["models"]
        ]
        translation = np.asarray([
            first["translation_error_m"] - second["translation_error_m"]
            for first, second in pairs
        ])
        rotation = np.asarray([
            first["rotation_error_deg"] - second["rotation_error_deg"]
            for first, second in pairs
        ])
        add = np.asarray([
            first["add_m"] - second["add_m"]
            for first, second in pairs
            if "add_m" in first and "add_m" in second
        ])
        output[name] = {
            "baseline": baseline,
            "paired_count": len(pairs),
            "translation_improvement_m": _stats(translation),
            "rotation_improvement_deg": _stats(rotation),
            "add_improvement_m": _stats(add),
            "translation_better_fraction": (
                None if not len(pairs) else float(np.mean(translation > 0))
            ),
            "rotation_better_fraction": (
                None if not len(pairs) else float(np.mean(rotation > 0))
            ),
            "add_better_fraction": (
                None if not len(add) else float(np.mean(add > 0))
            ),
            "both_better_fraction": (
                None if not len(pairs) else float(np.mean((translation > 0) & (rotation > 0)))
            ),
        }
    return output


def _plot(summary, names, bins, metric, ylabel, path):
    labels = [f"{low:g}–{high:g}" for low, high in zip(bins[:-1], bins[1:])]
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / len(names)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for index, name in enumerate(names):
        values = [
            item[metric]["mean"] for item in summary[name]["distance_bins"]
        ]
        values = [np.nan if value is None else value for value in values]
        axis.bar(x + (index - (len(names) - 1) / 2) * width, values, width, label=name)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Ground-truth camera depth (m)")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _blend(canvas, mask, color, opacity):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    canvas[mask] = np.clip(
        (1.0 - opacity) * canvas[mask] + opacity * np.asarray(color), 0, 255
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)


def _overlay_records(
    records, names, renderer, output_dir, maximum, opacity, requested_keys=(),
):
    available = [
        row for row in records if all(name in row["models"] for name in names)
    ]
    if maximum <= 0 or not available:
        return 0
    if requested_keys:
        requested_keys = set(requested_keys)
        selected = [row for row in available if row["key"] in requested_keys]
        missing = requested_keys - {row["key"] for row in selected}
        if missing:
            raise ValueError(
                "Requested overlay frames are unavailable in every model: "
                + ", ".join(sorted(missing))
            )
    else:
        available.sort(key=lambda row: row["distance_m"])
        indices = np.unique(
            np.linspace(0, len(available) - 1, min(maximum, len(available)), dtype=int)
        )
        selected = [available[int(index)] for index in indices]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        canvas = np.asarray(row["image"], dtype=np.uint8).copy()
        gt_mask, _ = renderer.render(row["gt_pose"], row["K"], canvas.shape[:2])
        _blend(canvas, gt_mask, GT_COLOR, opacity)
        lines = [f"GT green | depth={row['distance_m']:.1f}m"]
        for index, name in enumerate(names):
            value = row["models"][name]
            color = COLORS[index % len(COLORS)]
            mask, _ = renderer.render(value["pose"], row["K"], canvas.shape[:2])
            _blend(canvas, mask, color, opacity)
            lines.append(
                f"{name}: t={value['translation_error_m']:.2f}m "
                f"r={value['rotation_error_deg']:.1f}deg "
                f"ADD={value.get('add_m', float('nan')):.2f}m"
            )
        line_height = 24
        cv2.rectangle(
            canvas, (0, 0), (min(canvas.shape[1], 900), line_height * len(lines) + 8),
            (0, 0, 0), -1,
        )
        for index, text in enumerate(lines):
            color = GT_COLOR if index == 0 else COLORS[(index - 1) % len(COLORS)]
            cv2.putText(
                canvas, text, (10, 22 + line_height * index),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
            )
        Image.fromarray(canvas).save(
            overlay_dir / f"{row['key']}_depth{row['distance_m']:.1f}m.jpg",
            quality=92,
        )
    return len(selected)


def _write_rows(records, names, path):
    fields = ["scene_id", "im_id", "distance_m"]
    for name in names:
        fields.extend([
            f"{name}_available",
            f"{name}_translation_error_m",
            f"{name}_rotation_error_deg",
            f"{name}_add_m",
            f"{name}_score",
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "scene_id": record["scene_id"],
                "im_id": record["im_id"],
                "distance_m": record["distance_m"],
            }
            for name in names:
                value = record["models"].get(name)
                row[f"{name}_available"] = int(value is not None)
                if value is not None:
                    row[f"{name}_translation_error_m"] = value["translation_error_m"]
                    row[f"{name}_rotation_error_deg"] = value["rotation_error_deg"]
                    row[f"{name}_add_m"] = value.get("add_m", "")
                    row[f"{name}_score"] = value["score"]
            writer.writerow(row)


def main():
    args = parser().parse_args()
    models = _models(args.model)
    if args.baseline not in models:
        raise ValueError(f"Baseline {args.baseline!r} is not in {list(models)}")
    bins = [float(value) for value in args.distance_bins_m.split(",")]
    if len(bins) < 2 or any(a >= b for a, b in zip(bins[:-1], bins[1:])):
        raise ValueError("Distance bins must be strictly increasing")
    if not 0 <= args.overlay_opacity <= 1:
        raise ValueError("--overlay-opacity must be in [0,1]")
    overlay_keys = []
    for value in args.overlay_frame:
        try:
            scene_id, im_id = (int(item) for item in value.split(":"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid --overlay-frame {value!r}; expected SCENE_ID:IM_ID"
            ) from None
        overlay_keys.append(f"{scene_id:06d}_{im_id:06d}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    if not mesh.is_file():
        raise FileNotFoundError(f"Pass --mesh; default CAD is absent: {mesh}")
    vertices = _mesh_vertices(mesh, args.mesh_scale, args.center_mesh)
    predictions = {
        name: load_predictions(path, args.prediction_translation_unit)
        for name, path in models.items()
    }
    records, skipped = collect_records(
        args.dataset_dir, args.split, predictions, args.max_frames, vertices
    )
    names = list(models)
    summary = summarize(records, names, bins)
    paired = paired_comparisons(records, args.baseline, names)
    _write_rows(records, names, args.output_dir / "per_frame_metrics.csv")
    _plot(
        summary, names, bins, "translation_error_m", "Mean translation error (m)",
        args.output_dir / "translation_error_by_distance.png",
    )
    _plot(
        summary, names, bins, "rotation_error_deg", "Mean rotation error (deg)",
        args.output_dir / "rotation_error_by_distance.png",
    )
    _plot(
        summary, names, bins, "add_m", "Mean ADD (m)",
        args.output_dir / "add_by_distance.png",
    )
    overlays = 0
    if args.max_overlays > 0:
        renderer = CADRenderer(
            mesh, mesh_scale=args.mesh_scale, center_mesh=args.center_mesh
        )
        try:
            overlays = _overlay_records(
                records, names, renderer, args.output_dir,
                args.max_overlays, args.overlay_opacity, overlay_keys,
            )
        finally:
            renderer.close()
    report = {
        "format": "rgb_self_recovery_backbone_evaluation_v2",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "baseline": args.baseline,
        "models": {name: str(path) for name, path in models.items()},
        "gt_frame_count": len(records),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "overlays_written": overlays,
        "distance_bins_m": bins,
        "metrics": {
            "translation_error_m": "Euclidean object-origin translation error",
            "rotation_error_deg": "SO(3) geodesic rotation error",
            "add_m": "Mean corresponding CAD-vertex distance",
            "rmse": "sqrt(mean(per-frame metric^2))",
        },
        "mesh": str(mesh),
        "mesh_scale": args.mesh_scale,
        "center_mesh": args.center_mesh,
        "add_vertex_count": int(len(vertices)),
        "summary": summary,
        "paired_vs_baseline": paired,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
