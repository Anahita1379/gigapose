"""Evaluate prediction CSVs against available EPnP anchors without changing poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from tracking.geometry import rotation_error_deg

from .run import _load_frame_map, _load_label, _pose_from_row, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--epnp-root", type=Path, required=True)
    parser.add_argument("--translation-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--minimum-epnp-label-weight", type=float, default=0.5)
    parser.add_argument("--maximum-epnp-center-error-px", type=float, default=25.0)
    parser.add_argument("--minimum-epnp-bbox-iou", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    center = np.asarray(trimesh.load(args.mesh, process=False).vertices).mean(axis=0)
    frame_map = _load_frame_map(args.dataset_dir)
    scale = 0.001 if args.translation_unit == "mm" else 1.0
    rows = []
    groups = {"all": ([], []), "quality_approved": ([], [])}
    with args.predictions.open(newline="") as handle:
        for source in csv.DictReader(handle):
            key = (int(source["scene_id"]), int(source["im_id"]))
            if key not in frame_map:
                continue
            label, status = _load_label(args.epnp_root, frame_map[key], args)
            if label is None:
                continue
            prediction = _pose_from_row(source, scale)
            reference = np.eye(4)
            reference[:3, :] = np.asarray(label["T_camera_object_centered"], dtype=float)[:3, :]
            centered_translation = prediction[:3, 3] + prediction[:3, :3] @ center
            translation = float(np.linalg.norm(centered_translation - reference[:3, 3]))
            rotation = rotation_error_deg(prediction, reference)
            approved = bool(label.get("_approved"))
            groups["all"][0].append(translation); groups["all"][1].append(rotation)
            if approved:
                groups["quality_approved"][0].append(translation)
                groups["quality_approved"][1].append(rotation)
            rows.append({
                "scene_id": key[0], "im_id": key[1], "score": source.get("score", ""),
                "epnp_status": status, "quality_approved": int(approved),
                "translation_error_m": translation, "rotation_error_deg": rotation,
                "front_rear_flip_like": int(rotation >= 135.0),
            })
    if rows:
        with (args.output_dir / "per_frame.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report = {"format": "orientation_anchor_evaluation_v1", "predictions": str(args.predictions)}
    for name, (translation, rotation) in groups.items():
        report[name] = {
            "translation_error_m": _summary(translation),
            "rotation_error_deg": _summary(rotation),
            "front_rear_flip_like_count": int(np.sum(np.asarray(rotation) >= 135.0)),
            "front_rear_flip_like_fraction": None if not rotation else float(np.mean(np.asarray(rotation) >= 135.0)),
        }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
