"""Run a causal GRU-Kalman filter with component-wise learned GigaPose fallback."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.io import pose_to_csv_row, write_tracking_csv

from .dataset import MeasurementBundle
from .train import load_filter


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--max-correction-translation-m", type=float, default=5.0)
    value.add_argument("--max-correction-rotation-deg", type=float, default=60.0)
    value.add_argument("--minimum-translation-trust", type=float, default=0.5)
    value.add_argument("--minimum-rotation-trust", type=float, default=0.5)
    value.add_argument("--overwrite", action="store_true")
    return value


def _summary(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": len(array), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    args = parser().parse_args()
    if min(args.max_correction_translation_m, args.max_correction_rotation_deg) < 0:
        raise ValueError("Correction limits cannot be negative")
    if not 0 <= args.minimum_translation_trust <= 1:
        raise ValueError("--minimum-translation-trust must be in [0, 1]")
    if not 0 <= args.minimum_rotation_trust <= 1:
        raise ValueError("--minimum-rotation-trust must be in [0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = MeasurementBundle(args.data)
    model, checkpoint = load_filter(args.checkpoint, args.device)
    if checkpoint["context_feature_names"] != bundle.manifest["context_feature_names"]:
        raise ValueError("Checkpoint and measurement context schemas differ")
    arrays = bundle.arrays
    filtered = np.repeat(np.eye(4, dtype=np.float32)[None], len(bundle), axis=0)
    gain_values = np.zeros((len(bundle), 12), dtype=np.float32)
    trust_probabilities = np.ones((len(bundle), 2), dtype=np.float32)
    with torch.no_grad():
        for segment_id in np.unique(arrays["segment_id"]):
            indices = np.flatnonzero(arrays["segment_id"] == segment_id)
            measurement = torch.from_numpy(arrays["measurement_pose"][indices])[None].to(args.device)
            context = torch.from_numpy(arrays["context_features"][indices])[None].to(args.device)
            delta_time = torch.from_numpy(arrays["delta_time_s"][indices].astype(np.float32))[None].to(args.device)
            delta_time[:, 0] = 0.0
            valid = torch.ones(1, len(indices), dtype=torch.bool, device=args.device)
            baseline = torch.from_numpy(arrays["baseline_pose"][indices])[None].to(args.device)
            model_output = model(
                measurement,
                context,
                delta_time,
                valid,
                baseline_pose=baseline if model.use_fallback_heads else None,
            )
            poses, gains = model_output[:2]
            filtered[indices] = poses[0].cpu().numpy()
            gain_values[indices] = gains[0].cpu().numpy()
            if model.use_fallback_heads:
                trust_probabilities[indices] = torch.sigmoid(
                    model_output[2][0]
                ).cpu().numpy()

    output_rows, diagnostics = [], []
    raw_t, raw_r, final_t, final_r = [], [], [], []
    fallback_count = 0
    translation_fallback_count = 0
    rotation_fallback_count = 0
    by_camera = defaultdict(lambda: {"raw_t": [], "raw_r": [], "final_t": [], "final_r": []})
    for index in range(len(bundle)):
        measurement = arrays["measurement_pose"][index]
        baseline = arrays["baseline_pose"][index]
        candidate = filtered[index]
        correction_t = translation_error_m(candidate, measurement)
        correction_r = rotation_error_deg(candidate, measurement)
        finite = bool(np.isfinite(candidate).all())
        if model.use_fallback_heads:
            translation_safety_fallback = (
                not finite or correction_t > args.max_correction_translation_m
            )
            rotation_safety_fallback = (
                not finite or correction_r > args.max_correction_rotation_deg
            )
            translation_learned_fallback = (
                not translation_safety_fallback
                and trust_probabilities[index, 0] < args.minimum_translation_trust
            )
            rotation_learned_fallback = (
                not rotation_safety_fallback
                and trust_probabilities[index, 1] < args.minimum_rotation_trust
            )
            translation_fallback = (
                translation_safety_fallback or translation_learned_fallback
            )
            rotation_fallback = rotation_safety_fallback or rotation_learned_fallback
            pose = candidate.copy()
            # Preserve the previous cascade's correction guard: an extreme
            # learned correction returns to the selector measurement. Only a
            # low learned trust score selects original GigaPose.
            if translation_safety_fallback:
                pose[:3, 3] = measurement[:3, 3]
            elif translation_learned_fallback:
                pose[:3, 3] = baseline[:3, 3]
            if rotation_safety_fallback:
                pose[:3, :3] = measurement[:3, :3]
            elif rotation_learned_fallback:
                pose[:3, :3] = baseline[:3, :3]
            fallback = translation_fallback or rotation_fallback
        else:
            fallback = (
                not finite
                or correction_t > args.max_correction_translation_m
                or correction_r > args.max_correction_rotation_deg
            )
            translation_fallback = fallback
            rotation_fallback = fallback
            translation_safety_fallback = fallback
            rotation_safety_fallback = fallback
            translation_learned_fallback = False
            rotation_learned_fallback = False
            pose = measurement if fallback else candidate
        fallback_count += int(fallback)
        translation_fallback_count += int(translation_fallback)
        rotation_fallback_count += int(rotation_fallback)
        gt = arrays["ground_truth_pose"][index]
        rt = translation_error_m(measurement, gt)
        rr = rotation_error_deg(measurement, gt)
        candidate_t = translation_error_m(candidate, gt)
        candidate_r = rotation_error_deg(candidate, gt)
        baseline_t = translation_error_m(baseline, gt)
        baseline_r = rotation_error_deg(baseline, gt)
        ft = translation_error_m(pose, gt)
        fr = rotation_error_deg(pose, gt)
        raw_t.append(rt); raw_r.append(rr); final_t.append(ft); final_r.append(fr)
        camera = str(arrays["camera_id"][index])
        camera_values = by_camera[camera]
        camera_values["raw_t"].append(rt); camera_values["raw_r"].append(rr)
        camera_values["final_t"].append(ft); camera_values["final_r"].append(fr)
        output_rows.append(pose_to_csv_row(
            scene_id=int(arrays["scene_id"][index]),
            im_id=int(arrays["im_id"][index]),
            track_id=0,
            obj_id=1,
            confidence=float(arrays["measurement_score"][index]),
            pose_m=pose,
            mode=(
                "gigapose_translation_rotation_fallback"
                if translation_learned_fallback and rotation_learned_fallback
                else "gigapose_translation_fallback"
                if translation_learned_fallback
                else "gigapose_rotation_fallback"
                if rotation_learned_fallback
                else "measurement_safety_fallback"
                if translation_safety_fallback or rotation_safety_fallback
                else "filtered"
            ),
            source=(
                "learned_component_gigapose_fallback"
                if model.use_fallback_heads and (
                    translation_learned_fallback or rotation_learned_fallback
                )
                else "input_measurement_fallback"
                if fallback
                else "gru_kalman_translation_filter"
                if model.preserve_measurement_rotation
                else "gru_kalman_filter"
            ),
            elapsed_s=0.0,
        ))
        diagnostics.append({
            "scene_id": int(arrays["scene_id"][index]),
            "im_id": int(arrays["im_id"][index]),
            "source_run": str(arrays["source_run"][index]),
            "camera_id": camera,
            "source_frame": int(arrays["source_frame"][index]),
            "segment_id": int(arrays["segment_id"][index]),
            "sequence_reset": int(index == 0 or arrays["segment_id"][index] != arrays["segment_id"][index - 1]),
            "fallback": int(fallback),
            "translation_fallback": int(translation_fallback),
            "rotation_fallback": int(rotation_fallback),
            "translation_learned_gigapose_fallback": int(translation_learned_fallback),
            "rotation_learned_gigapose_fallback": int(rotation_learned_fallback),
            "translation_safety_measurement_fallback": int(translation_safety_fallback),
            "rotation_safety_measurement_fallback": int(rotation_safety_fallback),
            "translation_trust_probability": float(trust_probabilities[index, 0]),
            "rotation_trust_probability": float(trust_probabilities[index, 1]),
            "correction_translation_m": correction_t,
            "correction_rotation_deg": correction_r,
            "raw_translation_error_m": rt,
            "candidate_translation_error_m": candidate_t,
            "baseline_translation_error_m": baseline_t,
            "filtered_translation_error_m": ft,
            "raw_rotation_error_deg": rr,
            "candidate_rotation_error_deg": candidate_r,
            "baseline_rotation_error_deg": baseline_r,
            "filtered_rotation_error_deg": fr,
            "mean_pose_gain": float(gain_values[index, :6].mean()),
            "mean_velocity_gain": float(gain_values[index, 6:].mean()),
        })
    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    with (args.output_dir / "filter_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader(); writer.writerows(diagnostics)
    report = {
        "format": "rgb_self_recovery_gru_kalman_inference_v1",
        "frame_count": len(bundle),
        "segment_count": len(set(arrays["segment_id"].tolist())),
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / len(bundle),
        "translation_fallback_count": translation_fallback_count,
        "translation_fallback_fraction": translation_fallback_count / len(bundle),
        "rotation_fallback_count": rotation_fallback_count,
        "rotation_fallback_fraction": rotation_fallback_count / len(bundle),
        "use_fallback_heads": model.use_fallback_heads,
        "preserve_measurement_rotation": model.preserve_measurement_rotation,
        "raw_translation_error_m": _summary(raw_t),
        "filtered_translation_error_m": _summary(final_t),
        "raw_rotation_error_deg": _summary(raw_r),
        "filtered_rotation_error_deg": _summary(final_r),
        "by_camera": {
            camera: {
                "raw_translation_error_m": _summary(values["raw_t"]),
                "filtered_translation_error_m": _summary(values["final_t"]),
                "raw_rotation_error_deg": _summary(values["raw_r"]),
                "filtered_rotation_error_deg": _summary(values["final_r"]),
            }
            for camera, values in sorted(by_camera.items())
        },
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
