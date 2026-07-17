"""Select real-world labels using an optimized per-frame camera extrinsic.

This is an isolated alternative to ``select_real_label_candidates.py`` for an
extrinsic produced by ``optimize_camera_map_extrinsics.py`` with
``--use-epnp-label-extrinsics``.

For each EPnP label, let

    M = T_map_object
    E = T_camera_object
    C = M @ inv(E)

where ``C`` is the original camera-to-map transform derived from that label.
If the optimizer produced the left correction ``D``, this script uses

    G_aligned = G @ X_right
    C_opt = D @ C
    M_gigapose_opt = C_opt @ G_aligned

and compares ``M_gigapose_opt`` with ``M``, where ``G`` is the raw GigaPose
camera-object prediction and ``X_right`` is the object/CAD-frame alignment
estimated or loaded exactly like the original selector. Equivalently, it
compares ``G_aligned`` with the corrected camera-frame EPnP pose
``inv(C_opt) @ M``.

The frame transform is independent of the optimized-extrinsics file. Existing
extrinsics can therefore be reused without rerunning the optimizer.

Typical use::

    python -m fine_tuning.select_real_label_candidates_with_extrinsics \
      --gigapose-predictions /path/to/MultiHypothesis.csv \
      --dataset-dir /path/to/prepared_dataset \
      --epnp-root /path/to/EPnPv2_gt_mesh_z_hybrid_labels \
      --epnp-glob "*.json" \
      --epnp-strip-trailing-instance-id \
      --epnp-key-prefix image_ \
      --match-key image_stem \
      --optimized-extrinsics /path/to/optimized_extrinsics.json \
      --frame-transform-side right \
      --frame-transform-refine-iterations 5 \
      --epnp-map-pose-key T_map_object_raw \
      --epnp-camera-pose-key T_camera_object_centered \
      --epnp-translation-unit m \
      --output-dir /path/to/new_label_candidates
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning import select_real_label_candidates as base
from fine_tuning.optimize_camera_map_extrinsics import (
    matrix_3x4_or_4x4_to_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select GigaPose/EPnP pairs after applying an optimized EPnP-label "
            "camera-extrinsic correction."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/media/hdd2/ARCL_multicar_bags/camera_dataset"),
    )
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Date prefix to include. Repeatable.",
    )
    parser.add_argument("--gigapose-predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--epnp-root", type=Path, default=None)
    parser.add_argument("--epnp-glob", default="**/*")
    parser.add_argument("--epnp-key-field", default=None)
    parser.add_argument("--epnp-key-prefix", default="")
    parser.add_argument("--epnp-strip-trailing-instance-id", action="store_true")
    parser.add_argument(
        "--match-key",
        choices=(
            "auto",
            "scene_im",
            "image_stem",
            "image_name",
            "frame_id",
            "sim_time_ms",
        ),
        default="auto",
    )
    parser.add_argument(
        "--optimized-extrinsics",
        type=Path,
        required=True,
        help=(
            "optimized_extrinsics.json, or its containing optimization directory. "
            "It must come from --use-epnp-label-extrinsics mode."
        ),
    )
    parser.add_argument(
        "--frame-transform-json",
        type=Path,
        default=None,
        help=(
            "Optional precomputed GigaPose-to-EPnP frame transform. If omitted, "
            "estimate it from one-to-one prediction/label matches exactly like "
            "select_real_label_candidates."
        ),
    )
    parser.add_argument(
        "--epnp-map-pose-key",
        default="T_map_object_raw",
        help="Map-object pose field in every EPnP label.",
    )
    parser.add_argument(
        "--epnp-camera-pose-key",
        default="T_camera_object_centered",
        help="Camera-object pose field in every EPnP label.",
    )
    parser.add_argument(
        "--epnp-map-pose-unit",
        choices=("auto", "m", "mm"),
        default="auto",
    )
    parser.add_argument(
        "--epnp-camera-pose-unit",
        choices=("auto", "m", "mm"),
        default="auto",
    )
    parser.add_argument(
        "--epnp-translation-unit",
        choices=("auto", "m", "mm"),
        default=None,
        help=(
            "Compatibility alias that sets both EPnP pose units. Prefer the "
            "separate --epnp-map-pose-unit and --epnp-camera-pose-unit options."
        ),
    )
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-translation-error-mm", type=float, default=2000.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument("--max-candidates-per-key", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "gigaPose_datasets/results/real_world_label_candidates_with_extrinsics"
        ),
    )

    parser.add_argument(
        "--frame-transform-side",
        choices=("left", "right"),
        default="left",
        help=(
            "How to align GigaPose with EPnP. Use 'right' for "
            "T_aligned = T_gigapose @ X."
        ),
    )
    parser.add_argument(
        "--frame-transform-refine-iterations",
        type=int,
        default=0,
        help="Re-estimate the frame transform from inlier one-to-one pairs.",
    )
    parser.add_argument(
        "--frame-transform-inlier-translation-mm",
        type=float,
        default=None,
        help="Translation inlier threshold used during transform refinement.",
    )
    parser.add_argument(
        "--frame-transform-inlier-rotation-deg",
        type=float,
        default=None,
        help="Rotation inlier threshold used during transform refinement.",
    )
    return parser.parse_args()


def resolve_extrinsics_path(path: Path) -> Path:
    resolved = path / "optimized_extrinsics.json" if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find optimized extrinsics: {resolved}")
    return resolved


def rigid_transform_checks(T: np.ndarray, name: str) -> None:
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} is not a finite 4x4 transform")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row: {T[3]}")
    determinant = float(np.linalg.det(T[:3, :3]))
    if not np.isclose(determinant, 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation determinant is {determinant}, expected 1")


def load_extrinsic_correction(path: Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    resolved = resolve_extrinsics_path(path)
    data = json.loads(resolved.read_text())
    mode = str(data.get("optimization_mode", ""))
    key = "T_epnp_label_camera_correction_left_multiply"
    value = data.get(key)
    if value is None:
        raise ValueError(
            f"{resolved} does not contain {key}. Its optimization_mode is "
            f"{mode!r}; rerun optimize_camera_map_extrinsics with "
            "--use-epnp-label-extrinsics, or use the selector appropriate for "
            "that optimization mode."
        )
    if mode and mode != "epnp_label_camera_extrinsics":
        raise ValueError(
            f"{resolved} has optimization_mode={mode!r}, but this selector "
            "requires 'epnp_label_camera_extrinsics'."
        )
    correction = np.asarray(value, dtype=float).reshape(4, 4)
    rigid_transform_checks(correction, key)
    return correction, data, resolved


def records_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsn"):
        return base.load_json_records(path)
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def load_epnp_extrinsic_labels(
    roots: list[Path],
    glob_pattern: str,
    key_field: str | None,
    key_prefix: str,
    strip_trailing_instance_id: bool,
    map_pose_key: str,
    camera_pose_key: str,
    map_pose_unit: str,
    camera_pose_unit: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    for root in roots:
        for path in sorted(root.glob(glob_pattern)):
            if not path.is_file():
                continue
            try:
                records = records_from_path(path)
            except Exception as exc:
                skipped["unreadable_file"] += 1
                print(f"Skipping unreadable EPnP label file {path}: {exc}")
                continue
            if not records:
                skipped["unsupported_or_empty_file"] += 1
                continue

            for record_index, record in enumerate(records):
                try:
                    if map_pose_key not in record:
                        raise KeyError(map_pose_key)
                    if camera_pose_key not in record:
                        raise KeyError(camera_pose_key)
                    T_map_object = matrix_3x4_or_4x4_to_transform(
                        record[map_pose_key], map_pose_unit
                    )
                    T_camera_object = matrix_3x4_or_4x4_to_transform(
                        record[camera_pose_key], camera_pose_unit
                    )
                    rigid_transform_checks(T_map_object, map_pose_key)
                    rigid_transform_checks(T_camera_object, camera_pose_key)
                except KeyError as exc:
                    skipped[f"missing_pose_key:{exc.args[0]}"] += 1
                    continue
                except Exception as exc:
                    skipped["malformed_pose"] += 1
                    if skipped["malformed_pose"] <= 5:
                        print(f"Skipping malformed pose in {path}: {exc}")
                    continue

                key = base.row_key_from_mapping(record, key_field)
                if key is None:
                    key = base.normalize_key(path.stem)
                key = base.adjust_epnp_key(
                    key, key_prefix, strip_trailing_instance_id
                )
                T_map_camera_initial = T_map_object @ np.linalg.inv(
                    T_camera_object
                )
                labels.append(
                    {
                        "epnp_label_path": str(path),
                        "epnp_record_index": record_index,
                        "match_key": key,
                        "T_map_object": T_map_object,
                        "T_camera_object_original": T_camera_object,
                        "T_epnp": T_camera_object,
                        "T_map_camera_initial": T_map_camera_initial,
                    }
                )
    return labels, dict(skipped)


def pose_errors(T_pred: np.ndarray, T_target: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(T_pred[:3, 3] - T_target[:3, 3])),
        base.rotation_error_deg(T_pred[:3, :3], T_target[:3, :3]),
    )


def make_candidate_rows(
    preds_by_key: dict[str, list[dict[str, Any]]],
    epnp_by_key: dict[str, list[dict[str, Any]]],
    correction: np.ndarray,
    gigapose_frame_transform: np.ndarray | None,
    gigapose_frame_transform_side: str | None,
    max_candidates_per_key: int,
    extrinsics_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared_keys = sorted(set(preds_by_key) & set(epnp_by_key))
    for key in shared_keys:
        predictions = sorted(
            preds_by_key[key],
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )[:max_candidates_per_key]
        labels = epnp_by_key[key][:max_candidates_per_key]

        for label in labels:
            T_map_object = label["T_map_object"]
            T_camera_object_original = label["T_camera_object_original"]
            T_map_camera_initial = label["T_map_camera_initial"]
            T_map_camera_optimized = correction @ T_map_camera_initial
            T_camera_object_optimized = (
                np.linalg.inv(T_map_camera_optimized) @ T_map_object
            )

            for pred in predictions:
                T_gigapose_raw = pred["T_gigapose"]
                T_gigapose = (
                    base.apply_frame_transform(
                        T_gigapose_raw,
                        gigapose_frame_transform,
                        str(gigapose_frame_transform_side),
                    )
                    if gigapose_frame_transform is not None
                    else T_gigapose_raw
                )
                T_gigapose_map_optimized = (
                    T_map_camera_optimized @ T_gigapose
                )
                raw_t, raw_r = pose_errors(
                    T_gigapose_raw, T_camera_object_original
                )
                original_t, original_r = pose_errors(
                    T_gigapose, T_camera_object_original
                )
                optimized_t, optimized_r = pose_errors(
                    T_gigapose_map_optimized, T_map_object
                )
                camera_t, camera_r = pose_errors(
                    T_gigapose, T_camera_object_optimized
                )
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
                        # These camera-frame errors are used for ranking and
                        # selection and are exactly what the unchanged
                        # visualization script draws.
                        "translation_error_mm": camera_t,
                        "rotation_error_deg": camera_r,
                        # These expose the before/after comparison explicitly.
                        "raw_translation_error_mm": raw_t,
                        "raw_rotation_error_deg": raw_r,
                        "original_translation_error_mm": original_t,
                        "original_rotation_error_deg": original_r,
                        "optimized_camera_translation_error_mm": camera_t,
                        "optimized_camera_rotation_error_deg": camera_r,
                        "map_camera_translation_disagreement_mm": abs(
                            optimized_t - camera_t
                        ),
                        "map_camera_rotation_disagreement_deg": abs(
                            optimized_r - camera_r
                        ),
                        "T_gigapose_cam_obj": base.matrix_to_text(
                            T_gigapose_raw
                        ),
                        "T_gigapose_aligned_epnp_obj": base.matrix_to_text(
                            T_gigapose
                        ),
                        # Preserve the original selector's column contract, but
                        # put the extrinsics-corrected camera pose here so
                        # visualize_epnp_gigapose_comparison draws the same two
                        # poses used to calculate the reported errors.
                        "T_epnp_obj": base.matrix_to_text(
                            T_camera_object_optimized
                        ),
                        "T_epnp_camera_obj_original": base.matrix_to_text(
                            T_camera_object_original
                        ),
                        "T_epnp_camera_obj_optimized": base.matrix_to_text(
                            T_camera_object_optimized
                        ),
                        "T_epnp_map_obj": base.matrix_to_text(T_map_object),
                        "T_map_cam_initial": base.matrix_to_text(
                            T_map_camera_initial
                        ),
                        "T_map_cam_optimized": base.matrix_to_text(
                            T_map_camera_optimized
                        ),
                        "T_gigapose_map_obj_optimized": base.matrix_to_text(
                            T_gigapose_map_optimized
                        ),
                        "optimized_extrinsics_path": str(extrinsics_path),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["match_key"],
            float(row["translation_error_mm"]),
            float(row["rotation_error_deg"]),
            -float(row["score"]),
        )
    )
    return rows


def numeric_summary(
    rows: list[dict[str, Any]], source_key: str, output_prefix: str
) -> dict[str, float]:
    values = np.asarray(
        [float(row[source_key]) for row in rows], dtype=float
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{output_prefix}_mean": float("nan"),
            f"{output_prefix}_median": float("nan"),
            f"{output_prefix}_p90": float("nan"),
        }
    return {
        f"{output_prefix}_mean": float(np.mean(values)),
        f"{output_prefix}_median": float(np.median(values)),
        f"{output_prefix}_p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_candidates_per_key < 1:
        raise ValueError("--max-candidates-per-key must be positive")
    if args.epnp_translation_unit is not None:
        args.epnp_map_pose_unit = args.epnp_translation_unit
        args.epnp_camera_pose_unit = args.epnp_translation_unit

    correction, extrinsics_data, extrinsics_path = load_extrinsic_correction(
        args.optimized_extrinsics
    )
    dates = tuple(args.date or base.DEFAULT_DATES)
    frame_map = base.load_frame_map(args.frame_map, args.dataset_dir)
    predictions = base.load_gigapose_predictions(
        args.gigapose_predictions, frame_map, args.match_key
    )
    predictions = [
        row
        for row in predictions
        if float(row.get("score", 0.0)) >= args.min_score
    ]

    epnp_roots = base.discover_epnp_roots(
        args.source_root, dates, args.epnp_root
    )
    labels, label_skip_counts = load_epnp_extrinsic_labels(
        epnp_roots,
        args.epnp_glob,
        args.epnp_key_field,
        args.epnp_key_prefix,
        args.epnp_strip_trailing_instance_id,
        args.epnp_map_pose_key,
        args.epnp_camera_pose_key,
        args.epnp_map_pose_unit,
        args.epnp_camera_pose_unit,
    )
    if not labels:
        raise ValueError(
            "No EPnP labels containing both requested poses were loaded. Check "
            "--epnp-root, --epnp-glob, --epnp-map-pose-key, "
            "--epnp-camera-pose-key, and their units. "
            f"Skip counts: {label_skip_counts}"
        )

    preds_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    epnp_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        preds_by_key[str(row["match_key"])].append(row)
    for row in labels:
        epnp_by_key[str(row["match_key"])].append(row)

    if args.frame_transform_json is not None:
        gigapose_frame_transform = base.load_frame_transform(
            args.frame_transform_json
        )
        estimated_pairs = 0
        refinement_history: list[dict[str, Any]] = []
    else:
        one_to_one = base.initial_one_to_one_pairs(
            preds_by_key, epnp_by_key
        )
        gigapose_frame_transform = base.estimate_frame_transform(
            one_to_one, args.frame_transform_side
        )
        estimated_pairs = len(one_to_one)
        inlier_translation = (
            args.frame_transform_inlier_translation_mm
            if args.frame_transform_inlier_translation_mm is not None
            else args.max_translation_error_mm
        )
        inlier_rotation = (
            args.frame_transform_inlier_rotation_deg
            if args.frame_transform_inlier_rotation_deg is not None
            else args.max_rotation_error_deg
        )
        (
            gigapose_frame_transform,
            refinement_history,
        ) = base.refine_frame_transform(
            one_to_one,
            args.frame_transform_side,
            gigapose_frame_transform,
            args.frame_transform_refine_iterations,
            inlier_translation,
            inlier_rotation,
        )
    base.save_frame_transform(
        args.output_dir / "frame_transform_gigapose_to_epnp.json",
        gigapose_frame_transform,
        estimated_pairs,
        args.frame_transform_side,
    )

    all_candidates = make_candidate_rows(
        preds_by_key,
        epnp_by_key,
        correction,
        gigapose_frame_transform,
        args.frame_transform_side,
        args.max_candidates_per_key,
        extrinsics_path,
    )
    best_candidates = base.best_rows_per_label(all_candidates)
    selected = [
        row
        for row in best_candidates
        if float(row["translation_error_mm"])
        <= args.max_translation_error_mm
        and float(row["rotation_error_deg"])
        <= args.max_rotation_error_deg
    ]

    base.write_csv(args.output_dir / "all_candidate_pairs.csv", all_candidates)
    base.write_csv(
        args.output_dir / "best_candidate_per_epnp_label.csv", best_candidates
    )
    base.write_csv(args.output_dir / "selected_samples.csv", selected)

    shared_keys = set(preds_by_key) & set(epnp_by_key)
    report: dict[str, Any] = {
        "description": (
            "Selection applies the same estimated/loaded GigaPose-to-EPnP "
            "frame transform as select_real_label_candidates, then compares it "
            "with the extrinsics-corrected EPnP camera pose. The primary errors "
            "and visualization columns describe those same camera-frame poses."
        ),
        "source_root": str(args.source_root),
        "dates": dates,
        "epnp_roots": [str(path) for path in epnp_roots],
        "gigapose_predictions": str(args.gigapose_predictions),
        "optimized_extrinsics": str(extrinsics_path),
        "optimization_mode": extrinsics_data.get("optimization_mode"),
        "epnp_map_pose_key": args.epnp_map_pose_key,
        "epnp_camera_pose_key": args.epnp_camera_pose_key,
        "epnp_map_pose_unit": args.epnp_map_pose_unit,
        "epnp_camera_pose_unit": args.epnp_camera_pose_unit,
        "frame_transform_side": args.frame_transform_side,
        "frame_transform_source": (
            str(args.frame_transform_json)
            if args.frame_transform_json is not None
            else "estimated_from_current_one_to_one_pairs"
        ),
        "frame_transform_estimated_from_one_to_one_pairs": estimated_pairs,
        "frame_transform_refine_iterations": (
            args.frame_transform_refine_iterations
        ),
        "frame_transform_refinement_history": refinement_history,
        "min_score": args.min_score,
        "max_translation_error_mm": args.max_translation_error_mm,
        "max_rotation_error_deg": args.max_rotation_error_deg,
        "loaded_predictions_after_score_filter": len(predictions),
        "loaded_epnp_labels": len(labels),
        "epnp_label_skip_counts": label_skip_counts,
        "matched_keys": len(shared_keys),
        "candidate_pairs": len(all_candidates),
        "best_candidate_pairs": len(best_candidates),
        "selected_samples": len(selected),
    }
    for source_key, prefix in (
        ("raw_translation_error_mm", "best_raw_translation_error_mm"),
        ("raw_rotation_error_deg", "best_raw_rotation_error_deg"),
        ("original_translation_error_mm", "best_original_translation_error_mm"),
        ("original_rotation_error_deg", "best_original_rotation_error_deg"),
        ("translation_error_mm", "best_optimized_translation_error_mm"),
        ("rotation_error_deg", "best_optimized_rotation_error_deg"),
    ):
        report.update(numeric_summary(best_candidates, source_key, prefix))

    if all_candidates:
        max_translation_disagreement = max(
            float(row["map_camera_translation_disagreement_mm"])
            for row in all_candidates
        )
        max_rotation_disagreement = max(
            float(row["map_camera_rotation_disagreement_deg"])
            for row in all_candidates
        )
    else:
        max_translation_disagreement = float("nan")
        max_rotation_disagreement = float("nan")
    report["map_camera_translation_disagreement_mm_max"] = (
        max_translation_disagreement
    )
    report["map_camera_rotation_disagreement_deg_max"] = (
        max_rotation_disagreement
    )

    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2)
    )
    if (
        np.isfinite(max_translation_disagreement)
        and max_translation_disagreement > 1.0
    ) or (
        np.isfinite(max_rotation_disagreement)
        and max_rotation_disagreement > 0.01
    ):
        print(
            "WARNING: Corrected map-frame and camera-frame diagnostics differ "
            "more than expected. Selection uses the corrected camera-frame "
            "error so it remains identical to the visualization comparison. "
            "Inspect the two "
            "map_camera_*_disagreement fields for non-rigid/noisy input matrices."
        )
    print(json.dumps(report, indent=2))
    print(
        "Wrote extrinsics-aware selected samples to "
        f"{args.output_dir / 'selected_samples.csv'}"
    )


if __name__ == "__main__":
    main()
