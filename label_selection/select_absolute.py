"""Select GigaPose/EPnP pairs using unaltered absolute pose errors.

The only alignment performed here is the fixed, known raw-CAD to centered-CAD
origin conversion.  No transform is estimated from prediction/label pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from label_selection import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Absolute GigaPose-versus-EPnP comparison and pair selection."
    )
    parser.add_argument("--gigapose-predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--epnp-root", type=Path, required=True)
    parser.add_argument("--epnp-glob", default="*.json")
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
        default="image_stem",
    )
    parser.add_argument("--epnp-map-pose-key", default="T_map_object_raw")
    parser.add_argument(
        "--epnp-camera-pose-key", default="T_camera_object_centered"
    )
    parser.add_argument(
        "--epnp-map-pose-unit", choices=("auto", "m", "mm"), default="m"
    )
    parser.add_argument(
        "--epnp-camera-pose-unit", choices=("auto", "m", "mm"), default="m"
    )
    parser.add_argument(
        "--raw-object-center-m",
        type=float,
        nargs=3,
        default=common.DEFAULT_RAW_OBJECT_CENTER_M,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--prediction-translation-unit",
        choices=("mm", "m", "auto"),
        default="mm",
        help="Unit of the GigaPose CSV t column. BOP GigaPose output is normally mm.",
    )
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-candidates-per-key", type=int, default=20)
    parser.add_argument("--pairing-translation-scale-mm", type=float, default=1000.0)
    parser.add_argument("--pairing-rotation-scale-deg", type=float, default=10.0)
    parser.add_argument("--max-translation-error-mm", type=float, default=3000.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument("--max-roll-error-deg", type=float, default=None)
    parser.add_argument("--max-pitch-error-deg", type=float, default=None)
    parser.add_argument("--max-yaw-error-deg", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace these outputs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_name in (
        "all_candidate_pairs.csv",
        "assigned_pairs.csv",
        "selected_samples.csv",
        "selection_report.json",
    ):
        output_path = args.output_dir / output_name
        if output_path.exists() and args.overwrite:
            output_path.unlink()

    predictions, _ = common.load_predictions(
        args.gigapose_predictions,
        args.dataset_dir,
        args.frame_map,
        args.match_key,
        args.min_score,
        args.prediction_translation_unit,
    )
    labels, skipped_labels = common.load_labels(
        args.epnp_root,
        args.epnp_glob,
        args.epnp_key_field,
        args.epnp_key_prefix,
        args.epnp_strip_trailing_instance_id,
        args.epnp_map_pose_key,
        args.epnp_camera_pose_key,
        args.epnp_map_pose_unit,
        args.epnp_camera_pose_unit,
    )
    C = common.object_center_transform(args.raw_object_center_m)
    all_pairs, assigned, pair_counts = common.build_and_assign_pairs(
        predictions,
        labels,
        C,
        args.pairing_translation_scale_mm,
        args.pairing_rotation_scale_deg,
        args.max_candidates_per_key,
    )
    selected = [
        row
        for row in assigned
        if common.passes_thresholds(
            row,
            args.max_translation_error_mm,
            args.max_rotation_error_deg,
            args.max_roll_error_deg,
            args.max_pitch_error_deg,
            args.max_yaw_error_deg,
        )
    ]
    selected.sort(
        key=lambda row: (
            int(row["scene_id"]),
            int(row["im_id"]),
            float(row["pairing_cost"]),
        )
    )
    common.write_csv(args.output_dir / "all_candidate_pairs.csv", all_pairs)
    common.write_csv(args.output_dir / "assigned_pairs.csv", assigned)
    common.write_csv(args.output_dir / "selected_samples.csv", selected)
    thresholds = {
        "min_score": args.min_score,
        "max_translation_error_mm": args.max_translation_error_mm,
        "max_rotation_error_deg": args.max_rotation_error_deg,
        "max_roll_error_deg": args.max_roll_error_deg,
        "max_pitch_error_deg": args.max_pitch_error_deg,
        "max_yaw_error_deg": args.max_yaw_error_deg,
    }
    report = common.selection_report(
        predictions_path=args.gigapose_predictions,
        dataset_dir=args.dataset_dir,
        epnp_root=args.epnp_root,
        output_dir=args.output_dir,
        comparison_mode=common.COMPARISON_MODE,
        predictions=predictions,
        labels=labels,
        pair_counts=pair_counts,
        assigned=assigned,
        selected=selected,
        skipped_labels=skipped_labels,
        C=C,
        thresholds=thresholds,
        extra={"prediction_translation_unit": args.prediction_translation_unit},
    )
    common.save_json(args.output_dir / "selection_report.json", report)
    print(
        f"Selected {len(selected)}/{len(assigned)} one-to-one pairs using absolute "
        f"errors. No empirical prediction-label alignment was used."
    )
    print(args.output_dir / "selection_report.json")


if __name__ == "__main__":
    main()
