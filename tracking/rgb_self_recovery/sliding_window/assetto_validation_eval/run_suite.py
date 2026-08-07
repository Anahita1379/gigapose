"""Run all RGB recovery backbones sequentially on one prepared dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from .common import resolve_predictions


MODEL_SPECS = {
    "cnn_multicheckpoint": (
        "tracking.rgb_self_recovery.sliding_window.run",
        "cnn_multicheckpoint/best_pose.ckpt",
    ),
    "dino_frozen_multicheckpoint": (
        "tracking.rgb_self_recovery.sliding_window.run",
        "dino_frozen_multicheckpoint/best_pose.ckpt",
    ),
    "dino_last_block": (
        "tracking.rgb_self_recovery.sliding_window.run",
        "dino_last_block/best_pose.ckpt",
    ),
    "matching_unet": (
        "tracking.rgb_self_recovery.sliding_window.matching_unet.run",
        "matching_unet/best_pose.ckpt",
    ),
    "dino_matching_unet": (
        "tracking.rgb_self_recovery.sliding_window.dino_matching_unet.run",
        "dino_matching_unet/best_pose.ckpt",
    ),
    "dino_matching_unet_last_block": (
        "tracking.rgb_self_recovery.sliding_window.dino_matching_unet.run",
        "dino_matching_unet_last_block/best_pose.ckpt",
    ),
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--models-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--models", default=",".join(MODEL_SPECS))
    value.add_argument("--device", default="cuda")
    value.add_argument("--max-frames", type=int)
    value.add_argument("--window-size", type=int, default=5)
    value.add_argument("--per-frame", action="store_true")
    value.add_argument("--sequence-aware", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--sequence-max-frame-gap", type=int, default=1)
    value.add_argument("--sequence-max-time-gap-s", type=float, default=0.5)
    value.add_argument(
        "--orientation-gate-mode",
        choices=("off", "hard", "soft"),
        default="soft",
        help="Orientation handling for every model (default: soft reranking).",
    )
    value.add_argument(
        "--orientation-gates",
        dest="orientation_gate_mode",
        action="store_const",
        const="hard",
        help="Legacy alias for --orientation-gate-mode hard.",
    )
    value.add_argument(
        "--no-orientation-gates",
        dest="orientation_gate_mode",
        action="store_const",
        const="off",
        help="Legacy alias for --orientation-gate-mode off.",
    )
    value.add_argument(
        "--state-iou-confidence-weight",
        type=float,
        default=0.5,
        help="Conservative silhouette-IoU floor used only for tracker state.",
    )
    value.add_argument("--save-overlays", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--skip-missing", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Additional single token passed to every runner; repeat as needed.",
    )
    return value


def selected_models(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {list(MODEL_SPECS)}")
    if not names:
        raise ValueError("--models selected no models")
    return names


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    predictions = resolve_predictions(args.predictions)
    mesh = args.dataset_dir / "models" / "obj_000001.ply"
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    result = []
    for name in selected_models(args.models):
        module, relative_checkpoint = MODEL_SPECS[name]
        checkpoint = args.models_root / relative_checkpoint
        if not checkpoint.is_file():
            if args.skip_missing:
                print(f"Skipping {name}; checkpoint is absent: {checkpoint}")
                continue
            raise FileNotFoundError(checkpoint)
        command = [
            sys.executable,
            "-m",
            module,
            "--predictions",
            str(predictions),
            "--dataset-dir",
            str(args.dataset_dir),
            "--split",
            "test",
            "--checkpoint",
            str(checkpoint),
            "--mesh",
            str(mesh),
            "--output-dir",
            str(args.output_root / name),
            "--prediction-translation-unit",
            "mm",
            "--window-size",
            str(args.window_size),
            "--sequence-max-frame-gap",
            str(args.sequence_max_frame_gap),
            "--sequence-max-time-gap-s",
            str(args.sequence_max_time_gap_s),
            "--device",
            args.device,
            "--orientation-gate-mode",
            args.orientation_gate_mode,
            "--state-iou-confidence-weight",
            str(args.state_iou_confidence_weight),
        ]
        command.append("--sequence-aware" if args.sequence_aware else "--no-sequence-aware")
        if args.per_frame:
            command.append("--per-frame")
        if args.max_frames is not None:
            command.extend(["--max-frames", str(args.max_frames)])
        if args.save_overlays:
            command.append("--save-overlays")
        if args.overwrite:
            command.append("--overwrite")
        command.extend(args.runner_arg)
        result.append((name, command))
    return result


def main() -> None:
    args = parser().parse_args()
    planned = commands(args)
    if not planned:
        raise RuntimeError("No runnable model checkpoints were found")
    for name, command in planned:
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
