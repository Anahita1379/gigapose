"""Train the adaptive-gain GRU filter on consecutive GigaPose measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FilterClipDataset, MeasurementBundle, context_statistics
from .geometry import rotation_angle
from .model import GRUKalmanFilter


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--validation-data", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--clip-length", type=int, default=16)
    value.add_argument("--clip-stride", type=int, default=4)
    value.add_argument("--validation-stride", type=int, default=16)
    value.add_argument("--hidden-dim", type=int, default=128)
    value.add_argument("--context-hidden", type=int, default=32)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--rotation-scale-deg", type=float, default=20.0)
    value.add_argument("--translation-scale-m", type=float, default=1.0)
    value.add_argument("--measurement-preservation-weight", type=float, default=0.01)
    value.add_argument(
        "--use-fallback-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train independent filtered-vs-GigaPose translation and rotation heads.",
    )
    value.add_argument("--fallback-hidden", type=int, default=64)
    value.add_argument("--translation-fallback-weight", type=float, default=1.0)
    value.add_argument("--rotation-fallback-weight", type=float, default=1.0)
    value.add_argument("--translation-fallback-margin-m", type=float, default=0.0)
    value.add_argument("--rotation-fallback-margin-deg", type=float, default=0.0)
    value.add_argument(
        "--initialize-from-checkpoint",
        type=Path,
        help="Initialize the filter backbone (and heads, when present) from a checkpoint.",
    )
    value.add_argument(
        "--freeze-filter-backbone",
        action="store_true",
        help="Train only the two fallback heads.",
    )
    value.add_argument(
        "--preserve-measurement-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Copy each input rotation to the output and train only translation; "
            "use this after the adaptive temporal GRU selector."
        ),
    )
    value.add_argument("--gradient-clip", type=float, default=5.0)
    value.add_argument("--patience", type=int, default=10)
    value.add_argument("--min-delta", type=float, default=1e-4)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=20260803)
    value.add_argument("--allow-run-overlap", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def _validate_splits(train: MeasurementBundle, validation: MeasurementBundle, allow: bool):
    if train.manifest["context_feature_names"] != validation.manifest["context_feature_names"]:
        raise ValueError("Train/validation context schemas differ")
    train_runs = set(train.arrays["source_run"].tolist())
    validation_runs = set(validation.arrays["source_run"].tolist())
    overlap = sorted(train_runs & validation_runs)
    if overlap and not allow:
        raise ValueError(f"Train/validation physical-run leakage: {overlap}")
    return train_runs, validation_runs, overlap


def _step(model, batch, device, args, optimizer=None):
    measurement = batch["measurement_pose"].to(device)
    baseline = batch["baseline_pose"].to(device)
    target = batch["target_pose"].to(device)
    context = batch["context_features"].to(device)
    delta_time = batch["delta_time_s"].to(device)
    valid = batch["frame_valid"].to(device)
    model_output = model(
        measurement, context, delta_time, valid,
        baseline_pose=baseline if model.use_fallback_heads else None,
    )
    filtered, gains = model_output[:2]
    trust_logits = model_output[2] if model.use_fallback_heads else None
    # The first frame initializes the filter and cannot be improved temporally.
    loss_mask = valid.clone()
    loss_mask[:, 0] = False
    translation = torch.linalg.norm(filtered[..., :3, 3] - target[..., :3, 3], dim=-1)
    raw_translation = torch.linalg.norm(measurement[..., :3, 3] - target[..., :3, 3], dim=-1)
    rotation = rotation_angle(filtered[..., :3, :3], target[..., :3, :3])
    raw_rotation = rotation_angle(measurement[..., :3, :3], target[..., :3, :3])
    baseline_translation = torch.linalg.norm(
        baseline[..., :3, 3] - target[..., :3, 3], dim=-1
    )
    baseline_rotation = rotation_angle(
        baseline[..., :3, :3], target[..., :3, :3]
    )
    correction_translation = torch.linalg.norm(
        filtered[..., :3, 3] - measurement[..., :3, 3], dim=-1
    )
    correction_rotation = rotation_angle(
        filtered[..., :3, :3], measurement[..., :3, :3]
    )
    translation_loss = translation[loss_mask].mean() / args.translation_scale_m
    rotation_loss = rotation[loss_mask].mean() / np.deg2rad(args.rotation_scale_deg)
    preservation = correction_translation[loss_mask].mean() / args.translation_scale_m
    if not model.preserve_measurement_rotation:
        preservation = preservation + (
            correction_rotation[loss_mask].mean()
            / np.deg2rad(args.rotation_scale_deg)
        )
    pose_loss = translation_loss
    if not model.preserve_measurement_rotation:
        pose_loss = pose_loss + rotation_loss
    translation_trust_loss = torch.zeros((), device=device)
    rotation_trust_loss = torch.zeros((), device=device)
    translation_trust_accuracy = torch.zeros((), device=device)
    rotation_trust_accuracy = torch.zeros((), device=device)
    translation_fallback_precision = torch.zeros((), device=device)
    translation_fallback_recall = torch.zeros((), device=device)
    rotation_fallback_precision = torch.zeros((), device=device)
    rotation_fallback_recall = torch.zeros((), device=device)
    if model.use_fallback_heads:
        trust_mask = valid
        translation_target = (
            translation + args.translation_fallback_margin_m
            < baseline_translation
        ).float()
        rotation_target = (
            rotation + np.deg2rad(args.rotation_fallback_margin_deg)
            < baseline_rotation
        ).float()

        def balanced_bce(logits, target):
            selected_logits = logits[trust_mask]
            selected_target = target[trust_mask]
            positive = selected_target.sum()
            negative = selected_target.numel() - positive
            weights = torch.where(
                selected_target > 0.5,
                0.5 * selected_target.numel() / positive.clamp_min(1.0),
                0.5 * selected_target.numel() / negative.clamp_min(1.0),
            )
            return F.binary_cross_entropy_with_logits(
                selected_logits, selected_target, weight=weights
            )

        translation_trust_loss = balanced_bce(
            trust_logits[..., 0], translation_target
        )
        rotation_trust_loss = balanced_bce(
            trust_logits[..., 1], rotation_target
        )

        def trust_metrics(logits, target):
            predicted_trust = logits[trust_mask] >= 0
            target_trust = target[trust_mask] > 0.5
            predicted_fallback = ~predicted_trust
            target_fallback = ~target_trust
            true_positive = (predicted_fallback & target_fallback).sum().float()
            precision = true_positive / predicted_fallback.sum().clamp_min(1)
            recall = true_positive / target_fallback.sum().clamp_min(1)
            accuracy = (predicted_trust == target_trust).float().mean()
            return accuracy, precision, recall

        (
            translation_trust_accuracy,
            translation_fallback_precision,
            translation_fallback_recall,
        ) = trust_metrics(trust_logits[..., 0], translation_target)
        (
            rotation_trust_accuracy,
            rotation_fallback_precision,
            rotation_fallback_recall,
        ) = trust_metrics(trust_logits[..., 1], rotation_target)

    loss = pose_loss + args.measurement_preservation_weight * preservation
    if model.use_fallback_heads:
        loss = (
            loss
            + args.translation_fallback_weight * translation_trust_loss
            + args.rotation_fallback_weight * rotation_trust_loss
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite GRU-Kalman loss. Check the exported poses and numerical scales."
        )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if nonfinite_gradients:
            raise FloatingPointError(
                "Non-finite GRU-Kalman gradients in: "
                + ", ".join(nonfinite_gradients)
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
    count = int(loss_mask.sum())
    return {
        "loss": float(loss.detach()),
        "translation_error_m": float(translation[loss_mask].mean().detach()),
        "rotation_error_deg": float(torch.rad2deg(rotation[loss_mask]).mean().detach()),
        "raw_translation_error_m": float(raw_translation[loss_mask].mean().detach()),
        "raw_rotation_error_deg": float(torch.rad2deg(raw_rotation[loss_mask]).mean().detach()),
        "translation_improvement_m": float((raw_translation[loss_mask] - translation[loss_mask]).mean().detach()),
        "rotation_improvement_deg": float(torch.rad2deg(raw_rotation[loss_mask] - rotation[loss_mask]).mean().detach()),
        "mean_pose_gain": float(gains[..., :6][loss_mask].mean().detach()),
        "translation_trust_loss": float(translation_trust_loss.detach()),
        "rotation_trust_loss": float(rotation_trust_loss.detach()),
        "translation_trust_accuracy": float(translation_trust_accuracy.detach()),
        "rotation_trust_accuracy": float(rotation_trust_accuracy.detach()),
        "translation_fallback_precision": float(translation_fallback_precision.detach()),
        "translation_fallback_recall": float(translation_fallback_recall.detach()),
        "rotation_fallback_precision": float(rotation_fallback_precision.detach()),
        "rotation_fallback_recall": float(rotation_fallback_recall.detach()),
        "frames": count,
    }


def _epoch(model, loader, device, args, optimizer=None):
    model.train(optimizer is not None)
    totals, frames = {}, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            metrics = _step(model, batch, device, args, optimizer)
            count = metrics.pop("frames")
            frames += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / max(frames, 1) for key, value in totals.items()}


def _checkpoint(model, args, epoch, metrics, train_bundle):
    return {
        "format": (
            "rgb_self_recovery_gru_kalman_filter_v2"
            if model.use_fallback_heads
            else "rgb_self_recovery_gru_kalman_filter_v1"
        ),
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "epoch": epoch,
        "validation_metrics": metrics,
        "context_feature_names": train_bundle.manifest["context_feature_names"],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def load_filter(path: Path, device: str = "cpu"):
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") not in {
        "rgb_self_recovery_gru_kalman_filter_v1",
        "rgb_self_recovery_gru_kalman_filter_v2",
    }:
        raise ValueError(f"Unsupported learned-filter checkpoint: {path}")
    model = GRUKalmanFilter(**payload["model_config"])
    model.load_state_dict(
        payload["model_state"],
        strict=payload.get("format") == "rgb_self_recovery_gru_kalman_filter_v2",
    )
    return model.to(device).eval(), payload


def main() -> None:
    args = parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        args.device = "cpu"
    if min(args.translation_scale_m, args.rotation_scale_deg) <= 0:
        raise ValueError("Pose loss scales must be positive")
    if min(
        args.translation_fallback_weight,
        args.rotation_fallback_weight,
        args.translation_fallback_margin_m,
        args.rotation_fallback_margin_deg,
    ) < 0:
        raise ValueError("Fallback weights and margins cannot be negative")
    if args.freeze_filter_backbone and not args.use_fallback_heads:
        raise ValueError("--freeze-filter-backbone requires --use-fallback-heads")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = FilterClipDataset(args.data, clip_length=args.clip_length, stride=args.clip_stride)
    validation_data = FilterClipDataset(
        args.validation_data, clip_length=args.clip_length, stride=args.validation_stride
    )
    train_runs, validation_runs, overlap = _validate_splits(
        train_data.bundle, validation_data.bundle, args.allow_run_overlap
    )
    statistics = {
        key: torch.from_numpy(value)
        for key, value in context_statistics(train_data.bundle).items()
    }
    model = GRUKalmanFilter(
        train_data.bundle.context_dim,
        hidden_dim=args.hidden_dim,
        context_hidden=args.context_hidden,
        preserve_measurement_rotation=args.preserve_measurement_rotation,
        use_fallback_heads=args.use_fallback_heads,
        fallback_hidden=args.fallback_hidden,
        statistics=statistics,
    ).to(args.device)
    if args.use_fallback_heads:
        if np.allclose(
            train_data.bundle.arrays["baseline_pose"],
            train_data.bundle.arrays["measurement_pose"],
        ):
            raise ValueError(
                "Fallback heads require a distinct baseline_pose. Re-export with "
                "--fallback-baseline-predictions pointing to original GigaPose."
            )
    if args.initialize_from_checkpoint is not None:
        initial_payload = torch.load(args.initialize_from_checkpoint, map_location="cpu")
        if initial_payload.get("format") not in {
            "rgb_self_recovery_gru_kalman_filter_v1",
            "rgb_self_recovery_gru_kalman_filter_v2",
        }:
            raise ValueError(
                f"Unsupported initialization checkpoint: {args.initialize_from_checkpoint}"
            )
        incompatible = model.load_state_dict(initial_payload["model_state"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(f"Unexpected initialization parameters: {unexpected}")
        print(
            "Initialized from " + str(args.initialize_from_checkpoint)
            + f"; newly initialized parameters={list(incompatible.missing_keys)}"
        )
    if args.freeze_filter_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith((
                "translation_fallback_head", "rotation_fallback_head"
            ))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model, train_loader, args.device, args, optimizer)
        validation_metrics = _epoch(model, validation_loader, args.device, args)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        payload = _checkpoint(model, args, epoch, validation_metrics, train_data.bundle)
        torch.save(payload, args.output_dir / "last.ckpt")
        if validation_metrics["loss"] < best - args.min_delta:
            best = validation_metrics["loss"]
            stale = 0
            torch.save(payload, args.output_dir / "best.ckpt")
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={validation_metrics['loss']:.4f} "
            f"t={validation_metrics['translation_error_m']:.3f}m "
            f"r={validation_metrics['rotation_error_deg']:.2f}deg "
            f"improve_t={validation_metrics['translation_improvement_m']:.3f}m "
            f"improve_r={validation_metrics['rotation_improvement_deg']:.2f}deg "
            f"fallback_t={validation_metrics['translation_fallback_precision']:.2f}/"
            f"{validation_metrics['translation_fallback_recall']:.2f} "
            f"fallback_r={validation_metrics['rotation_fallback_precision']:.2f}/"
            f"{validation_metrics['rotation_fallback_recall']:.2f} "
            f"stale={stale}/{args.patience}", flush=True,
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "format": "rgb_self_recovery_gru_kalman_training_v1",
        "best_validation_loss": best,
        "epochs_completed": len(history),
        "train_runs": sorted(train_runs),
        "validation_runs": sorted(validation_runs),
        "run_overlap": overlap,
        "train_frames": len(train_data.bundle),
        "validation_frames": len(validation_data.bundle),
        "train_clips": len(train_data),
        "validation_clips": len(validation_data),
        "preserve_measurement_rotation": args.preserve_measurement_rotation,
        "use_fallback_heads": args.use_fallback_heads,
        "freeze_filter_backbone": args.freeze_filter_backbone,
        "recommended_checkpoint": str(args.output_dir / "best.ckpt"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
