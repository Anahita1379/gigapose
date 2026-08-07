"""Train isolated frozen- or last-block-DINO RGB self-recovery variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tracking.rgb_self_recovery.dataset import RGBRenderRecoveryDataset
from tracking.rgb_self_recovery.model import decode_outputs, rotation_geodesic_atan2
from tracking.rgb_self_recovery.train import (
    anti_flip_quality_loss,
    ranking_loss,
    safe_config,
)

from .checkpoint_selection import CHECKPOINT_NAMES, checkpoint_selection_values
from .dino_model import DINORGBRenderRecoveryNet


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--validation-data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dino-mode", choices=("frozen", "last_block"), required=True)
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--dino-input-size", type=int, default=224)
    p.add_argument(
        "--initialize-from", type=Path,
        help="Optional frozen-DINO checkpoint used to initialize last-block tuning.",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--dino-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--max-center-crop-px", type=float, default=56.0)
    p.add_argument("--max-log-depth", type=float, default=0.60)
    p.add_argument("--max-rotation-deg", type=float, default=70.0)
    p.add_argument("--center-weight", type=float, default=1.0)
    p.add_argument("--log-depth-weight", type=float, default=1.0)
    p.add_argument("--rotation-weight", type=float, default=1.0)
    p.add_argument("--confidence-weight", type=float, default=0.5)
    p.add_argument("--quality-weight", type=float, default=0.5)
    p.add_argument("--ranking-weight", type=float, default=0.5)
    p.add_argument("--ranking-margin", type=float, default=0.10)
    p.add_argument(
        "--anti-flip-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--anti-flip-weight", type=float, default=1.0)
    p.add_argument("--anti-flip-margin", type=float, default=0.25)
    p.add_argument("--anti-flip-min-angle-deg", type=float, default=150.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--logger", choices=("none", "wandb"), default="none")
    p.add_argument("--run-name")
    p.add_argument("--wandb-project", default="gigapose")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-offline", action="store_true")
    return p


def compute_loss(model, batch, args, device):
    rgb = batch["rgb"].to(device, non_blocking=True)
    observed = batch["observed_mask"].to(device, non_blocking=True)
    rendered = batch["rendered"].to(device, non_blocking=True)
    batch_size, candidates = rendered.shape[:2]
    rendered_flat = rendered.reshape(batch_size * candidates, *rendered.shape[2:])
    image, dino = model.encode_image(rgb, observed)
    image = image[:, None].expand(-1, candidates, -1, -1, -1).reshape(
        batch_size * candidates, *image.shape[1:]
    )
    dino = dino[:, None].expand(-1, candidates, -1).reshape(
        batch_size * candidates, -1
    )
    raw = model.forward_encoded(image, dino, rendered_flat)
    decoded = decode_outputs(
        raw,
        max_center_px=args.max_center_crop_px,
        max_log_depth=args.max_log_depth,
        max_rotation_deg=args.max_rotation_deg,
    )
    correctable = batch["correctable_targets"].to(device).reshape(-1)
    center_target = batch["center_targets"].to(device).reshape(-1, 2)
    log_depth_target = batch["log_depth_targets"].to(device).reshape(-1)
    rotation_target = batch["rotation_targets"].to(device).reshape(-1, 3)
    confidence_target = batch["confidence_targets"].to(device).reshape(-1)
    quality_target = batch["quality_targets"].to(device).reshape(-1)
    if correctable.any():
        center_loss = nn.functional.smooth_l1_loss(
            decoded["center_px"][correctable] / args.max_center_crop_px,
            center_target[correctable] / args.max_center_crop_px,
        )
        depth_loss = nn.functional.smooth_l1_loss(
            decoded["log_depth"][correctable] / args.max_log_depth,
            log_depth_target[correctable] / args.max_log_depth,
        )
        rotation_errors = rotation_geodesic_atan2(
            decoded["rotation_rad"][correctable], rotation_target[correctable]
        )
        rotation_loss = nn.functional.smooth_l1_loss(
            rotation_errors / math.radians(args.max_rotation_deg),
            torch.zeros_like(rotation_errors),
        )
        rotation_error_deg = torch.rad2deg(rotation_errors).mean()
        center_error_px = torch.linalg.vector_norm(
            decoded["center_px"][correctable] - center_target[correctable], dim=-1
        ).mean()
        log_depth_error = torch.abs(
            decoded["log_depth"][correctable] - log_depth_target[correctable]
        ).mean()
    else:
        zero = raw["quality_raw"].sum() * 0.0
        center_loss = depth_loss = rotation_loss = zero
        rotation_error_deg = center_error_px = log_depth_error = zero
    confidence_loss = nn.functional.binary_cross_entropy_with_logits(
        raw["confidence_logit"], confidence_target
    )
    quality_loss = nn.functional.smooth_l1_loss(decoded["quality"], quality_target)
    predicted_group = decoded["quality"].reshape(batch_size, candidates)
    target_group = quality_target.reshape(batch_size, candidates)
    ranking = ranking_loss(predicted_group, target_group, args.ranking_margin)
    anti_flip = anti_flip_quality_loss(
        predicted_group,
        target_group,
        rotation_target.reshape(batch_size, candidates, 3),
        margin=args.anti_flip_margin,
        minimum_angle_deg=args.anti_flip_min_angle_deg,
    )
    anti_flip_term = (
        args.anti_flip_weight * anti_flip["loss"]
        if args.anti_flip_training else anti_flip["loss"] * 0.0
    )
    total = (
        args.center_weight * center_loss
        + args.log_depth_weight * depth_loss
        + args.rotation_weight * rotation_loss
        + args.confidence_weight * confidence_loss
        + args.quality_weight * quality_loss
        + args.ranking_weight * ranking
        + anti_flip_term
    )
    predicted_best = predicted_group.argmin(dim=1)
    selected_quality = target_group[
        torch.arange(batch_size, device=device), predicted_best
    ].mean()
    return total, {
        "loss": float(total.detach()),
        "loss_center": float(center_loss.detach()),
        "loss_log_depth": float(depth_loss.detach()),
        "loss_rotation": float(rotation_loss.detach()),
        "loss_confidence": float(confidence_loss.detach()),
        "loss_quality": float(quality_loss.detach()),
        "loss_ranking": float(ranking.detach()),
        "loss_anti_flip": float(anti_flip["loss"].detach()),
        "anti_flip_pair_accuracy": float(anti_flip["pair_accuracy"].detach()),
        "anti_flip_margin_accuracy": float(anti_flip["margin_accuracy"].detach()),
        "anti_flip_quality_gap": float(anti_flip["quality_gap"].detach()),
        "anti_flip_pair_fraction": float(anti_flip["pair_fraction"].detach()),
        "center_error_crop_px": float(center_error_px.detach()),
        "log_depth_abs_error": float(log_depth_error.detach()),
        "rotation_error_deg": float(rotation_error_deg.detach()),
        "selected_target_quality": float(selected_quality.detach()),
        "oracle_target_quality": float(target_group.min(dim=1).values.mean().detach()),
    }


def main():
    args = parser().parse_args()
    if args.dino_learning_rate <= 0 or args.learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.anti_flip_weight < 0 or args.anti_flip_margin < 0:
        raise ValueError("Anti-flip weight and margin must be non-negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    train_data = RGBRenderRecoveryDataset(args.data, training=True, seed=args.seed)
    val_data = RGBRenderRecoveryDataset(
        args.validation_data, training=False, seed=args.seed
    )
    for key in ("crop_size", "crop_scale"):
        if train_data.manifest[key] != val_data.manifest[key]:
            raise ValueError(f"Train/validation {key} values differ")
    loaders = {
        "train": DataLoader(
            train_data, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        ),
        "val": DataLoader(
            val_data, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        ),
    }
    model = DINORGBRenderRecoveryNet(
        args.width,
        args.hidden_dim,
        dino_mode=args.dino_mode,
        dino_model=args.dino_model,
        dino_input_size=args.dino_input_size,
        dino_pretrained=True,
    )
    if args.initialize_from is not None:
        payload = torch.load(args.initialize_from, map_location="cpu")
        if payload.get("format") != "rgb_render_self_recovery_dino_v1":
            raise ValueError("--initialize-from must be a DINO recovery checkpoint")
        source = payload["model_config"]
        for key, expected in (
            ("width", args.width),
            ("hidden_dim", args.hidden_dim),
            ("dino_model", args.dino_model),
            ("dino_input_size", args.dino_input_size),
        ):
            if source.get(key) != expected:
                raise ValueError(
                    f"Initialization checkpoint {key}={source.get(key)!r}; "
                    f"current value is {expected!r}"
                )
        model.load_state_dict(payload["model_state"])
        print(f"Initialized from {args.initialize_from}")
    model = model.to(args.device)
    base_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("dino.")
    ]
    dino_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("dino.")
    ]
    groups = [{"params": base_parameters, "lr": args.learning_rate}]
    if dino_parameters:
        groups.append({"params": dino_parameters, "lr": args.dino_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_counts = model.trainable_parameter_counts()
    print(json.dumps({"dino_mode": args.dino_mode, **parameter_counts}, indent=2))

    wandb_run = None
    if args.logger == "wandb":
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or args.output_dir.name,
            dir=str(args.output_dir),
            mode="offline" if args.wandb_offline else "online",
            config={**safe_config(args), **parameter_counts},
        )
    history = []
    best_values = {name: float("inf") for name in CHECKPOINT_NAMES}
    best_epochs = {name: 0 for name in CHECKPOINT_NAMES}
    stale = 0
    started = time.perf_counter()

    def run_epoch(name):
        training = name == "train"
        model.train(training)
        sums, count = {}, 0
        for batch in loaders[name]:
            with torch.set_grad_enabled(training):
                loss, metrics = compute_loss(model, batch, args, args.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.gradient_clip,
                    )
                    optimizer.step()
            batch_count = int(batch["rgb"].shape[0])
            count += batch_count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * batch_count
        if not count:
            raise RuntimeError(f"{name} epoch contained no groups")
        return {key: value / count for key, value in sums.items()}

    def save(name, epoch, validation_metrics, criterion, criterion_value):
        torch.save(
            {
                "format": "rgb_render_self_recovery_dino_v1",
                "model_state": model.state_dict(),
                "model_config": {
                    "width": args.width,
                    "hidden_dim": args.hidden_dim,
                    "dino_mode": args.dino_mode,
                    "dino_model": args.dino_model,
                    "dino_input_size": args.dino_input_size,
                },
                "inference_config": {
                    "crop_size": train_data.manifest["crop_size"],
                    "crop_scale": train_data.manifest["crop_scale"],
                    "max_center_crop_px": args.max_center_crop_px,
                    "max_log_depth": args.max_log_depth,
                    "max_rotation_deg": args.max_rotation_deg,
                    "uses_observed_depth": False,
                },
                "epoch": epoch,
                "validation_loss": validation_metrics["loss"],
                "validation_metrics": validation_metrics,
                "selection_criterion": criterion,
                "selection_value": criterion_value,
                "arguments": safe_config(args),
                "parameter_counts": parameter_counts,
            },
            args.output_dir / name,
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch("train")
            with torch.no_grad():
                val_metrics = run_epoch("val")
            values = checkpoint_selection_values(
                val_metrics,
                max_center_px=args.max_center_crop_px,
                max_log_depth=args.max_log_depth,
                max_rotation_deg=args.max_rotation_deg,
            )
            loss_improved = values["loss"] < best_values["loss"] - args.min_delta
            stale = 0 if loss_improved else stale + 1
            for criterion, checkpoint_name in CHECKPOINT_NAMES.items():
                threshold = args.min_delta if criterion == "loss" else 0.0
                if values[criterion] < best_values[criterion] - threshold:
                    best_values[criterion] = values[criterion]
                    best_epochs[criterion] = epoch
                    save(
                        checkpoint_name,
                        epoch,
                        val_metrics,
                        criterion,
                        values[criterion],
                    )
                    if criterion == "loss":
                        # Preserve the checkpoint path used by older commands.
                        save("best.ckpt", epoch, val_metrics, criterion, values[criterion])
            save("last.ckpt", epoch, val_metrics, "last", values["loss"])
            row = {
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "val/pose_selection_score": values["pose"],
                **{f"best/{key}": value for key, value in best_values.items()},
                "stale_epochs": stale,
            }
            history.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={values['loss']:.6f} best={best_values['loss']:.6f} "
                f"rot={values['rotation']:.2f}deg "
                f"center={values['center']:.2f}px depth={values['depth']:.4f} "
                f"pose={values['pose']:.4f} "
                f"stale={stale}/{args.patience}"
            )
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (validation-loss patience).")
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    report = {
        "format": "rgb_render_self_recovery_dino_training_multi_checkpoint_v1",
        "best_validation_loss": best_values["loss"],
        "best_values": best_values,
        "best_epochs": best_epochs,
        "checkpoint_selection": {
            **CHECKPOINT_NAMES,
            "best": "best.ckpt (alias of best_loss.ckpt)",
            "pose_score": (
                "mean(center_error_crop_px/max_center_crop_px, "
                "log_depth_abs_error/max_log_depth, "
                "rotation_error_deg/max_rotation_deg)"
            ),
            "early_stopping_criterion": "validation loss",
        },
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "training_groups": len(train_data),
        "validation_groups": len(val_data),
        "best_checkpoint": str(args.output_dir / "best.ckpt"),
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
        "parameter_counts": parameter_counts,
        "history": history,
        "arguments": safe_config(args),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
