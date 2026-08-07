"""Train the isolated lightweight matching U-Net on existing recovery shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

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
from tracking.rgb_self_recovery.sliding_window.checkpoint_selection import (
    CHECKPOINT_NAMES,
    checkpoint_selection_values,
)

from .model import LightweightMatchingUNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-center-crop-px", type=float, default=56.0)
    parser.add_argument("--max-log-depth", type=float, default=0.60)
    parser.add_argument("--max-rotation-deg", type=float, default=70.0)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--log-depth-weight", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--heatmap-weight", type=float, default=0.10)
    parser.add_argument("--heatmap-sigma-px", type=float, default=4.0)
    parser.add_argument(
        "--anti-flip-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--anti-flip-weight", type=float, default=1.0)
    parser.add_argument("--anti-flip-margin", type=float, default=0.25)
    parser.add_argument("--anti-flip-min-angle-deg", type=float, default=150.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--logger", choices=("none", "wandb"), default="none")
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-offline", action="store_true")
    return parser


def heatmap_kl_loss(
    logits: torch.Tensor,
    center_target_px: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_center_px: float,
    sigma_px: float,
) -> torch.Tensor:
    """KL loss against a Gaussian center target on the decoded feature map."""

    if logits.ndim != 3:
        raise ValueError("Center heatmap logits must have shape (N, H, W).")
    if center_target_px.shape != (logits.shape[0], 2):
        raise ValueError("Center targets must have shape (N, 2).")
    if valid.shape != (logits.shape[0],):
        raise ValueError("Center validity mask must have shape (N,).")
    if max_center_px <= 0 or sigma_px <= 0:
        raise ValueError("Heatmap scales must be positive.")
    if not valid.any():
        return logits.sum() * 0.0

    selected = logits[valid]
    targets = torch.clamp(
        center_target_px[valid] / float(max_center_px), -1.0, 1.0
    )
    height, width = selected.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, dtype=logits.dtype, device=logits.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=logits.dtype, device=logits.device)
    distance_sq = (
        (x.view(1, 1, width) - targets[:, 0].view(-1, 1, 1)).square()
        + (y.view(1, height, 1) - targets[:, 1].view(-1, 1, 1)).square()
    )
    sigma_normalized = float(sigma_px) / float(max_center_px)
    target_logits = -0.5 * distance_sq / (sigma_normalized**2)
    target_distribution = torch.softmax(target_logits.flatten(1), dim=-1)
    predicted_log_distribution = torch.log_softmax(selected.flatten(1), dim=-1)
    return nn.functional.kl_div(
        predicted_log_distribution,
        target_distribution,
        reduction="batchmean",
    )


def compute_loss(
    model: LightweightMatchingUNet,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    observed = batch["observed_mask"].to(device, non_blocking=True)
    rendered = batch["rendered"].to(device, non_blocking=True)
    batch_size, candidates = rendered.shape[:2]
    rendered_flat = rendered.reshape(batch_size * candidates, *rendered.shape[2:])
    image_pyramid = model.encode_image(rgb, observed)
    image_pyramid = [
        feature[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            batch_size * candidates, *feature.shape[1:]
        )
        for feature in image_pyramid
    ]
    raw = model.forward_encoded(image_pyramid, rendered_flat)
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
    heatmap_loss = heatmap_kl_loss(
        raw["center_heatmap_logits"],
        center_target,
        correctable,
        max_center_px=args.max_center_crop_px,
        sigma_px=args.heatmap_sigma_px,
    )

    confidence_loss = nn.functional.binary_cross_entropy_with_logits(
        raw["confidence_logit"], confidence_target
    )
    quality_loss = nn.functional.smooth_l1_loss(
        decoded["quality"], quality_target
    )
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
        if args.anti_flip_training
        else anti_flip["loss"] * 0.0
    )
    total = (
        args.center_weight * center_loss
        + args.log_depth_weight * depth_loss
        + args.rotation_weight * rotation_loss
        + args.confidence_weight * confidence_loss
        + args.quality_weight * quality_loss
        + args.ranking_weight * ranking
        + args.heatmap_weight * heatmap_loss
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
        "loss_heatmap": float(heatmap_loss.detach()),
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


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "learning_rate",
        "max_center_crop_px",
        "max_log_depth",
        "max_rotation_deg",
        "heatmap_sigma_px",
        "patience",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("heatmap_weight", "anti_flip_weight", "anti_flip_margin"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    training_data = RGBRenderRecoveryDataset(args.data, training=True, seed=args.seed)
    validation_data = RGBRenderRecoveryDataset(
        args.validation_data, training=False, seed=args.seed
    )
    for key in ("crop_size", "crop_scale"):
        if training_data.manifest[key] != validation_data.manifest[key]:
            raise ValueError(f"Train/validation {key} values differ.")
    loaders = {
        "train": DataLoader(
            training_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        ),
        "validation": DataLoader(
            validation_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        ),
    }
    model = LightweightMatchingUNet(args.width, args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_counts = model.parameter_counts()
    print(json.dumps({"architecture": "lightweight_matching_unet", **parameter_counts}, indent=2))

    wandb_run = None
    if args.logger == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("--logger wandb requires wandb.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or args.output_dir.name,
            dir=str(args.output_dir),
            mode="offline" if args.wandb_offline else "online",
            config={**safe_config(args), **parameter_counts},
        )

    history: list[dict[str, Any]] = []
    best_values = {name: float("inf") for name in CHECKPOINT_NAMES}
    best_epochs = {name: 0 for name in CHECKPOINT_NAMES}
    stale = 0
    started = time.perf_counter()

    def run_epoch(name: str) -> dict[str, float]:
        training = name == "train"
        model.train(training)
        sums: dict[str, float] = {}
        groups = 0
        for batch in loaders[name]:
            with torch.set_grad_enabled(training):
                loss, metrics = compute_loss(model, batch, args, args.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    optimizer.step()
            count = int(batch["rgb"].shape[0])
            groups += count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * count
        if not groups:
            raise RuntimeError(f"{name} epoch contained no groups.")
        return {key: value / groups for key, value in sums.items()}

    def save(
        filename: str,
        epoch: int,
        validation_metrics: dict[str, float],
        criterion: str,
        criterion_value: float,
    ) -> None:
        torch.save(
            {
                "format": "rgb_render_self_recovery_matching_unet_v1",
                "model_state": model.state_dict(),
                "model_config": {"width": args.width, "hidden_dim": args.hidden_dim},
                "inference_config": {
                    "crop_size": training_data.manifest["crop_size"],
                    "crop_scale": training_data.manifest["crop_scale"],
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
            args.output_dir / filename,
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch("train")
            with torch.no_grad():
                validation_metrics = run_epoch("validation")
            values = checkpoint_selection_values(
                validation_metrics,
                max_center_px=args.max_center_crop_px,
                max_log_depth=args.max_log_depth,
                max_rotation_deg=args.max_rotation_deg,
            )
            loss_improved = values["loss"] < best_values["loss"] - args.min_delta
            stale = 0 if loss_improved else stale + 1
            for criterion, filename in CHECKPOINT_NAMES.items():
                threshold = args.min_delta if criterion == "loss" else 0.0
                if values[criterion] < best_values[criterion] - threshold:
                    best_values[criterion] = values[criterion]
                    best_epochs[criterion] = epoch
                    save(
                        filename,
                        epoch,
                        validation_metrics,
                        criterion,
                        values[criterion],
                    )
                    if criterion == "loss":
                        save("best.ckpt", epoch, validation_metrics, criterion, values[criterion])
            save("last.ckpt", epoch, validation_metrics, "last", values["loss"])
            row = {
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in validation_metrics.items()},
                "val/pose_selection_score": values["pose"],
                **{f"best/{key}": value for key, value in best_values.items()},
                "stale_epochs": stale,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={values['loss']:.6f} best={best_values['loss']:.6f} "
                f"rot={values['rotation']:.2f}deg center={values['center']:.2f}px "
                f"depth={values['depth']:.4f} pose={values['pose']:.4f} "
                f"heatmap={validation_metrics['loss_heatmap']:.4f} "
                f"stale={stale}/{args.patience}"
            )
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (validation-loss patience).")
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    report = {
        "format": "rgb_render_self_recovery_matching_unet_training_v1",
        "architecture": "dual_encoder_lightweight_matching_unet",
        "best_validation_loss": best_values["loss"],
        "best_values": best_values,
        "best_epochs": best_epochs,
        "checkpoint_selection": {
            **CHECKPOINT_NAMES,
            "best": "best.ckpt (alias of best_loss.ckpt)",
            "early_stopping_criterion": "validation loss",
        },
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "training_groups": len(training_data),
        "validation_groups": len(validation_data),
        "parameter_counts": parameter_counts,
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
        "history": history,
        "arguments": safe_config(args),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
