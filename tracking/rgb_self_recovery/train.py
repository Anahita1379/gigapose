"""Train the RGB/render-aware verifier and iterative residual head."""

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
from tracking.rgb_self_recovery.model import (
    RGBRenderRecoveryNet,
    decode_outputs,
    rotation_geodesic_atan2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8, help="Instance groups per batch.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=384)
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
    parser.add_argument(
        "--anti-flip-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable an explicit pairwise quality-margin loss between the "
            "correct pose and the generated near-180-degree candidate."
        ),
    )
    parser.add_argument("--anti-flip-weight", type=float, default=1.0)
    parser.add_argument("--anti-flip-margin", type=float, default=0.25)
    parser.add_argument(
        "--anti-flip-min-angle-deg",
        type=float,
        default=150.0,
        help="Minimum correction angle considered a front/rear flip candidate.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--logger", choices=("none", "wandb"), default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    return parser.parse_args()


def safe_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    target_difference = target[:, None, :] - target[:, :, None]
    predicted_difference = predicted[:, None, :] - predicted[:, :, None]
    valid = torch.abs(target_difference) > float(margin)
    if not valid.any():
        return predicted.sum() * 0.0
    direction = torch.sign(target_difference)
    return torch.relu(
        float(margin) - direction[valid] * predicted_difference[valid]
    ).mean()


def anti_flip_quality_loss(
    predicted_quality: torch.Tensor,
    target_quality: torch.Tensor,
    rotation_targets: torch.Tensor,
    *,
    margin: float,
    minimum_angle_deg: float,
) -> dict[str, torch.Tensor]:
    """Require the correct candidate to outrank its near-180-degree pair.

    Lower quality is better. The correct candidate is the minimum target-quality
    item in each instance group. The flip item is the candidate whose target
    correction is closest to pi among candidates above ``minimum_angle_deg``.
    This works with existing generated shards because candidate 1 is already an
    exact object-Z 180-degree perturbation and its SO(3) correction has norm pi.
    """

    if (
        predicted_quality.ndim != 2
        or target_quality.shape != predicted_quality.shape
    ):
        raise ValueError("Anti-flip quality tensors must both have shape (B, C).")
    if rotation_targets.shape != (*predicted_quality.shape, 3):
        raise ValueError("Anti-flip rotation targets must have shape (B, C, 3).")

    angles = torch.linalg.vector_norm(rotation_targets, dim=-1)
    flip_candidates = angles >= math.radians(float(minimum_angle_deg))
    valid_groups = flip_candidates.any(dim=1)
    zero = predicted_quality.sum() * 0.0
    if not valid_groups.any():
        return {
            "loss": zero,
            "pair_accuracy": zero,
            "margin_accuracy": zero,
            "quality_gap": zero,
            "pair_fraction": zero,
        }

    correct_indices = target_quality.argmin(dim=1)
    distance_from_pi = torch.abs(angles - math.pi)
    masked_distance = torch.where(
        flip_candidates,
        distance_from_pi,
        torch.full_like(distance_from_pi, float("inf")),
    )
    flip_indices = masked_distance.argmin(dim=1)
    batch_indices = torch.arange(
        predicted_quality.shape[0], device=predicted_quality.device
    )
    correct_quality = predicted_quality[batch_indices, correct_indices][valid_groups]
    flip_quality = predicted_quality[batch_indices, flip_indices][valid_groups]
    quality_gap = flip_quality - correct_quality
    return {
        "loss": torch.relu(float(margin) - quality_gap).mean(),
        "pair_accuracy": (quality_gap > 0.0).float().mean(),
        "margin_accuracy": (quality_gap >= float(margin)).float().mean(),
        "quality_gap": quality_gap.mean(),
        "pair_fraction": valid_groups.float().mean(),
    }


def compute_loss(
    model: RGBRenderRecoveryNet,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    observed = batch["observed_mask"].to(device, non_blocking=True)
    rendered = batch["rendered"].to(device, non_blocking=True)
    batch_size, candidates = rendered.shape[:2]
    rgb_flat = rgb[:, None].expand(-1, candidates, -1, -1, -1).reshape(
        batch_size * candidates, *rgb.shape[1:]
    )
    observed_flat = observed[:, None].expand(
        -1, candidates, -1, -1, -1
    ).reshape(batch_size * candidates, *observed.shape[1:])
    rendered_flat = rendered.reshape(
        batch_size * candidates, *rendered.shape[2:]
    )
    raw = model(rgb_flat, observed_flat, rendered_flat)
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
            decoded["rotation_rad"][correctable],
            rotation_target[correctable],
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
    quality_loss = nn.functional.smooth_l1_loss(
        decoded["quality"], quality_target
    )
    ranking = ranking_loss(
        decoded["quality"].reshape(batch_size, candidates),
        quality_target.reshape(batch_size, candidates),
        args.ranking_margin,
    )
    anti_flip = anti_flip_quality_loss(
        decoded["quality"].reshape(batch_size, candidates),
        quality_target.reshape(batch_size, candidates),
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
        + anti_flip_term
    )

    predicted_best = decoded["quality"].reshape(batch_size, candidates).argmin(dim=1)
    target_group = quality_target.reshape(batch_size, candidates)
    selected_target_quality = target_group[
        torch.arange(batch_size, device=device), predicted_best
    ].mean()
    oracle_quality = target_group.min(dim=1).values.mean()
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
        "anti_flip_margin_accuracy": float(
            anti_flip["margin_accuracy"].detach()
        ),
        "anti_flip_quality_gap": float(anti_flip["quality_gap"].detach()),
        "anti_flip_pair_fraction": float(anti_flip["pair_fraction"].detach()),
        "center_error_crop_px": float(center_error_px.detach()),
        "log_depth_abs_error": float(log_depth_error.detach()),
        "rotation_error_deg": float(rotation_error_deg.detach()),
        "selected_target_quality": float(selected_target_quality.detach()),
        "oracle_target_quality": float(oracle_quality.detach()),
    }


def main() -> None:
    args = parse_args()
    if args.anti_flip_weight < 0:
        raise ValueError("--anti-flip-weight must be non-negative.")
    if args.anti_flip_margin < 0:
        raise ValueError("--anti-flip-margin must be non-negative.")
    if not 0.0 < args.anti_flip_min_angle_deg <= 180.0:
        raise ValueError("--anti-flip-min-angle-deg must be in (0, 180].")
    print(
        "Anti-flip training: "
        + (
            "enabled "
            f"(weight={args.anti_flip_weight:g}, "
            f"margin={args.anti_flip_margin:g}, "
            f"minimum_angle={args.anti_flip_min_angle_deg:g} deg)"
            if args.anti_flip_training
            else "disabled"
        )
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    training_data = RGBRenderRecoveryDataset(
        args.data, training=True, seed=args.seed
    )
    validation_data = RGBRenderRecoveryDataset(
        args.validation_data, training=False, seed=args.seed
    )
    for key in ("crop_size", "crop_scale"):
        if training_data.manifest[key] != validation_data.manifest[key]:
            raise ValueError(f"Train/validation {key} values differ.")
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model = RGBRenderRecoveryNet(args.width, args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
            config={
                **safe_config(args),
                "training_groups": len(training_data),
                "validation_groups": len(validation_data),
                "uses_observed_depth": False,
            },
        )

    history = []
    best_loss = float("inf")
    stale = 0
    started = time.perf_counter()

    def run_epoch(loader, training: bool) -> dict[str, float]:
        model.train(training)
        sums: dict[str, float] = {}
        groups = 0
        for batch in loader:
            with torch.set_grad_enabled(training):
                loss, metrics = compute_loss(model, batch, args, args.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    optimizer.step()
            count = int(batch["rgb"].shape[0])
            groups += count
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * count
        if groups == 0:
            raise RuntimeError("Recovery epoch contained no groups.")
        return {name: value / groups for name, value in sums.items()}

    def save_checkpoint(name: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "format": "rgb_render_self_recovery_v1",
                "model_state": model.state_dict(),
                "model_config": {
                    "width": args.width,
                    "hidden_dim": args.hidden_dim,
                },
                "inference_config": {
                    "crop_size": training_data.manifest["crop_size"],
                    "crop_scale": training_data.manifest["crop_scale"],
                    "max_center_crop_px": args.max_center_crop_px,
                    "max_log_depth": args.max_log_depth,
                    "max_rotation_deg": args.max_rotation_deg,
                    "uses_observed_depth": False,
                },
                "epoch": epoch,
                "validation_loss": val_loss,
                "arguments": safe_config(args),
            },
            args.output_dir / name,
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(training_loader, True)
            with torch.no_grad():
                validation_metrics = run_epoch(validation_loader, False)
            val_loss = validation_metrics["loss"]
            improved = val_loss < best_loss - args.min_delta
            if improved:
                best_loss = val_loss
                stale = 0
                save_checkpoint("best.ckpt", epoch, val_loss)
            else:
                stale += 1
            save_checkpoint("last.ckpt", epoch, val_loss)
            row = {
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in validation_metrics.items()},
                "best_validation_loss": best_loss,
                "stale_epochs": stale,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={val_loss:.6f} best={best_loss:.6f} "
                f"rot={validation_metrics['rotation_error_deg']:.2f}deg "
                f"center={validation_metrics['center_error_crop_px']:.2f}px "
                f"flip_acc={validation_metrics['anti_flip_pair_accuracy']:.3f} "
                f"flip_margin={validation_metrics['anti_flip_margin_accuracy']:.3f} "
                f"stale={stale}/{args.patience}"
            )
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    report = {
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "uses_observed_depth": False,
        "training_groups": len(training_data),
        "validation_groups": len(validation_data),
        "best_checkpoint": str(args.output_dir / "best.ckpt"),
        "history": history,
        "arguments": safe_config(args),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
