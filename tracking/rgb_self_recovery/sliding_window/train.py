"""Train the CNN RGB/CAD verifier with geometry-aware checkpoint selection.

The network and loss implementation remain the original RGB self-recovery
implementation.  This isolated entry point adds checkpoint bookkeeping needed
by the sliding-window experiments without modifying the original trainer.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tracking.rgb_self_recovery.dataset import RGBRenderRecoveryDataset
from tracking.rgb_self_recovery.model import RGBRenderRecoveryNet
from tracking.rgb_self_recovery.train import compute_loss, parse_args, safe_config

from .checkpoint_selection import CHECKPOINT_NAMES, checkpoint_selection_values


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

    training_data = RGBRenderRecoveryDataset(args.data, training=True, seed=args.seed)
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
                "pose_checkpoint_score": (
                    "mean(center_px/max_center_px, log_depth/max_log_depth, "
                    "rotation_deg/max_rotation_deg)"
                ),
            },
        )

    history: list[dict[str, float | int]] = []
    best_values = {name: float("inf") for name in CHECKPOINT_NAMES}
    best_epochs = {name: 0 for name in CHECKPOINT_NAMES}
    stale = 0
    started = time.perf_counter()

    def run_epoch(loader: DataLoader, training: bool) -> dict[str, float]:
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

    def save_checkpoint(
        name: str,
        epoch: int,
        validation_metrics: dict[str, float],
        criterion: str,
        criterion_value: float,
    ) -> None:
        torch.save(
            {
                "format": "rgb_render_self_recovery_v1",
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
            },
            args.output_dir / name,
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(training_loader, True)
            with torch.no_grad():
                validation_metrics = run_epoch(validation_loader, False)
            values = checkpoint_selection_values(
                validation_metrics,
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
                    save_checkpoint(
                        checkpoint_name,
                        epoch,
                        validation_metrics,
                        criterion,
                        values[criterion],
                    )
                    if criterion == "loss":
                        # Backward-compatible alias used by existing commands.
                        save_checkpoint(
                            "best.ckpt", epoch, validation_metrics, criterion, values[criterion]
                        )
            save_checkpoint("last.ckpt", epoch, validation_metrics, "last", values["loss"])
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
                f"rot={values['rotation']:.2f}deg "
                f"center={values['center']:.2f}px depth={values['depth']:.4f} "
                f"pose={values['pose']:.4f} "
                f"flip_acc={validation_metrics['anti_flip_pair_accuracy']:.3f} "
                f"flip_margin={validation_metrics['anti_flip_margin_accuracy']:.3f} "
                f"stale={stale}/{args.patience}"
            )
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (validation-loss patience).")
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    report = {
        "format": "rgb_render_self_recovery_training_multi_checkpoint_v1",
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
        "uses_observed_depth": False,
        "training_groups": len(training_data),
        "validation_groups": len(validation_data),
        "best_checkpoint": str(args.output_dir / "best.ckpt"),
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
        "history": history,
        "arguments": safe_config(args),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
