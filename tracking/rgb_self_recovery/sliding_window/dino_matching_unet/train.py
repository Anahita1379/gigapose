"""Train DINO patch-feature matching U-Net on the existing recovery split."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tracking.rgb_self_recovery.dataset import RGBRenderRecoveryDataset
from tracking.rgb_self_recovery.train import safe_config
from tracking.rgb_self_recovery.sliding_window.checkpoint_selection import (
    CHECKPOINT_NAMES,
    checkpoint_selection_values,
)
from tracking.rgb_self_recovery.sliding_window.matching_unet.train import (
    _validate_args,
    build_parser,
    compute_loss,
)

from .model import DINOPatchMatchingUNet


def parser():
    value = build_parser()
    value.description = __doc__
    value.add_argument("--dino-mode", choices=("frozen", "last_block"), default="frozen")
    value.add_argument("--dino-model", default="dinov2_vits14")
    value.add_argument("--dino-input-size", type=int, default=224)
    value.add_argument("--dino-learning-rate", type=float, default=1e-5)
    value.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Optional DINO matching U-Net checkpoint used to initialize all "
            "model weights before training."
        ),
    )
    return value


def initialize_from_checkpoint(
    model: DINOPatchMatchingUNet,
    checkpoint: Path,
    *,
    width: int,
    hidden_dim: int,
    dino_model: str,
    dino_input_size: int,
) -> dict[str, Any]:
    """Load a compatible frozen or last-block DINO matching U-Net checkpoint."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("format") != "rgb_render_self_recovery_dino_matching_unet_v1":
        raise ValueError(
            "--initialize-from must be a DINO matching U-Net checkpoint"
        )
    source = payload.get("model_config", {})
    for key, expected in (
        ("width", width),
        ("hidden_dim", hidden_dim),
        ("dino_model", dino_model),
        ("dino_input_size", dino_input_size),
    ):
        if source.get(key) != expected:
            raise ValueError(
                f"Initialization checkpoint {key}={source.get(key)!r}; "
                f"current value is {expected!r}"
            )
    model.load_state_dict(payload["model_state"])
    return payload


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    if args.dino_learning_rate <= 0:
        raise ValueError("--dino-learning-rate must be positive.")
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
    model = DINOPatchMatchingUNet(
        args.width,
        args.hidden_dim,
        dino_mode=args.dino_mode,
        dino_model=args.dino_model,
        dino_input_size=args.dino_input_size,
        dino_pretrained=True,
    )
    if args.initialize_from is not None:
        payload = initialize_from_checkpoint(
            model,
            args.initialize_from,
            width=args.width,
            hidden_dim=args.hidden_dim,
            dino_model=args.dino_model,
            dino_input_size=args.dino_input_size,
        )
        source_mode = payload["model_config"].get("dino_mode", "unknown")
        print(
            f"Initialized from {args.initialize_from} "
            f"(source dino_mode={source_mode}, target dino_mode={args.dino_mode})"
        )
    model = model.to(args.device)
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("dino.")
    ]
    dino_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("dino.")
    ]
    groups = [{"params": base_parameters, "lr": args.learning_rate}]
    if dino_parameters:
        groups.append({"params": dino_parameters, "lr": args.dino_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_counts = model.parameter_counts()
    print(
        json.dumps(
            {
                "architecture": "dino_patch_matching_unet",
                "dino_mode": args.dino_mode,
                **parameter_counts,
            },
            indent=2,
        )
    )

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
        groups_seen = 0
        for batch in loaders[name]:
            with torch.set_grad_enabled(training):
                loss, metrics = compute_loss(model, batch, args, args.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        args.gradient_clip,
                    )
                    optimizer.step()
            count = int(batch["rgb"].shape[0])
            groups_seen += count
            for key, metric in metrics.items():
                sums[key] = sums.get(key, 0.0) + metric * count
        if not groups_seen:
            raise RuntimeError(f"{name} epoch contained no groups.")
        return {key: metric / groups_seen for key, metric in sums.items()}

    def save(
        filename: str,
        epoch: int,
        validation_metrics: dict[str, float],
        criterion: str,
        criterion_value: float,
    ) -> None:
        torch.save(
            {
                "format": "rgb_render_self_recovery_dino_matching_unet_v1",
                "model_state": model.state_dict(),
                "model_config": {
                    "width": args.width,
                    "hidden_dim": args.hidden_dim,
                    "dino_mode": args.dino_mode,
                    "dino_model": args.dino_model,
                    "dino_input_size": args.dino_input_size,
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
                **{f"train/{key}": metric for key, metric in train_metrics.items()},
                **{f"val/{key}": metric for key, metric in validation_metrics.items()},
                "val/pose_selection_score": values["pose"],
                **{f"best/{key}": metric for key, metric in best_values.items()},
                "stale_epochs": stale,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "dino_learning_rate": (
                    optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
                ),
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
        "format": "rgb_render_self_recovery_dino_matching_unet_training_v1",
        "architecture": "dino_patch_dual_encoder_matching_unet",
        "dino_mode": args.dino_mode,
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
