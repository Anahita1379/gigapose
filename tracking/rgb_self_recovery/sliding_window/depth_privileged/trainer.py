"""Shared training loops for three RGB-only depth-privileged students."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tracking.rgb_self_recovery.train import safe_config
from tracking.rgb_self_recovery.sliding_window.checkpoint_selection import (
    CHECKPOINT_NAMES,
    checkpoint_selection_values,
)
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.model import (
    DINOPatchMatchingUNet,
)
from tracking.rgb_self_recovery.sliding_window.matching_unet.train import (
    _validate_args,
    build_parser,
)

from .dataset import PrivilegedDepthRecoveryDataset
from .losses import (
    auxiliary_depth_objective,
    distillation_objective,
    forward_grouped,
    supervised_objective,
)
from .model import AuxiliaryDepthStudent, DepthConditionedTeacher


def parser(*, teacher=False, variant=None):
    value = build_parser()
    value.description = __doc__
    value.add_argument("--dino-mode", choices=("frozen", "last_block"), default="frozen")
    value.add_argument("--dino-model", default="dinov2_vits14")
    value.add_argument("--dino-input-size", type=int, default=224)
    value.add_argument("--dino-learning-rate", type=float, default=1e-5)
    value.add_argument("--maximum-depth-m", type=float, default=200.0)
    value.add_argument("--initialize-from", type=Path)
    if not teacher:
        value.set_defaults(
            teacher_checkpoint=None,
            auxiliary_depth_weight=0.0,
            distillation_output_weight=0.0,
            distillation_feature_weight=0.0,
        )
        if variant in {"teacher_student", "combined"}:
            value.add_argument("--teacher-checkpoint", type=Path, required=True)
            value.add_argument("--distillation-output-weight", type=float, default=0.5)
            value.add_argument("--distillation-feature-weight", type=float, default=0.1)
        if variant in {"auxiliary", "combined"}:
            value.add_argument("--auxiliary-depth-weight", type=float, default=0.25)
    return value


def model_kwargs(args, *, pretrained):
    return dict(
        width=args.width,
        hidden_dim=args.hidden_dim,
        dino_mode=args.dino_mode,
        dino_model=args.dino_model,
        dino_input_size=args.dino_input_size,
        dino_pretrained=pretrained,
    )


def model_config(args):
    return {
        "width": args.width,
        "hidden_dim": args.hidden_dim,
        "dino_mode": args.dino_mode,
        "dino_model": args.dino_model,
        "dino_input_size": args.dino_input_size,
    }


def _check_config(payload, args):
    source = payload.get("model_config", {})
    for name, expected in model_config(args).items():
        if source.get(name) != expected:
            raise ValueError(
                f"Checkpoint {name}={source.get(name)!r}; expected {expected!r}"
            )


def initialize_student(model, path, args):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "rgb_render_self_recovery_dino_matching_unet_v1":
        raise ValueError("Student initialization must be a DINO matching U-Net checkpoint")
    _check_config(payload, args)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    allowed = {name for name in missing if name.startswith("depth_")}
    if set(missing) != allowed or unexpected:
        raise ValueError(f"Incompatible initialization: missing={missing}, unexpected={unexpected}")


def load_teacher(path, args, device):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "rgb_render_self_recovery_depth_teacher_v1":
        raise ValueError("--teacher-checkpoint is not a privileged-depth teacher")
    _check_config(payload, args)
    teacher = DepthConditionedTeacher(
        **model_kwargs(args, pretrained=False),
        maximum_depth_m=float(payload["model_config"]["maximum_depth_m"]),
    )
    teacher.load_state_dict(payload["model_state"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def loaders(args):
    train = PrivilegedDepthRecoveryDataset(args.data, training=True, seed=args.seed)
    validation = PrivilegedDepthRecoveryDataset(
        args.validation_data, training=False, seed=args.seed
    )
    for key in ("crop_size", "crop_scale"):
        if train.manifest[key] != validation.manifest[key]:
            raise ValueError(f"Train/validation {key} values differ")
    values = {}
    for name, dataset in (("train", train), ("validation", validation)):
        values[name] = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )
    return train, validation, values


def optimizer_for(model, args):
    base, dino = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (dino if name.startswith("dino.") else base).append(parameter)
    groups = [{"params": base, "lr": args.learning_rate}]
    if dino:
        groups.append({"params": dino, "lr": args.dino_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _run_training(args, model, training_data, validation_data, data_loaders,
                  optimizer, step, save_payload, architecture):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.logger == "wandb":
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name or args.output_dir.name, dir=str(args.output_dir),
            mode="offline" if args.wandb_offline else "online",
            config=safe_config(args),
        )
    history, stale = [], 0
    best = {name: float("inf") for name in CHECKPOINT_NAMES}
    best_epochs = {name: 0 for name in CHECKPOINT_NAMES}
    started = time.perf_counter()

    def epoch(split):
        training = split == "train"
        model.train(training)
        sums, seen = {}, 0
        for batch in data_loaders[split]:
            with torch.set_grad_enabled(training):
                loss, metrics = step(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.gradient_clip,
                    )
                    optimizer.step()
            count = int(batch["rgb"].shape[0])
            seen += count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * count
        if not seen:
            raise RuntimeError(f"{split} epoch had no groups")
        return {key: value / seen for key, value in sums.items()}

    try:
        for epoch_index in range(1, args.epochs + 1):
            train_metrics = epoch("train")
            with torch.no_grad():
                validation_metrics = epoch("validation")
            values = checkpoint_selection_values(
                validation_metrics,
                max_center_px=args.max_center_crop_px,
                max_log_depth=args.max_log_depth,
                max_rotation_deg=args.max_rotation_deg,
            )
            improved = values["loss"] < best["loss"] - args.min_delta
            stale = 0 if improved else stale + 1
            for criterion, filename in CHECKPOINT_NAMES.items():
                threshold = args.min_delta if criterion == "loss" else 0.0
                if values[criterion] < best[criterion] - threshold:
                    best[criterion], best_epochs[criterion] = values[criterion], epoch_index
                    torch.save(save_payload(epoch_index, validation_metrics, criterion, values[criterion]), args.output_dir / filename)
                    if criterion == "loss":
                        torch.save(save_payload(epoch_index, validation_metrics, criterion, values[criterion]), args.output_dir / "best.ckpt")
            torch.save(save_payload(epoch_index, validation_metrics, "last", values["loss"]), args.output_dir / "last.ckpt")
            row = {
                "epoch": epoch_index,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in validation_metrics.items()},
                "val/pose_selection_score": values["pose"],
                "stale_epochs": stale,
            }
            history.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch_index)
            print(
                f"epoch={epoch_index:03d} train={train_metrics['loss']:.6f} "
                f"val={values['loss']:.6f} rot={values['rotation']:.2f}deg "
                f"center={values['center']:.2f}px depth={values['depth']:.4f} "
                f"stale={stale}/{args.patience}", flush=True,
            )
            if stale >= args.patience:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    report = {
        "format": "rgb_render_self_recovery_depth_privileged_training_v1",
        "architecture": architecture,
        "best_values": best,
        "best_epochs": best_epochs,
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "training_groups": len(training_data),
        "validation_groups": len(validation_data),
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
        "arguments": safe_config(args),
        "history": history,
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))


def train_teacher():
    args = parser(teacher=True).parse_args()
    _validate_args(args)
    if args.maximum_depth_m <= 0:
        raise ValueError("--maximum-depth-m must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    train, validation, data_loaders = loaders(args)
    model = DepthConditionedTeacher(
        **model_kwargs(args, pretrained=True), maximum_depth_m=args.maximum_depth_m
    )
    if args.initialize_from:
        initialize_student(model, args.initialize_from, args)
    model.to(args.device)
    optimizer = optimizer_for(model, args)

    def step(batch):
        raw, _encoding, batch_size, candidates = forward_grouped(
            model, batch, args.device, privileged_depth=True
        )
        loss, metrics = supervised_objective(
            raw, batch, args, args.device, batch_size, candidates
        )
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def payload(epoch, metrics, criterion, value):
        return {
            "format": "rgb_render_self_recovery_depth_teacher_v1",
            "model_state": model.state_dict(),
            "model_config": {**model_config(args), "maximum_depth_m": args.maximum_depth_m},
            "epoch": epoch, "validation_metrics": metrics,
            "selection_criterion": criterion, "selection_value": value,
            "arguments": safe_config(args),
        }

    _run_training(args, model, train, validation, data_loaders, optimizer, step, payload, "depth_conditioned_dino_matching_unet_teacher")


def train_student(variant):
    if variant not in {"auxiliary", "teacher_student", "combined"}:
        raise ValueError(variant)
    args = parser(teacher=False, variant=variant).parse_args()
    _validate_args(args)
    if args.maximum_depth_m <= 0 or min(
        args.auxiliary_depth_weight,
        args.distillation_output_weight,
        args.distillation_feature_weight,
    ) < 0:
        raise ValueError("Depth and distillation weights must be non-negative")
    needs_teacher = variant in {"teacher_student", "combined"}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    train, validation, data_loaders = loaders(args)
    student_class = AuxiliaryDepthStudent if variant in {"auxiliary", "combined"} else DINOPatchMatchingUNet
    model = student_class(**model_kwargs(args, pretrained=True))
    if args.initialize_from:
        initialize_student(model, args.initialize_from, args)
    model.to(args.device)
    teacher = load_teacher(args.teacher_checkpoint, args, args.device) if needs_teacher else None
    optimizer = optimizer_for(model, args)

    def step(batch):
        raw, encoding, batch_size, candidates = forward_grouped(
            model, batch, args.device
        )
        supervised, metrics = supervised_objective(
            raw, batch, args, args.device, batch_size, candidates
        )
        auxiliary = raw["quality_raw"].sum() * 0.0
        if variant in {"auxiliary", "combined"}:
            auxiliary = auxiliary_depth_objective(
                model, encoding, batch, args, args.device
            )
        distillation = output_distill = feature_distill = auxiliary * 0.0
        if teacher is not None:
            with torch.no_grad():
                teacher_raw, teacher_encoding, _, _ = forward_grouped(
                    teacher, batch, args.device, privileged_depth=True
                )
            distillation, output_distill, feature_distill = distillation_objective(
                raw, encoding, teacher_raw, teacher_encoding, args
            )
        total = supervised + args.auxiliary_depth_weight * auxiliary + distillation
        metrics.update(
            loss=float(total.detach()),
            loss_auxiliary_depth=float(auxiliary.detach()),
            loss_distillation=float(distillation.detach()),
            loss_distillation_output=float(output_distill.detach()),
            loss_distillation_feature=float(feature_distill.detach()),
        )
        return total, metrics

    def payload(epoch, metrics, criterion, value):
        state = model.inference_state_dict() if hasattr(model, "inference_state_dict") else model.state_dict()
        return {
            "format": "rgb_render_self_recovery_dino_matching_unet_v1",
            "model_state": state,
            "model_config": model_config(args),
            "inference_config": {
                "crop_size": train.manifest["crop_size"],
                "crop_scale": train.manifest["crop_scale"],
                "max_center_crop_px": args.max_center_crop_px,
                "max_log_depth": args.max_log_depth,
                "max_rotation_deg": args.max_rotation_deg,
                "uses_observed_depth": False,
            },
            "training_variant": variant,
            "privileged_depth_used_during_training": True,
            "depth_required_at_inference": False,
            "epoch": epoch, "validation_metrics": metrics,
            "selection_criterion": criterion, "selection_value": value,
            "arguments": safe_config(args),
        }

    _run_training(args, model, train, validation, data_loaders, optimizer, step, payload, f"dino_matching_unet_{variant}_depth_privileged_student")
