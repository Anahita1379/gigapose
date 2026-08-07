"""Train the eight-frame gated cross-attention candidate transformer."""

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

from tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.geometry import (
    rotation_angle,
    so3_exp,
    so3_log,
)

from .dataset import BundleCollection, TrainingWindowDataset, collection_statistics
from .model import GatedCandidateTransformer


FORMAT = "rgb_self_recovery_gated_candidate_transformer_v5"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, action="append", required=True,
                       help="Training candidate bundle; repeat to combine datasets.")
    value.add_argument("--validation-data", type=Path, action="append", required=True,
                       help="Validation candidate bundle; repeat if needed.")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--window-length", type=int, default=8)
    value.add_argument("--window-stride", type=int, default=1)
    value.add_argument("--validation-stride", type=int, default=8)
    value.add_argument("--model-dim", type=int, default=128)
    value.add_argument("--heads", type=int, default=4)
    value.add_argument("--feedforward-dim", type=int, default=256)
    value.add_argument("--cross-layers", type=int, default=2)
    value.add_argument("--dropout", type=float, default=0.1)
    value.add_argument(
        "--attention-future-lag",
        type=int,
        default=4,
        help="Maximum future-frame offset visible to each temporal query.",
    )
    value.add_argument("--epochs", type=int, default=120)
    value.add_argument("--batch-size", type=int, default=24)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--candidate-weight", type=float, default=1.0)
    value.add_argument("--expected-cost-weight", type=float, default=0.1)
    value.add_argument("--orientation-weight", type=float, default=0.5)
    value.add_argument("--translation-weight", type=float, default=1.0)
    value.add_argument("--rotation-weight", type=float, default=1.0)
    value.add_argument("--abstention-weight", type=float, default=0.25)
    value.add_argument("--translation-trust-weight", type=float, default=0.0,
                       help="Optional learned GigaPose fallback ablation (disabled by default).")
    value.add_argument("--translation-fallback-margin-m", type=float, default=0.0)
    value.add_argument("--rotation-trust-weight", type=float, default=0.0,
                       help="Optional learned GigaPose fallback ablation (disabled by default).")
    value.add_argument("--rotation-fallback-margin-deg", type=float, default=0.0)
    value.add_argument("--velocity-weight", type=float, default=0.1)
    value.add_argument("--acceleration-weight", type=float, default=0.05)
    value.add_argument("--jerk-weight", type=float, default=0.02)
    value.add_argument("--fixed-lag-regularization-weight", type=float, default=0.01)
    value.add_argument("--velocity-scale-mps", type=float, default=10.0)
    value.add_argument("--acceleration-scale-mps2", type=float, default=20.0)
    value.add_argument("--jerk-scale-mps3", type=float, default=100.0)
    value.add_argument("--translation-fixed-lag-layers", type=int, default=1)
    value.add_argument("--translation-position-scale-m", type=float, default=100.0)
    value.add_argument("--translation-relative-scale-m", type=float, default=10.0)
    value.add_argument("--maximum-translation-refinement-m", type=float, default=20.0)
    value.add_argument("--disable-translation-fixed-lag", action="store_true")
    value.add_argument("--translation-scale-m", type=float, default=1.0)
    value.add_argument("--rotation-scale-deg", type=float, default=20.0)
    value.add_argument("--orientation-correct-deg", type=float, default=90.0)
    value.add_argument(
        "--validation-orientation-gate-weight",
        type=float,
        default=1.0,
        help="Orientation log-probability weight used for selected-pose validation.",
    )
    value.add_argument("--abstention-improvement-margin", type=float, default=0.05)
    value.add_argument("--max-oracle-cost", type=float, default=20.0)
    value.add_argument("--gradient-clip", type=float, default=5.0)
    value.add_argument("--patience", type=int, default=12)
    value.add_argument("--min-delta", type=float, default=1e-4)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=20260804)
    value.add_argument("--allow-run-overlap", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def _validate_collections(train: BundleCollection, validation: BundleCollection, allow: bool):
    if train.candidate_feature_names != validation.candidate_feature_names:
        raise ValueError("Train/validation candidate feature schemas differ")
    if train.frame_feature_names != validation.frame_feature_names:
        raise ValueError("Train/validation frame feature schemas differ")
    if train.candidate_count != validation.candidate_count:
        raise ValueError("Train/validation candidate counts differ")
    overlap = sorted(train.runs & validation.runs)
    if overlap and not allow:
        raise ValueError(f"Physical run leakage between train and validation: {overlap}")
    return overlap


def _gather_candidate(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    suffix = values.shape[3:]
    gather_shape = (*indices.shape, 1, *suffix)
    expanded = indices[..., None]
    for _ in suffix:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand(gather_shape)
    return values.gather(2, expanded).squeeze(2)


def _translation_motion_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    frame_valid: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare translation derivatives while respecting gaps and padded frames."""
    zero = predicted.new_zeros(())
    if predicted.shape[1] < 2:
        return zero, zero, zero
    dt = (times[:, 1:] - times[:, :-1]).clamp_min(0.02)
    pair_valid = frame_valid[:, 1:] & frame_valid[:, :-1] & (dt <= 1.0)
    predicted_velocity = (predicted[:, 1:] - predicted[:, :-1]) / dt[..., None]
    target_velocity = (target[:, 1:] - target[:, :-1]) / dt[..., None]
    velocity_error = torch.linalg.vector_norm(
        predicted_velocity - target_velocity, dim=-1
    )
    velocity_loss = (
        velocity_error[pair_valid].mean() / args.velocity_scale_mps
        if pair_valid.any() else zero
    )

    if predicted.shape[1] < 3:
        return velocity_loss, zero, zero
    acceleration_dt = 0.5 * (dt[:, 1:] + dt[:, :-1])
    acceleration_valid = pair_valid[:, 1:] & pair_valid[:, :-1]
    predicted_acceleration = (
        predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
    ) / acceleration_dt[..., None]
    target_acceleration = (
        target_velocity[:, 1:] - target_velocity[:, :-1]
    ) / acceleration_dt[..., None]
    acceleration_error = torch.linalg.vector_norm(
        predicted_acceleration - target_acceleration, dim=-1
    )
    acceleration_loss = (
        acceleration_error[acceleration_valid].mean() / args.acceleration_scale_mps2
        if acceleration_valid.any() else zero
    )

    if predicted.shape[1] < 4:
        return velocity_loss, acceleration_loss, zero
    jerk_dt = 0.5 * (acceleration_dt[:, 1:] + acceleration_dt[:, :-1])
    jerk_valid = acceleration_valid[:, 1:] & acceleration_valid[:, :-1]
    predicted_jerk = (
        predicted_acceleration[:, 1:] - predicted_acceleration[:, :-1]
    ) / jerk_dt[..., None]
    target_jerk = (
        target_acceleration[:, 1:] - target_acceleration[:, :-1]
    ) / jerk_dt[..., None]
    jerk_error = torch.linalg.vector_norm(predicted_jerk - target_jerk, dim=-1)
    jerk_loss = (
        jerk_error[jerk_valid].mean() / args.jerk_scale_mps3
        if jerk_valid.any() else zero
    )
    return velocity_loss, acceleration_loss, jerk_loss


def _step(model, batch, device, args, optimizer=None):
    candidate = batch["candidate_features"].to(device)
    frame = batch["frame_features"].to(device)
    candidate_valid = batch["candidate_valid"].to(device)
    frame_valid = batch["frame_valid"].to(device)
    poses = batch["poses"].to(device)
    target_pose = batch["ground_truth_pose"].to(device)
    labels = batch["labels"].to(device)
    costs = batch["oracle_costs"].to(device)
    rotation_errors = batch["rotation_error_deg"].to(device)
    times = batch["time_s"].to(device)
    output = model(
        candidate,
        frame,
        candidate_valid,
        frame_valid,
        candidate_visual_features=(
            batch["candidate_visual_features"].to(device)
            if "candidate_visual_features" in batch else None
        ),
        frame_visual_features=(
            batch["frame_visual_features"].to(device)
            if "frame_visual_features" in batch else None
        ),
        candidate_poses=poses,
        time_s=times,
    )

    flat = frame_valid.reshape(-1)
    logits = output["candidate_logits"].reshape(-1, model.candidate_count)[flat]
    labels_flat = labels.reshape(-1)[flat]
    classification = F.cross_entropy(logits, labels_flat)
    probabilities = logits.softmax(dim=-1)
    cost_flat = costs.reshape(-1, model.candidate_count)[flat]
    valid_flat = candidate_valid.reshape(-1, model.candidate_count)[flat]
    safe_cost = torch.where(
        valid_flat, cost_flat.clamp(max=args.max_oracle_cost), torch.zeros_like(cost_flat)
    )
    expected_cost = (probabilities * safe_cost).sum(dim=-1).mean()

    orientation_logits = output["orientation_logits"]
    orientation_target = (rotation_errors < args.orientation_correct_deg).float()
    orientation_mask = candidate_valid & frame_valid[..., None]
    orientation_loss = F.binary_cross_entropy_with_logits(
        orientation_logits[orientation_mask], orientation_target[orientation_mask]
    )

    oracle_pose = _gather_candidate(poses, labels)
    oracle_residual = _gather_candidate(output["residual"], labels)
    oracle_sigma = _gather_candidate(output["log_sigma"], labels)
    fixed_lag_translation = (
        output["translation_fixed_lag_gate"][..., None]
        * output["translation_fixed_lag_delta"]
    )
    predicted_position = (
        oracle_pose[..., :3, 3]
        + oracle_residual[..., :3]
        + fixed_lag_translation
    )
    rotation_target = so3_log(
        oracle_pose[..., :3, :3].transpose(-1, -2) @ target_pose[..., :3, :3]
    )
    translation_error = torch.linalg.norm(
        predicted_position - target_pose[..., :3, 3], dim=-1
    ) / args.translation_scale_m
    rotation_error = torch.linalg.norm(
        oracle_residual[..., 3:] - rotation_target, dim=-1
    ) / np.deg2rad(args.rotation_scale_deg)
    translation_nll = (
        torch.exp(-oracle_sigma[..., 0]) * translation_error + oracle_sigma[..., 0]
    )[frame_valid].mean()
    rotation_nll = (
        torch.exp(-oracle_sigma[..., 1]) * rotation_error + oracle_sigma[..., 1]
    )[frame_valid].mean()

    baseline_cost = costs[..., 0]
    oracle_cost = costs.gather(2, labels[..., None]).squeeze(-1)
    abstention_target = (
        oracle_cost + args.abstention_improvement_margin < baseline_cost
    ).float()
    abstention_loss = F.binary_cross_entropy_with_logits(
        output["abstention_logits"][frame_valid], abstention_target[frame_valid]
    )

    # Ground truth is used only to supervise a deployable trust prediction. At
    # inference the trust heads, not ground truth, independently decide whether
    # refined translation and rotation should replace candidate-zero GigaPose.
    all_predicted_positions = (
        poses[..., :3, 3]
        + output["residual"][..., :3]
        + fixed_lag_translation[..., None, :]
    )
    all_translation_error = torch.linalg.vector_norm(
        all_predicted_positions - target_pose[..., None, :3, 3], dim=-1
    )
    baseline_translation_error = torch.linalg.vector_norm(
        poses[..., 0, :3, 3] - target_pose[..., :3, 3], dim=-1
    )
    translation_trust_target = (
        all_translation_error + args.translation_fallback_margin_m
        < baseline_translation_error[..., None]
    ).float().detach()
    translation_trust_mask = candidate_valid & frame_valid[..., None]
    translation_trust_loss = F.binary_cross_entropy_with_logits(
        output["translation_trust_logits"][translation_trust_mask],
        translation_trust_target[translation_trust_mask],
    )

    all_predicted_rotations = (
        poses[..., :3, :3] @ so3_exp(output["residual"][..., 3:])
    )
    all_rotation_error = torch.rad2deg(
        rotation_angle(
            all_predicted_rotations,
            target_pose[..., None, :3, :3],
        )
    )
    baseline_rotation_error = torch.rad2deg(
        rotation_angle(poses[..., 0, :3, :3], target_pose[..., :3, :3])
    )
    rotation_trust_target = (
        all_rotation_error + args.rotation_fallback_margin_deg
        < baseline_rotation_error[..., None]
    ).float().detach()
    rotation_trust_mask = candidate_valid & frame_valid[..., None]
    rotation_trust_loss = F.binary_cross_entropy_with_logits(
        output["rotation_trust_logits"][rotation_trust_mask],
        rotation_trust_target[rotation_trust_mask],
    )

    velocity_loss, acceleration_loss, jerk_loss = _translation_motion_losses(
        predicted_position,
        target_pose[..., :3, 3],
        times,
        frame_valid,
        args,
    )
    fixed_lag_regularization = (
        torch.linalg.vector_norm(fixed_lag_translation, dim=-1)[frame_valid].mean()
        / args.maximum_translation_refinement_m
    )

    # Deployment-oriented validation metrics use the candidate the model would
    # actually select, not the oracle candidate used to supervise residual NLL.
    selection_logits = output["candidate_logits"] + (
        args.validation_orientation_gate_weight
        * F.logsigmoid(output["orientation_logits"])
    )
    selected_indices = selection_logits.argmax(dim=-1)
    selected_pose = _gather_candidate(poses, selected_indices)
    selected_residual = _gather_candidate(output["residual"], selected_indices)
    selected_position = (
        selected_pose[..., :3, 3]
        + selected_residual[..., :3]
        + fixed_lag_translation
    )
    selected_rotation = (
        selected_pose[..., :3, :3] @ so3_exp(selected_residual[..., 3:])
    )
    selected_translation_error_m = torch.linalg.vector_norm(
        selected_position - target_pose[..., :3, 3], dim=-1
    )[frame_valid]
    selected_rotation_error_deg = torch.rad2deg(
        rotation_angle(selected_rotation, target_pose[..., :3, :3])
    )[frame_valid]

    loss = (
        args.candidate_weight * classification
        + args.expected_cost_weight * expected_cost
        + args.orientation_weight * orientation_loss
        + args.translation_weight * translation_nll
        + args.rotation_weight * rotation_nll
        + args.abstention_weight * abstention_loss
        + args.translation_trust_weight * translation_trust_loss
        + args.rotation_trust_weight * rotation_trust_loss
        + args.velocity_weight * velocity_loss
        + args.acceleration_weight * acceleration_loss
        + args.jerk_weight * jerk_loss
        + args.fixed_lag_regularization_weight * fixed_lag_regularization
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite transformer training loss")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

    selected = logits.argmax(dim=-1)
    return {
        "loss": float(loss.detach()),
        "candidate_loss": float(classification.detach()),
        "expected_cost": float(expected_cost.detach()),
        "orientation_loss": float(orientation_loss.detach()),
        "translation_nll": float(translation_nll.detach()),
        "rotation_nll": float(rotation_nll.detach()),
        "abstention_loss": float(abstention_loss.detach()),
        "translation_trust_loss": float(translation_trust_loss.detach()),
        "translation_trust_accuracy": float(
            ((output["translation_trust_logits"][translation_trust_mask] > 0)
             == (translation_trust_target[translation_trust_mask] > 0.5)).float().mean()
        ),
        "rotation_trust_loss": float(rotation_trust_loss.detach()),
        "rotation_trust_accuracy": float(
            ((output["rotation_trust_logits"][rotation_trust_mask] > 0)
             == (rotation_trust_target[rotation_trust_mask] > 0.5)).float().mean()
        ),
        "velocity_loss": float(velocity_loss.detach()),
        "acceleration_loss": float(acceleration_loss.detach()),
        "jerk_loss": float(jerk_loss.detach()),
        "fixed_lag_regularization": float(fixed_lag_regularization.detach()),
        "fixed_lag_gate_mean": float(
            output["translation_fixed_lag_gate"][frame_valid].mean().detach()
        ),
        "selected_translation_error_m_mean": float(
            selected_translation_error_m.mean().detach()
        ),
        "selected_translation_error_m_mse": float(
            selected_translation_error_m.square().mean().detach()
        ),
        "selected_rotation_error_deg_mean": float(
            selected_rotation_error_deg.mean().detach()
        ),
        "selected_rotation_error_deg_mse": float(
            selected_rotation_error_deg.square().mean().detach()
        ),
        "oracle_accuracy": float((selected == labels_flat).float().mean()),
        "orientation_accuracy": float(
            ((orientation_logits[orientation_mask] > 0) == (orientation_target[orientation_mask] > 0.5)).float().mean()
        ),
        "frames": int(flat.sum()),
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
    result = {key: value / max(frames, 1) for key, value in totals.items()}
    result["selected_translation_error_m_rmse"] = float(
        np.sqrt(result["selected_translation_error_m_mse"])
    )
    result["selected_rotation_error_deg_rmse"] = float(
        np.sqrt(result["selected_rotation_error_deg_mse"])
    )
    result["pose_score"] = (
        result["selected_translation_error_m_rmse"] / args.translation_scale_m
        + result["selected_rotation_error_deg_rmse"] / args.rotation_scale_deg
    )
    return result


def _checkpoint(model, args, epoch, metrics, train, validation):
    return {
        "format": FORMAT,
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "epoch": epoch,
        "validation_metrics": metrics,
        "candidate_feature_names": train.candidate_feature_names,
        "frame_feature_names": train.frame_feature_names,
        "saved_candidates": train.candidate_count,
        "train_runs": sorted(train.runs),
        "validation_runs": sorted(validation.runs),
        "arguments": {
            key: [str(item) for item in value] if isinstance(value, list)
            else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.attention_future_lag < args.window_length:
        raise ValueError(
            "--attention-future-lag must be in [0, --window-length)"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU")
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = TrainingWindowDataset(
        args.data, window_length=args.window_length, stride=args.window_stride
    )
    validation_data = TrainingWindowDataset(
        args.validation_data,
        window_length=args.window_length,
        stride=args.validation_stride,
    )
    train = train_data.collection
    validation = validation_data.collection
    overlap = _validate_collections(train, validation, args.allow_run_overlap)
    statistics = {
        key: torch.from_numpy(value)
        for key, value in collection_statistics(train).items()
    }
    model = GatedCandidateTransformer(
        train.candidate_dim,
        train.frame_dim,
        train.candidate_count,
        window_length=args.window_length,
        model_dim=args.model_dim,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        cross_layers=args.cross_layers,
        dropout=args.dropout,
        attention_future_lag=args.attention_future_lag,
        use_translation_fixed_lag=not args.disable_translation_fixed_lag,
        translation_fixed_lag_layers=args.translation_fixed_lag_layers,
        translation_position_scale_m=args.translation_position_scale_m,
        translation_relative_scale_m=args.translation_relative_scale_m,
        maximum_translation_refinement_m=args.maximum_translation_refinement_m,
        statistics=statistics,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )

    history, best_loss, best_pose, stale = [], float("inf"), float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model, train_loader, args.device, args, optimizer)
        validation_metrics = _epoch(model, validation_loader, args.device, args)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in validation_metrics.items()})
        history.append(row)
        payload = _checkpoint(model, args, epoch, validation_metrics, train, validation)
        torch.save(payload, args.output_dir / "last.ckpt")
        if validation_metrics["loss"] < best_loss - args.min_delta:
            best_loss = validation_metrics["loss"]
            stale = 0
            torch.save(payload, args.output_dir / "best.ckpt")
            torch.save(payload, args.output_dir / "best_loss.ckpt")
        else:
            stale += 1
        if validation_metrics["pose_score"] < best_pose:
            best_pose = validation_metrics["pose_score"]
            torch.save(payload, args.output_dir / "best_pose.ckpt")
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={validation_metrics['loss']:.4f} "
            f"pose={validation_metrics['pose_score']:.4f} "
            f"oracle_acc={validation_metrics['oracle_accuracy']:.3f} "
            f"orientation_acc={validation_metrics['orientation_accuracy']:.3f} "
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
        "format": "rgb_self_recovery_gated_candidate_transformer_training_v5",
        "best_validation_loss": best_loss,
        "best_validation_pose_score": best_pose,
        "pose_score_definition": (
            "selected translation RMSE / translation_scale_m + selected rotation "
            "RMSE / rotation_scale_deg; includes learned translation fixed lag"
        ),
        "epochs_completed": len(history),
        "train_frames": sum(len(bundle) for bundle in train.bundles),
        "validation_frames": sum(len(bundle) for bundle in validation.bundles),
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "train_runs": sorted(train.runs),
        "validation_runs": sorted(validation.runs),
        "run_overlap": overlap,
        "best_loss_checkpoint": str(args.output_dir / "best_loss.ckpt"),
        "best_checkpoint_alias": str(args.output_dir / "best.ckpt"),
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
