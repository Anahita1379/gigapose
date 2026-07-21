"""Train the optional SE(3) correction, confidence, and ranking head."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tracking.recovery import FEATURE_NAMES, PoseRecoveryHead, bounded_vector


REQUIRED_ARRAYS = (
    "features",
    "rotation_targets",
    "translation_targets",
    "confidence_targets",
    "quality_targets",
    "group_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=None,
        help=(
            "Optional independently generated validation NPZ. When omitted, "
            "--validation-fraction is held out from --data by instance group."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--groups-per-batch", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.20)
    parser.add_argument("--ranking-weight", type=float, default=0.20)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--max-rotation-deg", type=float, default=180.0)
    parser.add_argument("--max-translation-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--logger",
        choices=("none", "wandb"),
        default="none",
        help="Optional experiment logger. Local JSON/checkpoint saving always remains enabled.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name. Defaults to the output directory name.",
    )
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Write an offline W&B run without uploading during training.",
    )
    return parser.parse_args()


def _wandb_safe_config(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def _initialize_wandb(
    args: argparse.Namespace,
    data_metadata: dict[str, object],
    training_groups: int,
    validation_groups: int,
):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "--logger wandb requires the 'wandb' package in this environment."
        ) from exc
    config = _wandb_safe_config(vars(args))
    config.update(data_metadata)
    config.update(
        {
            "training_groups": int(training_groups),
            "validation_groups": int(validation_groups),
        }
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name or args.output_dir.name,
        dir=str(args.output_dir),
        mode="offline" if args.wandb_offline else "online",
        config=config,
    )


def _load_recovery_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        if "feature_names" not in payload.files:
            raise ValueError(f"{path} is missing feature_names.")
        names = tuple(str(value) for value in payload["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError(
                f"{path} feature definition differs from tracking.recovery."
            )
        arrays = {
            name: np.asarray(payload[name]).copy() for name in REQUIRED_ARRAYS
        }
    sample_count = int(arrays["features"].shape[0])
    if sample_count == 0:
        raise ValueError(f"{path} contains no recovery candidates.")
    inconsistent = {
        name: int(value.shape[0])
        for name, value in arrays.items()
        if int(value.shape[0]) != sample_count
    }
    if inconsistent:
        raise ValueError(
            f"{path} has inconsistent candidate counts: "
            f"features={sample_count}, others={inconsistent}"
        )
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            f"{path} features must have shape (N, {len(FEATURE_NAMES)})."
        )
    if arrays["rotation_targets"].shape != (sample_count, 3):
        raise ValueError(f"{path} rotation_targets must have shape (N, 3).")
    if arrays["translation_targets"].shape != (sample_count, 3):
        raise ValueError(f"{path} translation_targets must have shape (N, 3).")
    arrays["confidence_targets"] = np.asarray(
        arrays["confidence_targets"], dtype=np.float32
    ).reshape(-1)
    arrays["quality_targets"] = np.asarray(
        arrays["quality_targets"], dtype=np.float32
    ).reshape(-1)
    if arrays["confidence_targets"].shape != (sample_count,):
        raise ValueError(f"{path} confidence_targets must have shape (N,).")
    if arrays["quality_targets"].shape != (sample_count,):
        raise ValueError(f"{path} quality_targets must have shape (N,).")
    arrays["group_ids"] = np.asarray(
        arrays["group_ids"], dtype=np.int64
    ).reshape(-1)
    for name in (
        "features",
        "rotation_targets",
        "translation_targets",
        "confidence_targets",
        "quality_targets",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path} contains non-finite values in {name}.")
    if np.unique(arrays["group_ids"]).size == 0:
        raise ValueError(f"{path} contains no instance groups.")
    return arrays


def _load_depth_enabled(path: Path) -> bool | None:
    """Read optional generator metadata while accepting legacy NPZ files."""

    with np.load(path) as payload:
        if "depth_enabled" not in payload.files:
            return None
        value = np.asarray(payload["depth_enabled"])
    if value.size != 1:
        raise ValueError(f"{path} depth_enabled metadata must be scalar.")
    return bool(value.reshape(-1)[0])


def _remap_groups(
    group_ids: np.ndarray, start: int
) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(group_ids, return_inverse=True)
    remapped = inverse.astype(np.int64) + int(start)
    groups = np.arange(
        start,
        start + int(np.max(inverse)) + 1,
        dtype=np.int64,
    )
    return remapped, groups


def _prepare_recovery_data(
    training_path: Path,
    validation_path: Path | None,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    training = _load_recovery_data(training_path)
    training_depth_enabled = _load_depth_enabled(training_path)
    if validation_path is None:
        if not 0 < validation_fraction < 1:
            raise ValueError(
                "--validation-fraction must be in (0, 1) when "
                "--validation-data is omitted."
            )
        arrays = training
        unique_groups = np.unique(arrays["group_ids"])
        rng.shuffle(unique_groups)
        validation_count = max(
            1, int(round(len(unique_groups) * validation_fraction))
        )
        validation_groups = unique_groups[:validation_count]
        training_groups = unique_groups[validation_count:]
        if training_groups.size == 0:
            raise ValueError("Not enough groups for a train/validation split.")
        metadata: dict[str, object] = {
            "validation_strategy": "internal_group_holdout",
            "training_data": str(training_path),
            "validation_data": None,
            "training_candidates": int(
                np.isin(arrays["group_ids"], training_groups).sum()
            ),
            "validation_candidates": int(
                np.isin(arrays["group_ids"], validation_groups).sum()
            ),
            "training_depth_enabled": training_depth_enabled,
            "validation_depth_enabled": training_depth_enabled,
        }
        return arrays, training_groups, validation_groups, metadata

    if training_path.resolve() == validation_path.resolve():
        raise ValueError(
            "--validation-data must be a different file from --data."
        )
    validation = _load_recovery_data(validation_path)
    validation_depth_enabled = _load_depth_enabled(validation_path)
    if (
        training_depth_enabled is not None
        and validation_depth_enabled is not None
        and training_depth_enabled != validation_depth_enabled
    ):
        raise ValueError(
            "Recovery train/validation depth modes differ: "
            f"{training_path} depth_enabled={training_depth_enabled}, "
            f"{validation_path} depth_enabled={validation_depth_enabled}."
        )
    training_ids, training_groups = _remap_groups(
        training["group_ids"], start=0
    )
    validation_ids, validation_groups = _remap_groups(
        validation["group_ids"], start=len(training_groups)
    )
    training["group_ids"] = training_ids
    validation["group_ids"] = validation_ids
    arrays = {
        name: np.concatenate([training[name], validation[name]], axis=0)
        for name in REQUIRED_ARRAYS
    }
    metadata = {
        "validation_strategy": "external_dataset",
        "training_data": str(training_path),
        "validation_data": str(validation_path),
        "training_candidates": int(training["features"].shape[0]),
        "validation_candidates": int(validation["features"].shape[0]),
        "training_depth_enabled": training_depth_enabled,
        "validation_depth_enabled": validation_depth_enabled,
    }
    return arrays, training_groups, validation_groups, metadata


def _group_batches(
    groups: np.ndarray,
    selected_groups: np.ndarray,
    groups_per_batch: int,
    rng: np.random.Generator,
):
    values = selected_groups.copy()
    rng.shuffle(values)
    for start in range(0, len(values), groups_per_batch):
        group_chunk = values[start : start + groups_per_batch]
        yield np.flatnonzero(np.isin(groups, group_chunk))


def _loss(
    output: dict[str, torch.Tensor],
    rotation_target: torch.Tensor,
    translation_target: torch.Tensor,
    confidence_target: torch.Tensor,
    quality_target: torch.Tensor,
    groups: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_rotation = math.radians(args.max_rotation_deg)
    predicted_rotation = bounded_vector(output["rotation_raw"], max_rotation)
    predicted_translation = bounded_vector(
        output["translation_raw"], args.max_translation_m
    )
    rotation_loss = nn.functional.smooth_l1_loss(
        predicted_rotation / max(max_rotation, 1e-6),
        rotation_target / max(max_rotation, 1e-6),
    )
    translation_loss = nn.functional.smooth_l1_loss(
        predicted_translation / max(args.max_translation_m, 1e-6),
        translation_target / max(args.max_translation_m, 1e-6),
    )
    confidence_loss = nn.functional.binary_cross_entropy_with_logits(
        output["confidence_logit"], confidence_target
    )
    quality_loss = nn.functional.smooth_l1_loss(
        output["quality"], torch.log1p(quality_target)
    )
    ranking_terms = []
    for group in groups.unique():
        indices = torch.nonzero(groups == group, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        target = quality_target[indices]
        prediction = output["quality"][indices]
        differences = target[:, None] - target[None, :]
        better, worse = torch.where(differences < -1e-5)
        if better.numel():
            ranking_terms.append(
                nn.functional.relu(
                    args.ranking_margin
                    + prediction[better]
                    - prediction[worse]
                ).mean()
            )
    ranking_loss = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else output["quality"].sum() * 0.0
    )
    total = (
        args.rotation_weight * rotation_loss
        + args.translation_weight * translation_loss
        + args.confidence_weight * confidence_loss
        + args.quality_weight * quality_loss
        + args.ranking_weight * ranking_loss
    )
    return total, {
        "rotation": float(rotation_loss.detach()),
        "translation": float(translation_loss.detach()),
        "confidence": float(confidence_loss.detach()),
        "quality": float(quality_loss.detach()),
        "ranking": float(ranking_loss.detach()),
        "total": float(total.detach()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    arrays, training_groups, validation_groups, data_metadata = (
        _prepare_recovery_data(
            args.data,
            args.validation_data,
            args.validation_fraction,
            rng,
        )
    )
    print(
        "Recovery data: "
        f"strategy={data_metadata['validation_strategy']} "
        f"train_candidates={data_metadata['training_candidates']} "
        f"train_groups={training_groups.size} "
        f"validation_candidates={data_metadata['validation_candidates']} "
        f"validation_groups={validation_groups.size} "
        f"depth_enabled={data_metadata['training_depth_enabled']}"
    )
    training_mask = np.isin(arrays["group_ids"], training_groups)
    feature_mean = arrays["features"][training_mask].mean(axis=0)
    feature_std = arrays["features"][training_mask].std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    normalized_features = (arrays["features"] - feature_mean) / feature_std

    tensors = {
        "features": torch.as_tensor(normalized_features, dtype=torch.float32),
        "rotation": torch.as_tensor(arrays["rotation_targets"], dtype=torch.float32),
        "translation": torch.as_tensor(arrays["translation_targets"], dtype=torch.float32),
        "confidence": torch.as_tensor(arrays["confidence_targets"], dtype=torch.float32),
        "quality": torch.as_tensor(arrays["quality_targets"], dtype=torch.float32),
        "groups": torch.as_tensor(arrays["group_ids"], dtype=torch.long),
    }
    model = PoseRecoveryHead(len(FEATURE_NAMES), args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale = 0
    history = []

    def run_groups(
        selected: np.ndarray, training: bool
    ) -> dict[str, float]:
        model.train(training)
        metric_sums = {
            name: 0.0
            for name in (
                "rotation",
                "translation",
                "confidence",
                "quality",
                "ranking",
                "total",
            )
        }
        candidate_count = 0
        batches = _group_batches(
            arrays["group_ids"],
            selected,
            args.groups_per_batch,
            rng if training else np.random.default_rng(0),
        )
        for indices in batches:
            index = torch.as_tensor(indices, dtype=torch.long)
            batch = {
                key: value[index].to(args.device) for key, value in tensors.items()
            }
            with torch.set_grad_enabled(training):
                output = model(batch["features"])
                loss, batch_metrics = _loss(
                    output,
                    batch["rotation"],
                    batch["translation"],
                    batch["confidence"],
                    batch["quality"],
                    batch["groups"],
                    args,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            batch_size = int(indices.size)
            candidate_count += batch_size
            for name, value in batch_metrics.items():
                metric_sums[name] += value * batch_size
        if candidate_count == 0:
            raise RuntimeError("A recovery epoch contained no candidates.")
        return {
            name: value / candidate_count
            for name, value in metric_sums.items()
        }

    def save(name: str, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "max_rotation_rad": math.radians(args.max_rotation_deg),
                    "max_translation_m": args.max_translation_m,
                },
                "feature_names": FEATURE_NAMES,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "epoch": epoch,
                "validation_loss": validation_loss,
                "arguments": vars(args),
                "data_metadata": data_metadata,
            },
            args.output_dir / name,
        )

    wandb_run = _initialize_wandb(
        args,
        data_metadata,
        int(training_groups.size),
        int(validation_groups.size),
    )
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_groups(training_groups, True)
            validation_metrics = run_groups(validation_groups, False)
            train_loss = train_metrics["total"]
            val_loss = validation_metrics["total"]
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"val_{name}": value
                    for name, value in validation_metrics.items()
                },
            }
            history.append(history_row)
            save("last.ckpt", epoch, val_loss)
            improved = val_loss < best_loss - args.min_delta
            if improved:
                best_loss, stale = val_loss, 0
                save("best.ckpt", epoch, val_loss)
            else:
                stale += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        **{
                            f"train/loss_{name}": value
                            for name, value in train_metrics.items()
                        },
                        **{
                            f"val/loss_{name}": value
                            for name, value in validation_metrics.items()
                        },
                        "optimization/learning_rate": float(
                            optimizer.param_groups[0]["lr"]
                        ),
                        "early_stopping/best_val_loss": best_loss,
                        "early_stopping/stale_epochs": stale,
                        "early_stopping/improved": int(improved),
                    },
                    step=epoch,
                )
            print(
                f"epoch={epoch:03d} train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best_loss:.6f} "
                f"stale={stale}/{args.patience}"
            )
            if stale >= args.patience:
                print(
                    "Early stopping: validation recovery objective "
                    "stopped improving."
                )
                break
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )
        report = {
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "training_groups": int(training_groups.size),
            "validation_groups": int(validation_groups.size),
            **data_metadata,
            "logger": args.logger,
            "wandb_project": (
                args.wandb_project if args.logger == "wandb" else None
            ),
            "wandb_run_name": (
                args.run_name or args.output_dir.name
                if args.logger == "wandb"
                else None
            ),
            "best_checkpoint": str(args.output_dir / "best.ckpt"),
        }
        (args.output_dir / "run_report.json").write_text(
            json.dumps(report, indent=2)
        )
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_validation_loss": best_loss,
                    "epochs_completed": len(history),
                    "best_checkpoint": str(args.output_dir / "best.ckpt"),
                }
            )
        print(json.dumps(report, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
