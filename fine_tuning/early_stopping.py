"""Shared, opt-in early stopping utilities for fine-tuning entry points."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def pose_score(
    translation_error_mm: torch.Tensor,
    rotation_error_deg: torch.Tensor,
) -> torch.Tensor:
    """Combine metric translation and rotation errors into a pose score.

    A score change of 1.0 corresponds to either 1000 mm of translation error
    or 10 degrees of rotation error. Lower is better.
    """

    return translation_error_mm / 1000.0 + rotation_error_deg / 10.0


class StepAwareEarlyStopping(EarlyStopping):
    """Lightning early stopping that ignores checks before ``start_step``."""

    def __init__(self, *args: Any, start_step: int = 0, **kwargs: Any) -> None:
        self.start_step = int(start_step)
        self._activation_announced = False
        super().__init__(*args, **kwargs)

    def _should_skip_check(self, trainer: pl.Trainer) -> bool:
        return (
            int(trainer.global_step) < self.start_step
            or super()._should_skip_check(trainer)
        )

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if (
            not self._activation_announced
            and not trainer.sanity_checking
            and int(trainer.global_step) >= self.start_step
        ):
            if trainer.is_global_zero:
                print(
                    "EARLY STOPPING ACTIVE: "
                    f"step={int(trainer.global_step)} monitor={self.monitor} "
                    f"mode={self.mode} min_delta={abs(self.min_delta):g} "
                    f"patience={self.patience}",
                    flush=True,
                )
            self._activation_announced = True
        super().on_validation_end(trainer, pl_module)

    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        """Run Lightning's check and print the patience state on rank zero."""

        previous_wait = int(self.wait_count)
        previous_best = self.best_score.detach().clone()
        was_stopped = bool(trainer.should_stop)
        super()._run_early_stopping_check(trainer)

        if not trainer.is_global_zero or trainer.fast_dev_run:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current_value = float(current.detach().squeeze().cpu())
        best_changed = not torch.equal(
            previous_best.cpu(),
            self.best_score.detach().cpu(),
        )
        newly_stopped = bool(trainer.should_stop) and not was_stopped

        if best_changed:
            if torch.isfinite(previous_best).item() and previous_wait > 0:
                print(
                    "EARLY STOPPING PATIENCE RESET: "
                    f"step={int(trainer.global_step)} "
                    f"metric={current_value:.6g} wait=0/{self.patience}",
                    flush=True,
                )
            elif torch.isfinite(previous_best).item():
                print(
                    "EARLY STOPPING IMPROVED: "
                    f"step={int(trainer.global_step)} "
                    f"metric={current_value:.6g} wait=0/{self.patience}",
                    flush=True,
                )
            else:
                print(
                    "EARLY STOPPING BEST INITIALIZED: "
                    f"step={int(trainer.global_step)} "
                    f"metric={current_value:.6g} wait=0/{self.patience}",
                    flush=True,
                )
        elif int(self.wait_count) > previous_wait:
            label = (
                "EARLY STOPPING PATIENCE EXHAUSTED"
                if newly_stopped
                else "EARLY STOPPING PATIENCE"
            )
            print(
                f"{label}: step={int(trainer.global_step)} "
                f"metric={current_value:.6g} "
                f"wait={int(self.wait_count)}/{self.patience}",
                flush=True,
            )
        elif newly_stopped:
            # Covers finite/divergence/target thresholds, which do not consume
            # the patience counter.
            print(
                "EARLY STOPPING STOP CONDITION REACHED: "
                f"step={int(trainer.global_step)} "
                f"metric={current_value:.6g} "
                f"wait={int(self.wait_count)}/{self.patience}",
                flush=True,
            )


class StepAwareModelCheckpoint(ModelCheckpoint):
    """Best/last checkpointing that starts with the stopping window."""

    def __init__(self, *args: Any, start_step: int = 0, **kwargs: Any) -> None:
        self.start_step = int(start_step)
        super().__init__(*args, **kwargs)

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        return (
            int(trainer.global_step) < self.start_step
            or super()._should_skip_saving_checkpoint(trainer)
        )


def add_early_stopping_args(
    parser: argparse.ArgumentParser,
    *,
    default_monitor: str | None,
    default_mode: str,
    default_patience: int,
    default_min_delta: float,
    default_start_step: int,
) -> None:
    """Add the same early-stopping CLI to a training parser."""

    group = parser.add_argument_group("optional early stopping")
    group.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after the monitored full-validation metric becomes stale.",
    )
    group.add_argument(
        "--early-stopping-monitor",
        default=default_monitor,
        help="Epoch-aggregated validation metric to monitor.",
    )
    group.add_argument(
        "--early-stopping-mode",
        choices=("min", "max"),
        default=default_mode,
        help="Whether a lower or higher monitored value is better.",
    )
    group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=default_patience,
        help="Number of completed validation checks allowed without improvement.",
    )
    group.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=default_min_delta,
        help="Minimum absolute metric improvement that resets patience.",
    )
    group.add_argument(
        "--early-stopping-start-step",
        type=int,
        default=default_start_step,
        help="Do not update patience or stop before this optimizer step.",
    )
    group.add_argument(
        "--early-stopping-stopping-threshold",
        type=float,
        default=None,
        help="Optional target that stops immediately once reached.",
    )
    group.add_argument(
        "--early-stopping-divergence-threshold",
        type=float,
        default=None,
        help="Optional bad-value threshold that stops immediately once crossed.",
    )
    group.add_argument(
        "--early-stopping-check-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop if the monitored metric becomes NaN or infinite.",
    )
    group.add_argument(
        "--early-stopping-best-checkpoints",
        type=int,
        default=3,
        help=(
            "Number of best monitored checkpoints to retain when no existing "
            "best-checkpoint callback already monitors the same metric."
        ),
    )


def validate_early_stopping_args(args: argparse.Namespace) -> None:
    """Validate early-stopping arguments only when the feature is enabled."""

    if not args.early_stopping:
        return
    if not args.early_stopping_monitor:
        raise ValueError("--early-stopping-monitor must not be empty")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if not math.isfinite(args.early_stopping_min_delta):
        raise ValueError("--early-stopping-min-delta must be finite")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be nonnegative")
    if args.early_stopping_start_step < 0:
        raise ValueError("--early-stopping-start-step must be nonnegative")
    if args.early_stopping_best_checkpoints <= 0:
        raise ValueError("--early-stopping-best-checkpoints must be positive")
    for name in (
        "early_stopping_stopping_threshold",
        "early_stopping_divergence_threshold",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be finite")


def configure_early_stopping(
    trainer: pl.Trainer,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    logger=None,
) -> StepAwareEarlyStopping | None:
    """Attach best/last checkpointing and an early-stopping callback.

    If a specialized callback already saves top-k checkpoints for the same
    metric and mode, it is reused. Otherwise ``best-early-step*.ckpt`` files
    are added. In both cases ``last.ckpt`` is refreshed after every completed
    validation while early stopping is active.
    """

    if not args.early_stopping:
        return None
    validate_early_stopping_args(args)
    monitor = str(args.early_stopping_monitor)
    mode = str(args.early_stopping_mode)

    matching_checkpoint = None
    for callback in trainer.callbacks:
        if not isinstance(callback, ModelCheckpoint):
            continue
        if callback.monitor == monitor and callback.mode == mode:
            matching_checkpoint = callback
            break

    if matching_checkpoint is None:
        matching_checkpoint = StepAwareModelCheckpoint(
            dirpath=str(Path(output_dir) / "checkpoints"),
            filename="best-early-step{step:06d}",
            monitor=monitor,
            mode=mode,
            save_top_k=args.early_stopping_best_checkpoints,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True,
            start_step=args.early_stopping_start_step,
        )
        trainer.callbacks.append(matching_checkpoint)
    else:
        # Preserve the specialized name/top-k settings, but replace it with a
        # step-aware equivalent so a lucky pre-warmup validation cannot remain
        # the supposedly best stopping checkpoint.
        replacement = StepAwareModelCheckpoint(
            dirpath=matching_checkpoint.dirpath,
            filename=matching_checkpoint.filename,
            monitor=monitor,
            mode=mode,
            save_top_k=matching_checkpoint.save_top_k,
            save_last=True,
            save_weights_only=matching_checkpoint.save_weights_only,
            auto_insert_metric_name=matching_checkpoint.auto_insert_metric_name,
            verbose=matching_checkpoint.verbose,
            save_on_train_epoch_end=False,
            start_step=args.early_stopping_start_step,
        )
        callback_index = trainer.callbacks.index(matching_checkpoint)
        trainer.callbacks[callback_index] = replacement
        matching_checkpoint = replacement

    callback = StepAwareEarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=args.early_stopping_min_delta,
        patience=args.early_stopping_patience,
        start_step=args.early_stopping_start_step,
        strict=True,
        check_finite=args.early_stopping_check_finite,
        stopping_threshold=args.early_stopping_stopping_threshold,
        divergence_threshold=args.early_stopping_divergence_threshold,
        check_on_train_epoch_end=False,
        verbose=True,
        log_rank_zero_only=True,
    )
    trainer.callbacks.append(callback)

    if trainer.is_global_zero:
        print(
            "EARLY STOPPING CONFIGURED: "
            f"monitor={monitor} mode={mode} "
            f"min_delta={args.early_stopping_min_delta:g} "
            f"patience={args.early_stopping_patience} "
            f"start_step={args.early_stopping_start_step} "
            "stale_window_steps~="
            f"{args.early_stopping_patience * args.validation_interval}",
            flush=True,
        )
    if logger is not None:
        logger.info(
            "Early-stopping checkpoints: %s (best monitored plus last.ckpt)",
            Path(output_dir) / "checkpoints",
        )
    return callback
