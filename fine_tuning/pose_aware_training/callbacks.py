"""Human-readable metric history for pose-aware training."""

from __future__ import annotations

import csv
from pathlib import Path

import pytorch_lightning as pl


class PoseMetricHistoryCallback(pl.Callback):
    """Append important loss and pose metrics to a simple long-form CSV."""

    INTERESTING = (
        "loss",
        "monitor_translation_error_mm",
        "monitor_depth_abs_error_mm",
        "monitor_rotation_error_deg",
        "monitor_pose_score",
        "monitor_reprojection_error_px",
        "monitor_scale_",
        "soft_template_selected_rotation_deg",
        "soft_template_expected_rotation_deg",
        "soft_template_top1_diagonal",
    )

    def __init__(self, path: Path, train_interval: int = 50) -> None:
        self.path = Path(path)
        self.train_interval = int(train_interval)

    @staticmethod
    def _scalar(value):
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write(self, trainer: pl.Trainer, stage: str, prefix: str) -> None:
        if not trainer.is_global_zero:
            return
        rows = []
        for name, value in sorted(trainer.callback_metrics.items()):
            if not name.startswith(prefix):
                continue
            if not any(token in name for token in self.INTERESTING):
                continue
            scalar = self._scalar(value)
            if scalar is not None:
                rows.append(
                    {
                        "step": int(trainer.global_step),
                        "stage": stage,
                        "metric": name,
                        "value": f"{scalar:.10g}",
                    }
                )
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("step", "stage", "metric", "value")
            )
            if needs_header:
                writer.writeheader()
            writer.writerows(rows)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.train_interval > 0
            and trainer.global_step > 0
            and trainer.global_step % self.train_interval == 0
        ):
            self._write(trainer, "train", "train/")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write(trainer, "validation", "val/")
