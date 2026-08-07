"""Calibrate independent GigaPose fallback thresholds on validation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--diagnostics", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--maximum-fallback-fraction", type=float, default=0.10)
    value.add_argument("--minimum-fallback-precision", type=float, default=0.60)
    value.add_argument("--minimum-mean-improvement", type=float, default=0.0)
    value.add_argument("--max-correction-translation-m", type=float, default=100.0)
    value.add_argument("--max-correction-rotation-deg", type=float, default=45.0)
    return value


def calibrate_component(
    rows: list[dict[str, str]], component: str, unit: str, correction_limit: float,
    maximum_fraction: float, minimum_precision: float, minimum_improvement: float,
) -> dict:
    probability = np.asarray([
        float(row[f"{component}_trust_probability"]) for row in rows
    ])
    candidate = np.asarray([
        float(row[f"candidate_{component}_error_{unit}"]) for row in rows
    ])
    baseline = np.asarray([
        float(row[f"baseline_{component}_error_{unit}"]) for row in rows
    ])
    measurement = np.asarray([
        float(row[f"raw_{component}_error_{unit}"]) for row in rows
    ])
    correction = np.asarray([
        float(row[f"correction_{component}_{unit}"]) for row in rows
    ])
    safety = (~np.isfinite(correction)) | (correction > correction_limit)
    no_learned_fallback = np.where(safety, measurement, candidate)
    initial_mean = float(no_learned_fallback.mean())
    best = {
        "threshold": 0.0,
        "mean_error": initial_mean,
        "learned_fallback_count": 0,
        "learned_fallback_fraction": 0.0,
        "learned_fallback_precision": None,
    }
    # filter.py uses probability < threshold. nextafter includes a sample whose
    # probability is exactly equal to each observed candidate threshold.
    thresholds = np.r_[0.0, np.nextafter(np.unique(probability), np.inf), 1.0]
    fallback_is_better = baseline < candidate
    for threshold in thresholds:
        learned = (~safety) & (probability < threshold)
        count = int(learned.sum())
        fraction = count / len(rows)
        if fraction > maximum_fraction:
            continue
        precision = float(fallback_is_better[learned].mean()) if count else None
        if count and precision < minimum_precision:
            continue
        errors = np.where(safety, measurement, np.where(learned, baseline, candidate))
        mean_error = float(errors.mean())
        if mean_error < best["mean_error"]:
            best = {
                "threshold": float(threshold),
                "mean_error": mean_error,
                "learned_fallback_count": count,
                "learned_fallback_fraction": fraction,
                "learned_fallback_precision": precision,
            }
    best["mean_error_without_learned_fallback"] = initial_mean
    best["mean_improvement"] = initial_mean - best["mean_error"]
    best["safety_fallback_count"] = int(safety.sum())
    if best["mean_improvement"] < minimum_improvement:
        best.update({
            "threshold": 0.0,
            "mean_error": initial_mean,
            "mean_improvement": 0.0,
            "learned_fallback_count": 0,
            "learned_fallback_fraction": 0.0,
            "learned_fallback_precision": None,
            "disabled_reason": "validation improvement below required minimum",
        })
    return best


def main() -> None:
    args = parser().parse_args()
    for name in ("maximum_fallback_fraction", "minimum_fallback_precision"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    with args.diagnostics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Diagnostics CSV is empty")
    report = {
        "format": "rgb_self_recovery_component_fallback_calibration_v1",
        "validation_only": True,
        "frame_count": len(rows),
        "constraints": {
            "maximum_fallback_fraction": args.maximum_fallback_fraction,
            "minimum_fallback_precision": args.minimum_fallback_precision,
            "minimum_mean_improvement": args.minimum_mean_improvement,
            "max_correction_translation_m": args.max_correction_translation_m,
            "max_correction_rotation_deg": args.max_correction_rotation_deg,
        },
        "translation": calibrate_component(
            rows, "translation", "m", args.max_correction_translation_m,
            args.maximum_fallback_fraction, args.minimum_fallback_precision,
            args.minimum_mean_improvement,
        ),
        "rotation": calibrate_component(
            rows, "rotation", "deg", args.max_correction_rotation_deg,
            args.maximum_fallback_fraction, args.minimum_fallback_precision,
            args.minimum_mean_improvement,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
