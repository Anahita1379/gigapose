"""Recommend the best GigaPose model for a task from metric summary CSVs.

This script reads the outputs from ``fine_tuning.evaluate_gigapose_models``:

    overall_summary.csv
    camera_summary.csv

and produces decision-oriented rankings for pose + altitude/depth estimation.

Example:

    python -m fine_tuning.recommend_model_by_task \
      --metrics-dir gigaPose_datasets/results/final_results/metrics/all_models

Default task weights are intentionally biased toward translation/depth and pose:

    translation_error_mm_median: 0.30
    depth_error_mm_median:       0.25
    rotation_error_deg_median:   0.20
    center_error_px_median:      0.10
    add_mm_median:               0.10
    pred_bbox_iou_median:        0.025
    pred_mask_iou_median:        0.025

The final ``task_score`` is 0-100, where higher is better.  Each metric is
normalized within the set of candidate models, so the score is a relative
decision aid, not an absolute physical accuracy value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


DEFAULT_WEIGHTS = {
    "translation_error_mm_median": 0.30,
    "depth_error_mm_median": 0.25,
    "rotation_error_deg_median": 0.20,
    "center_error_px_median": 0.10,
    "add_mm_median": 0.10,
    "pred_bbox_iou_median": 0.025,
    "pred_mask_iou_median": 0.025,
}

LOWER_IS_BETTER_PREFIXES = (
    "translation_error",
    "depth_error",
    "rotation_error",
    "center_error",
    "gt_bbox_center_error",
    "add_",
)
HIGHER_IS_BETTER_PREFIXES = (
    "score",
    "pred_bbox_iou",
    "pred_mask_iou",
    "translation_recall",
    "rotation_recall",
)


def parse_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--weight must be formatted metric=value")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Metric name cannot be empty")
    try:
        weight = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid weight: {value}") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("Weights must be non-negative")
    return key, weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank models overall and per camera for pose + altitude/depth "
            "estimation using summary metrics."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing overall_summary.csv and camera_summary.csv.",
    )
    parser.add_argument("--overall-csv", type=Path, default=None)
    parser.add_argument("--camera-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <metrics-dir>/model_recommendation.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=parse_weight,
        default=None,
        help=(
            "Override/add a metric weight, e.g. "
            "--weight depth_error_mm_median=0.4. Repeatable."
        ),
    )
    parser.add_argument(
        "--only-weighted-metrics",
        action="store_true",
        help=(
            "If passed, replace defaults with only the --weight metrics. "
            "Otherwise --weight updates the defaults."
        ),
    )
    parser.add_argument(
        "--require-metric",
        action="append",
        default=None,
        help="Metric that must be present/non-NaN for a model to be ranked. Repeatable.",
    )
    parser.add_argument(
        "--min-evaluated-instances",
        type=int,
        default=1,
        help="Drop model/camera rows with fewer evaluated instances.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write task-score bar charts if matplotlib is available.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def metric_direction(metric: str) -> str:
    if metric.startswith(HIGHER_IS_BETTER_PREFIXES):
        return "higher"
    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return "lower"
    # Conservative default: errors usually dominate these summary CSVs.
    return "lower"


def normalize_weights(weights: dict[str, float], rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    available = set(rows[0].keys())
    valid = {}
    for metric, weight in weights.items():
        if weight <= 0 or metric not in available:
            continue
        vals = [to_float(row.get(metric)) for row in rows]
        if any(math.isfinite(v) for v in vals):
            valid[metric] = weight
    total = sum(valid.values())
    if total <= 0:
        return {}
    return {metric: weight / total for metric, weight in valid.items()}


def normalized_metric_scores(rows: list[dict[str, str]], metric: str) -> dict[int, float]:
    vals = [to_float(row.get(metric)) for row in rows]
    finite_vals = [v for v in vals if math.isfinite(v)]
    if not finite_vals:
        return {idx: math.nan for idx in range(len(rows))}

    best_direction = metric_direction(metric)
    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if math.isclose(vmin, vmax):
        return {idx: 1.0 if math.isfinite(vals[idx]) else math.nan for idx in range(len(rows))}

    scores = {}
    for idx, value in enumerate(vals):
        if not math.isfinite(value):
            scores[idx] = math.nan
            continue
        if best_direction == "higher":
            scores[idx] = (value - vmin) / (vmax - vmin)
        else:
            scores[idx] = (vmax - value) / (vmax - vmin)
    return scores


def model_passes_requirements(
    row: dict[str, str],
    required_metrics: list[str],
    min_evaluated_instances: int,
) -> tuple[bool, str]:
    evaluated = to_float(row.get("evaluated_instances"))
    if math.isfinite(evaluated) and evaluated < min_evaluated_instances:
        return False, f"evaluated_instances={evaluated:g} < {min_evaluated_instances}"
    for metric in required_metrics:
        if not math.isfinite(to_float(row.get(metric))):
            return False, f"missing required metric {metric}"
    return True, ""


def rank_rows(
    rows: list[dict[str, str]],
    weights: dict[str, float],
    required_metrics: list[str],
    min_evaluated_instances: int,
    group_label: str,
) -> list[dict[str, Any]]:
    usable_rows = []
    excluded = []
    for row in rows:
        ok, reason = model_passes_requirements(row, required_metrics, min_evaluated_instances)
        if ok:
            usable_rows.append(row)
        else:
            excluded.append((row, reason))

    normalized_weights = normalize_weights(weights, usable_rows)
    if not normalized_weights:
        raise ValueError(
            f"No weighted metrics are available for {group_label}. "
            f"Requested weights: {weights}"
        )

    per_metric_scores = {
        metric: normalized_metric_scores(usable_rows, metric)
        for metric in normalized_weights
    }

    ranked = []
    for idx, row in enumerate(usable_rows):
        weighted_sum = 0.0
        used_weight = 0.0
        detail = {}
        for metric, weight in normalized_weights.items():
            raw = to_float(row.get(metric))
            norm_score = per_metric_scores[metric][idx]
            detail[f"{metric}_raw"] = raw
            detail[f"{metric}_normalized_score"] = norm_score
            if math.isfinite(norm_score):
                weighted_sum += weight * norm_score
                used_weight += weight
        task_score = 100.0 * weighted_sum / used_weight if used_weight > 0 else math.nan
        ranked.append(
            {
                "rank": 0,
                "method": row.get("method", ""),
                "camera_id": row.get("camera_id", "overall"),
                "group": group_label,
                "task_score": task_score,
                "used_weight": used_weight,
                "evaluated_instances": to_float(row.get("evaluated_instances")),
                "prediction_rows": to_float(row.get("prediction_rows")),
                **detail,
            }
        )

    ranked.sort(key=lambda item: (-to_float(item["task_score"]), str(item["method"])))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    for row, reason in excluded:
        ranked.append(
            {
                "rank": "",
                "method": row.get("method", ""),
                "camera_id": row.get("camera_id", "overall"),
                "group": group_label,
                "task_score": math.nan,
                "used_weight": 0.0,
                "evaluated_instances": to_float(row.get("evaluated_instances")),
                "prediction_rows": to_float(row.get("prediction_rows")),
                "excluded_reason": reason,
            }
        )
    return ranked


def group_camera_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("camera_id", "unknown_camera"), []).append(row)
    return dict(sorted(groups.items()))


def compact_recommendations(
    overall_ranked: list[dict[str, Any]],
    camera_ranked: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    overall_valid = [row for row in overall_ranked if math.isfinite(to_float(row.get("task_score")))]
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in camera_ranked:
        if math.isfinite(to_float(row.get("task_score"))):
            by_camera.setdefault(str(row.get("camera_id")), []).append(row)

    return {
        "weights": weights,
        "overall_best": overall_valid[0] if overall_valid else None,
        "best_by_camera": {
            camera: sorted(items, key=lambda row: to_float(row["rank"]))[0]
            for camera, items in sorted(by_camera.items())
        },
    }


def write_markdown_report(
    path: Path,
    recommendation: dict[str, Any],
    overall_ranked: list[dict[str, Any]],
    camera_ranked: list[dict[str, Any]],
) -> None:
    lines = [
        "# Model recommendation for pose + altitude/depth estimation",
        "",
        "Higher `task_score` is better. Scores are relative to the models in this metrics folder.",
        "",
        "## Weights",
        "",
        "| Metric | Weight | Direction |",
        "|---|---:|---|",
    ]
    for metric, weight in recommendation["weights"].items():
        lines.append(f"| `{metric}` | {weight:.4f} | {metric_direction(metric)} is better |")

    lines.extend(["", "## Overall recommendation", ""])
    best = recommendation.get("overall_best")
    if best:
        lines.append(
            f"Best overall model: `{best['method']}` with task score "
            f"{to_float(best['task_score']):.2f}."
        )
    else:
        lines.append("No valid overall recommendation could be made.")

    lines.extend(["", "| Rank | Model | Task score | Evaluated instances |", "|---:|---|---:|---:|"])
    for row in overall_ranked:
        if not math.isfinite(to_float(row.get("task_score"))):
            continue
        lines.append(
            f"| {row['rank']} | `{row['method']}` | "
            f"{to_float(row['task_score']):.2f} | {to_float(row['evaluated_instances']):.0f} |"
        )

    lines.extend(["", "## Per-camera recommendation", ""])
    lines.append("| Camera | Best model | Task score |")
    lines.append("|---|---|---:|")
    for camera, row in recommendation.get("best_by_camera", {}).items():
        lines.append(f"| `{camera}` | `{row['method']}` | {to_float(row['task_score']):.2f} |")

    lines.extend(["", "## Full per-camera ranking", ""])
    lines.append("| Camera | Rank | Model | Task score | Evaluated instances |")
    lines.append("|---|---:|---|---:|---:|")
    for row in camera_ranked:
        if not math.isfinite(to_float(row.get("task_score"))):
            continue
        lines.append(
            f"| `{row['camera_id']}` | {row['rank']} | `{row['method']}` | "
            f"{to_float(row['task_score']):.2f} | {to_float(row['evaluated_instances']):.0f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def plot_scores(output_dir: Path, overall_ranked: list[dict[str, Any]], camera_ranked: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_overall = [row for row in overall_ranked if math.isfinite(to_float(row.get("task_score")))]
    if valid_overall:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar([row["method"] for row in valid_overall], [to_float(row["task_score"]) for row in valid_overall])
        ax.set_title("Overall model task score")
        ax.set_ylabel("task score, higher is better")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "overall_task_score.png", dpi=180)
        plt.close(fig)

    by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in camera_ranked:
        if math.isfinite(to_float(row.get("task_score"))):
            by_camera.setdefault(str(row.get("camera_id")), []).append(row)
    for camera, rows in by_camera.items():
        rows = sorted(rows, key=lambda row: to_float(row["rank"]))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar([row["method"] for row in rows], [to_float(row["task_score"]) for row in rows])
        ax.set_title(f"{camera} model task score")
        ax.set_ylabel("task score, higher is better")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{camera}_task_score.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir
    output_dir = args.output_dir or metrics_dir / "model_recommendation"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {} if args.only_weighted_metrics else dict(DEFAULT_WEIGHTS)
    if args.weight:
        weights.update(dict(args.weight))

    overall_rows = read_csv(args.overall_csv or metrics_dir / "overall_summary.csv")
    camera_rows = read_csv(args.camera_csv or metrics_dir / "camera_summary.csv")
    required_metrics = args.require_metric or []

    overall_ranked = rank_rows(
        overall_rows,
        weights,
        required_metrics,
        args.min_evaluated_instances,
        "overall",
    )

    camera_ranked_all: list[dict[str, Any]] = []
    for camera, rows in group_camera_rows(camera_rows).items():
        camera_ranked_all.extend(
            rank_rows(
                rows,
                weights,
                required_metrics,
                args.min_evaluated_instances,
                camera,
            )
        )

    normalized_weights = normalize_weights(weights, overall_rows)
    recommendation = compact_recommendations(overall_ranked, camera_ranked_all, normalized_weights)

    write_csv(output_dir / "model_recommendations_overall.csv", overall_ranked)
    write_csv(output_dir / "model_recommendations_by_camera.csv", camera_ranked_all)
    (output_dir / "model_recommendation.json").write_text(json.dumps(recommendation, indent=2))
    write_markdown_report(
        output_dir / "model_recommendation_report.md",
        recommendation,
        overall_ranked,
        camera_ranked_all,
    )

    if args.plot:
        plot_scores(output_dir / "plots", overall_ranked, camera_ranked_all)

    best = recommendation.get("overall_best")
    if best:
        print(
            f"Best overall model: {best['method']} "
            f"(task_score={to_float(best['task_score']):.2f})"
        )
    print("Best by camera:")
    for camera, row in recommendation.get("best_by_camera", {}).items():
        print(f"  {camera}: {row['method']} (task_score={to_float(row['task_score']):.2f})")
    print(f"Wrote recommendation outputs to {output_dir}")


if __name__ == "__main__":
    main()
