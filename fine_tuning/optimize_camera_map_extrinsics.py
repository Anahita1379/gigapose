"""Optimize map-to-camera extrinsic correction from selected real labels.

This script uses selected samples produced by:

    python -m fine_tuning.select_real_label_candidates ...

Expected selected CSV columns:

    T_gigapose_cam_obj
    T_epnp_obj

Interpretation:

- ``T_gigapose_cam_obj`` maps object coordinates into the GigaPose camera frame.
- ``T_epnp_obj`` maps object coordinates into the EPnPv2/map label frame.
- The initial extrinsic ``T_map_cam`` maps camera coordinates into the map frame.

The optimized model is:

    T_epnp_obj ≈ T_map_cam_optimized @ T_gigapose_cam_obj
    T_map_cam_optimized = exp(delta) @ T_map_cam_initial

The correction has a prior. By default translation correction is penalized more
strongly than rotation correction, because the current translation extrinsics
are assumed to be more reliable than the rotational extrinsics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-samples",
        type=Path,
        required=True,
        help="selected_samples.csv from select_real_label_candidates.py.",
    )
    parser.add_argument(
        "--initial-extrinsic",
        type=Path,
        required=True,
        help=(
            "JSON containing initial T_map_cam as key T_map_cam, matrix, or T. "
            "Translations are assumed to be in millimeters unless --initial-unit m."
        ),
    )
    parser.add_argument(
        "--initial-unit",
        choices=("mm", "m"),
        default="mm",
        help="Translation unit used in the initial extrinsic JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gigaPose_datasets/results/real_world_extrinsics"),
    )
    parser.add_argument(
        "--translation-sigma-mm",
        type=float,
        default=1000.0,
        help="Pose residual scale for object translation error.",
    )
    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=10.0,
        help="Pose residual scale for object rotation error.",
    )
    parser.add_argument(
        "--translation-prior-weight",
        type=float,
        default=10.0,
        help=(
            "Prior weight on changing the extrinsic translation. Larger values "
            "keep translation closer to the initial extrinsic."
        ),
    )
    parser.add_argument(
        "--rotation-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Prior weight on changing the extrinsic rotation. Lower than "
            "translation-prior-weight if rotation is expected to be less reliable."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    return parser.parse_args()


def text_to_matrix(value: str) -> np.ndarray:
    values = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if values.size != 16:
        raise ValueError(f"Expected 16 values for 4x4 matrix, got {values.size}")
    return values.reshape(4, 4)


def matrix_to_text(T: np.ndarray) -> str:
    return " ".join(f"{v:.12g}" for v in np.asarray(T, dtype=float).reshape(-1))


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-12:
        return np.eye(3) + W
    A = math.sin(theta) / theta
    B = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + A * W + B * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(math.acos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3)
    return (
        theta
        / (2.0 * math.sin(theta))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Small SE(3) update.

    xi = [rx, ry, rz, tx_mm, ty_mm, tz_mm].

    For an extrinsic correction this small-update form is enough and easier to
    reason about than the full SE(3) V matrix. Rotation is in radians,
    translation is in millimeters.
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = so3_exp(xi[:3])
    T[:3, 3] = xi[3:6]
    return T


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_pred @ R_gt.T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def load_initial_extrinsic(path: Path, unit: str) -> np.ndarray:
    data = json.loads(path.read_text())
    for key in ("T_map_cam", "matrix", "T", "T_world_camera", "T_map_camera"):
        if key in data:
            T = np.asarray(data[key], dtype=float).reshape(4, 4)
            if unit == "m":
                T[:3, 3] *= 1000.0
            return T
    raise ValueError(
        f"{path} must contain one of T_map_cam, matrix, T, T_world_camera, T_map_camera"
    )


def load_selected_samples(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                T_giga = text_to_matrix(row["T_gigapose_cam_obj"])
                T_epnp = text_to_matrix(row["T_epnp_obj"])
            except Exception as exc:
                print(f"Skipping malformed selected sample: {exc}")
                continue
            rows.append(
                {
                    **row,
                    "T_gigapose_cam_obj": T_giga,
                    "T_epnp_obj": T_epnp,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No valid selected samples found in {path}")
    return rows


def residual_vector(
    xi: np.ndarray,
    T_map_cam_initial: np.ndarray,
    samples: list[dict[str, Any]],
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    T_map_cam = T_delta @ T_map_cam_initial
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)

    for sample in samples:
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_epnp_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3]) / translation_sigma_mm
        r_res = so3_log(T_pred[:3, :3] @ T_gt[:3, :3].T) / rotation_sigma_rad
        residuals.extend(t_res.tolist())
        residuals.extend(r_res.tolist())

    # Priors on changing the extrinsic itself. Translation prior is intentionally
    # stronger by default.
    residuals.extend((xi[:3] * rotation_prior_weight).tolist())
    residuals.extend(((xi[3:6] / translation_sigma_mm) * translation_prior_weight).tolist())
    return np.asarray(residuals, dtype=float)


def compute_sample_errors(T_map_cam: np.ndarray, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_epnp_obj"]
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
            }
        )
    return rows


def summarize_errors(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out = {}
    for key in ("translation_error_mm", "rotation_error_deg"):
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{prefix}_{key}_mean"] = float(np.mean(vals))
        out[f"{prefix}_{key}_median"] = float(np.median(vals))
        out[f"{prefix}_{key}_p90"] = float(np.percentile(vals, 90))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise SystemExit(
            "This optimization template needs scipy. Install it in the gigapose "
            "environment with `pip install scipy` or `conda install scipy`."
        ) from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples(args.selected_samples, args.max_samples)
    T_initial = load_initial_extrinsic(args.initial_extrinsic, args.initial_unit)
    before_rows = compute_sample_errors(T_initial, samples)

    result = least_squares(
        residual_vector,
        x0=np.zeros(6, dtype=float),
        args=(
            T_initial,
            samples,
            args.translation_sigma_mm,
            args.rotation_sigma_deg,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        ),
        loss=args.robust_loss,
        max_nfev=500,
    )

    xi = result.x
    T_delta = se3_exp(xi)
    T_optimized = T_delta @ T_initial
    after_rows = compute_sample_errors(T_optimized, samples)

    write_csv(args.output_dir / "errors_before_optimization.csv", before_rows)
    write_csv(args.output_dir / "errors_after_optimization.csv", after_rows)

    extrinsics = {
        "description": "Optimized map-to-camera extrinsic correction from selected GigaPose/EPnPv2 samples.",
        "translation_unit": "mm",
        "T_map_cam_initial": T_initial.tolist(),
        "T_correction_left_multiply": T_delta.tolist(),
        "T_map_cam_optimized": T_optimized.tolist(),
        "correction_rotation_rpy_like_vector_rad": xi[:3].tolist(),
        "correction_translation_mm": xi[3:6].tolist(),
        "note": "Use T_map_cam_optimized as camera-to-map transform if your pipeline expects T_map_cam.",
    }
    (args.output_dir / "optimized_extrinsics.json").write_text(json.dumps(extrinsics, indent=2))

    report = {
        "selected_samples": len(samples),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "robust_loss": args.robust_loss,
        "translation_sigma_mm": args.translation_sigma_mm,
        "rotation_sigma_deg": args.rotation_sigma_deg,
        "translation_prior_weight": args.translation_prior_weight,
        "rotation_prior_weight": args.rotation_prior_weight,
        "correction_translation_norm_mm": float(np.linalg.norm(xi[3:6])),
        "correction_rotation_norm_deg": float(np.degrees(np.linalg.norm(xi[:3]))),
        **summarize_errors(before_rows, "before"),
        **summarize_errors(after_rows, "after"),
    }
    (args.output_dir / "optimization_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote optimized extrinsics to {args.output_dir / 'optimized_extrinsics.json'}")


if __name__ == "__main__":
    main()
