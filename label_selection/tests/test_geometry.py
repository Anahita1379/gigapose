from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from label_selection import common
from label_selection.optimize_extrinsics import main as optimize_extrinsics_main
from label_selection.optimize_extrinsics import residual_function, se3_exp
from label_selection.reselect_with_extrinsics import (
    main as reselect_with_extrinsics_main,
)
from label_selection.select_absolute import main as select_absolute_main


def transform(R: np.ndarray | None = None, t: tuple[float, float, float] = (0, 0, 0)):
    T = np.eye(4, dtype=float)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def label(identity: int, target: np.ndarray) -> dict:
    return {
        "match_key": "image_1",
        "epnp_label_path": f"/tmp/label_{identity}.json",
        "epnp_record_index": 0,
        "metadata_path": f"/tmp/front/metadata/sample_{identity}.yaml",
        "metadata_camera": "front",
        "T_camera_object_original": target,
        "T_map_object": target,
        "T_map_lidar": np.eye(4),
        "T_lidar_camera_prior": np.eye(4),
    }


def prediction(identity: int, pose: np.ndarray) -> dict:
    return {
        "match_key": "image_1",
        "scene_id": 1,
        "im_id": 1,
        "obj_id": 1,
        "instance_id": identity,
        "row_index": identity,
        "score": 0.9 - 0.1 * identity,
        "T_gigapose": pose,
    }


def test_known_center_conversion_preserves_physical_points():
    C = common.object_center_transform(common.DEFAULT_RAW_OBJECT_CENTER_M)
    R = Rotation.from_euler("xyz", [12, -4, 23], degrees=True).as_matrix()
    T_raw = transform(R, (300, -50, 20_000))
    T_centered = common.raw_pose_to_centered(T_raw, C)
    points_raw = np.asarray(
        [[-1000, -400, -300], [1200, 500, 700], [10, 20, 30]], dtype=float
    )
    points_centered = points_raw - C[:3, 3]
    camera_from_raw = (T_raw[:3, :3] @ points_raw.T).T + T_raw[:3, 3]
    camera_from_centered = (
        (T_centered[:3, :3] @ points_centered.T).T + T_centered[:3, 3]
    )
    assert np.allclose(camera_from_raw, camera_from_centered, atol=1e-9)


def test_absolute_pairing_is_one_to_one_and_uses_centered_pose():
    C = common.object_center_transform((0.1, 0.0, 0.0))
    first_raw = transform(t=(0, 0, 10_000))
    second_raw = transform(t=(2000, 0, 10_000))
    first_target = first_raw @ C
    second_target = second_raw @ C
    all_rows, assigned, counts = common.build_and_assign_pairs(
        [prediction(0, first_raw), prediction(1, second_raw)],
        [label(0, second_target), label(1, first_target)],
        C,
        pairing_translation_scale_mm=1000.0,
        pairing_rotation_scale_deg=10.0,
        max_candidates_per_key=20,
    )
    assert len(all_rows) == 4
    assert len(assigned) == 2
    assert counts["assigned_pairs"] == 2
    assert max(float(row["absolute_translation_error_mm"]) for row in assigned) < 1e-9
    identities = {
        (int(row["prediction_row_index"]), row["epnp_label_path"])
        for row in assigned
    }
    assert identities == {(0, "/tmp/label_1.json"), (1, "/tmp/label_0.json")}


def test_fixed_camera_lidar_optimizer_recovers_synthetic_transform():
    true_update = np.asarray(
        [0.01, -0.02, 0.005, 120.0, -45.0, 80.0], dtype=float
    )
    true_transform = se3_exp(true_update)
    samples = []
    for index in range(8):
        R = Rotation.from_euler(
            "xyz", [index, -0.5 * index, 2 * index], degrees=True
        ).as_matrix()
        G = transform(R, (300 * index, -100 * index, 15_000 + 400 * index))
        samples.append(
            {
                "T_gigapose_centered": G,
                "T_lidar_object_centered_target": true_transform @ G,
            }
        )
    result = least_squares(
        residual_function,
        x0=np.zeros(6),
        args=(
            np.eye(4),
            samples,
            [0, 1, 2],
            1000.0,
            np.deg2rad(10.0),
            0.0,
            0.0,
        ),
        loss="linear",
        max_nfev=500,
    )
    recovered = se3_exp(result.x)
    assert result.success
    assert np.linalg.norm(recovered[:3, 3] - true_transform[:3, 3]) < 1e-4
    assert (
        common.rotation_error_deg(recovered[:3, :3], true_transform[:3, :3])
        < 1e-6
    )


def test_corrected_camera_target_equation_is_consistent():
    C = common.object_center_transform(common.DEFAULT_RAW_OBJECT_CENTER_M)
    T_map_lidar = transform(
        Rotation.from_euler("z", 10, degrees=True).as_matrix(), (1000, 2000, 0)
    )
    T_lidar_camera = transform(
        Rotation.from_euler("xyz", [90, 0, 90], degrees=True).as_matrix(),
        (200, -100, 50),
    )
    T_camera_centered = transform(t=(500, 10, 20_000))
    T_map_raw = (
        T_map_lidar
        @ T_lidar_camera
        @ T_camera_centered
        @ np.linalg.inv(C)
    )
    recovered = (
        np.linalg.inv(T_lidar_camera)
        @ np.linalg.inv(T_map_lidar)
        @ T_map_raw
        @ C
    )
    assert np.allclose(recovered, T_camera_centered, atol=1e-8)


def test_absolute_selector_cli_end_to_end(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "frame_map.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": 1,
                    "im_id": 2,
                    "image_path": str(tmp_path / "image_123.png"),
                }
            ]
        )
    )
    predictions_path = tmp_path / "predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scene_id", "im_id", "obj_id", "score", "R", "t", "instance_id"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_id": 1,
                "im_id": 2,
                "obj_id": 1,
                "score": 0.9,
                "R": "1 0 0 0 1 0 0 0 1",
                "t": "0 0 10000",
                "instance_id": 0,
            }
        )

    C = common.object_center_transform(common.DEFAULT_RAW_OBJECT_CENTER_M)
    centered_target_mm = transform(t=(0, 0, 10_000)) @ C
    centered_target_m = centered_target_mm.copy()
    centered_target_m[:3, 3] *= 0.001
    map_object_m = np.eye(4)
    map_object_m[:3, 3] = [10.0, 20.0, 0.5]
    metadata_dir = tmp_path / "front" / "metadata"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "sample_123_0.yaml"
    metadata_path.write_text(
        "t_map_lidar:\n"
        "  - [1, 0, 0, 10]\n"
        "  - [0, 1, 0, 20]\n"
        "  - [0, 0, 1, 0]\n"
        "  - [0, 0, 0, 1]\n"
        "t_lidar_camera_prior:\n"
        "  - [1, 0, 0, 0]\n"
        "  - [0, 1, 0, 0]\n"
        "  - [0, 0, 1, 0]\n"
        "  - [0, 0, 0, 1]\n"
    )
    epnp_root = tmp_path / "epnp"
    epnp_root.mkdir()
    (epnp_root / "123_0.json").write_text(
        json.dumps(
            {
                "metadata_path": str(metadata_path),
                "T_map_object_raw": map_object_m[:3].tolist(),
                "T_camera_object_centered": centered_target_m[:3].tolist(),
            }
        )
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_absolute",
            "--gigapose-predictions",
            str(predictions_path),
            "--dataset-dir",
            str(dataset_dir),
            "--epnp-root",
            str(epnp_root),
            "--epnp-glob",
            "*.json",
            "--epnp-strip-trailing-instance-id",
            "--epnp-key-prefix",
            "image_",
            "--match-key",
            "image_stem",
            "--epnp-map-pose-unit",
            "m",
            "--epnp-camera-pose-unit",
            "m",
            "--prediction-translation-unit",
            "mm",
            "--max-translation-error-mm",
            "1",
            "--max-rotation-error-deg",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )
    select_absolute_main()
    selected = common.read_csv(output_dir / "selected_samples.csv")
    report = json.loads((output_dir / "selection_report.json").read_text())
    assert len(selected) == 1
    assert float(selected[0]["absolute_translation_error_mm"]) < 1e-9
    assert report["empirical_prediction_label_alignment_used"] is False
    assert report["prediction_translation_unit"] == "mm"

    calibration_dir = tmp_path / "calibration"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_extrinsics",
            "--selected-samples",
            str(output_dir / "selected_samples.csv"),
            "--camera-id",
            "front",
            "--translation-prior-weight",
            "0",
            "--rotation-prior-weight",
            "0",
            "--robust-loss",
            "linear",
            "--min-samples",
            "1",
            "--output-dir",
            str(calibration_dir),
        ],
    )
    optimize_extrinsics_main()
    calibration = json.loads(
        (calibration_dir / "optimized_extrinsics.json").read_text()
    )
    assert calibration["optimization_mode"] == common.CALIBRATION_MODE
    assert calibration["camera_id"] == "front"
    assert calibration["optimizer_success"] is True

    reselected_dir = tmp_path / "reselected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reselect_with_extrinsics",
            "--gigapose-predictions",
            str(predictions_path),
            "--dataset-dir",
            str(dataset_dir),
            "--epnp-root",
            str(epnp_root),
            "--epnp-glob",
            "*.json",
            "--epnp-strip-trailing-instance-id",
            "--epnp-key-prefix",
            "image_",
            "--match-key",
            "image_stem",
            "--epnp-map-pose-unit",
            "m",
            "--epnp-camera-pose-unit",
            "m",
            "--prediction-translation-unit",
            "mm",
            "--optimized-extrinsics",
            str(calibration_dir / "optimized_extrinsics.json"),
            "--camera-id",
            "front",
            "--max-translation-error-mm",
            "1",
            "--max-rotation-error-deg",
            "1",
            "--output-dir",
            str(reselected_dir),
        ],
    )
    reselect_with_extrinsics_main()
    reselected = common.read_csv(reselected_dir / "selected_samples.csv")
    assert len(reselected) == 1
    assert reselected[0]["comparison_mode"] == common.POST_CALIBRATION_MODE
    assert (
        float(reselected[0]["absolute_translation_error_mm"]) < 1e-5
    )
