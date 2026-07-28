from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def _pose(angle, translation):
    cosine, sine = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [
        [cosine, -sine, 0],
        [sine, cosine, 0],
        [0, 0, 1],
    ]
    value[:3, 3] = translation
    return value


def test_robust_fixed_extrinsic_recovers_synthetic_transform(tmp_path):
    true_extrinsic = _pose(0.12, [0.55, -0.08, 0.24])
    prior = _pose(0.16, [0.70, 0.02, 0.18])
    rows = []
    for index in range(8):
        map_lidar = _pose(-0.03 * index, [5 + index, -2, 0.5])
        camera_object = _pose(
            0.2 + 0.11 * index,
            [-2 + 0.5 * index, 0.3, 18 + 2 * index],
        )
        target = map_lidar @ true_extrinsic @ camera_object
        rows.append(
            {
                "prediction_row": index,
                "independent_support": True,
                "T_map_lidar": map_lidar.tolist(),
                "T_lidar_camera_initial": prior.tolist(),
                "T_camera_object_centered_gigapose": camera_object.tolist(),
                "T_map_object_centered_epnp": target.tolist(),
                "T_map_object_centered_refined": target.tolist(),
            }
        )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "extrinsic.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "teacher_pipeline.optimize_extrinsic",
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
            "--translation-prior-weight",
            "0",
            "--rotation-prior-weight",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    np.testing.assert_allclose(
        result["T_lidar_camera_optimized"], true_extrinsic, atol=1e-5
    )
    assert result["after_translation_error_m_median"] < 1e-5
    assert result["after_rotation_error_deg_median"] < 1e-5
    assert result["calibration_valid_for_reuse"] is True


def test_implausible_correction_is_not_exposed_for_reuse(tmp_path):
    prior = _pose(0.0, [0.0, 0.0, 0.0])
    true_extrinsic = _pose(0.0, [8.0, 0.0, 0.0])
    rows = []
    for index in range(4):
        map_lidar = _pose(0.0, [float(index), 0.0, 0.0])
        camera_object = _pose(0.0, [0.0, 0.0, 20.0 + index])
        target = map_lidar @ true_extrinsic @ camera_object
        rows.append(
            {
                "prediction_row": index,
                "independent_support": True,
                "T_map_lidar": map_lidar.tolist(),
                "T_lidar_camera_initial": prior.tolist(),
                "T_camera_object_centered_gigapose": camera_object.tolist(),
                "T_map_object_centered_epnp": target.tolist(),
            }
        )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    output = tmp_path / "extrinsic.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "teacher_pipeline.optimize_extrinsic",
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
            "--translation-prior-weight",
            "0",
            "--rotation-prior-weight",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    result = json.loads(output.read_text())
    assert result["calibration_valid_for_reuse"] is False
    assert "T_lidar_camera_optimized" not in result
    assert "T_lidar_camera_candidate" in result
