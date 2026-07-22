from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fine_tuning.optimize_camera_lidar_extrinsics_centered import (
    apply_centered_object_convention,
    raw_from_centered_transform,
    residual_vector_centered,
)


def transform(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def translation_mm_to_m(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=float).copy()
    output[:3, 3] *= 0.001
    return output


def matrix_text(value: np.ndarray) -> str:
    return " ".join(f"{item:.17g}" for item in np.asarray(value).reshape(-1))


class CenteredCameraLidarOptimizerTest(unittest.TestCase):
    def test_raw_to_centered_translation_is_saved_in_millimetres(self) -> None:
        conversion = raw_from_centered_transform(
            np.asarray([-0.2411941141, 0.0009010172, 0.3329219520])
        )
        np.testing.assert_allclose(
            conversion[:3, 3], [-241.1941141, 0.9010172, 332.9219520]
        )
        np.testing.assert_allclose(conversion[:3, :3], np.eye(3))

    def test_centering_right_multiplies_raw_map_pose(self) -> None:
        center = np.asarray([0.2, -0.1, 0.3])
        conversion = raw_from_centered_transform(center)
        raw_pose = transform(rotation_z(0.7), [1000.0, 2000.0, 3000.0])
        samples = [{"T_target_obj": raw_pose.copy(), "target_pose_key": "raw"}]

        apply_centered_object_convention(samples, conversion)

        centered = samples[0]["T_map_object_centered"]
        np.testing.assert_allclose(centered[:3, :3], raw_pose[:3, :3])
        np.testing.assert_allclose(
            centered[:3, 3],
            raw_pose[:3, 3] + raw_pose[:3, :3] @ (center * 1000.0),
        )
        np.testing.assert_allclose(samples[0]["T_target_obj"], raw_pose @ conversion)
        np.testing.assert_allclose(
            samples[0]["T_target_obj_image"], raw_pose @ conversion
        )

    def test_exact_synthetic_calibration_has_zero_residual(self) -> None:
        conversion = raw_from_centered_transform(np.asarray([0.2, -0.1, 0.3]))
        T_map_lidar = transform(rotation_z(-0.15), [4000.0, -3000.0, 800.0])
        T_lidar_camera = transform(rotation_z(0.25), [600.0, -120.0, 240.0])
        T_camera_object_centered = transform(rotation_z(0.9), [-1500.0, 500.0, 22000.0])
        T_lidar_object_centered = T_lidar_camera @ T_camera_object_centered
        T_map_object_raw = (
            T_map_lidar @ T_lidar_object_centered @ np.linalg.inv(conversion)
        )
        samples = [
            {
                "T_target_obj": T_map_object_raw,
                "target_pose_key": "T_map_object_raw",
                "T_map_lidar": T_map_lidar,
                "T_gigapose_cam_obj": T_camera_object_centered,
                "K": None,
            }
        ]
        apply_centered_object_convention(samples, conversion)

        residual = residual_vector_centered(
            np.zeros(6),
            T_lidar_camera,
            samples,
            translation_sigma_mm=1000.0,
            translation_residual_components="xyz",
            rotation_sigma_deg=10.0,
            image_center_weight=0.0,
            image_center_sigma_px=50.0,
            projection_model="metadata",
            translation_prior_weight=0.0,
            rotation_prior_weight=0.0,
        )

        # Six pose residuals plus six explicitly zero correction-prior terms.
        self.assertEqual(residual.shape, (12,))
        np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

    def test_cli_recovers_calibration_and_saves_both_directions(self) -> None:
        center_m = np.asarray([-0.2411941141, 0.0009010172, 0.3329219520])
        conversion = raw_from_centered_transform(center_m)
        T_lidar_camera_true = transform(rotation_z(0.20), [620.0, -130.0, 250.0])
        T_lidar_camera_prior = transform(rotation_z(0.22), [720.0, -80.0, 225.0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_dir = root / "session" / "front" / "metadata"
            label_dir = root / "session" / "front" / "labels"
            metadata_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            selected_path = root / "selected.csv"
            output_dir = root / "output"
            selected_rows = []

            for index in range(5):
                T_map_lidar = transform(
                    rotation_z(-0.08 + 0.02 * index),
                    [4000.0 + 250.0 * index, -2000.0, 500.0],
                )
                T_camera_object_centered = transform(
                    rotation_z(0.35 + 0.18 * index),
                    [
                        -1800.0 + 700.0 * index,
                        350.0,
                        16000.0 + 2500.0 * index,
                    ],
                )
                T_lidar_object_centered = T_lidar_camera_true @ T_camera_object_centered
                T_map_object_raw = (
                    T_map_lidar @ T_lidar_object_centered @ np.linalg.inv(conversion)
                )
                metadata_path = metadata_dir / f"sample_{index}_0.yaml"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "t_map_lidar": translation_mm_to_m(T_map_lidar).tolist(),
                            "t_lidar_camera_prior": translation_mm_to_m(
                                T_lidar_camera_prior
                            ).tolist(),
                            "camera_intrinsics": {
                                "k": [1000, 0, 1000, 0, 1000, 200, 0, 0, 1],
                                "d": [0.1, 0.01, -0.01, 0.001],
                                "distortion_model": "equidistant",
                            },
                        }
                    )
                )
                label_path = label_dir / f"{index}_0.json"
                label_path.write_text(
                    json.dumps(
                        {
                            "T_map_object_raw": translation_mm_to_m(
                                T_map_object_raw
                            ).tolist()
                        }
                    )
                )
                selected_rows.append(
                    {
                        "match_key": f"image_{index}",
                        "scene_id": 1,
                        "im_id": index,
                        "instance_id": 0,
                        "score": 0.9,
                        "T_gigapose_cam_obj": matrix_text(T_camera_object_centered),
                        "epnp_label_path": str(label_path),
                        "sample_metadata_path": str(metadata_path),
                    }
                )

            with selected_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(selected_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(selected_rows)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fine_tuning.optimize_camera_lidar_extrinsics_centered",
                    "--selected-samples",
                    str(selected_path),
                    "--output-dir",
                    str(output_dir),
                    "--epnp-map-pose-unit",
                    "m",
                    "--max-nfev",
                    "100",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            report = json.loads((output_dir / "optimization_report.json").read_text())
            saved = json.loads((output_dir / "optimized_extrinsics.json").read_text())

            self.assertTrue(report["optimizer_success"])
            self.assertLess(report["after_translation_error_mm_median"], 1e-4)
            self.assertLess(report["after_rotation_error_deg_median"], 1e-5)
            T_lidar_camera_saved = np.asarray(
                saved["T_lidar_camera_optimized"], dtype=float
            )
            T_camera_lidar_saved = np.asarray(
                saved["T_camera_lidar_optimized"], dtype=float
            )
            np.testing.assert_allclose(
                T_lidar_camera_saved, T_lidar_camera_true, atol=1e-5
            )
            np.testing.assert_allclose(
                T_camera_lidar_saved @ T_lidar_camera_saved,
                np.eye(4),
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
