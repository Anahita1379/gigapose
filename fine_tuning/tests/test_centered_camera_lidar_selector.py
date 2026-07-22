from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fine_tuning.select_real_label_candidates_with_centered_camera_lidar_extrinsics import (
    load_centered_calibration,
    make_candidate_rows,
)


def transform(translation: list[float]) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[:3, 3] = translation
    return value


class CenteredCameraLidarSelectorTest(unittest.TestCase):
    def test_candidate_uses_raw_to_centered_target_without_frame_alignment(self) -> None:
        T_lidar_camera = transform([600.0, -100.0, 250.0])
        T_map_lidar = transform([4000.0, -2000.0, -34000.0])
        T_object_raw_centered = transform([-241.0, 1.0, 333.0])
        T_camera_object_centered = transform([500.0, 200.0, 22000.0])
        T_map_object_raw = (
            T_map_lidar
            @ T_lidar_camera
            @ T_camera_object_centered
            @ np.linalg.inv(T_object_raw_centered)
        )
        predictions = {
            "image_1": [
                {
                    "T_gigapose": T_camera_object_centered,
                    "scene_id": 1,
                    "im_id": 1,
                    "obj_id": 1,
                    "instance_id": 1,
                    "row_index": 0,
                    "score": 0.9,
                }
            ]
        }
        labels = {
            "image_1": [
                {
                    "T_map_lidar": transform([4000.0, -2000.0, 100.0]),
                    "T_map_lidar_target": T_map_lidar,
                    "T_map_object": T_map_object_raw,
                    "T_camera_object_original": T_camera_object_centered,
                    "T_lidar_camera_prior": T_lidar_camera,
                    "epnp_label_path": "/tmp/label.json",
                    "epnp_record_index": 0,
                    "metadata_path": "/tmp/sample.yaml",
                    "corrected_lidar_map_z_mm": -34000.0,
                    "target_lidar_z_replacement_mm": -34100.0,
                }
            ]
        }

        rows = make_candidate_rows(
            predictions,
            labels,
            T_lidar_camera,
            T_object_raw_centered,
            None,
            "right",
            20,
            Path("/tmp/optimized_extrinsics.json"),
        )

        self.assertEqual(len(rows), 1)
        self.assertLess(float(rows[0]["translation_error_mm"]), 1e-9)
        self.assertLess(float(rows[0]["rotation_error_deg"]), 1e-9)
        self.assertEqual(rows[0]["frame_transform_applied"], "none")
        self.assertEqual(
            rows[0]["T_gigapose_cam_obj"],
            rows[0]["T_gigapose_aligned_epnp_obj"],
        )

    def test_candidate_applies_explicit_right_alignment_when_requested(self) -> None:
        T_lidar_camera = transform([600.0, -100.0, 250.0])
        T_map_lidar = transform([4000.0, -2000.0, -34000.0])
        T_object_raw_centered = transform([-241.0, 1.0, 333.0])
        T_camera_object_centered = transform([500.0, 200.0, 22000.0])
        frame_transform = transform([120.0, -40.0, 25.0])
        T_gigapose_raw = T_camera_object_centered @ np.linalg.inv(frame_transform)
        T_map_object_raw = (
            T_map_lidar
            @ T_lidar_camera
            @ T_camera_object_centered
            @ np.linalg.inv(T_object_raw_centered)
        )
        predictions = {
            "image_1": [
                {
                    "T_gigapose": T_gigapose_raw,
                    "scene_id": 1,
                    "im_id": 1,
                    "obj_id": 1,
                    "row_index": 0,
                    "score": 0.9,
                }
            ]
        }
        labels = {
            "image_1": [
                {
                    "T_map_lidar": T_map_lidar,
                    "T_map_lidar_target": T_map_lidar,
                    "T_map_object": T_map_object_raw,
                    "T_camera_object_original": T_camera_object_centered,
                    "epnp_label_path": "/tmp/label.json",
                    "epnp_record_index": 0,
                    "metadata_path": "/tmp/sample.yaml",
                }
            ]
        }

        rows = make_candidate_rows(
            predictions,
            labels,
            T_lidar_camera,
            T_object_raw_centered,
            frame_transform,
            "right",
            20,
            Path("/tmp/optimized_extrinsics.json"),
        )

        self.assertLess(float(rows[0]["translation_error_mm"]), 1e-9)
        self.assertEqual(rows[0]["frame_transform_applied"], "right")
        self.assertNotEqual(
            rows[0]["T_gigapose_cam_obj"],
            rows[0]["T_gigapose_aligned_epnp_obj"],
        )

    def test_unsafe_large_correction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized_extrinsics.json"
            path.write_text(
                json.dumps(
                    {
                        "optimization_mode": (
                            "direct_map_lidar_raw_object_to_centered_gigapose"
                        ),
                        "T_lidar_camera_optimized": np.eye(4).tolist(),
                        "correction_translation_mm": [0.0, 0.0, 34000.0],
                        "max_reusable_correction_mm": 2000.0,
                        "calibration_valid_for_reuse": False,
                        "preprocessing": {
                            "T_object_raw_object_centered": np.eye(4).tolist()
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Refusing unsafe calibration"):
                load_centered_calibration(path, allow_unsafe=False)


if __name__ == "__main__":
    unittest.main()
