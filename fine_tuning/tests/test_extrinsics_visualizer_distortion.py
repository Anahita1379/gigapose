from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fine_tuning.visualize_epnp_gigapose_comparison_extrinsics import (
    finite_summary,
    load_metadata_camera,
    project_points,
    project_pose_center,
)


class ExtrinsicsVisualizerDistortionTest(unittest.TestCase):
    def test_equidistant_projection_uses_theta_radius(self) -> None:
        K = np.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        point = np.asarray([[1.0, 0.0, 1.0]])
        identity = np.eye(4)

        uv, valid = project_points(
            point,
            identity,
            K,
            np.zeros(4),
            "equidistant",
            "metadata",
        )

        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(uv[0, 0], 100.0 * np.arctan(1.0), places=10)
        self.assertAlmostEqual(uv[0, 1], 0.0, places=10)

    def test_pinhole_mode_ignores_equidistant_coefficients(self) -> None:
        K = np.asarray([[200.0, 0.0, 20.0], [0.0, 300.0, 30.0], [0.0, 0.0, 1.0]])
        point = np.asarray([[1.0, 2.0, 4.0]])

        uv, valid = project_points(
            point,
            np.eye(4),
            K,
            np.asarray([0.2, -0.1, 0.05, 0.01]),
            "equidistant",
            "pinhole",
        )

        self.assertTrue(bool(valid[0]))
        np.testing.assert_allclose(uv[0], [70.0, 180.0])

    def test_metadata_camera_loads_equidistant_coefficients(self) -> None:
        text = """
camera_intrinsics:
  k: [1000, 0, 100, 0, 1001, 50, 0, 0, 1]
  d: [0.1, 0.2, -0.1, 0.05]
  distortion_model: equidistant
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.yaml"
            path.write_text(text)
            K, D, model = load_metadata_camera(path)

        self.assertEqual(model, "equidistant")
        np.testing.assert_allclose(K, [[1000, 0, 100], [0, 1001, 50], [0, 0, 1]])
        np.testing.assert_allclose(D, [0.1, 0.2, -0.1, 0.05])

    def test_positive_delta_v_means_pose_center_is_below_detection(self) -> None:
        K = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 1.0, 10.0]
        center, valid = project_pose_center(
            pose,
            K,
            np.zeros(4),
            "equidistant",
            "metadata",
        )

        self.assertTrue(valid)
        self.assertIsNotNone(center)
        detection_center = np.asarray([50.0, 55.0])
        delta = center - detection_center
        self.assertGreater(delta[1], 0.0)

    def test_finite_summary_reports_median(self) -> None:
        summary = finite_summary([1.0, 3.0, 100.0, float("nan")])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 3.0)


if __name__ == "__main__":
    unittest.main()
