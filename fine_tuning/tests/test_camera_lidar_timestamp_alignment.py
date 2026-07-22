from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from fine_tuning.optimize_camera_lidar_extrinsics import (
    load_metadata_lidar_trajectory,
    load_yaml,
    matrix_from_yaml_key,
    prepare_target_map_lidar_transforms,
)


def transform_with_x_m(x_m: float) -> list[list[float]]:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = x_m
    return transform.tolist()


def write_metadata(
    metadata_dir: Path,
    image_ns: int,
    lidar_ns: int | None,
    x_m: float,
    index: int,
) -> Path:
    timestamps = {"image_ns": image_ns, "lidar_ns": lidar_ns}
    data = {
        "timestamps": timestamps,
        "t_map_lidar": transform_with_x_m(x_m),
    }
    path = metadata_dir / f"sample_{image_ns}_{index}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class CameraLidarTimestampAlignmentTest(unittest.TestCase):
    def test_interpolates_the_missing_timestamp_from_adjacent_anchors(self) -> None:
        image_0 = 3_902_128_165_726
        lidar_0 = 3_902_096_138_922
        image_missing = 3_902_418_113_941
        image_1 = 3_902_627_385_066
        lidar_1 = 3_902_555_773_649
        expected_lidar = 3_902_363_096_265

        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_dir = Path(temporary_directory) / "session" / "front" / "metadata"
            metadata_dir.mkdir(parents=True)
            write_metadata(metadata_dir, image_0, lidar_0, 0.0, 0)
            missing_path = write_metadata(
                metadata_dir, image_missing, None, 50.0, 1
            )
            write_metadata(metadata_dir, image_1, lidar_1, 100.0, 2)

            timestamps, _, diagnostics, timestamp_info = (
                load_metadata_lidar_trajectory(
                    metadata_dir,
                    interpolate_missing_lidar_timestamps=True,
                    timestamp_max_imputation_gap_ms=600.0,
                )
            )

            missing_info = timestamp_info[str(missing_path.resolve())]
            self.assertEqual(
                missing_info["lidar_timestamp_status"], "interpolated"
            )
            self.assertEqual(
                missing_info["lidar_timestamp_ns"], expected_lidar
            )
            self.assertIn(expected_lidar, timestamps.tolist())
            self.assertEqual(
                diagnostics[
                    "trajectory_records_with_interpolated_lidar_timestamp"
                ],
                1,
            )
            self.assertEqual(
                diagnostics[
                    "trajectory_records_with_unresolved_lidar_timestamp"
                ],
                0,
            )

    def test_dense_trajectory_is_then_interpolated_to_image_time(self) -> None:
        image_0 = 3_902_128_165_726
        lidar_0 = 3_902_096_138_922
        image_missing = 3_902_418_113_941
        image_1 = 3_902_627_385_066
        lidar_1 = 3_902_555_773_649
        expected_lidar = 3_902_363_096_265

        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_dir = Path(temporary_directory) / "session" / "front" / "metadata"
            metadata_dir.mkdir(parents=True)
            write_metadata(metadata_dir, image_0, lidar_0, 0.0, 0)
            missing_path = write_metadata(
                metadata_dir, image_missing, None, 50.0, 1
            )
            write_metadata(metadata_dir, image_1, lidar_1, 100.0, 2)

            metadata = load_yaml(missing_path)
            raw_pose = matrix_from_yaml_key(metadata, "t_map_lidar", "m")
            samples = [
                {
                    "T_map_lidar": raw_pose,
                    "metadata_path": str(missing_path),
                    "metadata": metadata,
                    "label_data": {},
                    "target_pose_path": "synthetic.json",
                }
            ]
            summary = prepare_target_map_lidar_transforms(
                samples,
                target_lidar_z_mode="raw",
                allow_missing_corrected_lidar_z=False,
                timestamp_alignment="interpolate_metadata",
                timestamp_max_bracket_gap_ms=350.0,
                timestamp_fallback="error",
                interpolate_missing_lidar_timestamps=True,
                timestamp_max_imputation_gap_ms=600.0,
            )

            pose_alpha = (image_missing - expected_lidar) / float(
                lidar_1 - expected_lidar
            )
            expected_x_mm = (50.0 + pose_alpha * 50.0) * 1000.0
            self.assertEqual(
                samples[0]["lidar_timestamp_status"], "interpolated"
            )
            self.assertEqual(samples[0]["lidar_timestamp_ns"], expected_lidar)
            self.assertEqual(
                samples[0]["timestamp_alignment_status"], "interpolated"
            )
            self.assertAlmostEqual(
                samples[0]["T_map_lidar_time_aligned"][0, 3],
                expected_x_mm,
                places=6,
            )
            self.assertEqual(
                summary["samples_with_interpolated_lidar_timestamp"], 1
            )
            self.assertEqual(summary["samples_timestamp_interpolated"], 1)

    def test_refuses_an_excessively_wide_anchor_interval(self) -> None:
        base = 4_000_000_000_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_dir = Path(temporary_directory) / "session" / "front" / "metadata"
            metadata_dir.mkdir(parents=True)
            write_metadata(metadata_dir, base, base - 10_000_000, 0.0, 0)
            missing_path = write_metadata(
                metadata_dir, base + 1_000_000_000, None, 1.0, 1
            )
            write_metadata(
                metadata_dir,
                base + 2_000_000_000,
                base + 1_990_000_000,
                2.0,
                2,
            )

            _, _, diagnostics, timestamp_info = load_metadata_lidar_trajectory(
                metadata_dir,
                interpolate_missing_lidar_timestamps=True,
                timestamp_max_imputation_gap_ms=500.0,
            )

            self.assertEqual(
                timestamp_info[str(missing_path.resolve())][
                    "lidar_timestamp_status"
                ],
                "missing:anchor_gap_too_wide",
            )
            self.assertEqual(
                diagnostics["trajectory_lidar_timestamp_imputation_failures"][
                    "anchor_gap_too_wide"
                ],
                1,
            )

    def test_refuses_to_cross_a_timestamp_reset(self) -> None:
        base = 5_000_000_000_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_dir = Path(temporary_directory) / "session" / "front" / "metadata"
            metadata_dir.mkdir(parents=True)
            write_metadata(metadata_dir, base, base - 10_000_000, 0.0, 0)
            missing_path = write_metadata(
                metadata_dir, base + 100_000_000, None, 1.0, 1
            )
            # The image clock increases while the LiDAR clock moves backward.
            write_metadata(
                metadata_dir,
                base + 200_000_000,
                base - 20_000_000,
                2.0,
                2,
            )

            _, _, diagnostics, timestamp_info = load_metadata_lidar_trajectory(
                metadata_dir,
                interpolate_missing_lidar_timestamps=True,
                timestamp_max_imputation_gap_ms=1000.0,
            )

            self.assertEqual(
                timestamp_info[str(missing_path.resolve())][
                    "lidar_timestamp_status"
                ],
                "missing:timestamp_reset_or_discontinuity",
            )
            self.assertEqual(
                diagnostics[
                    "trajectory_timestamp_reset_or_discontinuity_boundaries"
                ],
                1,
            )

    def test_refuses_ambiguous_duplicate_image_timestamp_anchors(self) -> None:
        base = 5_500_000_000_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_dir = Path(temporary_directory) / "session" / "front" / "metadata"
            metadata_dir.mkdir(parents=True)
            write_metadata(metadata_dir, base, base - 10_000_000, 0.0, 0)
            write_metadata(metadata_dir, base, base - 20_000_000, 0.0, 1)
            missing_path = write_metadata(
                metadata_dir, base + 100_000_000, None, 1.0, 2
            )
            write_metadata(
                metadata_dir,
                base + 200_000_000,
                base + 190_000_000,
                2.0,
                3,
            )

            _, _, diagnostics, timestamp_info = load_metadata_lidar_trajectory(
                metadata_dir,
                interpolate_missing_lidar_timestamps=True,
                timestamp_max_imputation_gap_ms=1000.0,
            )

            self.assertEqual(
                timestamp_info[str(missing_path.resolve())][
                    "lidar_timestamp_status"
                ],
                "missing:conflicting_anchor_timestamps",
            )
            self.assertEqual(
                diagnostics["trajectory_conflicting_anchor_timestamps"], 1
            )

    def test_never_uses_anchors_from_another_session(self) -> None:
        base = 6_000_000_000_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "session_a" / "front" / "metadata"
            second_dir = root / "session_b" / "front" / "metadata"
            first_dir.mkdir(parents=True)
            second_dir.mkdir(parents=True)
            write_metadata(first_dir, base, base - 10_000_000, 0.0, 0)
            missing_path = write_metadata(
                first_dir, base + 100_000_000, None, 1.0, 1
            )
            write_metadata(
                second_dir,
                base + 200_000_000,
                base + 190_000_000,
                2.0,
                0,
            )

            _, _, _, timestamp_info = load_metadata_lidar_trajectory(
                first_dir,
                interpolate_missing_lidar_timestamps=True,
                timestamp_max_imputation_gap_ms=1000.0,
            )

            self.assertEqual(
                timestamp_info[str(missing_path.resolve())][
                    "lidar_timestamp_status"
                ],
                "missing:not_bracketed",
            )


if __name__ == "__main__":
    unittest.main()
