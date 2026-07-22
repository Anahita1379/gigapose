from __future__ import annotations

import unittest

from tracking.select_real_label_candidates_with_centered_camera_lidar_extrinsics import (
    parse_args,
)


class CenteredCalibrationTrackingSelectorTest(unittest.TestCase):
    def test_tracking_arguments_are_parsed_and_geometry_arguments_forwarded(self) -> None:
        args, forwarded = parse_args(
            [
                "--tracked-predictions",
                "tracked.csv",
                "--optimized-extrinsics",
                "optimized_extrinsics.json",
                "--output-dir",
                "selection",
                "--allowed-tracking-modes",
                "normal",
                "--min-tracking-confidence",
                "0.65",
                "--dataset-dir",
                "dataset",
                "--max-translation-error-mm",
                "3000",
            ]
        )

        self.assertEqual(args.allowed_tracking_modes, ["normal"])
        self.assertEqual(args.min_tracking_confidence, 0.65)
        self.assertEqual(
            forwarded,
            [
                "--dataset-dir",
                "dataset",
                "--max-translation-error-mm",
                "3000",
            ],
        )

    def test_direct_gigapose_argument_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--tracked-predictions",
                    "tracked.csv",
                    "--optimized-extrinsics",
                    "optimized_extrinsics.json",
                    "--output-dir",
                    "selection",
                    "--gigapose-predictions",
                    "wrong.csv",
                ]
            )


if __name__ == "__main__":
    unittest.main()
