from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tracking.compare_predictions_without_gt import (
    associate_frame,
    compare_models,
    load_predictions,
    main,
    temporal_rows,
)


IDENTITY = "1 0 0 0 1 0 0 0 1"


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scene_id",
        "im_id",
        "obj_id",
        "score",
        "R",
        "t",
        "instance_id",
        "rank",
        "track_id",
        "tracking_mode",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row(
    im_id: int,
    translation_mm: tuple[float, float, float],
    instance_id: int,
    *,
    score: float = 0.8,
    rank: int = 0,
    track_id: int | str = "",
) -> dict[str, object]:
    return {
        "scene_id": 1,
        "im_id": im_id,
        "obj_id": 1,
        "score": score,
        "R": IDENTITY,
        "t": " ".join(str(value) for value in translation_mm),
        "instance_id": instance_id,
        "rank": rank,
        "track_id": track_id,
        "tracking_mode": "normal",
        "source": "unit_test",
    }


class GroundTruthFreeComparisonTests(unittest.TestCase):
    def test_multi_hypothesis_loader_keeps_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi.csv"
            write_predictions(
                path,
                [
                    row(1, (1000, 0, 10000), 11, score=0.95, rank=1),
                    row(1, (0, 0, 10000), 11, score=0.80, rank=0),
                ],
            )
            predictions = load_predictions(path, "reference", "mm")
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].rank, 0)
        np.testing.assert_allclose(predictions[0].pose[:3, 3], [0, 0, 10])

    def test_hungarian_pose_association_handles_reordered_cars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.csv"
            tracked_path = root / "tracked.csv"
            write_predictions(
                reference_path,
                [
                    row(1, (-2000, 0, 20000), 10),
                    row(1, (3000, 0, 20000), 11),
                ],
            )
            write_predictions(
                tracked_path,
                [
                    row(1, (3100, 0, 20100), 1, track_id=1),
                    row(1, (-1900, 0, 19900), 0, track_id=0),
                ],
            )
            reference = load_predictions(reference_path, "reference", "mm")
            tracked = load_predictions(tracked_path, "tracked", "mm")
            matches, missing_reference, missing_tracked = associate_frame(
                reference,
                tracked,
                max_center_distance=0.35,
                max_relative_translation=0.5,
                max_cost=1.0,
            )
        self.assertEqual(len(matches), 2)
        self.assertEqual(missing_reference, [])
        self.assertEqual(missing_tracked, [])
        paired_x = {
            (
                round(reference[reference_index].pose[0, 3], 1),
                round(tracked[tracked_index].pose[0, 3], 1),
            )
            for reference_index, tracked_index, _, _ in matches
        }
        self.assertEqual(paired_x, {(-2.0, -1.9), (3.0, 3.1)})

    def test_comparison_reports_changes_not_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.csv"
            tracked_path = root / "tracked.csv"
            write_predictions(reference_path, [row(1, (0, 0, 10000), 5)])
            write_predictions(
                tracked_path,
                [row(1, (300, 400, 11000), 0, track_id=0)],
            )
            reference = load_predictions(reference_path, "reference", "mm")
            tracked = load_predictions(tracked_path, "tracked", "mm")
            rows, coverage = compare_models(
                "reference",
                reference,
                "tracked",
                tracked,
                {},
                {},
                max_center_distance=0.35,
                max_relative_translation=0.5,
                max_cost=1.0,
            )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["translation_change_m"], np.sqrt(1.25))
        self.assertAlmostEqual(rows[0]["abs_depth_change_m"], 1.0)
        self.assertNotIn("translation_error_m", rows[0])
        self.assertEqual(coverage["matched_predictions"], 1)

    def test_temporal_metrics_use_stable_model_track(self) -> None:
        comparison_rows = []
        for im_id, x in ((1, 0.0), (2, 1.0), (3, 3.0)):
            comparison_rows.append(
                {
                    "model": "tracked",
                    "scene_id": 1,
                    "im_id": im_id,
                    "model_track_id": 7,
                    "_reference_rotation": np.eye(3).reshape(-1).tolist(),
                    "_model_rotation": np.eye(3).reshape(-1).tolist(),
                    "_reference_translation": [float(im_id - 1), 0.0, 10.0],
                    "_model_translation": [x, 0.0, 10.0],
                }
            )
        transitions = temporal_rows(comparison_rows)
        self.assertEqual(len(transitions), 2)
        self.assertAlmostEqual(
            transitions[-1]["model_translation_acceleration_m_per_frame2"], 1.0
        )
        self.assertAlmostEqual(
            transitions[-1]["reference_translation_acceleration_m_per_frame2"],
            0.0,
        )

    def test_end_to_end_writes_reports_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.csv"
            tracked_path = root / "tracked.csv"
            output = root / "output"
            write_predictions(
                reference_path,
                [
                    row(1, (0, 0, 10000), 10),
                    row(2, (100, 0, 10100), 11),
                    row(3, (200, 0, 10200), 12),
                ],
            )
            write_predictions(
                tracked_path,
                [
                    row(1, (0, 0, 10000), 0, score=0.7, track_id=0),
                    row(2, (120, 0, 10120), 0, score=0.8, track_id=0),
                    row(3, (220, 0, 10220), 0, score=0.9, track_id=0),
                ],
            )
            argv = [
                "tracking.compare_predictions_without_gt",
                "--model",
                f"reference={reference_path}",
                "--model",
                f"tracked={tracked_path}",
                "--output-dir",
                str(output),
                "--confidence-thresholds",
                "0.0",
                "0.8",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            expected = (
                "matched_pose_changes.csv",
                "overall_summary.csv",
                "coverage.csv",
                "by_confidence.csv",
                "temporal_summary.csv",
                "summary.json",
                "REPORT.md",
                "plots/translation_change_vs_distance.png",
                "plots_vs_confidence/summary_vs_confidence.png",
            )
            for relative_path in expected:
                self.assertTrue((output / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
