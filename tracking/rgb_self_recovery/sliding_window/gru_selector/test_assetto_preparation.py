"""Unit tests for the isolated multi-run Assetto preparation policy."""

from __future__ import annotations

import unittest

from .prepare_assetto_dataset import (
    assign_scene_ids,
    balanced_validation_subset,
    resolve_runs,
    selected_runs,
)


class AssettoPreparationTests(unittest.TestCase):
    def test_roles_are_disjoint(self):
        available = {"train_a", "train_b", "validation", "evaluation"}
        train = selected_runs(available, "train", "validation", "evaluation")
        validation = selected_runs(available, "validation", "validation", "evaluation")
        evaluation = selected_runs(available, "evaluation", "validation", "evaluation")
        self.assertEqual(train, {"train_a", "train_b"})
        self.assertEqual(validation, {"validation", "evaluation"})
        self.assertFalse(train & validation)
        self.assertFalse(train & evaluation)
        self.assertEqual(evaluation, {"evaluation"})

    def test_benchmark_role_keeps_every_available_run(self):
        available = {"benchmark_a", "benchmark_b", "benchmark_c"}
        selected = selected_runs(
            available, "benchmark", "not_present", "also_not_present"
        )
        self.assertEqual(selected, available)

    def test_explicit_runs_override_old_role_holds(self):
        available = {"old_validation", "old_evaluation", "new_front", "new_rear"}
        training, source = resolve_runs(
            available, "train", "old_validation", "old_evaluation", "all"
        )
        validation, validation_source = resolve_runs(
            available,
            "validation",
            "old_validation",
            "old_evaluation",
            "new_front,new_rear",
        )
        self.assertEqual(training, available)
        self.assertEqual(source, "explicit_all")
        self.assertEqual(validation, {"new_front", "new_rear"})
        self.assertEqual(validation_source, "explicit_include")

    def test_explicit_include_rejects_unknown_run(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_runs({"known"}, "benchmark", "x", "y", "missing")

    def test_each_run_camera_gets_one_scene(self):
        rows = [
            {"key": ("run_b", 2, "rear")},
            {"key": ("run_a", 1, "front")},
            {"key": ("run_a", 2, "front")},
            {"key": ("run_a", 2, "rear")},
        ]
        mapping = assign_scene_ids(rows)
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[("run_a", "front")], 1)
        self.assertNotEqual(mapping[("run_a", "front")], mapping[("run_a", "rear")])

    def test_validation_cap_preserves_consecutive_frames(self):
        rows = [
            {"key": ("run", frame, "front"), "bin_name": "near" if frame < 5 else "far"}
            for frame in range(10)
        ]
        selected, report = balanced_validation_subset(rows, 4)
        frames = [row["key"][1] for row in selected]
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[-1] - frames[0], 3)
        self.assertEqual(sum(report[0]["distance_bin_counts"].values()), 4)

    def test_validation_cap_can_fill_from_multiple_long_segments(self):
        rows = [
            {"key": ("run", frame, "rear"), "bin_name": "near"}
            for frame in [*range(0, 4), *range(10, 14), *range(20, 24)]
        ]
        selected, report = balanced_validation_subset(rows, 10)
        frames = [row["key"][1] for row in selected]
        self.assertEqual(len(frames), 10)
        self.assertEqual(len(set(frames)), 10)
        self.assertEqual(
            report[0]["selection_strategy"], "largest_consecutive_segments"
        )


if __name__ == "__main__":
    unittest.main()
