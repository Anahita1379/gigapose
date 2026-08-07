"""Unit tests for command/path assembly in the isolated evaluation package."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from .common import resolve_predictions
from .evaluate_suite import command as evaluation_command
from .run_suite import MODEL_SPECS, commands as recovery_commands, selected_models


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "dataset"
    (dataset / "models").mkdir(parents=True)
    (dataset / "models" / "obj_000001.ply").write_text("ply\n")
    predictions = tmp_path / "gp" / "predictions" / "gpMultiHypothesis.csv"
    predictions.parent.mkdir(parents=True)
    predictions.write_text("scene_id,im_id,R,t,score\n")
    models = tmp_path / "models"
    for _name, (_module, relative) in MODEL_SPECS.items():
        checkpoint = models / relative
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint")
    return dataset, predictions.parent.parent, models, tmp_path / "runs"


class WrapperTests(unittest.TestCase):
    def test_resolve_predictions_accepts_csv_and_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            csv_path = tmp_path / "predictions" / "resultMultiHypothesis.csv"
            csv_path.parent.mkdir()
            csv_path.write_text("scene_id,im_id,R,t,score\n")
            self.assertEqual(resolve_predictions(csv_path), csv_path)
            self.assertEqual(resolve_predictions(tmp_path), csv_path)

    def test_selected_models_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown models"):
            selected_models("cnn,unknown")

    def test_recovery_commands_cover_all_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions, models, runs = _fixture_paths(Path(directory))
            args = argparse.Namespace(
                dataset_dir=dataset,
                predictions=predictions,
                models_root=models,
                output_root=runs,
                models=",".join(MODEL_SPECS),
                device="cuda",
                max_frames=7,
                window_size=5,
                per_frame=False,
                sequence_aware=True,
                sequence_max_frame_gap=1,
                sequence_max_time_gap_s=0.5,
                orientation_gate_mode="soft",
                state_iou_confidence_weight=0.5,
                save_overlays=True,
                overwrite=False,
                skip_missing=False,
                dry_run=True,
                runner_arg=[],
            )
            planned = recovery_commands(args)
            self.assertEqual([name for name, _command in planned], list(MODEL_SPECS))
            self.assertTrue(
                all(
                    "--window-size" in value and "--max-frames" in value
                    for _, value in planned
                )
            )
            self.assertTrue(
                all(
                    "--orientation-gate-mode" in value
                    and value[value.index("--orientation-gate-mode") + 1] == "soft"
                    and "--state-iou-confidence-weight" in value
                    and "--sequence-aware" in value
                    and value[value.index("--sequence-max-frame-gap") + 1] == "1"
                    and value[value.index("--sequence-max-time-gap-s") + 1] == "0.5"
                    for _, value in planned
                )
            )

    def test_per_frame_mode_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions, models, runs = _fixture_paths(Path(directory))
            args = argparse.Namespace(
                dataset_dir=dataset,
                predictions=predictions,
                models_root=models,
                output_root=runs,
                models="cnn_multicheckpoint",
                device="cpu",
                max_frames=2,
                window_size=1,
                per_frame=True,
                sequence_aware=True,
                sequence_max_frame_gap=1,
                sequence_max_time_gap_s=0.5,
                orientation_gate_mode="soft",
                state_iou_confidence_weight=0.5,
                save_overlays=False,
                overwrite=False,
                skip_missing=False,
                dry_run=True,
                runner_arg=[],
            )
            command = recovery_commands(args)[0][1]
            self.assertIn("--per-frame", command)
            self.assertEqual(command[command.index("--window-size") + 1], "1")

    def test_evaluation_command_includes_raw_and_recovery_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            dataset, predictions, _models, runs = _fixture_paths(tmp_path)
            for name in MODEL_SPECS:
                output = runs / name / "tracked_predictions.csv"
                output.parent.mkdir(parents=True)
                output.write_text("scene_id,im_id,R,t,score\n")
            args = argparse.Namespace(
                dataset_dir=dataset,
                gigapose_predictions=predictions,
                runs_root=runs,
                output_dir=tmp_path / "comparison",
                models=",".join(MODEL_SPECS),
                max_overlays=20,
                max_frames=None,
                skip_missing=False,
                overwrite=False,
                dry_run=True,
            )
            value = evaluation_command(args)
            specifications = [
                value[index + 1]
                for index, token in enumerate(value)
                if token == "--model"
            ]
            self.assertTrue(specifications[0].startswith("gigapose="))
            self.assertEqual(len(specifications), 1 + len(MODEL_SPECS))


if __name__ == "__main__":
    unittest.main()
