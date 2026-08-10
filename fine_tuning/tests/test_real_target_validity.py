from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# The helper under test does not use WebDataset. Keep this focused unit test
# runnable in lightweight environments where the full inference dependency is
# intentionally absent.
try:
    import webdataset  # noqa: F401
except ImportError:
    sys.modules["webdataset"] = types.ModuleType("webdataset")
try:
    from bop_toolkit_lib import pycoco_utils  # noqa: F401
except ImportError:
    bop_module = types.ModuleType("bop_toolkit_lib")
    bop_module.pycoco_utils = types.ModuleType("pycoco_utils")
    sys.modules["bop_toolkit_lib"] = bop_module
    sys.modules["bop_toolkit_lib.pycoco_utils"] = bop_module.pycoco_utils

from Assetto_data_prep.prepare_grounded_sam_inference import load_reference_bboxes
from fine_tuning.optimize_camera_lidar_extrinsics_centered import (
    filter_explicitly_invalid_targets,
)
from fine_tuning.select_real_label_candidates import epnp_target_validity


class RealTargetValidityTest(unittest.TestCase):
    def test_validity_reader_supports_both_updated_spellings(self) -> None:
        self.assertIs(epnp_target_validity({"is_valid_target": False}), False)
        self.assertIs(
            epnp_target_validity({"target_validity": {"is_valid": True}}), True
        )
        self.assertIsNone(epnp_target_validity({}))

    def test_optimizer_rejects_only_explicit_false_by_default(self) -> None:
        samples = [
            {"label_data": {"is_valid_target": True}, "name": "valid"},
            {"label_data": {"is_valid_target": False}, "name": "cybertruck"},
            {"label_data": {}, "name": "legacy"},
        ]
        kept, rejected = filter_explicitly_invalid_targets(samples, False)
        self.assertEqual([row["name"] for row in kept], ["valid", "legacy"])
        self.assertEqual(rejected, 1)

    def test_dataset_reference_auto_excludes_invalid_but_keeps_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"detector_bbox_xywh": [1, 1, 10, 10], "is_valid_target": True},
                {"detector_bbox_xywh": [2, 2, 10, 10], "is_valid_target": False},
                {"detector_bbox_xywh": [3, 3, 10, 10]},
            ]
            for index, record in enumerate(records):
                (root / f"100{index}_0.json").write_text(json.dumps(record))

            auto = load_reference_bboxes(root, "all", "auto")
            required = load_reference_bboxes(root, "all", "require")
            ignored = load_reference_bboxes(root, "all", "ignore")

            self.assertEqual(set(auto), {"1000", "1002"})
            self.assertEqual(set(required), {"1000"})
            self.assertEqual(set(ignored), {"1000", "1001", "1002"})


if __name__ == "__main__":
    unittest.main()
