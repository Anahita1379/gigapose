from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from teacher_pipeline.build_observations import build_from_selected
from teacher_pipeline.geometry import CENTER_RAW_M


def _pose(translation):
    value = np.eye(4)
    value[:3, 3] = translation
    return value


def test_selected_epnp_and_yaml_contract(tmp_path):
    dataset = tmp_path / "dataset"
    labels = tmp_path / "session" / "front" / "EPnPv2_gt_mesh_z_hybrid_labels"
    metadata = tmp_path / "session" / "front" / "metadata"
    dataset.mkdir()
    labels.mkdir(parents=True)
    metadata.mkdir(parents=True)

    T_map_lidar_m = _pose([10.0, 20.0, 1.0])
    T_lidar_camera_m = _pose([0.4, -0.1, 0.3])
    T_map_object_raw_m = _pose([30.0, 40.0, 2.0])
    T_camera_object_centered_m = _pose([1.0, 2.0, 50.0])
    metadata_path = metadata / "sample_123_0.yaml"
    metadata_path.write_text(
        json.dumps(
            {
                "t_map_lidar": T_map_lidar_m.tolist(),
                "t_lidar_camera_prior": T_lidar_camera_m.tolist(),
                "camera_intrinsics": {
                    "k": [1000, 0, 640, 0, 1000, 360, 0, 0, 1],
                    "d": [0.1, 0.01],
                    "distortion_model": "equidistant",
                },
            }
        )
    )
    label_path = labels / "123_0.json"
    label_path.write_text(
        json.dumps(
            {
                "T_map_object_raw": T_map_object_raw_m.tolist(),
                "T_camera_object_centered": T_camera_object_centered_m.tolist(),
                "metadata_path": str(metadata_path),
            }
        )
    )
    (dataset / "frame_map.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": 1,
                    "im_id": 2,
                    "source_root": "/recordings/session",
                    "camera_id": "front",
                    "image_path": "/images/image_123.jpg",
                    "mask_path": "/masks/image_123.png",
                    "instances": [{"instance_id": 7, "bbox": [1, 2, 30, 40]}],
                }
            ]
        )
    )
    predictions_path = tmp_path / "predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene_id", "im_id", "instance_id", "track_id", "score", "R", "t"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_id": 1,
                "im_id": 2,
                "instance_id": 7,
                "track_id": 42,
                "score": 0.9,
                "R": "1 0 0 0 1 0 0 0 1",
                "t": "1000 2000 50000",
            }
        )
    selected_path = tmp_path / "selected.csv"
    with selected_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_id",
                "im_id",
                "prediction_row_index",
                "epnp_label_path",
                "epnp_record_index",
                "sample_metadata_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_id": 1,
                "im_id": 2,
                "prediction_row_index": 0,
                "epnp_label_path": str(label_path),
                "epnp_record_index": 0,
                "sample_metadata_path": str(metadata_path),
            }
        )

    args = argparse.Namespace(
        predictions=predictions_path,
        selected_samples=selected_path,
        dataset_dir=dataset,
        frame_map=None,
        tracks=None,
        epnp_root=None,
        metadata_root=None,
        center_raw=CENTER_RAW_M.tolist(),
        epnp_map_pose_key="T_map_object_raw",
        epnp_camera_pose_key="T_camera_object_centered",
        epnp_map_pose_unit="m",
        epnp_camera_pose_unit="m",
        prediction_translation_unit="mm",
        gigapose_object_origin="raw",
        strict=True,
    )
    observations, report = build_from_selected(args)

    assert report["skipped_count"] == 0
    assert len(observations) == 1
    row = observations[0]
    assert row["track_id"] == "42"
    assert row["track_id_source"] == "prediction.track_id"
    assert row["bbox_xywh"] == [1, 2, 30, 40]
    assert row["distortion_model"] == "equidistant"
    np.testing.assert_allclose(row["T_map_lidar"], T_map_lidar_m)
    np.testing.assert_allclose(row["T_lidar_camera_initial"], T_lidar_camera_m)
    np.testing.assert_allclose(row["T_map_object_raw_epnp"], T_map_object_raw_m)
    expected_centered = T_map_object_raw_m.copy()
    expected_centered[:3, 3] += CENTER_RAW_M
    np.testing.assert_allclose(row["T_map_object_centered_epnp"], expected_centered)
    # Raw GigaPose millimetres become metres, then receive the known center shift.
    np.testing.assert_allclose(
        np.asarray(row["T_camera_object_centered_gigapose"])[:3, 3],
        np.asarray([1.0, 2.0, 50.0]) + CENTER_RAW_M,
    )
