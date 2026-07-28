from __future__ import annotations

import numpy as np

from teacher_pipeline.evaluate import (
    _load_extrinsic,
    _validate_trajectory_contract,
    evaluate_rows,
    overall_summary,
    save_overlays,
    save_plots,
    summarize_distance_bins,
)
import json
import pytest


def _pose(angle_deg=0.0, translation=(0.0, 0.0, 0.0)):
    angle = np.radians(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [
        [cosine, -sine, 0],
        [sine, cosine, 0],
        [0, 0, 1],
    ]
    value[:3, 3] = translation
    return value


def test_before_after_metrics_and_distance_plots(tmp_path):
    target = _pose(0, [0, 0, 20])
    row = {
        "scene_id": 1,
        "im_id": 2,
        "track_id": 4,
        "T_map_lidar": np.eye(4).tolist(),
        "T_lidar_camera_initial": np.eye(4).tolist(),
        "T_camera_object_centered_gigapose": _pose(10, [0, 0, 22]).tolist(),
        "T_camera_object_centered_epnp": target.tolist(),
        "T_map_object_centered_epnp": target.tolist(),
        "T_map_object_centered_refined": _pose(2, [0, 0, 20.5]).tolist(),
    }
    metrics, skipped = evaluate_rows([row])

    assert not skipped
    assert len(metrics) == 1
    metric = metrics[0]
    assert np.isclose(metric["reference_distance_m"], 20)
    assert np.isclose(metric["original_translation_error_m"], 2)
    assert np.isclose(metric["final_translation_error_m"], 0.5)
    assert np.isclose(metric["translation_improvement_m"], 1.5)
    assert np.isclose(metric["original_rotation_error_deg"], 10)
    assert np.isclose(metric["final_rotation_error_deg"], 2)
    assert np.isclose(metric["rotation_improvement_deg"], 8)

    bins = summarize_distance_bins(metrics, [0, 20, 40, float("inf")])
    assert len(bins) == 1
    assert bins[0]["distance_bin"] == "20-40"
    assert bins[0]["translation_improvement_m_positive_fraction"] == 1
    summary = overall_summary(metrics)
    assert summary["translation_improvement_m_positive_fraction"] == 1

    save_plots(metrics, bins, tmp_path, dpi=60)
    assert (tmp_path / "translation_error_by_distance.png").is_file()
    assert (tmp_path / "rotation_error_by_distance.png").is_file()
    assert (tmp_path / "improvement_histograms.png").is_file()


def test_three_way_overlay_writes_image_with_mock_renderer(tmp_path, monkeypatch):
    import tracking.rendering
    from PIL import Image

    captured_poses = []

    class FakeRenderer:
        def __init__(self, *_args, **_kwargs):
            pass

        def render(self, pose, _K, image_shape):
            captured_poses.append(np.asarray(pose).copy())
            mask = np.zeros(image_shape, dtype=bool)
            mask[10:20, 15:25] = True
            return mask, np.zeros(image_shape, dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr(tracking.rendering, "CADRenderer", FakeRenderer)
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (64, 48), (80, 80, 80)).save(image_path)
    target = _pose(0, [0, 0, 20])
    rows = [
        {
            "image_path": str(image_path),
            "K": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
            "T_map_lidar": np.eye(4).tolist(),
            "T_lidar_camera_initial": np.eye(4).tolist(),
            "T_camera_object_centered_gigapose": _pose(5, [0, 0, 21]).tolist(),
            "T_camera_object_centered_epnp": target.tolist(),
            "T_map_object_centered_epnp": target.tolist(),
            "T_map_object_centered_refined": _pose(1, [0, 0, 20.2]).tolist(),
        }
    ]
    metrics, _ = evaluate_rows(rows)
    output = tmp_path / "overlays"
    written, failures = save_overlays(
        rows,
        metrics,
        output,
        tmp_path / "mesh.ply",
        1.0,
        "raw",
        None,
        1,
    )

    assert written == 1
    assert not failures
    assert len(captured_poses) == 3
    assert len(list(output.glob("*.jpg"))) == 1
    assert (output / "overlay_index.csv").is_file()
    assert (output / "overlay_report.json").is_file()


def test_invalid_old_extrinsic_is_rejected(tmp_path):
    path = tmp_path / "extrinsic.json"
    path.write_text(
        json.dumps(
            {
                "T_lidar_camera_optimized": np.eye(4).tolist(),
                "correction_translation_m": [-6.3, -1.7, -34.2],
            }
        )
    )
    with pytest.raises(ValueError, match="implausible"):
        _load_extrinsic(path)


def test_stale_mesh_z_trajectory_is_rejected():
    rows = [
        {
            "epnp_label_path": (
                "/session/front/EPnPv2_gt_mesh_z_hybrid_labels/1.json"
            )
        }
    ]
    with pytest.raises(ValueError, match="old incompatible adapter"):
        _validate_trajectory_contract(rows)
