from types import SimpleNamespace

import numpy as np

from tracking.rgb_self_recovery.inference import RecoveryResult
from tracking.rgb_self_recovery.map_window_recovery.map_selector import MapWindowSelector


def _track(path):
    x = np.arange(101, dtype=float)
    center = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    np.savez_compressed(
        path, center_xyz=center, s=x,
        tangent_xyz=np.tile([1., 0., 0.], (len(x), 1)),
        lateral_xyz=np.tile([0., 1., 0.], (len(x), 1)),
        normal_xyz=np.tile([0., 0., 1.], (len(x), 1)),
        left_width=np.full(len(x), 5.), right_width=np.full(len(x), 5.),
        closed=np.asarray(False), track_length_m=np.asarray(100.),
    )


def _args(path):
    return SimpleNamespace(
        window_size=5, window_candidates=4, window_fps=10,
        window_unary_weight=1, window_max_relative_speed_mps=80,
        window_speed_sigma_mps=10, window_acceleration_sigma_mps2=20,
        window_rotation_sigma_deg=30,
        window_angular_acceleration_sigma_deg_s2=180,
        window_anchor_translation_sigma_m=.5,
        window_anchor_rotation_sigma_deg=8,
        window_continuous_refinement=False,
        track_map=path, map_extrinsics=None, map_metadata_root=None,
        map_transform_unit="m", map_weight=10, map_motion_weight=0,
        map_boundary_sigma_m=.5, map_height_sigma_m=.5,
        map_expected_center_height_m=0,
        map_heading_sigma_deg=20, map_car_width_m=2,
        map_forward_axis="x", map_allow_reverse=False,
        map_speed_sigma_mps=5, map_acceleration_sigma_mps2=10,
        map_lateral_speed_sigma_mps=2, map_backward_speed_sigma_mps=2,
        map_hard_boundary_violation_m=float("inf"),
    )


def _result(y, error, source):
    pose = np.eye(4)
    pose[:3, 3] = [20, y, 1]
    return RecoveryResult(
        pose=pose, source=source, confidence=.9, quality=error,
        silhouette_iou=.8, total_error=error,
        delta_center_crop_px=np.zeros(2), delta_log_depth=0,
        delta_rotation_rad=np.zeros(3), rendered_mask_crop=np.zeros((4, 4), bool),
    )


def test_soft_map_reranking_prefers_in_bounds_candidate(tmp_path):
    path = tmp_path / "track.npz"
    _track(path)
    selector = MapWindowSelector(_args(path))
    frame = SimpleNamespace(
        im_id=0,
        metadata={"T_map_lidar": np.eye(4).tolist(),
                  "T_lidar_camera": np.eye(4).tolist()},
    )
    selected, _, diagnostics = selector.select(
        0, frame, [_result(10, 0, "off_track"), _result(0, .2, "on_track")]
    )
    assert selected.source.startswith("on_track")
    assert diagnostics["map_available"] == 1
    assert diagnostics["map_boundary_violation_m"] == 0
