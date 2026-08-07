from types import SimpleNamespace

import numpy as np

from tracking.rgb_self_recovery.inference import RecoveryResult
from tracking.rgb_self_recovery.sliding_window.optimizer import SlidingWindowSelector


def _args(**overrides):
    values = dict(
        window_size=5, window_candidates=4, window_fps=10,
        window_unary_weight=1, window_max_relative_speed_mps=30,
        window_speed_sigma_mps=5, window_acceleration_sigma_mps2=3,
        window_rotation_sigma_deg=30,
        window_angular_acceleration_sigma_deg_s2=180,
        window_anchor_translation_sigma_m=.5,
        window_anchor_rotation_sigma_deg=8,
        window_continuous_refinement=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(x, error, source):
    pose = np.eye(4)
    pose[0, 3] = x
    pose[2, 3] = 20
    return RecoveryResult(
        pose=pose, source=source, confidence=.9, quality=error,
        silhouette_iou=.8, total_error=error,
        delta_center_crop_px=np.zeros(2), delta_log_depth=0,
        delta_rotation_rad=np.zeros(3), rendered_mask_crop=np.zeros((4, 4), bool),
    )


def test_window_rejects_low_unary_large_motion_outlier():
    selector = SlidingWindowSelector(_args())
    for index in range(4):
        frame = SimpleNamespace(im_id=index, metadata={})
        selected, _, _ = selector.select(0, frame, [_result(index, .1, f"good{index}")])
        assert selected.source.startswith("good")
    frame = SimpleNamespace(im_id=4, metadata={})
    selected, _, diagnostics = selector.select(
        0, frame, [_result(25, 0., "rgb_outlier"), _result(4, .25, "coherent")]
    )
    assert selected.source.startswith("coherent")
    assert diagnostics["window_size_used"] == 5


def test_clear_prevents_window_leakage_between_scenes():
    selector = SlidingWindowSelector(_args())
    frame = SimpleNamespace(im_id=0, metadata={})
    selector.select(0, frame, [_result(0, 0, "old_scene")])
    selector.clear()
    selected, _, diagnostics = selector.select(
        0, frame, [_result(50, 0, "new_scene")]
    )
    assert selected.source.startswith("new_scene")
    assert diagnostics["window_size_used"] == 1


def test_selector_retains_pre_smoothing_visual_anchor():
    selector = SlidingWindowSelector(_args(window_continuous_refinement=True))
    for index in range(3):
        frame = SimpleNamespace(im_id=index, metadata={})
        candidate = _result(index, 0.1, f"candidate{index}")
        selector.select(7, frame, [candidate])
    anchor = selector.selected_anchor(7)
    assert anchor is not None
    assert anchor.source == "candidate2"
    selector.remove(7)
    assert selector.selected_anchor(7) is None


def test_window_size_one_is_genuine_per_frame_selection():
    selector = SlidingWindowSelector(_args(window_size=1, window_continuous_refinement=True))
    first = SimpleNamespace(im_id=0, metadata={})
    selector.select(0, first, [_result(0, 0.0, "old")])
    second = SimpleNamespace(im_id=1, metadata={})
    selected, _, diagnostics = selector.select(
        0,
        second,
        [_result(100, 0.0, "current_best"), _result(1, 0.5, "temporally_close")],
    )
    assert selected.source.startswith("current_best")
    assert diagnostics["window_size_used"] == 1
    assert diagnostics["window_speed_cost"] == 0.0
    assert diagnostics["window_acceleration_cost"] == 0.0
    assert diagnostics["window_continuous_refinement"] == 0
