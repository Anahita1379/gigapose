import pytest

from tracking.rgb_self_recovery.sliding_window.checkpoint_selection import (
    CHECKPOINT_NAMES,
    checkpoint_selection_values,
)


def test_checkpoint_selection_normalizes_center_and_rotation_equally():
    values = checkpoint_selection_values(
        {
            "loss": 0.2,
            "center_error_crop_px": 28.0,
            "log_depth_abs_error": 0.3,
            "rotation_error_deg": 35.0,
        },
        max_center_px=56.0,
        max_log_depth=0.6,
        max_rotation_deg=70.0,
    )
    assert values == {
        "loss": 0.2,
        "rotation": 35.0,
        "center": 28.0,
        "depth": 0.3,
        "pose": 0.5,
    }
    assert set(CHECKPOINT_NAMES) == {"loss", "rotation", "center", "depth", "pose"}


def test_checkpoint_selection_rejects_invalid_scale():
    with pytest.raises(ValueError, match="must be positive"):
        checkpoint_selection_values(
            {
                "loss": 0.2,
                "center_error_crop_px": 2.0,
                "log_depth_abs_error": 0.1,
                "rotation_error_deg": 3.0,
            },
            max_center_px=0.0,
            max_log_depth=0.6,
            max_rotation_deg=70.0,
        )
