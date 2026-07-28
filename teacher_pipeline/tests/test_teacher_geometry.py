import numpy as np

from teacher_pipeline.geometry import (
    CENTER_RAW_M,
    centered_to_raw_pose,
    raw_to_centered_pose,
    so3_log,
)


def test_origin_conversion_roundtrip():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    assert np.allclose(raw_to_centered_pose(centered_to_raw_pose(pose)), pose)
    assert np.allclose(
        centered_to_raw_pose(pose)[:3, 3], pose[:3, 3] - CENTER_RAW_M
    )


def test_so3_log_is_stable_at_180_degrees():
    rotation = np.diag([-1.0, -1.0, 1.0])
    assert np.isclose(np.linalg.norm(so3_log(rotation)), np.pi)
