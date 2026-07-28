"""Explicit pose conventions and small SE(3) helpers."""
from __future__ import annotations
import numpy as np

CENTER_RAW_M = np.array([-0.2411941141, 0.0009010172, 0.3329219520], dtype=float)

def as_pose(value) -> np.ndarray:
    a = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.isfinite(a).all(): raise ValueError("pose contains non-finite values")
    return a

def centered_to_raw_pose(T_camera_object_centered, center_raw=CENTER_RAW_M):
    T = as_pose(T_camera_object_centered).copy(); T[:3, 3] -= T[:3, :3] @ np.asarray(center_raw)
    return T

def raw_to_centered_pose(T_camera_object_raw, center_raw=CENTER_RAW_M):
    T = as_pose(T_camera_object_raw).copy(); T[:3, 3] += T[:3, :3] @ np.asarray(center_raw)
    return T

def matrix(value, name):
    if value in (None, "", "null"): raise ValueError(f"missing {name}")
    return as_pose(value)

def so3_log(R):
    R = np.asarray(R, float).reshape(3, 3)
    vector = np.asarray(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    sine = 0.5 * float(np.linalg.norm(vector))
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arctan2(sine, cosine))
    if sine < 1e-8:
        if cosine > 0:
            return 0.5 * vector
        values, vectors = np.linalg.eigh(0.5 * (R + np.eye(3)))
        axis = vectors[:, int(np.argmax(values))]
        dominant = int(np.argmax(np.abs(axis)))
        axis *= 1.0 if axis[dominant] >= 0 else -1.0
        return axis * theta
    return vector * (theta / (2.0 * sine))

def so3_exp(v):
    v=np.asarray(v,float).reshape(3); theta=np.linalg.norm(v)
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if theta < 1e-8: return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3)+np.sin(theta)/theta*K+(1-np.cos(theta))/(theta*theta)*(K@K)

def pose_error(A, B):
    return np.r_[A[:3,3]-B[:3,3], so3_log(A[:3,:3] @ B[:3,:3].T)]
