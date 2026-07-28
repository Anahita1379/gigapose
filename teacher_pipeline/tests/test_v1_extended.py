from __future__ import annotations
import argparse
import numpy as np
from teacher_pipeline.geometry import centered_to_raw_pose
from teacher_pipeline.v1_extended.prepare_track_map import prepare
from teacher_pipeline.v1_extended.refine_track_trajectories import optimize_track
from teacher_pipeline.v1_extended.track_map import TrackMap


def _write_straight_track(path):
    center=np.column_stack([np.arange(0,101,dtype=float),np.zeros(101),np.zeros(101)])
    data=prepare(center,np.full(101,5.0),np.full(101,5.0),closed=False); np.savez_compressed(path,**data)


def test_track_projection_and_reconstruction(tmp_path):
    path=tmp_path/"track.npz"; _write_straight_track(path); track=TrackMap(path)
    projection=track.project([25.2,2.0,.3]); assert abs(projection.s-25)<1.1; assert abs(projection.d-2)<1e-6
    pose=track.pose(25,2,0,np.eye(3)); np.testing.assert_allclose(pose[:3,3],[25,2,0],atol=1e-6)


def test_physical_optimizer_emits_full_state(tmp_path):
    path=tmp_path/"track.npz"; _write_straight_track(path); track=TrackMap(path); rows=[]
    for index in range(8):
        centered=np.eye(4); centered[:3,3]=[10+2*index,.15*(-1)**index,0]
        rows.append({"scene_id":1,"im_id":index,"timestamp_ns":index*100_000_000,"track_id":4,"bbox_xywh":[0,0,100,100],"gigapose_score":.9,"T_map_lidar":np.eye(4).tolist(),"T_lidar_camera_initial":np.eye(4).tolist(),"T_camera_object_raw_gigapose":centered_to_raw_pose(centered).tolist(),"T_camera_object_centered_gigapose":centered.tolist(),"sources":[]})
    args=argparse.Namespace(pose_source="raw",fps=10,max_iterations=80,reference_bbox_height_px=100,sigma_s_base_m=.35,sigma_s_per_range=.025,sigma_d_base_m=.2,sigma_d_per_range=.008,sigma_yaw_base_deg=2,sigma_yaw_per_range_deg=.04,sigma_boundary_m=.15,min_lidar_points=8,lidar_reliable_range_m=60,sigma_lidar_m=.35,sigma_motion_s_m=.25,sigma_velocity_mps=2,sigma_motion_d_m=.12,sigma_lateral_velocity_mps=1,sigma_yaw_rate_deg=8,sigma_acceleration_mps2=8,sigma_jerk_mps3=12)
    output,report=optimize_track(rows,track,None,args); assert report["success"]; assert len(output)==8
    assert all(row["teacher_status"]=="physical_track_optimized" for row in output); assert all("longitudinal_velocity_mps" in row and "yaw_offset_rad" in row for row in output)
    assert np.std([row["track_d_m"] for row in output])<.15
