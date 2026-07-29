from __future__ import annotations
import argparse
import numpy as np
from scipy.spatial import cKDTree
from teacher_pipeline.geometry import centered_to_raw_pose
from teacher_pipeline.v1_extended.prepare_track_map import prepare
from teacher_pipeline.v1_extended.compare_versions import compare_pairs, match_rows
from teacher_pipeline.v1_extended.build_hybrid_teacher import merge
from teacher_pipeline.v1_extended.build_full_observations import (
    _build_track_id_lookup,
    _track_id_for_prediction,
)
from teacher_pipeline.v1_extended.extract_track_map_from_surface import (
    _apply_planar_transform,
    _correct_unsupported_route_detours,
    _fit_planar_transform,
    _select_guided_cycle,
)
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


def test_version_comparison_matches_and_scores_common_frames():
    target=np.eye(4); target[:3,3]=[10,0,0]
    raw_centered=np.eye(4); raw_centered[:3,3]=[12,0,0]
    base={"scene_id":1,"im_id":2,"track_id":"4","timestamp_ns":1000000000,"T_map_lidar":np.eye(4).tolist(),"T_lidar_camera_initial":np.eye(4).tolist(),"T_camera_object_raw_gigapose":centered_to_raw_pose(raw_centered).tolist(),"T_camera_object_centered_gigapose":raw_centered.tolist(),"T_camera_object_centered_epnp":target.tolist(),"T_map_object_centered_epnp":target.tolist()}
    v1={**base,"T_map_object_centered_refined":np.array([[1,0,0,10.8],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float).tolist()}
    extended={**base,"track_id":4,"T_map_object_centered_refined":np.array([[1,0,0,10.2],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float).tolist()}
    pairs,missing_v1,missing_extended=match_rows([v1],[extended]); assert len(pairs)==1; assert not missing_v1; assert not missing_extended
    metrics,skipped=compare_pairs(pairs); assert not skipped; assert np.isclose(metrics[0]["v1_translation_error_m"],.8); assert np.isclose(metrics[0]["extended_translation_error_m"],.2); assert metrics[0]["extended_minus_v1_translation_improvement_m"]>0


def test_recorded_path_selects_main_loop_over_parallel_pit_loop():
    main=np.asarray([[0,0],[0,20],[20,20],[20,0]])
    pit=np.asarray([[5,0],[5,16],[15,16],[15,0]])
    guide=np.asarray([[0,2],[0,10],[0,18],[10,20],[18,20]],dtype=float)
    chosen,diagnostics,selected=_select_guided_cycle([main,pit],guide,.5)
    np.testing.assert_array_equal(chosen,main); assert len(diagnostics)==2; assert selected["candidate_index"]==0


def test_planar_track_registration_recovers_yaw_and_translation():
    angle=np.linspace(0.0,2.0*np.pi,500,endpoint=False)
    candidate=np.column_stack([120.0*np.cos(angle)+15.0*np.cos(2.0*angle),70.0*np.sin(angle)])
    expected=np.asarray([np.deg2rad(8.0),-42.0,31.0])
    guide=_apply_planar_transform(candidate[::4],expected)
    estimated,distances=_fit_planar_transform(candidate,guide,30.0)
    assert np.median(distances)<0.5
    assert abs(np.rad2deg(estimated[0]-expected[0]))<0.5
    assert np.linalg.norm(estimated[1:]-expected[1:])<1.0


def test_dense_guide_replaces_local_skeleton_detour():
    ordered=np.asarray(
        [[x,0] for x in range(6)]
        + [[5,y] for y in range(1,11)]
        + [[x,10] for x in range(6,21)]
        + [[20,y] for y in range(9,-1,-1)]
        + [[x,0] for x in range(21,101)],dtype=float
    )
    guide=np.asarray([[x,0] for x in np.linspace(0,100,201)],dtype=float)
    sequence=[(1,index,index*100_000_000) for index in range(len(guide))]
    corrected,corrections=_correct_unsupported_route_detours(
        ordered,guide,sequence,threshold_px=2,max_gap_s=1,max_frame_gap=2
    )
    assert len(corrections)==1
    assert np.max(cKDTree(corrected).query(guide)[0])<1.5


def test_hybrid_teacher_prefers_selected_v1_and_fills_remaining_rows():
    pose_v1=np.eye(4); pose_v1[0,3]=1
    pose_extended=np.eye(4); pose_extended[0,3]=2
    extended=[
        {"scene_id":1,"im_id":1,"track_id":"0","T_map_object_centered_refined":pose_extended.tolist(),"track_s_m":10,"sources":["track_map"]},
        {"scene_id":1,"im_id":2,"track_id":"0","T_map_object_centered_refined":pose_extended.tolist(),"track_s_m":11,"sources":["track_map"]},
    ]
    selected=[{"scene_id":1,"im_id":1,"track_id":0,"T_map_object_centered_refined":pose_v1.tolist(),"sources":["epnp_hybrid"]}]
    output,count,fallback=merge(selected,extended,.9,.5)
    assert count==1 and fallback==0 and len(output)==2
    assert output[0]["T_map_object_centered_refined"]==pose_v1.tolist()
    assert output[0]["track_s_m"] is None
    assert output[1]["T_map_object_centered_refined"]==pose_extended.tolist()
    assert output[1]["track_s_m"]==11


def test_raw_prediction_can_borrow_only_track_id_from_refined_tracking_row():
    tracking = [{
        "scene_id": 1, "im_id": 4, "instance_id": 7, "track_id": 3,
        "R": "refined rotation must not be consumed",
        "t": "refined translation must not be consumed",
    }]
    by_instance, by_frame = _build_track_id_lookup(tracking)
    raw_prediction = {
        "scene_id": 1, "im_id": 4,
        "R": "raw rotation", "t": "raw translation",
    }
    track_id, source = _track_id_for_prediction(
        raw_prediction, by_instance, by_frame
    )
    assert track_id == 3
    assert source == "track_ids_from.unique_frame"
    assert raw_prediction["R"] == "raw rotation"
    assert raw_prediction["t"] == "raw translation"
