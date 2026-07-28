"""Export complete physical-teacher labels for student training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from teacher_pipeline.geometry import as_pose, centered_to_raw_pose
from teacher_pipeline.trajectory import load


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trajectories",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--extrinsics"); p.add_argument("--extrinsics-version",default="v1_extended_fixed_0"); a=p.parse_args()
    output=Path(a.output_dir); output.mkdir(parents=True,exist_ok=True); optimized=None
    if a.extrinsics: optimized=as_pose(json.loads(Path(a.extrinsics).read_text())["T_lidar_camera_optimized"])
    labels=[]
    for row in load(a.trajectories):
        if row.get("image_path") in (None,"") and row.get("bbox_xywh") in (None,""): continue
        map_object=as_pose(row["T_map_object_centered_refined"]); map_lidar=as_pose(row["T_map_lidar"]); lidar_camera=optimized if optimized is not None else as_pose(row["T_lidar_camera_initial"])
        camera_object=np.linalg.inv(map_lidar@lidar_camera)@map_object
        labels.append({
            "session_id":row.get("session_id",row.get("scene_id")),"camera":row.get("camera","unknown"),"timestamp_ns":row.get("timestamp_ns"),"track_id":row.get("track_id"),
            "image_path":row.get("image_path"),"bbox_xywh":row.get("bbox_xywh"),"mask_path":row.get("mask_path"),
            "T_camera_object_centered":camera_object.tolist(),"T_camera_object_raw":centered_to_raw_pose(camera_object).tolist(),"T_map_object_centered":map_object.tolist(),
            "track_s_m":row.get("track_s_m"),"track_d_m":row.get("track_d_m"),"longitudinal_velocity_mps":row.get("longitudinal_velocity_mps"),"lateral_velocity_mps":row.get("lateral_velocity_mps"),"longitudinal_acceleration_mps2":row.get("longitudinal_acceleration_mps2"),"yaw_offset_rad":row.get("yaw_offset_rad"),
            "translation_std_m":row.get("translation_std_m",[.5,.5,.8]),"rotation_std_deg":row.get("rotation_std_deg",[3,3,6]),"teacher_confidence":row.get("teacher_confidence",.5),"teacher_status":row.get("teacher_status","physical_track_optimized"),"sources":row.get("sources",[]),
            "gigapose_translation_correction_m":row.get("gigapose_translation_correction_m"),"gigapose_rotation_correction_deg":row.get("gigapose_rotation_correction_deg"),"camera_lidar_extrinsic_version":a.extrinsics_version,
        })
    (output/"labels.json").write_text(json.dumps(labels,indent=2)); (output/"labels.jsonl").write_text("\n".join(json.dumps(label) for label in labels)+("\n" if labels else "")); print(f"Exported {len(labels)} visible labels -> {output}")

if __name__ == "__main__": main()
