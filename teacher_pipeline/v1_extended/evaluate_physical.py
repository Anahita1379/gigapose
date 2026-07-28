"""Evaluate raw/aligned/refined accuracy and physical trajectory quality."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from teacher_pipeline.geometry import as_pose, pose_error, raw_to_centered_pose
from teacher_pipeline.io import write_json, write_rows
from teacher_pipeline.trajectory import load
from .track_map import TrackMap


def _error(predicted,target):
    value=pose_error(target,predicted); return float(np.linalg.norm(value[:3])),float(np.degrees(np.linalg.norm(value[3:])))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trajectories",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--track-map",type=Path); p.add_argument("--held-out",action="store_true"); p.add_argument("--boundary-tolerance-m",type=float,default=.05); a=p.parse_args()
    rows=load(a.trajectories); track=TrackMap(a.track_map) if a.track_map else None; metrics=[]
    grouped={}
    for row in rows:
        grouped.setdefault(str(row.get("track_id")),[]).append(row)
        target_value=row.get("T_map_object_centered_epnp")
        if target_value is None: continue
        target=as_pose(target_value); base=as_pose(row["T_map_lidar"])@as_pose(row["T_lidar_camera_initial"])
        raw=base@raw_to_centered_pose(as_pose(row["T_camera_object_raw_gigapose"])); aligned=base@as_pose(row["T_camera_object_centered_gigapose"]); final=as_pose(row["T_map_object_centered_refined"])
        raw_t,raw_r=_error(raw,target); aligned_t,aligned_r=_error(aligned,target); final_t,final_r=_error(final,target)
        violation=0.0
        if track and row.get("track_s_m") is not None:
            sample=track.sample(float(row["track_s_m"])); d=float(row["track_d_m"]); violation=max(0.0,d-sample["right_width"],-sample["left_width"]-d)
        metrics.append({"scene_id":row.get("scene_id"),"im_id":row.get("im_id"),"track_id":row.get("track_id"),"raw_translation_error_m":raw_t,"raw_rotation_error_deg":raw_r,"aligned_translation_error_m":aligned_t,"aligned_rotation_error_deg":aligned_r,"final_translation_error_m":final_t,"final_rotation_error_deg":final_r,"track_boundary_violation_m":violation})
    accelerations=[]; jerks=[]
    for values in grouped.values():
        values.sort(key=lambda row:float(row.get("timestamp_ns",row.get("im_id",0)))); acc=np.asarray([float(row.get("longitudinal_acceleration_mps2",0) or 0) for row in values]); accelerations.extend(np.abs(acc).tolist()); jerks.extend(np.abs(np.diff(acc)).tolist())
    summary={"format":"teacher_v1_extended_evaluation","held_out":bool(a.held_out),"reference_count":len(metrics),"interpretation":"held-out" if a.held_out else "in-sample; not a generalization estimate"}
    for prefix in ("raw","aligned","final"):
        for metric in ("translation_error_m","rotation_error_deg"):
            values=np.asarray([row[f"{prefix}_{metric}"] for row in metrics]); summary[f"{prefix}_{metric}_median"]=float(np.median(values)) if len(values) else None; summary[f"{prefix}_{metric}_p90"]=float(np.percentile(values,90)) if len(values) else None
    summary.update(track_boundary_tolerance_m=a.boundary_tolerance_m,track_boundary_violation_fraction=float(np.mean([row["track_boundary_violation_m"]>a.boundary_tolerance_m for row in metrics])) if metrics else None,track_boundary_violation_m_max=float(np.max([row["track_boundary_violation_m"] for row in metrics])) if metrics else None,absolute_acceleration_mps2_median=float(np.median(accelerations)) if accelerations else None,absolute_jerk_per_step_mps2_median=float(np.median(jerks)) if jerks else None)
    a.output_dir.mkdir(parents=True,exist_ok=True); write_rows(a.output_dir/"physical_per_frame.csv",metrics); write_json(a.output_dir/"physical_summary.json",summary); print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
