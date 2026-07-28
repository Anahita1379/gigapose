"""Export visible, uncertainty-aware labels for student training."""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
from .trajectory import load
from .geometry import as_pose, centered_to_raw_pose

def main():
    p=argparse.ArgumentParser(); p.add_argument("--trajectories",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--extrinsics",help="optimized extrinsic JSON"); p.add_argument("--extrinsics-version",default="fixed_iteration_0"); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); labels=[]; optimized=None
    if a.extrinsics: optimized=as_pose(json.loads(Path(a.extrinsics).read_text())["T_lidar_camera_optimized"])
    for r in load(a.trajectories):
        if r.get("image_path") in (None,"") and r.get("bbox_xywh") in (None,""): continue
        Tm=as_pose(r["T_map_object_centered_refined"]); Tml=as_pose(r["T_map_lidar"]); Tlc=optimized if optimized is not None else as_pose(r["T_lidar_camera_initial"]); Tc=np.linalg.inv(Tml@Tlc)@Tm; Tr=centered_to_raw_pose(Tc)
        label={"session_id":r.get("session_id",r.get("scene_id")),"camera":r.get("camera","unknown"),"timestamp_ns":r.get("timestamp_ns"),"track_id":int(float(r.get("track_id",0))),"image_path":r.get("image_path"),"bbox_xywh":r.get("bbox_xywh"),"mask_path":r.get("mask_path"),"T_camera_object_centered":Tc.tolist(),"T_camera_object_raw":Tr.tolist(),"T_map_object_centered":Tm.tolist(),"track_s_m":r.get("track_s_m"),"track_d_m":r.get("track_d_m"),"translation_std_m":r.get("translation_std_m",[.5,.5,.8]),"rotation_std_deg":r.get("rotation_std_deg",[3,3,6]),"teacher_confidence":r.get("teacher_confidence",.5),"teacher_status":r.get("teacher_status","refined_observed"),"sources":r.get("sources",[]),"camera_lidar_extrinsic_version":a.extrinsics_version}
        labels.append(label)
    (out/"labels.json").write_text(json.dumps(labels,indent=2)); (out/"labels.jsonl").write_text("\n".join(json.dumps(x) for x in labels)+"\n")
if __name__ == "__main__": main()
