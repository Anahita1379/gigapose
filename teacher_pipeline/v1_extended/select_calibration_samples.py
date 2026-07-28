"""Select independently supported, well-spread refined calibration frames."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from teacher_pipeline.io import write_json, write_rows
from teacher_pipeline.trajectory import load


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trajectories",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--min-lidar-points",type=int,default=8); p.add_argument("--min-range-m",type=float,default=20); p.add_argument("--max-range-m",type=float,default=60); p.add_argument("--max-samples",type=int,default=200); p.add_argument("--min-separation-m",type=float,default=10); a=p.parse_args()
    rows=load(a.trajectories); candidates=[]
    for row in rows:
        count=int(float(row.get("lidar_point_count",0) or 0)); range_m=float(row.get("lidar_range_m",0) or 0)
        independent=(count>=a.min_lidar_points and (range_m<=0 or a.min_range_m<=range_m<=a.max_range_m)) or bool(row.get("manual_pose_support"))
        if not independent: continue
        score=float(row.get("gigapose_score",0) or 0); purity=float(row.get("cluster_purity",1) or 1); uncertainty=float(np.linalg.norm(row.get("translation_std_m",[.5,.5,.5])))
        candidates.append((-(score*purity)/(uncertainty+.05),row))
    candidates.sort(key=lambda item:item[0]); selected=[]
    for _,row in candidates:
        s=float(row.get("track_s_m",0) or 0)
        if selected and min(abs(s-float(old.get("track_s_m",0) or 0)) for old in selected)<a.min_separation_m: continue
        copy=dict(row); copy["calibration_independent_support"]=True; selected.append(copy)
        if len(selected)>=a.max_samples: break
    if not selected: raise ValueError("No independently supported calibration rows. Extended V1 requires reliable LiDAR or manual support; EPnP/GigaPose-only trajectories are circular targets.")
    write_rows(a.output,selected); write_json(a.output.with_suffix(".report.json"),{"candidate_count":len(candidates),"selected_count":len(selected),"independent_support_required":True})
    print(f"Selected {len(selected)} independent calibration samples -> {a.output}")

if __name__ == "__main__": main()
