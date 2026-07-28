"""Verify that track-map and metadata map coordinates actually coincide."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from teacher_pipeline.geometry import as_pose, raw_to_centered_pose
from teacher_pipeline.trajectory import load
from .track_map import TrackMap


def main():
    p=argparse.ArgumentParser(); p.add_argument("--observations",type=Path,required=True); p.add_argument("--track-map",type=Path,required=True); p.add_argument("--max-median-distance-m",type=float,default=15); a=p.parse_args()
    track=TrackMap(a.track_map); distances=[]
    for row in load(a.observations):
        camera=raw_to_centered_pose(as_pose(row["T_camera_object_raw_gigapose"])); map_object=as_pose(row["T_map_lidar"])@as_pose(row["T_lidar_camera_initial"])@camera; distances.append(track.project(map_object[:3,3]).distance)
    summary={"count":len(distances),"nearest_track_distance_m_median":float(np.median(distances)),"nearest_track_distance_m_p90":float(np.percentile(distances,90)),"threshold_m":a.max_median_distance_m,"aligned":bool(np.median(distances)<=a.max_median_distance_m)}; print(json.dumps(summary,indent=2))
    if not summary["aligned"]: raise SystemExit("Track map and metadata map frames are not aligned. Adjust surface extraction --axis-order/--translation; do not run physical refinement yet.")

if __name__ == "__main__": main()
