"""Verify that track-map and metadata map coordinates actually coincide."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from teacher_pipeline.geometry import as_pose, raw_to_centered_pose
from teacher_pipeline.trajectory import load
from .track_map import TrackMap


def main():
    p=argparse.ArgumentParser(); p.add_argument("--observations",type=Path,required=True); p.add_argument("--track-map",type=Path,required=True); p.add_argument("--max-median-distance-m",type=float,default=15); p.add_argument("--max-p90-distance-m",type=float,default=20); a=p.parse_args()
    track=TrackMap(a.track_map); rows=load(a.observations)
    tree=cKDTree(track.center[:,:2]); ego=[]; raw_objects=[]; seen=set()
    for row in rows:
        token=(row.get("scene_id"),row.get("im_id"))
        if token not in seen:
            seen.add(token); ego.append(as_pose(row["T_map_lidar"])[:2,3])
        camera=raw_to_centered_pose(as_pose(row["T_camera_object_raw_gigapose"])); map_object=as_pose(row["T_map_lidar"])@as_pose(row["T_lidar_camera_initial"])@camera; raw_objects.append(map_object[:2,3])
    ego_distances=tree.query(np.asarray(ego))[0]; raw_distances=tree.query(np.asarray(raw_objects))[0]
    summary={
        "unique_ego_pose_count":len(ego_distances),
        "ego_to_track_horizontal_distance_m_median":float(np.median(ego_distances)),
        "ego_to_track_horizontal_distance_m_p90":float(np.percentile(ego_distances,90)),
        "raw_gigapose_object_to_track_horizontal_distance_m_median":float(np.median(raw_distances)),
        "raw_gigapose_object_to_track_horizontal_distance_m_p90":float(np.percentile(raw_distances,90)),
        "threshold_m":a.max_median_distance_m,
        "p90_threshold_m":a.max_p90_distance_m,
        "aligned":bool(np.median(ego_distances)<=a.max_median_distance_m and np.percentile(ego_distances,90)<=a.max_p90_distance_m),
        "alignment_decision_source":"unique_T_map_lidar_horizontal_positions",
    }; print(json.dumps(summary,indent=2))
    if not summary["aligned"]: raise SystemExit("Track map and metadata map frames are not aligned. Adjust surface extraction --axis-order/--translation; do not run physical refinement yet.")

if __name__ == "__main__": main()
