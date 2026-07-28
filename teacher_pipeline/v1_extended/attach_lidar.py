"""Join independently generated LiDAR-cluster measurements to observations."""
from __future__ import annotations
import argparse
from pathlib import Path
from teacher_pipeline.io import arr, read_rows, write_json, write_rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--observations",type=Path,required=True); p.add_argument("--lidar-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--strict",action="store_true"); a=p.parse_args()
    observations=read_rows(a.observations); lidar=read_rows(a.lidar_manifest)
    def key(row): return (str(row.get("scene_id","")),str(row.get("im_id","")),str(row.get("track_id","")))
    by_key={key(row):row for row in lidar}; matched=0; missing=[]
    for row in observations:
        measurement=by_key.get(key(row)) or by_key.get((str(row.get("scene_id","")),str(row.get("im_id","")),""))
        if measurement is None: missing.append(key(row)); continue
        for field in ("lidar_cluster_path","lidar_point_count","lidar_range_m","cluster_purity","lidar_association_confidence"):
            if measurement.get(field) not in (None,""): row[field]=measurement[field]
        for field in ("lidar_centroid","lidar_centroid_map"):
            if measurement.get(field) not in (None,""): row[field]=arr(measurement[field],(3,)).tolist()
        matched+=1
    if a.strict and missing: raise ValueError(f"No LiDAR manifest row for {len(missing)} observations; first={missing[0]}")
    write_rows(a.output,observations); write_json(a.output.with_suffix(".report.json"),{"observations":len(observations),"lidar_rows":len(lidar),"matched":matched,"missing":len(missing)})
    print(f"Attached LiDAR measurements to {matched}/{len(observations)} observations")

if __name__ == "__main__": main()
