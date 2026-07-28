"""Map-frame initialization and robust fixed-extrinsic smoothing."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .geometry import as_pose, so3_exp, so3_log
from .io import write_rows

def load(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def map_pose(r, extrinsic=None):
    Tml=as_pose(r["T_map_lidar"]); Tlc=as_pose(extrinsic if extrinsic is not None else r["T_lidar_camera_initial"]); Tc=as_pose(r["T_camera_object_centered_gigapose"])
    return Tml@Tlc@Tc

def main():
    p=argparse.ArgumentParser(); p.add_argument("--observations",required=True); p.add_argument("--output",required=True); p.add_argument("--extrinsics",help="JSON from optimize_extrinsic; uses optimized T_lidar_camera"); a=p.parse_args(); rows=load(a.observations); rows.sort(key=lambda r:(str(r.get("track_id")),float(r.get("timestamp_ns",r.get("im_id",0))))); extrinsic=None
    if a.extrinsics:
        data=json.loads(Path(a.extrinsics).read_text()); extrinsic=data.get("T_lidar_camera_optimized")
    for r in rows:
        T=map_pose(r,extrinsic); r["T_map_object_centered_initial"]=T.tolist(); r["T_map_object_centered_refined"]=T.tolist(); r["teacher_status"]="initialized"; r["sources"]=["gigapose","map"]
    write_rows(a.output,rows)
if __name__ == "__main__": main()

def project_track(T, track):
    p=T[:3,3]; c=track["center_xyz"]; d=np.linalg.norm(c-p,axis=1); i=int(np.argmin(d)); delta=p-c[i]; lat=np.asarray(track["lateral_xyz"])[i]; normal=np.asarray(track["normal_xyz"])[i]
    return float(track["s"][i]), float(delta@lat), float(delta@normal), i

def smooth(rows, iterations=4, alpha=.45, epnp_anchor_weight=.35):
    by={}
    for r in rows: by.setdefault(str(r.get("track_id")),[]).append(r)
    for values in by.values():
        values.sort(key=lambda r:float(r.get("timestamp_ns",r.get("im_id",0))))
        poses=[as_pose(r["T_map_object_centered_initial"]) for r in values]
        for _ in range(iterations):
            nxt=[x.copy() for x in poses]
            for i in range(1,len(poses)-1):
                # Median position suppresses isolated GigaPose depth jumps.
                neigh=np.stack([poses[i-1][:3,3],poses[i][:3,3],poses[i+1][:3,3]])
                target=np.median(neigh,axis=0); nxt[i][:3,3]=(1-alpha)*poses[i][:3,3]+alpha*target
                R=poses[i][:3,:3]; d=so3_log(poses[i-1][:3,:3]@R.T)+so3_log(poses[i+1][:3,:3]@R.T); angle=np.linalg.norm(d)
                if angle>1e-8: nxt[i][:3,:3]=R # retain robust orientation in v1
            for i, row in enumerate(values):
                anchor_value = row.get("T_map_object_centered_epnp")
                if anchor_value is None or epnp_anchor_weight <= 0:
                    continue
                anchor = as_pose(anchor_value)
                weight = float(np.clip(epnp_anchor_weight, 0.0, 1.0))
                nxt[i][:3, 3] = (
                    (1.0 - weight) * nxt[i][:3, 3]
                    + weight * anchor[:3, 3]
                )
                delta = so3_log(anchor[:3, :3] @ nxt[i][:3, :3].T)
                nxt[i][:3, :3] = so3_exp(weight * delta) @ nxt[i][:3, :3]
            poses=nxt
        for r,T in zip(values,poses):
            anchor_sources = ["epnp_hybrid"] if r.get("T_map_object_centered_epnp") is not None else []
            r["T_map_object_centered_refined"]=T.tolist(); r["teacher_status"]="refined_observed"; r["sources"]=sorted(set(r.get("sources",[])+["temporal"]+anchor_sources))
            initial=as_pose(r["T_map_object_centered_initial"]); r["gigapose_translation_correction_m"]=(T[:3,3]-initial[:3,3]).tolist(); r["translation_std_m"]=(np.std([x[:3,3] for x in poses],axis=0)+.02).tolist(); r["rotation_std_deg"]=[2.,2.,4.]
    return rows
