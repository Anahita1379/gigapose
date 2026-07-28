"""Validate/normalize a centerline into the searchable track-map format.

The supplied Track_info assets are surface meshes. A centerline/raceline is
still required for s,d coordinates and must be exported separately as CSV or
NPZ (x,y,z[,left_width,right_width]).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--centerline", required=True); p.add_argument("--output", required=True); a=p.parse_args()
    src=Path(a.centerline)
    if src.suffix.lower()==".npz":
        d=np.load(src); xyz=np.asarray(d["xyz"],float); left=d["left_width"] if "left_width" in d else np.full(len(xyz),6.0); right=d["right_width"] if "right_width" in d else np.full(len(xyz),6.0)
    else:
        rows=list(csv.DictReader(src.open())); names=rows[0].keys(); xyz=np.array([[float(r.get("x",r.get("center_x"))),float(r.get("y",r.get("center_y"))),float(r.get("z",r.get("center_z")))] for r in rows]); left=np.array([float(r.get("left_width",6)) for r in rows]); right=np.array([float(r.get("right_width",6)) for r in rows])
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2: raise ValueError("centerline must contain at least two xyz samples")
    order=np.argsort(np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))])
    xyz=xyz[order]; left=np.asarray(left)[order]; right=np.asarray(right)[order]
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    tangent=np.gradient(xyz,axis=0); tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    normal=np.tile([0.,0.,1.],(len(xyz),1)); lateral=np.cross(normal,tangent); lateral/=np.maximum(np.linalg.norm(lateral,axis=1,keepdims=True),1e-9)
    np.savez(a.output, s=s, center_xyz=xyz, tangent_xyz=tangent, lateral_xyz=lateral, normal_xyz=normal, left_width=left, right_width=right)

if __name__ == "__main__": main()

