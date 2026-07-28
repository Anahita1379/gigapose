"""Run fixed-extrinsic temporal refinement; optional centerline projection."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .trajectory import load, smooth, project_track
from .geometry import as_pose
from .io import write_rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--trajectories",required=True); p.add_argument("--output",required=True); p.add_argument("--track-map",help="prepared .npz centerline map"); p.add_argument("--iterations",type=int,default=4); p.add_argument("--epnp-anchor-weight",type=float,default=.35,help="Weight in [0,1] of independently supported EPnP map poses during smoothing."); a=p.parse_args();
    if not 0 <= a.epnp_anchor_weight <= 1: raise ValueError("--epnp-anchor-weight must be in [0,1]")
    rows=smooth(load(a.trajectories),a.iterations,epnp_anchor_weight=a.epnp_anchor_weight)
    track=None
    if a.track_map:
        with np.load(a.track_map) as d: track={k:d[k] for k in d.files}
        for r in rows:
            s,d,h,_=project_track(as_pose(r["T_map_object_centered_refined"]),track); r.update(track_s_m=s,track_d_m=d,height_m=h); r["sources"]=sorted(set(r["sources"]+["track_geometry"]))
            r["teacher_confidence"]=float(np.clip(1/(1+np.linalg.norm([d])/3),.2,.98))
    write_rows(a.output,rows)
if __name__ == "__main__": main()
