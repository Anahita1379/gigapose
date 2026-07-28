"""Compare baseline V1 and physical V1-extended on common observations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from teacher_pipeline.evaluate import _blend_mask
from teacher_pipeline.geometry import as_pose, centered_to_raw_pose, pose_error, raw_to_centered_pose
from teacher_pipeline.io import write_json, write_rows
from teacher_pipeline.trajectory import load


def _track_token(value):
    if value in (None, ""): return ""
    try: return str(int(float(value)))
    except (TypeError, ValueError): return str(value)


def _key(row):
    return (int(row.get("scene_id", 0)), int(row.get("im_id", 0)), _track_token(row.get("track_id")))


def match_rows(v1_rows, extended_rows):
    exact={_key(row):row for row in extended_rows}; frame_groups={}
    for row in extended_rows: frame_groups.setdefault(_key(row)[:2],[]).append(row)
    pairs=[]; unmatched=[]; used=set()
    for row in v1_rows:
        match=exact.get(_key(row)); mode="scene_im_track"
        if match is None:
            candidates=frame_groups.get(_key(row)[:2],[])
            if len(candidates)==1: match=candidates[0]; mode="unique_scene_im"
        if match is None: unmatched.append(_key(row)); continue
        pairs.append((row,match,mode)); used.add(id(match))
    return pairs,unmatched,[row for row in extended_rows if id(row) not in used]


def _error(predicted,target):
    residual=pose_error(target,predicted)
    return float(np.linalg.norm(residual[:3])),float(np.degrees(np.linalg.norm(residual[3:])))


def _raw_map_pose(row):
    camera=raw_to_centered_pose(as_pose(row["T_camera_object_raw_gigapose"]))
    return as_pose(row["T_map_lidar"])@as_pose(row["T_lidar_camera_initial"])@camera


def compare_pairs(pairs):
    output=[]; skipped=[]
    for index,(v1,extended,mode) in enumerate(pairs):
        try:
            target_value=extended.get("T_map_object_centered_epnp") or v1.get("T_map_object_centered_epnp")
            if target_value is None: raise ValueError("no EPnP map reference")
            target=as_pose(target_value); raw=_raw_map_pose(extended); v1_pose=as_pose(v1["T_map_object_centered_refined"]); extended_pose=as_pose(extended["T_map_object_centered_refined"])
            raw_t,raw_r=_error(raw,target); v1_t,v1_r=_error(v1_pose,target); ext_t,ext_r=_error(extended_pose,target); delta_t,delta_r=_error(extended_pose,v1_pose)
            camera_target=extended.get("T_camera_object_centered_epnp") or v1.get("T_camera_object_centered_epnp")
            distance=float(np.linalg.norm(as_pose(camera_target)[:3,3])) if camera_target is not None else float(np.linalg.norm(target[:3,3]-as_pose(extended["T_map_lidar"])[:3,3]))
            output.append({"pair_index":index,"scene_id":extended.get("scene_id"),"im_id":extended.get("im_id"),"track_id":extended.get("track_id"),"match_mode":mode,"reference_distance_m":distance,"image_path":extended.get("image_path") or v1.get("image_path"),"raw_translation_error_m":raw_t,"v1_translation_error_m":v1_t,"extended_translation_error_m":ext_t,"raw_rotation_error_deg":raw_r,"v1_rotation_error_deg":v1_r,"extended_rotation_error_deg":ext_r,"extended_minus_v1_translation_improvement_m":v1_t-ext_t,"extended_minus_v1_rotation_improvement_deg":v1_r-ext_r,"v1_to_extended_translation_change_m":delta_t,"v1_to_extended_rotation_change_deg":delta_r})
        except Exception as exc: skipped.append({"pair_index":index,"error_type":type(exc).__name__,"error":str(exc)})
    return output,skipped


def distance_bins(metrics,edges):
    rows=[]
    for lower,upper in zip(edges[:-1],edges[1:]):
        selected=[row for row in metrics if lower<=row["reference_distance_m"]<upper]
        if not selected: continue
        result={"distance_bin":f"{lower:g}+" if np.isinf(upper) else f"{lower:g}-{upper:g}","count":len(selected)}
        for version in ("raw","v1","extended"):
            for metric in ("translation_error_m","rotation_error_deg"):
                values=np.asarray([row[f"{version}_{metric}"] for row in selected]); result[f"{version}_{metric}_median"]=float(np.median(values)); result[f"{version}_{metric}_p90"]=float(np.percentile(values,90))
        rows.append(result)
    return rows


def _time_seconds(rows):
    values=np.asarray([float(row.get("timestamp_ns",row.get("im_id",i))) for i,row in enumerate(rows)])
    differences=np.diff(values); positive=differences[differences>0]
    if len(positive) and np.median(positive)>1e6: return (values-values[0])*1e-9
    if len(positive) and np.median(positive)>1e3: return (values-values[0])*1e-3
    return values-values[0]


def temporal_summary(rows):
    grouped={}
    for row in rows:
        if row.get("T_map_object_centered_refined") is not None:
            key=(
                _track_token(row.get("track_id")),
                row.get("teacher_track_segment_index", 0),
            )
            grouped.setdefault(key,[]).append(row)
    speed=[]; acceleration=[]; jerk=[]; steps=[]
    state_speed=[]; state_longitudinal_acceleration=[]; state_longitudinal_jerk=[]
    for values in grouped.values():
        values.sort(key=lambda row:float(row.get("timestamp_ns",row.get("im_id",0))))
        if len(values)<2: continue
        t=_time_seconds(values); dt=np.diff(t); valid=dt>1e-6
        if not np.all(valid): continue
        steps.extend(dt)
        if len(values)>=3:
            p=np.stack([as_pose(row["T_map_object_centered_refined"])[:3,3] for row in values]); velocity=np.diff(p,axis=0)/dt[:,None]; mid_dt=.5*(dt[:-1]+dt[1:]); acc=np.diff(velocity,axis=0)/mid_dt[:,None]
            speed.extend(np.linalg.norm(velocity,axis=1)); acceleration.extend(np.linalg.norm(acc,axis=1))
            if len(acc)>1:
                jerk_dt=.5*(mid_dt[:-1]+mid_dt[1:]); jerk.extend(np.linalg.norm(np.diff(acc,axis=0)/jerk_dt[:,None],axis=1))
        has_physical_state=all(
            row.get("longitudinal_velocity_mps") is not None
            and row.get("lateral_velocity_mps") is not None
            and row.get("longitudinal_acceleration_mps2") is not None
            for row in values
        )
        if has_physical_state:
            velocity=np.asarray([
                [float(row["longitudinal_velocity_mps"]),float(row["lateral_velocity_mps"])]
                for row in values
            ])
            acc=np.asarray([float(row["longitudinal_acceleration_mps2"]) for row in values])
            state_speed.extend(np.linalg.norm(velocity,axis=1)); state_longitudinal_acceleration.extend(np.abs(acc))
            if len(acc)>1: state_longitudinal_jerk.extend(np.abs(np.diff(acc)/dt))
    def stats(values,prefix):
        values=np.asarray(values); return {f"{prefix}_median":float(np.median(values)) if len(values) else None,f"{prefix}_p90":float(np.percentile(values,90)) if len(values) else None}
    return {"track_count":len({_track_token(row.get("track_id")) for row in rows}),"segment_count":len(grouped),"primary_temporal_metric_source":"finite_differences_of_refined_map_pose_within_segments",**stats(steps,"frame_dt_s"),**stats(speed,"pose_speed_mps"),**stats(acceleration,"pose_acceleration_mps2"),**stats(jerk,"pose_jerk_mps3"),**stats(state_speed,"state_speed_mps"),**stats(state_longitudinal_acceleration,"state_longitudinal_acceleration_mps2"),**stats(state_longitudinal_jerk,"state_longitudinal_jerk_mps3")}


def _plots(metrics,bins,output):
    labels=[row["distance_bin"] for row in bins]; x=np.arange(len(labels)); width=.25
    for metric,ylabel,name in (("translation_error_m","Median translation error (m)","translation_comparison_by_distance.png"),("rotation_error_deg","Median rotation error (deg)","rotation_comparison_by_distance.png")):
        fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.3),4.8))
        for offset,version,color in ((-width,"raw","#d9534f"),(0,"v1","#f0ad4e"),(width,"extended","#35a853")):
            ax.bar(x+offset,[row[f"{version}_{metric}_median"] for row in bins],width,label=version,color=color)
        ax.set_xticks(x,labels); ax.set_xlabel("EPnP reference distance (m)"); ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output/name,dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2)); axes[0].hist([row["extended_minus_v1_translation_improvement_m"] for row in metrics],bins=25); axes[1].hist([row["extended_minus_v1_rotation_improvement_deg"] for row in metrics],bins=25)
    axes[0].set_xlabel("Extended improvement over V1 (m)"); axes[1].set_xlabel("Extended improvement over V1 (deg)")
    for axis in axes: axis.axvline(0,color="black",linewidth=1); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"extended_vs_v1_improvement.png",dpi=160); plt.close(fig)


def _extrinsic(path,row):
    if path is None: return as_pose(row["T_lidar_camera_initial"])
    return as_pose(json.loads(path.read_text())["T_lidar_camera_optimized"])


def overlays(pairs,metrics,output,mesh,mesh_scale,origin,v1_extrinsics,extended_extrinsics,max_count):
    from tracking.rendering import CPUSilhouetteRenderer
    output.mkdir(parents=True,exist_ok=True); renderer=CPUSilhouetteRenderer(mesh,mesh_scale=mesh_scale,center_mesh=False); written=[]; failures=[]
    ranked=sorted(metrics,key=lambda row:-abs(row["extended_minus_v1_translation_improvement_m"]))
    try:
        for metric in ranked[:max_count]:
            try:
                v1,extended,_=pairs[int(metric["pair_index"])]; image=np.asarray(Image.open(metric["image_path"]).convert("RGB")); K=np.asarray(extended.get("K") or v1.get("K"),dtype=float).reshape(3,3); map_lidar=as_pose(extended["T_map_lidar"])
                raw=raw_to_centered_pose(as_pose(extended["T_camera_object_raw_gigapose"])); v1_camera=np.linalg.inv(map_lidar@_extrinsic(v1_extrinsics,v1))@as_pose(v1["T_map_object_centered_refined"]); ext_camera=np.linalg.inv(map_lidar@_extrinsic(extended_extrinsics,extended))@as_pose(extended["T_map_object_centered_refined"]); target_value=extended.get("T_camera_object_centered_epnp") or v1.get("T_camera_object_centered_epnp"); target=as_pose(target_value) if target_value is not None else np.linalg.inv(map_lidar@_extrinsic(extended_extrinsics,extended))@as_pose(extended["T_map_object_centered_epnp"])
                poses=[raw,v1_camera,ext_camera,target]
                if origin=="raw": poses=[centered_to_raw_pose(pose) for pose in poses]
                canvas=image.copy()
                for pose,color in zip(poses,((235,65,55),(245,180,35),(40,210,90),(50,110,240))): _blend_mask(canvas,renderer.render(pose,K,image.shape[:2])[0],color,.32)
                name=f"{int(metric['scene_id'] or 0):06d}_{int(metric['im_id'] or 0):06d}_T{_track_token(metric['track_id'])}.jpg"; path=output/name; Image.fromarray(canvas).save(path,quality=92); written.append({**metric,"overlay_path":str(path)})
            except Exception as exc: failures.append({"pair_index":metric["pair_index"],"error_type":type(exc).__name__,"error":str(exc)})
    finally: renderer.close()
    write_rows(output/"overlay_index.csv",written); write_json(output/"overlay_report.json",{"written":len(written),"failures":failures,"legend":{"red":"raw GigaPose","yellow":"V1","green":"V1 extended","blue":"EPnP"}})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--v1-trajectories",type=Path,required=True); p.add_argument("--extended-trajectories",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--v1-extrinsics",type=Path); p.add_argument("--extended-extrinsics",type=Path); p.add_argument("--distance-bins",nargs="+",type=float,default=[0,20,40,60,80,100,float("inf")]); p.add_argument("--mesh",type=Path); p.add_argument("--mesh-scale",type=float,default=1.0); p.add_argument("--mesh-object-origin",choices=("raw","centered"),default="raw"); p.add_argument("--max-overlays",type=int,default=100); p.add_argument("--held-out",action="store_true"); a=p.parse_args()
    v1_rows=load(a.v1_trajectories); extended_rows=load(a.extended_trajectories); pairs,unmatched_v1,unmatched_extended=match_rows(v1_rows,extended_rows); metrics,skipped=compare_pairs(pairs)
    if not metrics: raise ValueError("No matched V1/extended rows have an EPnP reference")
    bins=distance_bins(metrics,a.distance_bins); a.output_dir.mkdir(parents=True,exist_ok=True); write_rows(a.output_dir/"per_frame_comparison.csv",metrics); write_rows(a.output_dir/"distance_bin_comparison.csv",bins); _plots(metrics,bins,a.output_dir)
    summary={"format":"teacher_v1_vs_v1_extended","held_out":bool(a.held_out),"interpretation":"held-out comparison" if a.held_out else "in-sample comparison; EPnP may have influenced both outputs","v1_rows":len(v1_rows),"extended_rows":len(extended_rows),"matched_rows":len(pairs),"evaluated_rows":len(metrics),"unmatched_v1_rows":len(unmatched_v1),"unmatched_extended_rows":len(unmatched_extended),"skipped":skipped[:100],"v1_temporal":temporal_summary(v1_rows),"extended_temporal":temporal_summary(extended_rows)}
    for version in ("raw","v1","extended"):
        for metric in ("translation_error_m","rotation_error_deg"):
            values=np.asarray([row[f"{version}_{metric}"] for row in metrics]); summary[f"{version}_{metric}_median"]=float(np.median(values)); summary[f"{version}_{metric}_p90"]=float(np.percentile(values,90))
    summary["extended_better_translation_fraction"]=float(np.mean([row["extended_minus_v1_translation_improvement_m"]>0 for row in metrics])); summary["extended_better_rotation_fraction"]=float(np.mean([row["extended_minus_v1_rotation_improvement_deg"]>0 for row in metrics]))
    if a.mesh: overlays(pairs,metrics,a.output_dir/"overlays",a.mesh,a.mesh_scale,a.mesh_object_origin,a.v1_extrinsics,a.extended_extrinsics,a.max_overlays)
    write_json(a.output_dir/"summary.json",summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
