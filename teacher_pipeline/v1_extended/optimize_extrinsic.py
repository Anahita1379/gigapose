"""Optimize a fixed extrinsic against independently supported refined poses."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from teacher_pipeline.geometry import as_pose, pose_error, so3_exp
from teacher_pipeline.io import write_json
from teacher_pipeline.trajectory import load


def _update(prior, xi):
    delta=np.eye(4); delta[:3,:3]=so3_exp(xi[:3]); delta[:3,3]=xi[3:]; return delta@prior


def main():
    p=argparse.ArgumentParser(); p.add_argument("--samples",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--translation-sigma-m",type=float,default=.35); p.add_argument("--rotation-sigma-deg",type=float,default=5); p.add_argument("--translation-prior-sigma-m",type=float,default=.20); p.add_argument("--rotation-prior-sigma-deg",type=float,default=2); p.add_argument("--max-correction-translation-m",type=float,default=.5); p.add_argument("--max-correction-rotation-deg",type=float,default=5); p.add_argument("--max-iterations",type=int,default=200); a=p.parse_args()
    rows=load(a.samples)
    usable=[row for row in rows if row.get("calibration_independent_support") is True and row.get("T_map_object_centered_refined") is not None]
    if len(usable)<6: raise ValueError(f"Need at least 6 independently supported refined samples; found {len(usable)}")
    priors=np.stack([as_pose(row["T_lidar_camera_initial"]) for row in usable]); prior=priors[0]
    if np.max(np.linalg.norm(priors[:,:3,3]-prior[:3,3],axis=1))>.05: raise ValueError("Samples do not share one fixed initial camera-LiDAR calibration")
    rotation_sigma=np.radians(a.rotation_sigma_deg); rotation_prior=np.radians(a.rotation_prior_sigma_deg)
    def residual(xi):
        transform=_update(prior,xi); values=[]
        for row in usable:
            camera_pose=row.get("T_camera_object_centered_physical_input",row["T_camera_object_centered_gigapose"])
            predicted=as_pose(row["T_map_lidar"])@transform@as_pose(camera_pose)
            error=pose_error(as_pose(row["T_map_object_centered_refined"]),predicted)
            values.extend((error[:3]/a.translation_sigma_m).tolist()); values.extend((error[3:]/rotation_sigma).tolist())
        values.extend((xi[:3]/rotation_prior).tolist()); values.extend((xi[3:]/a.translation_prior_sigma_m).tolist()); return np.asarray(values)
    result=least_squares(residual,np.zeros(6),loss="soft_l1",max_nfev=a.max_iterations); optimized=_update(prior,result.x)
    translation_norm=float(np.linalg.norm(result.x[3:])); rotation_deg=float(np.degrees(np.linalg.norm(result.x[:3]))); within=translation_norm<=a.max_correction_translation_m and rotation_deg<=a.max_correction_rotation_deg
    def errors(transform):
        values=[]
        for row in usable:
            camera_pose=row.get("T_camera_object_centered_physical_input",row["T_camera_object_centered_gigapose"])
            error=pose_error(as_pose(row["T_map_object_centered_refined"]),as_pose(row["T_map_lidar"])@transform@as_pose(camera_pose))
            values.append([np.linalg.norm(error[:3]),np.degrees(np.linalg.norm(error[3:]))])
        return np.asarray(values)
    before,after=errors(prior),errors(optimized)
    report={"format":"teacher_v1_extended_fixed_extrinsic_v1","optimization_mode":"independent_refined_trajectory_targets","sample_count":len(usable),"optimizer_success":bool(result.success),"correction_within_safety_limits":bool(within),"calibration_valid_for_iteration":bool(result.success and within),"calibration_valid_for_reuse":False,"reuse_note":"Requires held-out and cross-session stability validation before reuse.","before_translation_error_m_median":float(np.median(before[:,0])),"after_translation_error_m_median":float(np.median(after[:,0])),"before_rotation_error_deg_median":float(np.median(before[:,1])),"after_rotation_error_deg_median":float(np.median(after[:,1])),"correction_translation_m":result.x[3:].tolist(),"correction_translation_norm_m":translation_norm,"correction_rotvec_rad":result.x[:3].tolist(),"correction_rotation_deg":rotation_deg,"T_lidar_camera_initial":prior.tolist(),"T_lidar_camera_candidate":optimized.tolist(),"T_camera_lidar_candidate":np.linalg.inv(optimized).tolist()}
    if within and result.success: report.update(T_lidar_camera_optimized=optimized.tolist(),T_camera_lidar_optimized=np.linalg.inv(optimized).tolist())
    write_json(a.output,report)
    if not within: raise SystemExit(f"Rejected extended extrinsic correction: {translation_norm:.3f} m, {rotation_deg:.3f} deg; diagnostic report written")
    print(f"Extended calibration: translation median {np.median(before[:,0]):.3f} -> {np.median(after[:,0]):.3f} m")

if __name__ == "__main__": main()
