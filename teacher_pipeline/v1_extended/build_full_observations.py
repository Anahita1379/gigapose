"""Build full tracked sequences; attach sparse EPnP anchors without subsampling."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from fine_tuning.optimize_camera_lidar_extrinsics import corrected_lidar_map_z_mm, load_metadata_camera, load_yaml, matrix_from_yaml_key
from fine_tuning.select_real_label_candidates import load_json_records
from teacher_pipeline.build_observations import _pose_m, _prediction_pose_m
from teacher_pipeline.geometry import CENTER_RAW_M, raw_to_centered_pose
from teacher_pipeline.io import read_rows, write_json, write_rows


def _build_track_id_lookup(rows):
    """Index tracking rows without making their refined poses teacher inputs."""
    by_instance = {}
    by_frame = {}
    for row in rows:
        if row.get("scene_id") in (None, "") or row.get("im_id") in (None, ""):
            continue
        key = (int(row["scene_id"]), int(row["im_id"]))
        by_frame.setdefault(key, []).append(row)
        if row.get("instance_id") not in (None, ""):
            by_instance[(key[0], key[1], str(row["instance_id"]))] = row
    return by_instance, by_frame


def _track_id_for_prediction(prediction, by_instance, by_frame):
    """Return only an association ID; never copy R/t from the tracking row."""
    if prediction.get("track_id") not in (None, ""):
        return prediction["track_id"], "prediction.track_id"
    key = (int(prediction["scene_id"]), int(prediction["im_id"]))
    instance_id = prediction.get("instance_id")
    if instance_id not in (None, ""):
        match = by_instance.get((key[0], key[1], str(instance_id)))
        if match is not None and match.get("track_id") not in (None, ""):
            return match["track_id"], "track_ids_from.instance"
    candidates = [
        row for row in by_frame.get(key, [])
        if row.get("track_id") not in (None, "")
    ]
    if len(candidates) == 1:
        return candidates[0]["track_id"], "track_ids_from.unique_frame"
    if instance_id not in (None, ""):
        return instance_id, "prediction.instance_id"
    return None, "missing"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",type=Path,required=True,help="One retained pose hypothesis per object/frame. Use the original GigaPose CSV with --track-ids-from for a raw-pose experiment.")
    p.add_argument("--track-ids-from",type=Path,help="Optional tracking CSV used only for track_id association. R, t, and score always remain from --predictions.")
    p.add_argument("--dataset-dir",type=Path,required=True); p.add_argument("--anchor-observations",type=Path,help="Optional sparse corrected EPnP observations from V1")
    p.add_argument("--epnp-root",type=Path,help="Complete EPnP-hybrid label directory. Labels are matched directly by frame timestamp; selected_samples.csv is not needed.")
    p.add_argument("--metadata-root",type=Path); p.add_argument("--prediction-translation-unit",choices=("auto","m","mm"),default="mm"); p.add_argument("--gigapose-object-origin",choices=("raw","centered"),default="raw")
    p.add_argument("--center-raw",nargs=3,type=float,default=CENTER_RAW_M.tolist()); p.add_argument("--output",type=Path,required=True); p.add_argument("--strict",action="store_true"); a=p.parse_args()
    predictions=read_rows(a.predictions); frames=json.loads((a.dataset_dir/"frame_map.json").read_text()); frame_by_key={(int(row["scene_id"]),int(row["im_id"])):row for row in frames}
    track_id_rows=read_rows(a.track_ids_from) if a.track_ids_from else []
    track_ids_by_instance,track_ids_by_frame=_build_track_id_lookup(track_id_rows)
    labels_by_timestamp={}
    if a.epnp_root:
        for path in sorted(a.epnp_root.glob("*.json")):
            timestamp=path.stem.rsplit("_",1)[0]
            for record_index,label in enumerate(load_json_records(path)):
                labels_by_timestamp.setdefault(timestamp,[]).append((path,record_index,label))
    anchors=read_rows(a.anchor_observations) if a.anchor_observations else []; anchor_by_key={(int(row["scene_id"]),int(row["im_id"])):row for row in anchors}
    z_anchors={}
    for row in anchors:
        if row.get("corrected_lidar_map_z_m") is not None: z_anchors.setdefault(int(row["scene_id"]),[]).append((int(row["im_id"]),float(row["corrected_lidar_map_z_m"])))
    for values in z_anchors.values(): values.sort()
    def corrected_z(scene_id,im_id):
        values=z_anchors.get(scene_id,[])
        if not values: return None
        x=np.asarray([value[0] for value in values]); y=np.asarray([value[1] for value in values]); return float(np.interp(im_id,x,y))
    output=[]; skipped=[]
    for index,prediction in enumerate(predictions):
        try:
            scene_id,im_id=int(prediction["scene_id"]),int(prediction["im_id"]); frame=frame_by_key[(scene_id,im_id)]
            track_id,track_id_source=_track_id_for_prediction(prediction,track_ids_by_instance,track_ids_by_frame)
            if a.track_ids_from and track_id is None:
                raise ValueError(f"No unambiguous track_id for scene_id={scene_id}, im_id={im_id}")
            metadata_path=Path(frame["sample_metadata_path"])
            if a.metadata_root: metadata_path=a.metadata_root/metadata_path.name
            centered,raw=_prediction_pose_m(prediction,a.prediction_translation_unit,a.gigapose_object_origin,np.asarray(a.center_raw))
            timestamp=Path(frame.get("frame_file",str(im_id))).stem.replace("image_","").split("_")[0]
            anchor=anchor_by_key.get((scene_id,im_id),{})
            label_choice=None
            candidates=labels_by_timestamp.get(timestamp,[])
            if candidates:
                def label_cost(candidate):
                    label=candidate[2]
                    if "T_camera_object_centered" not in label: return float("inf")
                    pose=_pose_m(label["T_camera_object_centered"],"m")
                    return float(np.linalg.norm(pose[:3,3]-centered[:3,3]))
                label_choice=min(candidates,key=label_cost)
                label_path,label_index,label=label_choice
                map_raw=_pose_m(label["T_map_object_raw"],"m")
                anchor={"T_camera_object_centered_epnp":_pose_m(label["T_camera_object_centered"],"m").tolist(),"T_map_object_raw_epnp":map_raw.tolist(),"T_map_object_centered_epnp":raw_to_centered_pose(map_raw,np.asarray(a.center_raw)).tolist(),"epnp_label_path":str(label_path),"epnp_record_index":label_index}
            metadata=load_yaml(metadata_path); map_lidar=matrix_from_yaml_key(metadata,"t_map_lidar","m"); lidar_camera=matrix_from_yaml_key(metadata,"t_lidar_camera_prior","m"); map_lidar[:3,3]*=.001; lidar_camera[:3,3]*=.001
            raw_map_lidar=map_lidar.copy()
            if label_choice is not None:
                corrected_mm=corrected_lidar_map_z_mm(label_choice[2]); z=None if corrected_mm is None else float(corrected_mm)*.001
            else: z=corrected_z(scene_id,im_id)
            if z is not None: map_lidar[2,3]=z
            instance={}; instances=frame.get("instances",[])
            if len(instances)==1: instance=instances[0]
            if anchor.get("T_map_lidar") is not None:
                map_lidar=np.asarray(anchor["T_map_lidar"],dtype=float).reshape(4,4)
            row={"scene_id":scene_id,"im_id":im_id,"session_id":frame.get("source_root"),"camera":frame.get("camera_id"),"timestamp_ns":timestamp,"track_id":track_id,"track_id_source":track_id_source,"image_path":frame.get("image_path"),"mask_path":frame.get("mask_path"),"bbox_xywh":instance.get("bbox"),"prediction_row":index,"gigapose_score":float(prediction.get("score",0) or 0),"T_camera_object_centered_gigapose":centered.tolist(),"T_camera_object_raw_gigapose":raw.tolist(),"T_map_lidar":map_lidar.tolist(),"T_map_lidar_raw_metadata":raw_map_lidar.tolist(),"corrected_lidar_map_z_m":z,"target_lidar_z_mode":"epnp_corrected" if anchor else ("interpolated_epnp_corrected" if z is not None else "raw"),"T_lidar_camera_initial":lidar_camera.tolist(),"object_origin_convention":"centered","center_raw_m":list(a.center_raw),"sources":["gigapose","map","tracking_id"]}
            camera=load_metadata_camera(metadata); row.update(K=camera[0].tolist() if camera else None,distortion=camera[1].tolist() if camera else [],distortion_model=camera[2] if camera else "unknown",sample_metadata_path=str(metadata_path))
            for field in ("T_camera_object_centered_epnp","T_map_object_raw_epnp","T_map_object_centered_epnp","epnp_label_path","epnp_record_index"):
                if anchor.get(field) is not None: row[field]=anchor[field]
            row["epnp_reference_available"]=bool(anchor)
            row["independent_support"]=False
            if anchor: row["sources"].append("epnp_hybrid_anchor")
            output.append(row)
        except Exception as exc:
            skipped.append({"prediction_row":index,"error_type":type(exc).__name__,"error":str(exc)})
            if a.strict: raise
    write_rows(a.output,output); write_json(a.output.with_suffix(".report.json"),{"format":"teacher_v1_extended_full_observations","prediction_rows":len(predictions),"observations_written":len(output),"track_ids_from":str(a.track_ids_from) if a.track_ids_from else None,"track_id_source_counts":{source:sum(row.get("track_id_source")==source for row in output) for source in sorted({row.get("track_id_source") for row in output})},"epnp_files_discovered":sum(len(values) for values in labels_by_timestamp.values()),"anchor_rows":sum(row.get("T_map_object_centered_epnp") is not None for row in output),"corrected_z_rows":sum(row.get("corrected_lidar_map_z_m") is not None for row in output),"skipped_count":len(skipped),"skipped":skipped[:100]})
    print(f"Wrote {len(output)} full-sequence observations; attached EPnP to {sum(row.get('T_map_object_centered_epnp') is not None for row in output)} rows -> {a.output}")

if __name__ == "__main__": main()
