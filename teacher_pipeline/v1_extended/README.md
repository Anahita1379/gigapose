# Teacher V1 extended: physical track-coordinate teacher

This package is a superset of `teacher_pipeline.v1`. It retains observation
building, fixed-extrinsic initialization, calibration, student export,
evaluation, plots, and CPU CAD overlays, and adds the complete physical first
teacher described in the supplied design.

## What is genuinely different

For every existing `track_id`, the extended refiner jointly optimizes the full
offline sequence with state

```text
[s, d, v_s, v_d, a_s, delta_yaw]
```

The objective contains quality-dependent GigaPose position/yaw terms,
longitudinal and lateral dynamics, acceleration and jerk penalties, track
boundaries, surface pose reconstruction, and gated LiDAR centroid anchors.
Because the complete tracklet is solved together, future observations constrain
earlier weak frames. The result is a newly reconstructed map pose sequence, not
an XYZ median filter.

The default physical input is untouched raw GigaPose converted only by the
known raw-to-centered CAD-origin transform. The older aligned-selection pose is
available only with `--pose-source teacher-input`.

## Stage 0: create a track map from the available PLY

The supplied Assetto Corsa PLY has no explicit centerline. The extractor
rasterizes its road surface, finds the closed medial-axis loop, recovers surface
height, estimates widths, and resamples it:

```bash
python3 -m teacher_pipeline.v1_extended.extract_track_map_from_surface \
  --surface-ply gigaPose_datasets/datasets/Track_info/sim_track_info/fn_lagunaseca2026_track_info/track_scene.ply \
  --axis-order xzy \
  --resolution-m 0.5 \
  --output <run>/track_map.npz
```

This asset was tested locally and produced an approximately 3638 m closed loop.
Always inspect `<run>/track_map.png`; automatic extraction can select a pit lane
or branch on a different mesh. `xzy` converts the PLY's Y-up AC coordinates to
a Z-up map. Use `--translation X Y Z` if the metadata map has an offset.

Before refinement, prove the frames coincide:

```bash
python3 -m teacher_pipeline.v1_extended.validate_track_alignment \
  --observations <run>/full_observations.jsonl \
  --track-map <run>/track_map.npz
```

Do not continue if this fails. A surface mesh alone cannot reveal an unknown
map-frame rigid transform.

## Stage 1: observations and optional LiDAR support

Build the full trajectory table directly from the tracking output and complete
EPnP-hybrid directory. `selected_samples.csv` is not required:

```bash
python3 -m teacher_pipeline.v1_extended.build_full_observations \
  --predictions <tracking-run>/tracked_predictions.csv \
  --dataset-dir <prepared-gigapose-dataset> \
  --epnp-root <recording>/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --metadata-root <recording>/front/metadata \
  --output <run>/full_observations.jsonl \
  --strict
```

The builder keeps every tracked frame, matches EPnP labels by frame timestamp,
and uses every available corrected-Z label. When only a sparse EPnP subset is
available, `--anchor-observations <v1>/observations.jsonl` remains a fallback
and corrected LiDAR-map Z is interpolated between those anchors.

If cluster measurements exist, attach a CSV/JSONL manifest containing
`scene_id, im_id, track_id, lidar_point_count, lidar_centroid` (LiDAR frame) or
`lidar_centroid_map`, plus optional `lidar_range_m`, `cluster_purity`, and
`lidar_association_confidence`:

```bash
python3 -m teacher_pipeline.v1_extended.attach_lidar \
  --observations <run>/full_observations.jsonl \
  --lidar-manifest <clusters>/lidar_manifest.csv \
  --output <run>/observations_with_lidar.jsonl
```

The files currently identified in `Track_info` contain no LiDAR clusters. The
physical refiner can run without them, but the result then lacks an independent
range anchor and must not be used to recalibrate the extrinsic.

## Stage 2: physical offline trajectory optimization

Initialize map poses with the current fixed extrinsic, then optimize complete
tracklets:

```bash
python3 -m teacher_pipeline.v1_extended.trajectory \
  --observations <run>/observations_with_lidar.jsonl \
  --output <run>/initial_physical_iteration_0.jsonl

python3 -m teacher_pipeline.v1_extended.refine_track_trajectories \
  --trajectories <run>/initial_physical_iteration_0.jsonl \
  --track-map <run>/track_map.npz \
  --pose-source raw \
  --output <run>/refined_physical_iteration_0.jsonl
```

Important knobs include the range-dependent GigaPose sigmas, motion/velocity,
acceleration/jerk, boundary, LiDAR reliability range, and LiDAR sigma. Defaults
are starting values, not learned sensor noise; tune them on held-out data.

## Stage 3: non-circular calibration (optional)

Only independently supported frames may become calibration targets:

```bash
python3 -m teacher_pipeline.v1_extended.select_calibration_samples \
  --trajectories <run>/refined_physical_iteration_0.jsonl \
  --output <run>/calibration_samples_iteration_0.jsonl

python3 -m teacher_pipeline.v1_extended.optimize_extrinsic \
  --samples <run>/calibration_samples_iteration_0.jsonl \
  --output <run>/extrinsic_physical_iteration_1.json
```

The selector deliberately fails when there is no reliable LiDAR/manual support.
Unlike baseline V1, the target is `T_map_object_centered_refined`, never the
per-frame EPnP pose. The tighter default correction limit is 0.5 m / 5 degrees,
and output explicitly remains unapproved for cross-session reuse until held-out
stability is demonstrated.

After a valid calibration, rerun initialization and physical refinement with
the frozen iteration-1 transform. Two or three alternating rounds are usually
enough; stop based on held-out metrics, not training residual.

## Stage 4: evaluation and export

The baseline evaluator and CPU overlays remain available as
`teacher_pipeline.v1_extended.evaluate`. Physical evaluation separately reports
raw GigaPose, aligned GigaPose, final pose, boundary violations, acceleration,
and jerk:

```bash
python3 -m teacher_pipeline.v1_extended.evaluate_physical \
  --trajectories <run>/refined_physical_iteration_1.jsonl \
  --track-map <run>/track_map.npz \
  --output-dir <run>/physical_evaluation

python3 -m teacher_pipeline.v1_extended.export_student_labels \
  --trajectories <run>/refined_physical_iteration_1.jsonl \
  --extrinsics <run>/extrinsic_physical_iteration_1.json \
  --output-dir <run>/student_labels
```

Exported labels include camera/map transforms, `s,d,v_s,v_d,a_s,delta_yaw`,
uncertainty/confidence fields, correction magnitude, source provenance, and the
extrinsic version.

## Scope boundary

This implements the complete *first physical teacher*: top-hypothesis pose,
existing IDs, fixed-extrinsic track-coordinate optimization, dynamics, dynamic
measurement weighting, boundaries/surface, LiDAR anchors, independent sample
selection, and staged calibration. The supplied roadmap explicitly assigns
image box/silhouette residuals and robust outlier models to V2, temporal
multi-hypothesis selection and point-to-CAD LiDAR to V3, and multi-camera/ID
recovery to V4. Those are therefore not mislabeled as V1 features here.
