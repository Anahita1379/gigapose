# GigaPose teacher pipeline

This folder is an isolated, staged teacher pipeline. It reuses the project’s
existing EPnP-hybrid and per-sample YAML contracts; no new metadata CSV is
required.

## Inputs used by the preferred path

The observation builder joins four existing sources:

```text
GigaPose MultiHypothesis.csv (or tracked_predictions.csv)
    prediction pose, score, instance ID, optional persistent track_id

selected_samples.csv
    prediction_row_index -> epnp_label_path + epnp_record_index

EPnPv2_gt_mesh_z_hybrid_labels/*.json
    T_map_object_raw, T_camera_object_centered, metadata_path

metadata/sample_*.yaml
    t_map_lidar, t_lidar_camera_prior, camera_intrinsics
```

`selected_samples.csv` and `--predictions` must refer to the same prediction
CSV: `prediction_row_index` is a physical row index into that file. If label
selection used the original GigaPose CSV, pass that same CSV here. To add
persistent identities from a separate tracking run, also pass
`--tracks tracked_predictions.csv`; rows are joined by
`scene_id, im_id, instance_id`.

The canonical teacher convention is `T_camera_object_centered`, in metres.
GigaPose CSV poses are treated as raw-CAD poses by default, matching
`label_selection/common.py`, and are converted using the known CAD-center
offset. Use `--gigapose-object-origin centered` only if that particular CSV was
already generated in the centered object frame.

## Run one stage at a time

Run from the repository root. First build observations:

```bash
python3 -m teacher_pipeline.build_observations \
  --predictions <the-same-MultiHypothesis.csv-used-for-selection> \
  --selected-samples <selection-output>/selected_samples.csv \
  --dataset-dir <prepared-gigapose-dataset> \
  --tracks <optional-tracking-output>/tracked_predictions.csv \
  --prediction-translation-unit mm \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --output <run>/observations.jsonl \
  --strict
```

The optional `--dataset-dir` supplies `frame_map.json`, image paths, masks, and
boxes. The EPnP JSON and YAML paths normally come directly from
`selected_samples.csv`. If the dataset was moved, override their directories:

```bash
--epnp-root <new>/EPnPv2_gt_mesh_z_hybrid_labels \
--metadata-root <new>/metadata
```

Inspect `<run>/observations.report.json`. In particular,
`instance_id_track_fallbacks` should ideally be zero. A nonzero value means no
persistent track ID was available for those rows, so temporal grouping may be
incorrect.

Initialize map-frame trajectories while holding the metadata extrinsic fixed:

```bash
python3 -m teacher_pipeline.trajectory \
  --observations <run>/observations.jsonl \
  --output <run>/initial_trajectories_iteration_0.jsonl
```

Refine with temporal smoothing and independently supported EPnP anchors:

```bash
python3 -m teacher_pipeline.refine_trajectories \
  --trajectories <run>/initial_trajectories_iteration_0.jsonl \
  --epnp-anchor-weight 0.35 \
  --output <run>/refined_iteration_0.jsonl
```

Estimate one fixed `T_lidar_camera` from the EPnP-hybrid map anchors:

```bash
python3 -m teacher_pipeline.optimize_extrinsic \
  --trajectories <run>/refined_iteration_0.jsonl \
  --output <run>/extrinsic_iteration_1.json
```

Reinitialize and refine with the new fixed transform:

```bash
python3 -m teacher_pipeline.trajectory \
  --observations <run>/observations.jsonl \
  --extrinsics <run>/extrinsic_iteration_1.json \
  --output <run>/initial_trajectories_iteration_1.jsonl

python3 -m teacher_pipeline.refine_trajectories \
  --trajectories <run>/initial_trajectories_iteration_1.jsonl \
  --epnp-anchor-weight 0.35 \
  --output <run>/refined_iteration_1.jsonl
```

Finally export visible student labels:

```bash
python3 -m teacher_pipeline.export_student_labels \
  --trajectories <run>/refined_iteration_1.jsonl \
  --extrinsics <run>/extrinsic_iteration_1.json \
  --extrinsics-version iteration_1 \
  --output-dir <run>/student_labels
```

## Before/after evaluation and plots

Evaluate original GigaPose versus the final V1 trajectory against the EPnP
map-pose reference:

```bash
python3 -m teacher_pipeline.evaluate \
  --trajectories <run>/refined_iteration_1.jsonl \
  --extrinsics <run>/extrinsic_iteration_1.json \
  --output-dir <run>/evaluation
```

This writes:

```text
summary.json
per_frame_metrics.csv
distance_bin_metrics.csv
translation_error_by_distance.png
rotation_error_by_distance.png
improvement_histograms.png
```

The default distance bins are `0, 20, 40, 60, 80, 100, infinity` metres.
Override them, including a final upper edge, with for example:

```bash
--distance-bins 0 10 20 40 60 80 120 inf
```

Positive `translation_improvement_m` and `rotation_improvement_deg` mean V1 is
closer to the EPnP target than the original GigaPose pose. The CSV also records
the magnitude of the V1 correction relative to original GigaPose.

Add three-way CAD overlays when the original raw-origin CAD is available:

```bash
python3 -m teacher_pipeline.evaluate \
  --trajectories <run>/refined_iteration_1.jsonl \
  --extrinsics <run>/extrinsic_iteration_1.json \
  --mesh gigaPose_datasets/datasets/<dataset>/models/obj_000001.ply \
  --mesh-object-origin raw \
  --max-overlays 100 \
  --output-dir <run>/evaluation
```

Overlay colors are red for original GigaPose, green for V1 final, and blue for
the EPnP target. The evaluator converts centered poses back to the known raw
CAD origin before rendering a raw-origin mesh. If the mesh vertices themselves
were already recentered, use `--mesh-object-origin centered`.

An evaluation on the same EPnP samples used for anchoring/calibration is an
in-sample measurement. It shows whether the optimization fit those samples,
not whether it generalizes. The evaluator records this warning in
`summary.json`.

For a genuine held-out evaluation:

1. Calibrate on one session or time segment.
2. Build a separate observation file from EPnP samples that were not used for
   calibration.
3. Initialize those observations with the frozen calibration.
4. Refine them with `--epnp-anchor-weight 0`, so EPnP is only an evaluation
   reference.
5. Run the evaluator with `--held-out`.

```bash
python3 -m teacher_pipeline.trajectory \
  --observations <heldout>/observations.jsonl \
  --extrinsics <calibration>/extrinsic_iteration_1.json \
  --output <heldout>/initial.jsonl

python3 -m teacher_pipeline.refine_trajectories \
  --trajectories <heldout>/initial.jsonl \
  --epnp-anchor-weight 0 \
  --output <heldout>/refined.jsonl

python3 -m teacher_pipeline.evaluate \
  --trajectories <heldout>/refined.jsonl \
  --extrinsics <calibration>/extrinsic_iteration_1.json \
  --held-out \
  --output-dir <heldout>/evaluation
```

## Track map assets

The available assets under
`gigaPose_datasets/datasets/Track_info/sim_track_info/` contain
`track_scene.ply` surface meshes and scene/material metadata. They do not
contain an explicit centerline, raceline, longitudinal `s`, or left/right
width table. Therefore the commands above run without `--track-map` and still
use map-frame ego/object poses, but do not claim centerline coordinates.

If a centerline is later exported as
`x,y,z,left_width,right_width`, prepare and use it with:

```bash
python3 -m teacher_pipeline.prepare_track_map \
  --centerline <centerline.csv> \
  --output <run>/track_map.npz

python3 -m teacher_pipeline.refine_trajectories \
  --trajectories <run>/initial_trajectories_iteration_1.jsonl \
  --track-map <run>/track_map.npz \
  --output <run>/refined_iteration_1.jsonl
```

Direct `track_scene.ply` surface projection and automatic centerline extraction
are not part of this first version.

## Legacy generic input

For synthetic/custom data, the original generic mode remains available:

```bash
python3 -m teacher_pipeline.build_observations \
  --predictions <predictions.csv> \
  --metadata <custom-metadata.csv> \
  --output <run>/observations.jsonl
```
