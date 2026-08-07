# GTSAM complete-sequence pose optimization

This package is an isolated GTSAM alternative to `sequence_factor_graph`.
It does not modify the SciPy graph, candidate exporter, cascades, transformers,
or their outputs.

It optimizes every contiguous `segment_id` as a complete batch, so offline
optimization uses both past and future frames. Ground truth is never passed to
the optimizer. If a benchmark bundle contains ground truth, it is used only for
the before/after fields in `run_report.json`.

## What GTSAM optimizes

Each frame has separate `Point3` translation and `Rot3` orientation variables.
The graph contains robust factors for:

- the selected RGB/mask/CAD candidate translation and rotation;
- an optional external translation-only/full-pose/transformer trajectory;
- translation acceleration over three frames;
- translation jerk over four frames;
- angular acceleration over three frames on SO(3).

Candidate selection is an alternating hard max-mixture approximation. RGB/CAD
cost, distance from the current graph pose, and bidirectional neighbor motion
choose one candidate per frame; GTSAM then continuously optimizes the complete
segment. This repeats for `--outer-iterations` rounds or until assignments stop
changing.

## Install

GTSAM 4.2 wheels require NumPy 1.x. NumPy 2 can cause a native crash rather
than a Python exception. Install into the environment that already runs this
repository:

```bash
conda activate gigapose
cd /home/anahita/gigapose

python3 -m pip install -r \
  tracking/rgb_self_recovery/sliding_window/sequence_factor_graph_gtsam/requirements.txt

python3 -c 'import gtsam, numpy; print("GTSAM OK; NumPy", numpy.__version__)'
python3 -m unittest \
  tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.test_gtsam_factor_graph -v
```

If the existing environment needs NumPy 2 for another dependency, clone it
before installing GTSAM:

```bash
conda create --name gigapose-gtsam --clone gigapose
conda activate gigapose-gtsam
python3 -m pip install -r \
  tracking/rgb_self_recovery/sliding_window/sequence_factor_graph_gtsam/requirements.txt
```

GTSAM optimization is CPU-based. RGB/CAD candidate export can use CUDA.

## Required inputs

The optimizer needs:

1. An RGB/CAD candidate bundle (`candidates.npz` plus `manifest.json`). Each
   frame must contain candidate camera-object poses, validity, source, visual
   `total_error`, timestamp, and contiguous `segment_id`.
2. For prior mode, one BOP-style tracking CSV with `scene_id,im_id,R,t` and
   exactly one pose per candidate-bundle frame.
3. Correct translation units for the prior (`mm` or `m`).
4. Optionally, one world-camera transform per frame. This is strongly preferred
   for a moving real camera because acceleration in camera coordinates includes
   ego motion.

RGB images, masks, intrinsics, CAD, GigaPose hypotheses, and the frozen RGB/CAD
checkpoint are needed only when creating a candidate bundle. They are not read
again during graph optimization.

## Assetto Corsa benchmark

The regenerated benchmark bundle already exists, so candidate export does not
need to be repeated:

```bash
cd /home/anahita/gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export CANDIDATES="$ROOT/gru_candidates"
export GRAPH="$ROOT/sequence_factor_graph_gtsam"
export TRANSLATION_PRIOR="$ROOT/cascade_translation_only/tracked_predictions.csv"
export FULL_PRIOR="$ROOT/cascade_full_pose/tracked_predictions.csv"
```

### Translation-only cascade prior

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$TRANSLATION_PRIOR" \
  --prior-translation-unit mm \
  --prior-components translation \
  --output-dir "$GRAPH/translation_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite
```

### Full-pose cascade prior

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$FULL_PRIOR" \
  --prior-translation-unit mm \
  --prior-components full \
  --output-dir "$GRAPH/full_pose_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite
```

The two commands use the same graph. `--prior-components translation` ignores
the `R` field that must still be present in a BOP CSV. `full` adds both
translation and rotation factors.
Translation and orientation prior strengths can be controlled independently:

```text
--prior-translation-sigma-m 5.0
--prior-rotation-sigma-deg 45.0
```

A smaller sigma trusts that component more. For a translation-only cascade,
the default weak 45-degree rotation prior allows RGB/CAD and temporal rotation
factors to revise orientation. To nearly ignore its orientation, use a larger
value such as `--prior-rotation-sigma-deg 90`.

### Evaluate both priors

```bash
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_benchmark_assetto_20260803_pose_regenerated-test_gigapose_rgb_recovery_benchmark_pose_regenerated_20260803MultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model cascade_translation_only="$ROOT/cascade_translation_only" \
  --model cascade_full_pose="$ROOT/cascade_full_pose" \
  --model gtsam_translation_prior="$GRAPH/translation_prior" \
  --model gtsam_full_pose_prior="$GRAPH/full_pose_prior" \
  --baseline cascade_translation_only \
  --prediction-translation-unit mm \
  --mesh "$DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$GRAPH/comparison" \
  --overwrite
```

| Distance | Translation-prior GTSAM | Full-pose-prior GTSAM | Better |
|---|---:|---:|---|
| 0–20 m | 0.842 m | **0.838 m** | Essentially tied |
| 20–40 m | **1.083 m** | 1.094 m | Translation prior |
| 40–60 m | **1.935 m** | 2.005 m | Translation prior |
| 60–80 m | **3.774 m** | 4.004 m | Translation prior |
| 80–100 m | **4.651 m** | 4.895 m | Translation prior |
| 100–120 m | **5.457 m** | 6.480 m | Translation prior |
| 120–140 m OOD | **5.971 m** | 8.278 m | Translation prior |

```bash
export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export GRAPH="$ROOT/sequence_factor_graph_gtsam"
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_benchmark_assetto_20260803_pose_regenerated-test_gigapose_rgb_recovery_benchmark_pose_regenerated_20260803MultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model cascade_translation_only="$ROOT/cascade_translation_only" \
  --model cascade_full_pose="$ROOT/cascade_full_pose" \
  --model gtsam_translation_prior="$GRAPH/translation_prior" \
  --model gtsam_full_pose_prior="$GRAPH/full_pose_prior" \
  --model gtsam_translation_cascade_full_components="$GRAPH/translation_cascade_full_components" \
  --baseline cascade_translation_only \
  --prediction-translation-unit mm \
  --mesh "$DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$GRAPH/comparison_full_components" \
  --overwrite
```

## Real-world data

### Prepare the runtime candidate bundle

The existing training/benchmark exporter requires ground truth. Use the new
runtime exporter below for a real sequence:

```bash
export REAL_DATASET="/absolute/path/to/bop_or_webdataset"
export REAL_GP="/absolute/path/to/gigapose_multihypothesis.csv"
export RGB_CAD_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"
export REAL_MESH="$REAL_DATASET/models/obj_000001.ply"
export REAL_RUN="$PWD/gigaPose_datasets/results/real_gtsam_sequence"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.export_runtime_candidates \
  --dataset-dir "$REAL_DATASET" \
  --split test \
  --predictions "$REAL_GP" \
  --prediction-translation-unit mm \
  --checkpoint "$RGB_CAD_CKPT" \
  --mesh "$REAL_MESH" \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$REAL_RUN/candidates" \
  --device cuda \
  --overwrite
```

The dataset must provide one target detection/mask, RGB image, camera
intrinsics, frame ordering, and preferably timestamps for every retained
frame. With one car, no multi-object identity association is needed. Missing
or ambiguous detections create skipped frames and therefore segment breaks.

### Run with either real-world prior

```bash
export REAL_TRANSLATION_PRIOR="/absolute/path/to/translation_only/tracked_predictions.csv"
export REAL_FULL_PRIOR="/absolute/path/to/full_pose/tracked_predictions.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$REAL_RUN/candidates" \
  --mode prior \
  --prior-predictions "$REAL_TRANSLATION_PRIOR" \
  --prior-translation-unit mm \
  --prior-components translation \
  --output-dir "$REAL_RUN/gtsam_translation_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$REAL_RUN/candidates" \
  --mode prior \
  --prior-predictions "$REAL_FULL_PRIOR" \
  --prior-translation-unit mm \
  --prior-components full \
  --output-dir "$REAL_RUN/gtsam_full_pose_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite
```

Without ground truth, `run_report.json` contains null `initial_metrics` and
`final_metrics`. Optimization and diagnostics still run normally.

### Moving-camera world-frame optimization

For real driving data, supply a JSONL with one transform per candidate frame.
Either representation below is accepted:

```json
{"scene_id": 1, "im_id": 10, "T_world_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
{"scene_id": 1, "im_id": 11, "T_map_lidar": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "T_lidar_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

Then append this argument to either command:

```bash
--world-camera-transforms "$REAL_RUN/world_camera_transforms.jsonl"
```

The graph converts every candidate and prior using
`T_world_object = T_world_camera @ T_camera_object`, optimizes in the world
frame, and exports the final CSV back in camera coordinates. The transform file
must cover every candidate-bundle frame. Use one fixed, independently validated
LiDAR-camera extrinsic; do not use an extrinsic fitted from the same candidate
errors without treating that as a separate experiment.

## Outputs and safety

Every run writes:

- `tracked_predictions.csv`: final camera-object poses in millimetres;
- `initial_predictions.csv`: exact input-prior poses;
- `diagnostics.jsonl`: candidate source and per-frame pose correction;
- `run_report.json`: configuration, segment status, coordinate frame, timing,
  and optional benchmark metrics.

An outer update is rejected if it is non-finite, worsens the factor-graph
error by more than five percent, changes any translation by more than the
configured safety limit, or exceeds the rotation safety limit. A rejected
first update falls back to the original prior; a rejected later update retains
the last accepted trajectory.

Start with the defaults. More outer iterations only repeat candidate
assignment and continuous optimization; they do not guarantee better ground
truth accuracy. Always compare `best` benchmark errors, distance bins, ADD,
and overlays rather than judging only the internal graph objective.
