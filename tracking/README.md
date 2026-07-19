# Self-correcting GigaPose tracking

This folder is an isolated tracking and pose-recovery package. It does not
modify GigaPose training, inference, dataset preparation, or any existing
fine-tuning code.

The goal is not merely to propagate yesterday's pose. The tracker repeatedly
asks whether the current pose still agrees with the current image, searches
alternative poses when it does not, and invokes fresh global GigaPose
hypotheses when the track is uncertain or lost.

## What is implemented

For every detected car, the runtime pipeline combines:

1. a short-window constant-velocity pose proposal;
2. sparse Lucas-Kanade optical-flow center/scale/in-plane-rotation update;
3. fresh top-K GigaPose hypotheses from a `MultiHypothesis.csv`;
4. an explicit 180-degree object-frame flip hypothesis;
5. optional EPnP/correspondence candidates supplied as a compatible CSV;
6. center, depth, translation, and rotation perturbations;
7. CAD rendering and derivative-free local pose refinement;
8. independent silhouette, boundary, bbox, depth, GigaPose-score, and weak
   motion evidence;
9. confidence-gated `normal`, `uncertain`, and `lost` states;
10. periodic global GigaPose safety checks, even for apparently healthy tracks;
11. optional learned full-SE(3) recovery, confidence, quality, and ranking head.

Multi-car identity is maintained with Hungarian bbox/center association. A
wrong motion prediction is never accepted only because it agrees with the
previous pose: motion has a deliberately small score weight, and confidence is
computed from current-image evidence without the motion term.

All internal poses are:

- `T_camera_object`, meaning OpenCV camera-from-object;
- right-handed proper rotations;
- translations in **metres**.

GigaPose/BOP input and output CSV translations use millimetres by default.

## Adaptive compute cascade

The default configuration implements:

- **Normal:** one flow or sliding-window proposal and one local refinement
  iteration.
- **Periodic safety frame:** add the best fresh GigaPose hypothesis every 15
  processed frames.
- **Uncertain:** use up to three fresh hypotheses, flip and moderate
  perturbations, two refinement iterations, and beam width two.
- **Lost/new:** use up to five fresh hypotheses, flipped and broad
  perturbations, three refinement iterations, and beam width three.

`tracking/configs/fast.json` removes normal-frame local refinement and narrows
the uncertain/lost beam. It is useful for latency testing but is less capable
of correcting subtle drift.

CAD evidence is rendered only inside a padded detection crop and at reduced
resolution (`render_scale`), then mapped back to the original image coordinate
system. Intrinsics are shifted and scaled with that crop, so projected poses,
bboxes, masks, depth, and saved overlays remain in original-image coordinates.

## Package layout

- `config.py`: all thresholds, weights, and adaptive-cascade settings.
- `types.py`: frames, detections, hypotheses, scores, and tracks.
- `geometry.py`: stable SE(3)/SO(3) operations and projection.
- `io.py`: `frame_map.json`, WebDataset, and GigaPose CSV adapters.
- `association.py`: multi-object association.
- `flow.py`: cheap optical-flow pose update.
- `hypotheses.py`: local/global/flip/recovery proposal generation.
- `rendering.py`: lazy EGL CAD renderer.
- `scoring.py`: image evidence and confidence.
- `refinement.py`: local coordinate-search CAD alignment.
- `tracker.py`: state machine and track lifecycle.
- `recovery.py`: optional learned recovery network and runtime adapter.
- `run_tracking.py`: complete tracking CLI.
- `generate_recovery_dataset.py`: synthesize recovery failures from BOP GT.
- `train_recovery.py`: train residual, confidence, quality, and ranking losses.
- `evaluate_tracking.py`: evaluate tracked CSV against WebDataset BOP GT.
- `visualization.py`: per-frame overlays.
- `tests/`: renderer-free focused tests.

## Prerequisites

Run inside the same Conda environment used by GigaPose. CAD scoring needs
NumPy, SciPy, OpenCV, Pillow, trimesh, pyrender, PyOpenGL/EGL, and the packages
already required by GigaPose. The learned recovery option additionally uses
PyTorch.

The dataset must contain:

- `<dataset>/frame_map.json` in chronological order;
- `<dataset>/<split>/key_to_shard.json`;
- shard camera JSON containing `cam_K`;
- an integer instance mask or per-instance bbox in each frame-map row;
- optionally metric depth in the shard;
- a CAD with the same origin, axes, and units used by GigaPose.

Do **not** use `--center-mesh` unless GigaPose inference used that same centered
CAD convention. Changing the CAD origin creates a systematic overlay shift.

## Step 1: produce fresh GigaPose hypotheses

Run your normal GigaPose test/inference pipeline with multiple hypotheses and
keep its `MultiHypothesis.csv`. The tracker accepts the standard columns:

```text
scene_id,im_id,obj_id,score,R,t,time,instance_id
```

Rows sharing `(scene_id, im_id, instance_id)` are treated as top-K hypotheses
for one detected car and sorted by score.

An EPnP or correspondence pipeline can be included without changing tracker
code by writing the same CSV schema and passing:

```bash
--auxiliary-predictions <epnp_candidates.csv> \
--auxiliary-translation-unit mm
```

Auxiliary rows with the same `instance_id` and `obj_id` are merged into that
car's candidate group. Repeat `--auxiliary-predictions` for multiple sources.

The current CLI is an **offline** implementation: the CSV contains global
measurements for all frames, but normal frames only consume a fresh hypothesis
at the safety interval. For a live system, call
`AdaptivePoseTracker.process_frame(frame, fresh_groups)` and supply an empty
list on normal frames or invoke GigaPose only when the state machine requests a
global measurement. The explicit live gate is:

```python
run_global = tracker.needs_global_measurement(len(frame.detections))
fresh_groups = run_gigapose(frame) if run_global else []
result = tracker.process_frame(frame, fresh_groups)
```

## Step 2: smoke-test tracking and inspect overlays

Example using the included Assetto inference paths:

```bash
python -m tracking.run_tracking \
  --predictions gigaPose_datasets/results/large_assettocorsa_ot2block_pose_aware_ist_tran_residual_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_large_assettocorsa_ot2block_pose_aware_ist_tran_residual_benchmark_new_datasetMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset \
  --split test \
  --config tracking/configs/default.json \
  --output-dir gigaPose_datasets/results/tracking_smoke \
  --max-frames 100 \
  --save-overlays \
  --no-depth \
  --overwrite
```

If your CSV translations are already metres, add:

```bash
--prediction-translation-unit m
```

If depth is missing or unreliable, add `--no-depth`. The depth term is then
removed from the evidence normalization; missing depth is not incorrectly
treated as zero error.

Inspect:

- `overlays/*.jpg`: red original top-1 GigaPose CAD/bbox, green tracked
  CAD/bbox, and yellow observed detection bbox;
- `tracked_predictions.csv`: final pose and tracking confidence;
- `candidate_diagnostics.csv`: state, source, all evidence terms, timing;
- `resolved_tracker_config.json`: exact settings used;
- `run_report.json`: counts and wall-clock time.

The overlay label is:

```text
T<track id> <state> <confidence> <winning hypothesis source>
```

The red GigaPose bbox is derived from the CAD silhouette rendered at the raw
top-1 GigaPose pose because the standard MultiHypothesis CSV does not contain a
bbox column. The green bbox is derived from the final tracked CAD silhouette.
This makes center, depth/scale, and rotation changes directly visible.

## Step 3: tune deterministic recovery before learning

Start with `default.json`. Important parameters are:

- `state.high_confidence` / `low_confidence`: state transitions;
- `state.safety_interval`: processed frames between global safety checks;
- `score.silhouette`, `edge`, `bbox`, `depth`: current-image evidence;
- `score.motion`: keep weak so a wrong pose can be rejected;
- `hypotheses.flip_axis`: object-frame axis for the 180-degree alternative;
- `refinement.refine_full_rotation`: also search camera X/Y rotation. It is
  slower; enable it if pitch/elevation errors are common.
- `render_scale`, `render_crop_padding_frac`, `render_crop_min_side_px`:
  rendering latency versus fine boundary detail.

The default flip axis is object Z. Confirm it visually for your CAD coordinate
frame. A CAD with a different vertical/longitudinal convention may require X
or Y.

## Step 4: optionally train the recovery head

This is separate from GigaPose/IST training. First generate training records
from a split containing BOP GT:

```bash
python -m tracking.generate_recovery_dataset \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_new_dataset \
  --split train_pbr_web_gsam_clean \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output gigaPose_datasets/results/tracking_recovery_data/train.npz \
  --max-frames 100 \
  --perturbations-per-instance 10
```

This deliberately creates:

- clean/almost-correct candidates;
- center and log-depth errors;
- moderate and large full-rotation errors;
- exact 180-degree flips;
- randomly occluded observed masks.

Each candidate is rendered and converted to current-image evidence. Targets
are the left-applied SO(3) residual and camera-frame translation residual:

\[
\Delta R = R^*R_c^\top,\qquad
\Delta\omega=\log(\Delta R),\qquad
\Delta t=t^*-t_c.
\]

The network also learns a binary good-pose confidence target and a scalar
quality target. The within-instance ranking loss forces a lower predicted
quality value for the better candidate. At runtime this quality orders the
recovery proposals and contributes to their measurement confidence; it is not
a logging-only output. Every resulting pose is still CAD-rendered and rescored.

Train it:

```bash
python -m tracking.train_recovery \
  --data gigaPose_datasets/results/tracking_recovery_data/train.npz \
  --output-dir gigaPose_datasets/results/tracking_recovery_head \
  --epochs 100 \
  --groups-per-batch 24 \
  --learning-rate 2e-4 \
  --patience 15 \
  --device cuda
```

`best.ckpt` is selected by the complete validation recovery objective;
`last.ckpt` is always the latest epoch. The train/validation split is by object
instance group, so perturbations of one instance cannot leak across splits.

Then enable the learned correction:

```bash
python -m tracking.run_tracking \
  ...same tracking arguments... \
  --recovery-checkpoint gigaPose_datasets/results/tracking_recovery_head/best.ckpt
```

The learned head runs only for uncertain/lost tracks. Its proposal is still
rendered, scored, and allowed to lose to a deterministic/global hypothesis.

## Step 5: full run

After the smoke overlays confirm CAD origin, axes, intrinsics, masks, and depth:

```bash
python -m tracking.run_tracking \
  --predictions <MultiHypothesis.csv> \
  --dataset-dir <prepared_dataset> \
  --split test \
  --mesh <obj_XXXXXX.ply> \
  --config tracking/configs/default.json \
  --output-dir <tracking_result_dir> \
  --save-overlays \
  --overlay-every 10 \
  --overwrite
```

For speed benchmarking, use:

```bash
--config tracking/configs/fast.json --no-depth
```

## Step 6: evaluate synthetic/labeled data

```bash
python -m tracking.evaluate_tracking \
  --predictions <tracking_result_dir>/tracked_predictions.csv \
  --dataset-dir <prepared_dataset_with_gt> \
  --split test \
  --min-confidence 0.0 \
  --output-dir <tracking_result_dir>/metrics
```

This writes per-instance translation error in metres, rotation error in
degrees, missed GT count, false-positive count, and mean/median/p90 summaries.
Run again with a higher `--min-confidence` to measure selective accuracy.

## Confidence semantics

The output `score` is a new tracking confidence in `[0,1]`; it is not the raw
GigaPose score. It combines current-frame silhouette, edge, bbox, optional
depth, and the GigaPose measurement score. The weak motion term affects
candidate ranking but is excluded from confidence. A small margin between the
best two candidates slightly lowers confidence.

## What can and cannot recover

The tracker can recover from a wrong initial pose when at least one global,
flipped, perturbed, or learned proposal enters the correct basin and current
image evidence distinguishes it. It is specifically designed not to blindly
continue a wrong pose.

Recovery is still not mathematically guaranteed. It can fail when all top-K
hypotheses are wrong, masks switch between visually identical cars, the CAD
coordinate convention is wrong, depth is invalid, or front/rear symmetry is
indistinguishable in the image. Periodic global proposals and multi-object
association reduce these risks but cannot eliminate them.

## Tests

The focused tests do not require EGL:

```bash
python -m unittest discover -s tracking/tests -v
python -m compileall -q tracking
```

They cover SO(3) near 180 degrees, RLE masks, millimetre conversion and top-K
grouping, optical flow, association, CAD evidence ordering, and identity
propagation.
