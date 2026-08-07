# Absolute-orientation bootstrap and fixed-lag polarity correction

This package does not modify the original GRU selector, GRU--Kalman cascade,
SciPy graph, GTSAM graph, or their checkpoints. Its recommended entry point is
an **anchored replacement selector** that fixes absolute orientation before any
downstream filtering. A postprocessor remains available for diagnostics, but
it should not be the production path.

## Recommended: rerun from the anchored selector

The anchored selector reuses the trained GRU logits but jointly selects among
all 16 RGB/CAD candidates over a five-frame fixed-lag window. Its calibration-
safe default uses camera-facing polarity, GigaPose consensus, RGB/CAD, and
temporal costs. EPnP pose and map-heading costs default to zero and remain
available only for explicit ablations and diagnostics. It can also construct a
180-degree alternative while preserving the CAD center rather than incorrectly
rotating around an off-center raw mesh origin.

The default camera rules are `front:away,stereo_left:away,rear:toward`. They
compare the transformed CAD forward axis with the camera-to-object viewing ray,
so they do not require a camera/LiDAR extrinsic or EPnP pose.

```bash
export ANCHORED_SELECTOR="$RUN/selector_epnp_map_anchored"

python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.select_anchored \
  --data "$RUN/candidates" \
  --checkpoint "$SELECTOR_CKPT" \
  --dataset-dir "$DATASET" \
  --mesh "$MESH" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --epnp-rotation-weight 0 \
  --epnp-translation-weight 0 \
  --map-heading-weight 0 \
  --camera-facing-weight 50 \
  --camera-facing-rules front:away,stereo_left:away,rear:toward \
  --window-length 5 \
  --flip-axis z \
  --object-forward-axis x \
  --output-dir "$ANCHORED_SELECTOR" \
  --device cuda \
  --overwrite
```

Use `$ANCHORED_SELECTOR/tracked_predictions.csv` in place of the original
selector CSV when exporting GRU--Kalman measurements. Then rerun the
translation-only cascade and SciPy graph from scratch.

```bash
export ANCHORED_MEASUREMENTS="$RUN/selector_epnp_map_anchored_measurements"
export ANCHORED_CASCADE="$RUN/translation_cascade_epnp_map_anchored"
export ANCHORED_GRAPH="$RUN/scipy_graph_epnp_map_anchored"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$ANCHORED_SELECTOR/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$ANCHORED_MEASUREMENTS" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$ANCHORED_MEASUREMENTS" \
  --checkpoint "$TRANSLATION_KF_CKPT" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$ANCHORED_CASCADE" \
  --device cuda \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$RUN/candidates" \
  --mode prior \
  --prior-predictions "$ANCHORED_CASCADE/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --prior-rotation-sigma-deg 10 \
  --output-dir "$ANCHORED_GRAPH" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite
```

The 10-degree graph rotation prior is intentional: the translation-only
cascade preserves anchored orientation, while the full SE(3) graph still sees
all original candidates and otherwise could select a front/rear-flipped
rotation again.

Audit both selector and graph outputs without altering either pose file:

```bash
for item in \
  "selector=$ANCHORED_SELECTOR/tracked_predictions.csv" \
  "cascade=$ANCHORED_CASCADE/tracked_predictions.csv" \
  "graph=$ANCHORED_GRAPH/tracked_predictions.csv"
do
  name="${item%%=*}"
  predictions="${item#*=}"
  python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --mesh "$MESH" \
    --epnp-root "$EPNP_ROOT" \
    --output-dir "$RUN/anchor_evaluation_$name" \
    --overwrite
done
```

The output score is the trained GRU probability of the selected source
candidate. It is explicitly not an accuracy probability.

## Diagnostic postprocessor

The older entry point reads one of the existing tracking CSVs and writes a new
CSV whose translations are identical to the input. Only a front/rear half-turn
is applied. It is useful for diagnosing polarity, but it cannot undo a graph
that already optimized around incorrect candidate orientation and translation.

If used, run it **after the final translation optimizer**. Running it before a
full SE(3) factor graph is insufficient because that graph can select a flipped
RGB/CAD rotation again.

## Method

For each input orientation, the two polarity states are

\[
R_t^{(0)}=R_t,\qquad R_t^{(1)}=R_tR_z(\pi).
\]

The unary cost can combine:

- quality-approved `EPnPv2_gt_mesh_z_hybrid_labels` orientation;
- map heading from the raceline tangent embedded in those labels;
- an aligned `track_map.npz` when embedded tangents are unavailable;
- a robust medoid of the top GigaPose hypotheses;
- RGB/CAD candidate costs from the existing candidate bundle.

A five-frame fixed-lag binary optimizer adds rotation-motion and polarity-switch
penalties. It observes four future frames before committing each ordinary
decision, and it backfills the first buffered window when the offline output is
written.

EPnP is accepted by default only when `label_weight >= 0.5`, center error is at
most 25 pixels, and projected bounding-box IoU is at least 0.25. Rejected EPnP
pose anchors are not used. Their embedded raceline tangent may still provide a
map-heading constraint because it is independent of the EPnP image fit.

## Current real-world front run

From the repository root:

```bash
export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/20260518v1v4_front"
export DATASET="$PWD/gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4"
export CANDIDATES="$RUN/candidates"
export EPNP_ROOT="$PWD/gigaPose_datasets/datasets/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels"
export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_real_20260518v1v4_front_translation_residual_rerun/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_gigapose_real_20260518v1v4_front_translation_residual_rerunMultiHypothesis.csv"
export OPTIMIZED_POSES="$RUN/scipy_graph/tracked_predictions.csv"
export ORIENTATION_RUN="$RUN/scipy_graph_orientation_bootstrapped"

python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.run \
  --predictions "$OPTIMIZED_POSES" \
  --candidates "$CANDIDATES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --window-length 5 \
  --flip-axis z \
  --object-forward-axis x \
  --output-dir "$ORIENTATION_RUN" \
  --overwrite
```

The corrected poses are:

```text
$ORIENTATION_RUN/tracked_predictions.csv
```

Diagnostics are saved as:

```text
$ORIENTATION_RUN/orientation_diagnostics.csv
$ORIENTATION_RUN/run_report.json
```

`translation_preservation_max_error_m` must be zero. Inspect the before/after
EPnP rotation summaries and the number of frames assigned flipped polarity
before using the result for calibration.

## Diagnose the selector itself

To see the correction before translation optimization, change only the input and
output paths:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.run \
  --predictions "$RUN/selector_fixed_lag/tracked_predictions.csv" \
  --candidates "$CANDIDATES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --output-dir "$RUN/selector_orientation_bootstrapped" \
  --overwrite
```

This second command is diagnostic. For the production translation-cascade plus
SciPy result, apply the postprocessor after SciPy as shown in the first command.

## Track information

The track assets are under the case-sensitive path:

```text
gigaPose_datasets/datasets/Track_info/
```

For this Putnam recording, every hybrid EPnP JSON already contains the relevant
raceline tangent, and the sample metadata contains `t_map_lidar` and
`t_lidar_camera_prior`. The script therefore uses the map-heading term without a
separate map file.

For another dataset, pass an already extracted and visually validated map:

```bash
--track-map /path/to/aligned/track_map.npz
```

Do not pass `track_scene.ply` directly. It is a surface mesh, not a registered
centerline. Convert it using the teacher pipeline's track-map extraction tools,
validate its alignment, and then pass the resulting NPZ. The Putnam surface is:

```text
gigaPose_datasets/datasets/Track_info/sim_track_info/putnam_park-no_chicanes_track_info/track_scene.ply
```

## Overlay the corrected result

```bash
python3 -m fine_tuning.overlay_gigapose_predictions \
  --predictions "$ORIENTATION_RUN/tracked_predictions.csv" \
  --dataset-dir "$DATASET" \
  --split test \
  --mesh "$DATASET/models/obj_000001.ply" \
  --translation-scale 0.001 \
  --max-images 214 \
  --output-dir "$ORIENTATION_RUN/overlays"
```

The overlay renderer uses a pinhole projection. Small edge discrepancies can be
caused by the real equidistant camera model, but a 180-degree front/rear reversal
is still meaningful.
