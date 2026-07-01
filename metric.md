# GigaPose / Assetto Corsa benchmark metrics

This note describes how to evaluate and compare GigaPose models on the prepared
Assetto Corsa benchmark/inference datasets.

The short version:

1. `test.py` runs inference over every sample in the prepared dataset split.
2. GigaPose writes per-batch `.npz` files and final BOP-style prediction CSVs.
3. `fine_tuning.split_predictions_by_camera` can split those outputs into
   `front/`, `rear/`, etc.
4. `fine_tuning.overlay_gigapose_predictions` makes visual CAD overlays.
5. `fine_tuning.compare_gigapose_predictions` compares two model prediction CSVs
   against the Assetto-derived GT stored in the WebDataset shards.
6. `fine_tuning.plot_prediction_comparison` plots histograms, recall curves, and
   paired model-improvement plots.
7. `fine_tuning.visualize_prediction_gt_comparison` draws GT, baseline, and
   fine-tuned boxes together for qualitative comparison.

## Quick metric workflow

| Goal | Command/module | Main output |
|---|---|---|
| Run one model on the benchmark dataset | `python test.py ...` | Prediction `.npz`, CSV, and `MultiHypothesis.csv` files |
| Split predictions by camera | `python -m fine_tuning.split_predictions_by_camera` | `predictions/by_camera/<camera>/` folders |
| Make CAD overlays for visual checking | `python -m fine_tuning.overlay_gigapose_predictions` | Overlay images + `prediction_overlay_report.json` |
| Compare two models numerically | `python -m fine_tuning.compare_gigapose_predictions` | Per-instance CSV, summary JSON/CSV, plots-ready outputs |
| Compare many models numerically | `python -m fine_tuning.evaluate_gigapose_models` | Aggregate, camera, pairwise, and best-model tables |
| Recommend the best model for pose/depth use | `python -m fine_tuning.recommend_model_by_task` | Overall and per-camera model rankings using weighted task scores |
| Plot multi-model summary tables | `python -m fine_tuning.plot_model_summary` | Overall and camera-by-camera bar charts from `overall_summary.csv` and `camera_summary.csv` |
| Plot numeric metric outputs | `python -m fine_tuning.plot_prediction_comparison` | Histograms, recall curves, model-improvement plots |
| Visualize GT vs two models | `python -m fine_tuning.visualize_prediction_gt_comparison` | Images with GT/baseline/fine-tuned boxes together |
| Visualize many models per GT car | `python -m fine_tuning.visualize_multi_model_per_car` | Side-by-side panels, one panel per car instance |

## Metric output files

When evaluating multiple models, the most useful files are:

| File | What it tells you |
|---|---|
| `overall_summary.csv` / `overall_summary.json` | Overall model ranking across the benchmark split |
| `camera_summary.csv` / `camera_summary.json` | Same metrics grouped by camera, e.g. front vs rear |
| `all_instance_metrics.csv` | One row per matched GT/model prediction instance |
| `pairwise_instance_comparison.csv` | Per-instance model-vs-model deltas |
| `best_model_per_instance.csv` | Which model wins each instance under the chosen metric |
| `plots/` | Histogram/recall/comparison figures from `plot_prediction_comparison` |
| `side_by_side_visuals/` | Qualitative multi-model images from `visualize_multi_model_per_car` |
| `prediction_overlay_report.json` | Overlay render/debug stats such as visible pose counts and rendered pixels |

## Metric groups at a glance

| Metric family | Examples | Lower or higher is better | What it checks |
|---|---|---|---|
| GigaPose confidence | `score`, score mean/median | Higher | Model confidence, not direct geometric correctness |
| Translation/depth | translation error, depth error, RMSE | Lower | Whether the estimated car location is close to Assetto-derived GT |
| Rotation | angular error in degrees | Lower | Whether the car orientation is correct |
| Image-plane alignment | center error, bbox IoU | Center error lower; IoU higher | Whether the projected prediction lands on the correct car in the image |
| Rendered silhouette overlap | rendered mask IoU | Higher | Whether the CAD silhouette projected from the predicted pose overlaps the GT/rendered mask |
| ADD-style geometry | ADD / ADD-S-style distance if available | Lower | 3D model-point alignment under the estimated pose |
| Camera-specific summary | per-camera mean/median metrics | Depends on metric | Finds problems isolated to `front`, `rear`, stereo, etc. |

## Recommended comparison table for reports

For a clean report, make one table per benchmark run with columns like:

| Model | Evaluated instances | Mean score | Median translation error | Median rotation error | Mean bbox IoU | Mean mask IoU | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| original |  |  |  |  |  |  | baseline pretrained GigaPose |
| finetune |  |  |  |  |  |  | IST-only, AE+IST, or other training notes |
| finetune2 |  |  |  |  |  |  | second checkpoint/config |

Then include a camera-by-camera version using `camera_summary.csv`, because the
rear camera can behave very differently from the front/stereo cameras when
objects are far away or tiny.

To turn those summaries into a direct model choice for pose + altitude/depth
estimation, run:

```bash
python -m fine_tuning.recommend_model_by_task \
  --metrics-dir gigaPose_datasets/results/final_results/metrics/all_models \
  --plot
```

This writes:

```text
model_recommendation/model_recommendations_overall.csv
model_recommendation/model_recommendations_by_camera.csv
model_recommendation/model_recommendation.json
model_recommendation/model_recommendation_report.md
model_recommendation/plots/
```

The default scoring uses median translation error, median depth error, median
rotation error, projected-center error, ADD, bbox IoU, and mask IoU. Here,
`depth_error_mm_median` is used as the altitude/depth proxy; if you later add a
true altitude metric, pass custom weights with `--weight`.

## What inference already does

When you run:

```bash
python test.py \
  test_dataset_name=assettocorsa_benchmark \
  model.checkpoint_path='PATH/TO/CHECKPOINT.ckpt' \
  run_id=assettocorsa_IST_only_benchmark \
  name_exp=large_assettocorsa
```

GigaPose loads:

```text
gigaPose_datasets/datasets/assettocorsa_benchmark/test/
```

and iterates over the WebDataset `test` split. In practice, that means it uses
all images/samples that were written by:

```bash
python -m Assetto_data_prep.prepare_inference ...
```

If you prepared with:

```bash
--cameras all
```

then the single benchmark dataset contains all selected camera types. GigaPose
does not automatically save one prediction directory per camera; it saves one
combined result set.

Typical outputs are:

```text
gigaPose_datasets/results/<experiment>/predictions/0.npz
gigaPose_datasets/results/<experiment>/predictions/1.npz
...
gigaPose_datasets/results/<experiment>/predictions/<name>.csv
gigaPose_datasets/results/<experiment>/predictions/<name>MultiHypothesis.csv
```

The `.npz` files are intermediate per-batch predictions. The `.csv` files are
the final BOP-style prediction tables.

## Split inference outputs by camera

The prepared Assetto dataset contains:

```text
gigaPose_datasets/datasets/assettocorsa_benchmark/frame_map.json
```

That file maps each `(scene_id, im_id)` back to `camera_id`, e.g. `front`,
`rear`, etc.

Use:

```bash
python -m fine_tuning.split_predictions_by_camera \
  --predictions gigaPose_datasets/results/<experiment>/predictions \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark
```

This splits both:

- final `.csv` files
- intermediate `.npz` files

into:

```text
gigaPose_datasets/results/<experiment>/predictions/by_camera/front/
gigaPose_datasets/results/<experiment>/predictions/by_camera/rear/
...
```

This is useful for camera-by-camera visual inspection and camera-specific
metrics.

## Visual overlays

To visually inspect whether a model’s pose estimates look reasonable, render
the CAD model back into the image plane using the predicted pose:

```bash
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/<experiment>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/overlays/<experiment> \
  --min-score 0.05
```

The script writes side-by-side images:

```text
left: original RGB
right: RGB + rendered CAD prediction overlay
```

It also writes:

```text
prediction_overlay_report.json
```

with per-image prediction counts, scores, rendered pixel counts, and visible
pose counts.

If you want overlays per camera, first split the predictions, then run the
overlay script once per camera CSV:

```bash
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/<experiment>/predictions/by_camera/front/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/overlays/<experiment>/front \
  --min-score 0.05
```

Repeat for `rear`, `left`, `right`, etc.

Notes:

- Very low-score predictions can look reversed or nonsensical. For qualitative
  reports, use a threshold such as `--min-score 0.05` or `--min-score 0.1`.
- The overlay script handles the common unit mismatch by keeping the mesh in
  meters and converting BOP/GigaPose millimeter translations to meters during
  rendering.

## Ground truth source

The comparison scripts use the Assetto-derived reference poses written into the
prepared WebDataset:

```text
test/*.gt.json
test/*.gt_info.json
test/*.camera.json
```

These are not hand-labeled external measurements. They are derived from the
Assetto camera parameters and object transforms used during data preparation.
For controlled simulator evaluation, this is the right practical “GT”.

## Numeric comparison between two models

Use `fine_tuning.compare_gigapose_predictions` to compare, for example, the
original model vs a fine-tuned model:

the prediction files are: 
```bash
--baseline-predictions 
--finetuned-predictions
--finetuned-predictions
--finetuned-predictions

```

```bash
python -m fine_tuning.compare_gigapose_predictions \
  --baseline-predictions gigaPose_datasets/results/<original_run>/predictions/<original>MultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/<finetuned_run>/predictions/<finetuned>MultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/metrics/original_vs_finetuned
```

Outputs:

```text
fine_tuning/metrics/original_vs_finetuned/per_instance_metrics.csv
fine_tuning/metrics/original_vs_finetuned/summary_metrics.csv
fine_tuning/metrics/original_vs_finetuned/summary_metrics.json
fine_tuning/metrics/original_vs_finetuned/paired_instance_comparison.csv
fine_tuning/metrics/original_vs_finetuned/paired_summary.json
```

The important files are:

- `summary_metrics.csv`: one row per model with aggregate metrics.
- `paired_instance_comparison.csv`: matched original-vs-fine-tuned rows for the
  same GT instance.
- `paired_summary.json`: concise “which model is better?” summary.

The script handles MultiHypothesis CSVs by keeping the best hypothesis per
detection, then matching predictions to GT per image. This avoids directly
trusting global `instance_id` as GT order, which can be wrong for multi-car
images.

## Plot comparison results

After running the numeric comparison:

```bash
python -m fine_tuning.plot_prediction_comparison \
  --comparison-dir fine_tuning/metrics/original_vs_finetuned
```

This writes plots to:

```text
fine_tuning/metrics/original_vs_finetuned/plots/
```

Useful plots include:

- translation error histogram
- rotation error histogram
- projected-center error histogram
- ADD histogram
- recall curves
- score-vs-error plots
- paired improvement histograms
- paired scatter plots

For paired scatter plots, points below the diagonal mean the fine-tuned model has
lower error than the baseline.

## Visual GT vs original vs fine-tuned comparison

To generate images with three boxes:

- green: Assetto-derived GT
- red: baseline/original model
- blue: fine-tuned model

run:

```bash
python -m fine_tuning.visualize_prediction_gt_comparison \
  --baseline-predictions gigaPose_datasets/results/<original_run>/predictions/<original>MultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/<finetuned_run>/predictions/<finetuned>MultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/metrics/original_vs_finetuned/visual_overlays \
  --max-images 100 \
  --sort-by image
```

To inspect where fine-tuning helped most:

```bash
--sort-by translation_improvement
```

or:

```bash
--sort-by center_improvement
```

## Metrics currently computed

### Prediction score

GigaPose’s confidence-like score from the prediction CSV.

Higher is better, but it is not the same as pose accuracy. It is useful for
thresholding and for checking whether low-score predictions are the ones that
look visually wrong.

### Translation error

```text
|| t_pred - t_gt ||_2
```

Units: millimeters.

Lower is better.

This measures 3D position error of the object origin/pose translation.

### Depth error

```text
abs(z_pred - z_gt)
```

Units: millimeters.

Lower is better.

This is useful because pose methods can sometimes get image-plane alignment
roughly right while depth is still very wrong.

### Rotation error

Computed from:

```text
R_delta = R_pred * R_gt^T
angle = arccos((trace(R_delta) - 1) / 2)
```

Units: degrees.

Lower is better.

This is the standard geodesic SO(3) angle between predicted and reference
rotation.

### Projected-center error

Project both predicted and GT object origins into the image:

```text
u_pred = K * t_pred
u_gt   = K * t_gt
error = || u_pred - u_gt ||_2
```

Units: pixels.

Lower is better.

This is often very interpretable for driving/camera data because it asks:
“Did the predicted object center land in the right place in the image?”

### GT bbox center error

Distance between the projected predicted origin and the center of the GT visible
bounding box.

Units: pixels.

Lower is better.

This is less pure than projected-center error, but helpful when judging whether
the prediction is on the correct visible car in multi-car images.

### ADD

Average Distance of Model Points:

```text
ADD = mean_i || (R_pred * x_i + t_pred) - (R_gt * x_i + t_gt) ||
```

Units: millimeters.

Lower is better.

This is close to a standard 6D pose metric used in BOP-style evaluations. The
current script samples mesh vertices from the CAD `.ply`.

For strongly symmetric objects, ADD-S may be more appropriate than ADD. The race
car is not fully symmetric because front/back are visually different, so ADD is
usually more informative here.

## Recommended headline metrics for reports

For the overall model comparison, report:

- median translation error, mm
- median rotation error, degrees
- median projected-center error, px
- median ADD, mm
- recall at useful thresholds
- paired improvement fraction

Example threshold recalls:

```text
translation_recall_100mm
translation_recall_250mm
translation_recall_500mm
translation_recall_1000mm
rotation_recall_5deg
rotation_recall_10deg
rotation_recall_20deg
rotation_recall_45deg
```

For camera-by-camera summaries, compute the same metrics on each camera’s split
CSV.

## Camera-by-camera performance

First split predictions:

```bash
python -m fine_tuning.split_predictions_by_camera \
  --predictions gigaPose_datasets/results/<experiment>/predictions \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark
```

Then compare models per camera. Example for `front`:

```bash
python -m fine_tuning.compare_gigapose_predictions \
  --baseline-predictions gigaPose_datasets/results/<original_run>/predictions/by_camera/front/<original>MultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/<finetuned_run>/predictions/by_camera/front/<finetuned>MultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/metrics/original_vs_finetuned/front
```

Repeat for `rear`, `left`, `right`, etc.

This gives a clear camera-by-camera answer like:

```text
front: fine-tuned improves median center error by X px
rear: fine-tuned worsens median rotation by Y deg
left: no meaningful change
```

## How to compare more than two models

Use `fine_tuning.evaluate_gigapose_models` to compare any number of models in
one run:
# gigaPose_datasets/results/final_results/large_assettocorsa_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv

# gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv
# gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv

# gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_original_benchmark_runMultiHypothesis.csv

```bash
python -m fine_tuning.evaluate_gigapose_models \
  --model original=gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_original_benchmark_runMultiHypothesis.csv \
  --model IST_only_new=gigaPose_datasets/results/final_results/large_assettocorsa_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_only_old=gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_AE=gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/final_results/metrics/all_models
```


New: 
```bash
python -m fine_tuning.evaluate_gigapose_models \
--model original=gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_original_benchmark_runMultiHypothesis.csv \
  --model IST_only_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_only_withoutRear=gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_only_old_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_AE_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv \
  --model IST_AE_withoutRear=gigaPose_datasets/results/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/final_results/metrics/all_models_new
```


```bash
python -m fine_tuning.evaluate_gigapose_models \
  --model IST_only_withoutRear=gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_only_old_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_AE_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv \
  --model IST_AE_withoutRear=gigaPose_datasets/results/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/final_results/metrics/top4_models
```




for only two models:
```bash
python -m fine_tuning.evaluate_gigapose_models \
  --model original=gigaPose_datasets/results/large_assettocorsa_original_benchmark_with_max_depthenchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_original_benchmark_with_max_depthenchmarkMultiHypothesis.csv \
  --model new_IST_only=gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/final_results/metrics/2_models_maxdepth_new
```


Outputs:

```text
all_instance_metrics.csv
overall_summary.csv
camera_summary.csv
pairwise_instance_comparison.csv
best_model_per_instance.csv
overall_summary.json
camera_summary.json
```

Use:

- `overall_summary.csv` for the headline ranking.
- `camera_summary.csv` for front/rear/etc. performance.
- `pairwise_instance_comparison.csv` for all model-vs-model paired rows.
- `best_model_per_instance.csv` to see which model wins per GT car under each
  metric.

Recommended table columns:

```text
model
camera
evaluated_instances
score_median
translation_error_mm_median
rotation_error_deg_median
center_error_px_median
add_mm_median
translation_recall_500mm
rotation_recall_20deg
center_error_px_recall_50px  # if added later
pred_bbox_iou_median
pred_mask_iou_median
pred_mask_iou_recall_0.5
```

## Multi-model side-by-side visualizations

Use `fine_tuning.visualize_multi_model_per_car` when you have more than two
models and want one image per frame, with one panel per car:


<!-- 
original: gigaPose_datasets/results/large_assettocorsa_original_benchmark_with_max_depthenchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_original_benchmark_with_max_depthenchmarkMultiHypothesis.csv

---------------- these two were trained without rear images ------------
gigaPose_datasets/results/assettocorsa_ist_only_run_noRear/checkpoints/epoch=10-step=7000.ckpt
=> gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv

gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_noRear/checkpoints/epoch=28-step=20000.ckpt 
=> gigaPose_datasets/results/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv

--------------------------------------------------------------------

---- trained with rear images ------------
gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv

gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv

gigaPose_datasets/results/final_results/large_assettocorsa_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv

-->
```bash
python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_withoutRear=gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_only_old_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_AE_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv \
  --model IST_AE_withoutRear=gigaPose_datasets/results/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/final_results/metrics/top4_models/side_by_side_visuals \
  --max-images 300
```

```bash
python -m fine_tuning.visualize_multi_model_per_car \
--model IST_AE_withoutRear=gigaPose_datasets/results/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_only_old_withRear=gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_corrected_IST_only_benchmarkMultiHypothesis.csv \
  --model IST_only_withoutRear=gigaPose_datasets/results/large_assettocorsa_new_IST_only_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_new_IST_only_benchmark_with_max_depthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/final_results/metrics/top3_models/side_by_side_visuals \
  --max-images 300
```
<!-- finetune2: t=842mm R=18.3deg c=42px
translation is off by 842 mm
rotation is off by 18.3 degrees
projected center is off by 42 pixels

 -->
If an image has one GT car, the output has one full-image panel. If it has three
GT cars, the output has three full-image panels next to each other. Each panel
corresponds to one GT car and overlays GT plus all model predictions paired to
that car.

## IoU metrics

There are two possible IoU definitions:

### 2D bbox IoU

Render the predicted CAD pose, compute its visible 2D bbox, and compare to the
GT visible bbox:

```text
IoU = area(pred_bbox ∩ gt_bbox) / area(pred_bbox ∪ gt_bbox)
```

Higher is better.

This is easy to interpret, but it does not fully measure 6D pose. A pose can
have good 2D bbox IoU and still have wrong rotation/depth.

### Mask IoU

Render the predicted CAD silhouette and compare it to the GT/generated visible
mask.

Higher is better.

This is stronger than bbox IoU and often very useful for qualitative pose
alignment, but it requires rendering predicted masks for every evaluated pose.
Because it is slower than basic pose metrics, enable it explicitly:

```bash
--rendered-iou
```

Rendered IoU metrics:

- `pred_bbox_iou`
- `pred_mask_iou`
- `pred_bbox_iou_recall_0.25`
- `pred_bbox_iou_recall_0.5`
- `pred_bbox_iou_recall_0.75`
- `pred_mask_iou_recall_0.25`
- `pred_mask_iou_recall_0.5`
- `pred_mask_iou_recall_0.75`

## RMSE metrics

RMSE can be reported for translation, depth, center error, or ADD:

```text
RMSE = sqrt(mean(error^2))
```

RMSE is more sensitive to outliers than median error. For pose estimation,
report both:

- median error: robust “typical case”
- RMSE or mean error: outlier-sensitive “bad failures matter”

The current scripts report means and medians. If needed, RMSE can be added to
`summarize()` in `fine_tuning.compare_gigapose_predictions`.

## Suggested final benchmark report structure

For each model:

1. Overall summary table
   - number of evaluated instances
   - median translation error
   - median rotation error
   - median projected-center error
   - median ADD
   - recall at key thresholds

2. Camera-by-camera table
   - same metrics split by `front`, `rear`, etc.

3. Plots
   - error histograms
   - recall curves
   - paired improvement plots

4. Qualitative examples
   - several raw prediction overlays
   - several GT vs original vs fine-tuned overlays
   - include both good and bad examples

5. Notes about limitations
   - Assetto GT is simulator-derived
   - oracle masks vs detector masks should be reported separately
   - low-score predictions should be filtered or reported as failures depending
     on the benchmark goal

## Minimal command checklist

Prepare benchmark:

```bash
python -m Assetto_data_prep.prepare_inference \
  --source-root "$BENCHMARK" \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa_benchmark \
  --cameras all \
  --frame-stride 5 \
  --overwrite
```

Run inference for each model:

```bash
python test.py \
  test_dataset_name=assettocorsa_benchmark \
  model.checkpoint_path='PATH/TO/CHECKPOINT.ckpt' \
  run_id=MODEL_RUN_ID \
  name_exp=large_assettocorsa
```

Split outputs by camera:

```bash
python -m fine_tuning.split_predictions_by_camera \
  --predictions gigaPose_datasets/results/<experiment>/predictions \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark
```

Make visual overlays:

```bash
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/<experiment>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/overlays/<experiment> \
  --min-score 0.05
```

Compare two models:

```bash
python -m fine_tuning.compare_gigapose_predictions \
  --baseline-predictions gigaPose_datasets/results/<original_run>/predictions/<original>MultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/<finetuned_run>/predictions/<finetuned>MultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/metrics/original_vs_finetuned
```

Plot:

```bash
python -m fine_tuning.plot_prediction_comparison \
  --comparison-dir fine_tuning/metrics/original_vs_finetuned
```

Visual GT/model comparison:

```bash
python -m fine_tuning.visualize_prediction_gt_comparison \
  --baseline-predictions gigaPose_datasets/results/<original_run>/predictions/<original>MultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/<finetuned_run>/predictions/<finetuned>MultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/metrics/original_vs_finetuned/visual_overlays \
  --max-images 100
```
