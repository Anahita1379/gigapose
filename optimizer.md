# Real-data extrinsic optimization notes

This document explains how to use the real-data GigaPose/EPnPv2 selected samples
to estimate a camera calibration correction, and how to visualize/debug the result.

The short version:

- Use `EPnPv2_gt_mesh_z_hybrid_labels` or another EPnP folder that contains both:
  - `T_camera_object_centered`
  - `T_map_object_raw`
- Use `fine_tuning.select_real_label_candidates` to select good GigaPose/EPnP camera-frame matches.
- Combine selected samples per camera.
- Run `fine_tuning.optimize_camera_map_extrinsics` in metadata mode.
- Because map `z` conventions are inconsistent in the current data, optimize with `xy` translation residuals and an image-plane residual.
- Visualize before/after boxes on the actual image.

Do not mix cameras in one optimization run. Optimize `front`, `rear`, and `stereo_left` separately.

---

## 1. Why the optimizer uses metadata

Each frame is captured from a moving ego vehicle. Therefore there is no single fixed
`T_map_camera` for the whole run.

The metadata provides:

```yaml
t_map_lidar
t_lidar_camera_prior
```

So the per-frame camera pose is:

```text
T_map_camera_i = t_map_lidar_i @ t_lidar_camera_prior_i
```

The optimizer estimates a fixed correction to the camera calibration:

```text
T_lidar_camera_optimized =
    T_lidar_camera_correction_left_multiply @ t_lidar_camera_prior
```

Then for each frame:

```text
T_map_camera_optimized_i =
    t_map_lidar_i @ T_lidar_camera_optimized
```

This is why lidar appears in the optimizer: the lidar pose is the bridge between the moving map frame and the fixed camera calibration.

---

## 2. Which EPnP label folder to use

For candidate selection and optimization, prefer:

```text
EPnPv2_gt_mesh_z_hybrid_labels
```

or another label folder that contains:

```text
T_camera_object_centered
T_map_object_raw
```

The plain folder:

```text
EPnPv2_labels
```

may only contain `T_camera_object_centered`, which is useful for camera-frame comparison but not enough for map-camera extrinsic optimization.

You can check a label folder with:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels")
keys = set()
for f in sorted(root.glob("*.json"))[:20]:
    data = json.load(open(f))
    keys.update(k for k in data if "T_" in k or "map" in k.lower() or "pose" in k.lower())
for k in sorted(keys):
    print(k)
PY
```

You want to see:

```text
T_camera_object_centered
T_map_object_raw
```

---

## 3. Select good GigaPose label candidates

Use the right-side object-frame correction and robust refinement:

```bash
python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_data/large_real_20260505v1_front_gsam_v4_finetuned/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_real_20260505v1_front_gsam_v4_finetunedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 3 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260505v1_front_gsam_v4_finetuned/label_candidates_hybrid_right_refined
```

Important idea:

```text
T_epnp_camera_object ≈ T_gigapose_camera_object @ X_object
```

This is why we use:

```bash
--frame-transform-side right
```

The old left-side model:

```text
T_epnp ≈ X @ T_gigapose
```

caused the overlay to lag behind the moving car.

The output folder contains:

```text
all_candidate_pairs.csv
best_candidate_per_epnp_label.csv
selected_samples.csv
selection_report.json
frame_transform_gigapose_to_epnp.json
```

Use `selected_samples.csv` for optimization.

---

## 4. Visualize selected candidates

Before optimizing extrinsics, make sure the selected samples look good:

```bash
python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_data/large_real_20260505v1_front_gsam_v4_finetuned/label_candidates_hybrid_right_refined/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260505v1_front_gsam_v4_finetuned/label_candidates_hybrid_right_refined/visual_selected \
  --max-images 100 \
  --draw-mask-bbox
```

Colors:

```text
green = EPnPv2 camera pose
blue  = GigaPose aligned to EPnPv2 object frame
yellow = GSAM/detection bbox
```

If these do not look reasonable, do not proceed to extrinsic optimization.

---

## 5. Combine selected samples

If you have multiple selected-sample files for the same camera, combine them:

```bash
python -m fine_tuning.combine_selected_samples \
  --input path/to/session1/selected_samples.csv \
  --input path/to/session2/selected_samples.csv \
  --input path/to/session3/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```

Only combine files from the same camera:

```text
front + front + front      OK
rear + rear + rear         OK
front + rear               not OK
```

The combiner adds:

```text
source_selected_csv
```

so you can trace each row back to the original run.

---

## 6. Why map-z causes problems

The current data has inconsistent `z` conventions.

Example:

```text
T_map_object_raw z  ≈ -35 to -38 m
ground_truth_pose z ≈ -33 to -37 m
t_map_lidar z       ≈ -3 to +1 m
```

`T_map_object_raw` and `ground_truth_pose` are in the same altitude convention.
`t_map_lidar z` is not.

If the optimizer uses full 3D translation residuals:

```text
[x_error, y_error, z_error]
```

it tries to fix the fake z error by creating a huge camera z correction, for example:

```text
tz = -15 m
tz = -34 m
```

Those corrections are not physical and should not be used.

For the current data, use:

```bash
--translation-residual-components xy
```

This optimizes horizontal map position and rotation while ignoring the inconsistent map `z` residual.

There is a second, separate z issue during projection. The earlier
`metadata_lidar` z mode did this:

```text
object_z_for_projection = metadata t_map_lidar z
```

That makes the object sit at the ego/lidar height. It can look acceptable on
flat road, but it is wrong on hills and downhills because it destroys the
object's relative height.

The safer projection mode is:

```bash
--image-center-map-z-mode ego_relative
```

That uses:

```text
object_z_for_projection =
    metadata t_map_lidar z
    + (EPnP object map z - metadata ground_truth_pose z)
```

In plain English: keep the target car's height relative to the ego car, but
express that height in the same z convention as `t_map_lidar`. This is the mode
to try when boxes are above/below the car on slopes.

---

## 7. Why we add an image-plane residual

Using only map `x/y` and rotation is still not enough. The optimizer can improve map residuals while moving the projected box off the image.

To prevent that, the optimizer now supports an image-center residual:

```text
project(T_map_object_raw through corrected extrinsic)
≈ project(selected GigaPose pose)
```

The full objective becomes:

```text
map x/y object error
+ map/object rotation error
+ image center projection error
+ strong prior on extrinsic correction
```

This keeps the optimized projection visually tied to the selected car.

---

## 8. Run extrinsic optimization

Recommended command for the current front-camera data:

```bash
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --use-sample-metadata \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --translation-residual-components xy \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --image-center-map-z-mode ego_relative \
  --translation-prior-weight 5000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy_image
```

Meaning of the important options:

| option | meaning |
|---|---|
| `--use-sample-metadata` | Load per-sample `t_map_lidar` and `t_lidar_camera_prior` from metadata YAML. |
| `--epnp-map-pose-key T_map_object_raw` | Use map-frame object pose from EPnP labels. |
| `--translation-residual-components xy` | Ignore broken map-z residual. |
| `--image-center-weight 5` | Add image-plane guardrail. |
| `--image-center-sigma-px 50` | Scale image center residual by 50 px. |
| `--image-center-map-z-mode ego_relative` | For image-center residual only, preserve target-vs-ego height while using metadata lidar z convention. |
| `--translation-prior-weight 5000` | Strongly discourage moving camera translation. |
| `--rotation-prior-weight 100` | Discourage large rotation changes. |

The optimizer writes:

```text
optimized_extrinsics.json
optimization_report.json
errors_before_optimization.csv
errors_after_optimization.csv
```

In sample-metadata mode, the important output is:

```json
T_lidar_camera_correction_left_multiply
```

Apply it as:

```text
T_lidar_camera_optimized =
    T_lidar_camera_correction_left_multiply @ t_lidar_camera_prior

T_map_camera_optimized =
    t_map_lidar @ T_lidar_camera_optimized
```

---

## 9. Plot numeric optimization results

```bash
python -m fine_tuning.plot_extrinsic_optimization \
  --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy_image \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv
```

This creates:

```text
plots/translation_error_mm_hist.png
plots/rotation_error_deg_hist.png
plots/before_after_summary_bars.png
plots/per_sample_translation_error_mm.png
plots/per_sample_rotation_error_deg.png
plots/extrinsic_correction_components.png
plots/extrinsic_axes_before_after.png
plots/plot_summary.json
```

Watch for huge correction values. A plausible correction should not move the camera by many meters.

---

## 10. Visualize before/after boxes on real images

```bash
python -m fine_tuning.visualize_extrinsic_optimization_on_images \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy_image \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy_image/image_overlays_z_ego_relative \
  --map-z-mode ego_relative \
  --draw-gigapose \
  --draw-detection-bbox \
  --write-debug-projections \
  --max-images 100
```

Colors:

```text
red    = map pose projected with original metadata extrinsic
blue   = map pose projected with optimized extrinsic
green  = selected GigaPose pose
yellow = detection bbox
```

The debug CSV:

```text
overlay_index.csv
```

contains:

```text
original_center_u
original_center_v
original_center_depth_mm
optimized_center_u
optimized_center_v
optimized_center_depth_mm
gigapose_center_u
gigapose_center_v
gigapose_center_depth_mm
raw_map_object_z_m
adjusted_map_object_z_m
metadata_lidar_z_m
metadata_ground_truth_pose_z_m
object_minus_ego_z_m
adjusted_minus_lidar_z_m
```

Use this to understand why a box is invisible:

- negative depth: behind camera
- huge `u`/`v`: off image
- negative `v`: above image
- very large `v`: below image

For a `2064 x 400` image, the center should usually satisfy:

```text
0 <= center_u <= 2064
0 <= center_v <= 400
```

---

## 11. How to judge success

A good optimized result should satisfy:

- Blue box is visible.
- Blue box is closer to the car/detection/GigaPose than red.
- Optimized center is not thrown outside the image.
- Correction translation is small.
- Correction rotation is plausible, not tens of degrees unless you know the prior was very wrong.

If blue disappears or is worse than red:

- do not use that optimized extrinsic;
- increase `--image-center-weight`;
- increase `--rotation-prior-weight`;
- increase `--translation-prior-weight`;
- check whether selected samples are mixed from different cameras;
- check whether map z conventions are still leaking into the residual.

Example stronger visual guardrail:

```bash
--image-center-weight 20 \
--rotation-prior-weight 500 \
--translation-prior-weight 10000
```

---

## 12. Current recommendation

For the current data, do not trust pure map-frame optimization without image residuals.

Use:

```bash
--translation-residual-components xy
--image-center-weight 5
--image-center-map-z-mode ego_relative
```

Then verify on images. The visual overlay is the deciding sanity check.
