# Assetto Corsa Training and Data-Cleaning Modifications

## Purpose

This document summarizes the changes made to prepare, clean, validate, and
fine-tune the Assetto Corsa dataset for GigaPose. It describes the behavior and
reasoning rather than reproducing implementation code.

## 1. Dataset preparation and train/validation splitting

The preparation workflow converts the selected Assetto Corsa recording
sessions into GigaPose-compatible WebDataset splits. Each stored sample keeps
the complete set of files needed by GigaPose, including RGB, depth, camera
intrinsics, poses, object annotations, visible masks, and sample metadata.

Two splitting strategies are supported:

- **Session split:** complete recording sessions are assigned to either
  training or validation.
- **Random-frame split:** validation frames are selected from across all
  supplied sessions and enabled cameras. A selected frame is removed from the
  training split, so there is no train/validation overlap.

The random-frame mode supports:

- An exact requested number of validation images.
- A fixed random seed for repeatable selection.
- Sampling across all source sessions and all cameras when all cameras are
  enabled.

A validation unit is a complete camera frame, not one individual car. All
opponent instances and all required GigaPose data belonging to that frame are
stored together in the selected split.

The random selection occurs before write-time eligibility filters such as
depth and minimum-mask requirements. If a selected frame has no eligible
instances after those filters, the number of samples actually written can be
lower than the requested selection count.

## 2. Grounded-SAM detection stage

Grounded-SAM was used as an independent visual sanity check of the prepared
ground-truth data. The purpose was not to replace the Assetto Corsa ground
truth. It was used to identify frames where the stored ground truth claims that
more visible cars exist than an independent image detector can find.

### 2.1 Input

The Grounded-SAM runner reads RGB images directly from every `shard-*.tar` file
in a prepared WebDataset split. Both JPEG and PNG RGB members are supported.

The runner scans all shards and reports:

- Number of shards found.
- Number of RGB images found.
- Number of RGB images per shard.
- Image-like members with unsupported names.

Images can be sampled using a frame stride or a maximum-image limit for quick
debugging. Full cleaning runs should omit the maximum-image limit.

### 2.2 Detection

GroundingDINO performs open-vocabulary box detection using a text prompt such
as `race car.`. SAM2 then produces an independent segmentation mask for each
accepted box.

GroundingDINO is an open-vocabulary detector rather than an exact CAD-model
classifier. A “race car” prompt can therefore detect visually similar cars and
does not guarantee that the detected vehicle is the exact training CAD model.
The detector is used for visible-object counting and sanity checking, not for
fine-grained vehicle identity.

### 2.3 Detection cleanup

Several cleanup stages are applied before a detection is counted:

- GroundingDINO box-score and text-score thresholds.
- Optional label-substring filtering.
- Minimum and maximum box-area fractions.
- Minimum and maximum box aspect ratios.
- Box non-maximum suppression to remove duplicate boxes.
- SAM2 minimum mask-pixel threshold.
- Mask-IoU non-maximum suppression to remove duplicate masks.

The ego-car filter removes detections in a configured lower-middle image
region. It can be restricted to selected camera IDs, normally the front
camera. Camera-specific filtering requires a `frame_map.json` or
`frame_map.csv`; without a camera map, the script cannot infer front versus
rear from intrinsics because those cameras use the same intrinsic matrix.

### 2.4 Outputs

The Grounded-SAM stage writes:

- One integer instance-mask PNG per processed frame.
- One JSON file per frame with accepted objects.
- A combined metadata JSON.
- A detector-style JSON containing every accepted detection.
- An optional RGB overlay directory.
- A run report containing thresholds, scan results, processed-image count,
  detection count, and skipped-image counts.

An image reported as `no_valid_masks` produced zero accepted detections after
all filtering stages. The image and empty metadata can still be written, but
its detection count is zero.

The runner also accepts a CSV containing a `key` column. This makes it possible
to rerun Grounded-SAM only on samples listed in `rejected_samples.csv` and
produce targeted overlays for visual inspection.

## 3. Grounded-SAM WebDataset cleaning

The cleaning stage compares the independent Grounded-SAM count with the number
of visible ground-truth cars stored in each WebDataset sample.

The ground-truth visible count is read from the visible-mask metadata when
available, with the ground-truth object list used as a fallback.

Grounded-SAM detections can first be filtered by a minimum score. The default
cleaning rule is:

> Reject a sample when the number of visible ground-truth cars is greater than
> the number of accepted Grounded-SAM detections.

Examples:

| Visible GT cars | GSAM detections | Result |
| ---: | ---: | --- |
| 3 | 4 | Kept |
| 3 | 3 | Kept |
| 3 | 2 | Rejected |
| 1 | 0 | Rejected |
| 0 | 0 | Kept |

Extra independent detections are tolerated. Missing independent detections
cause rejection because they can indicate an incorrect GT object count,
incorrect projection, invisible annotation, or a frame too ambiguous for
reliable supervision.

An optional absolute minimum Grounded-SAM count can also be required. With its
default value of zero, a zero-detection frame is rejected only when its visible
GT count is greater than zero.

### 3.1 Rewriting the clean split

Cleaning creates a new WebDataset split. It does not modify the original split.
For every kept key, all members belonging to that sample are copied together,
so RGB, depth, masks, camera information, poses, and metadata remain complete.

The kept samples are repacked into new shards. Consequently, the number of TAR
files in a cleaned split does not need to match the number in the source split.
Fewer output shards do not imply that most images were removed; they usually
mean that the retained samples were repacked using a different samples-per-
shard limit.

The cleaner writes:

- `kept_samples.csv`
- `rejected_samples.csv`
- `key_to_shard.json`
- `filter_report.json`
- Repacked `shard-*.tar` files

Each CSV row records the sample key, scene and image IDs, GT visible count,
Grounded-SAM count, source shard, and keep/reject decision.

## 4. Camera and geometry checks

The available front and rear cameras use identical intrinsics, so the intrinsic
matrix alone cannot identify which of those cameras produced a frame. Camera
identity must come from source metadata or a frame map.

A separate opponent-ground-truth projection visualizer was added for the newer
recording schema. It uses opponent transforms, camera transforms, intrinsics,
RGB images, and the CAD model directly. It does not require the legacy
`bboxes_3d.csv` file or original instance masks.

The visualizer can render the predicted/projected CAD silhouette and compare
its box with the recorder-provided box. This is useful for confirming that
camera calibration, coordinate conventions, opponent transforms, and CAD
orientation agree with the source images.

## 5. Mixed-camera training preprocessing

Front/rear images are `2064 × 400`, while stereo images are `2064 × 760`.
Previously, a mixed-resolution batch retained only the majority resolution,
which could silently skip minority-camera frames.

The updated training and validation preprocessing uses a shared
`2064 × 760` canvas instead.

### 5.1 Symmetric front/rear padding

Each `2064 × 400` front or rear frame receives:

- 180 invalid rows above the image.
- 180 invalid rows below the image.

The original image therefore occupies rows `[180, 580)` in the padded canvas.

Stereo frames are already `2064 × 760` and do not require padding.

### 5.2 Geometry preservation

Padding is translation, not resizing. No source pixels are stretched or
resampled merely to match the full-frame canvas.

For front/rear images, all geometry is shifted consistently:

- The intrinsic principal point moves from `cy = 200` to `cy = 380`.
- Modal bounding boxes move down by 180 pixels.
- Amodal bounding boxes move down by 180 pixels.
- RGB, depth, segmentation, and binary instance masks receive the same
  symmetric padding.
- Object poses remain unchanged because the camera rays are preserved by the
  corresponding principal-point update.

The original resolution and padding offset are retained so visualizations can
be converted back to the native image coordinates.

## 6. Valid-pixel and supervision masks

Every training image now has a binary validity mask:

- `1`: usable original image pixel.
- `0`: padded or deliberately ignored pixel.

For front/rear images, the outer 258 columns on each side are marked invalid.
This leaves exactly 1,548 valid columns, or 75% of the original 2,064-pixel
width.

The front/rear valid region in padded coordinates is:

- Horizontal: `[258, 1806)`
- Vertical: `[180, 580)`

Stereo images retain their complete native image area as valid.

The object supervision mask is combined with the validity mask. As a result:

- Padding cannot become an object correspondence.
- The dark front/rear side regions cannot become object correspondences.
- Invalid locations do not contribute to IST regression supervision.
- Invalid locations do not contribute to AE contrastive supervision.
- Padded depth is zero.
- Crops may geometrically contain invalid areas, but those areas are black in
  the masked network input and are excluded from supervision.

The crop transform itself is kept unchanged. This avoids shifting or
distorting an object crop merely because it intersects an invalid boundary.

## 7. Validation behavior during fine-tuning

### 7.1 Lightweight validation

The normal validation stage runs at the configured validation interval, for
example every 250 optimizer steps. It traverses the validation loader and
calculates the applicable training losses and diagnostic metrics.

Validation instance selection is deterministic. At least one instance from
each frame in a validation batch is selected before remaining crop slots are
filled. This preserves frame coverage while retaining the configured
object-crop memory limit.

The lightweight visual outputs include:

- Ground-truth template/image correspondences.
- An IST alignment overlay on the real, unmasked object crop.

The IST overlay aggregates predicted relative scale and in-plane rotation and
uses patch correspondences for translation. It is a direct diagnostic of IST,
not a claim of an independently recovered complete 6D pose.

### 7.2 Optional heavy validation

An optional heavy visual-validation stage can be enabled separately. Typical
settings run it every 1,000 optimizer steps on a small deterministic set of
distinct validation frames.

At steps where heavy validation runs, normal validation still runs as well.
The heavy stage:

- Runs with gradients disabled.
- Does not calculate or contribute a training loss.
- Does not update weights.
- Uses frozen/current model components for template retrieval and matching.
- Uses the current trained IST state.
- Runs RANSAC and full pose recovery.
- Renders the CAD at the recovered pose over the original RGB image.

Template IST features are regenerated at every heavy-validation event so they
do not become stale while IST continues to train.

Rendering uses the padded intrinsics and canvas internally, then crops the
result back to the original image extent for presentation.

### 7.3 Saved validation results

Local validation images are written under:

`gigaPose_datasets/results/<run-name>/validation_images/`

The directory includes lightweight correspondence images, lightweight IST
alignment images, and optional heavy CAD-overlay images. Filenames include the
optimizer step, validation batch where relevant, and distributed rank.

When W&B logging is enabled, the principal image streams are:

- `vis/val_gt_samples`
- `vis/val_ist_overlay`
- `vis/val_heavy_cad_overlay`

Validation losses and errors are logged separately under `val/...` metric
names.

## 8. Fine-tuning controls

The fine-tuning entry point supports:

- Training IST only, AE only, or both networks.
- Separate IST and AE learning rates.
- Configurable batch size and worker count.
- Configurable validation and checkpoint intervals.
- Single-GPU or multi-GPU execution.
- TensorBoard, W&B, or disabled experiment logging.
- Match-similarity and patch-threshold overrides.
- Optional heavy validation interval, image count, and CAD path.

IST-only training updates only IST parameters. Frozen model components can
still participate in validation inference and full pose recovery.

## 9. Validation-data integrity checks

The prepared training and validation splits were checked for:

- Missing foreground depth.
- Foreground mask pixels without valid depth.
- Invalid rotation determinants.
- Dataset and instance counts.

The reported checks showed no missing foreground depth and rotation
determinants extremely close to `+1`, indicating valid proper rotations.

These checks establish internal data consistency. They do not replace visual
CAD projection checks, which are still needed to catch a globally consistent
but incorrect coordinate convention, calibration, or pose interpretation.

## 10. Important operational notes

- Use the original split as Grounded-SAM input. Rejected images are absent from
  the cleaned split.
- Use the cleaned split explicitly as the training or validation split when
  cleaned data is desired.
- A cleaned split can legitimately contain fewer TAR shards because samples
  are repacked.
- Grounded-SAM is an independent sanity filter, not ground truth and not an
  exact CAD classifier.
- Front/rear camera identity cannot be recovered from intrinsics alone.
- Native-resolution inference remains possible because training padding is a
  geometry-preserving full-frame batching operation performed before
  fixed-size object crops are produced.
- If inference itself mixes full-image resolutions in one batch, it must apply
  geometrically consistent padding and validity masks. If cameras are processed
  in native-resolution homogeneous batches, padding is unnecessary.
- The outer-side validity policy should also be respected during inference if
  detections can occur in those ignored regions.
- Diagnostic images showing bottom-only padding, symmetric padding, the
  75%-valid side policy, and the resulting binary masks are stored under
  `fine_tuning/padding_mask_examples/`.
- Static syntax and formatting checks were run after the modifications. A full
  GPU/EGL smoke test is still required in the actual GigaPose environment.
