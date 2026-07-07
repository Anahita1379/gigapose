Notes: 


IST_AE runs and the IST lr: 
newest: assettocorsa_ist_penultimate_last_ae_noRear_again  ISTlr1e5
going to use the checkpoint:  gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt
large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv

older: assettocorsa_ist_penultimate_last_ae_noRear  ISTlr5e6
gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_noRear/checkpoints/epoch=28-step=20000.ckpt
large_assettocorsa_IST_AE_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv


oldest: assettocorsa_ist_penultimate_last_ae_corrected ISTlr5e6
gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt
assettocorsa_older_ist_2layerAE_benchmark
large_assettocorsa_older_ist_2layerAE_benchmark
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_older_ist_2layerAE_benchmark/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_older_ist_2layerAE_benchmarkMultiHypothesis.csv

First make sure they all have ran on the same dataset
python test.py \
  test_dataset_name=assettocorsa_benchmark_with_max_depth \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt'" \
  run_id=assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depth \
  name_exp=large_assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depth


So now, the correct format is: 
oldest: assettocorsa_ist_penultimate_last_ae_corrected ISTlr5e6
gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt


comparing the three IST_AE and the IST_only using evaluate_gigapose_models
```bash
python -m fine_tuning.evaluate_gigapose_models \
--model IST_AE_withoutRear_istLr1e5=gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_AE_withoutRear_istLr5e6=gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_AE_withRear_istLr5e6=gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depthMultiHypothesis.csv  \
  --model IST_only_current=gigaPose_datasets/results/last_picks/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
--split test \
--rendered-iou \
--output-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_July6_2217



python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/last_picks/metrics/IST_AE_July6_2217/all_instance_metrics.csv --comparison-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_July6_2217  --output-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_July6_2217/plots

  python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_July6_2217

```


new code for evaluation: 
```bash
python -m fine_tuning.evaluate_pose_errors_by_distance \
  --model IST_AE_withoutRear_istLr1e5=gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_AE_withoutRear_istLr5e6=gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv \
  --model IST_AE_withRear_istLr5e6=gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_ist_penultimate_last_ae_corrected_benchmark_with_max_depthMultiHypothesis.csv  \
  --model IST_only_current=gigaPose_datasets/results/last_picks/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --distance-bin-m 10 \
  --output-dir gigaPose_datasets/results/final_results/metrics/IST_IST_AE_July6_2151/pose_distance

```


For now it seems the best model is actually: 
older: assettocorsa_ist_penultimate_last_ae_noRear  ISTlr5e6
gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_noRear/checkpoints/epoch=28-step=20000.ckpt
large_assettocorsa_IST_AE_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_benchmark_with_max_depthMultiHypothesis.csv

-------------------------------------------------------
no, i think it should be:
newest: assettocorsa_ist_penultimate_last_ae_noRear_again  ISTlr1e5
going to use the checkpoint:  gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt
large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv





For plots, comparision and recommandation:
```bash
python -m fine_tuning.plot_prediction_comparison \
  --input-csv fine_tuning/prediction_gt_comparison/per_instance_metrics.csv \
  --output-dir fine_tuning/prediction_gt_comparison/plots

  python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/last_picks/metrics/2ckt_diff/all_instance_metrics.csv   --comparison-dir gigaPose_datasets/results/last_picks/metrics/2ckt_diff  --output-dir gigaPose_datasets/results/last_picks/metrics/2ckt_diff/plots

  python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/2ckt_diff


  python -m fine_tuning.recommend_model_by_task \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/2ckt_diff \
  --plot
```

final benchmark testing: 
newest: assettocorsa_ist_penultimate_last_ae_noRear_again  ISTlr1e5
going to use the checkpoint:  gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt
large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv

Now, final benchmarck testing: 
  python test.py \
  test_dataset_name=assettocorsa_benchmark_with_max_depth \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt'" \
  run_id=final_AC_gigapose_IST1e5_AE_noRear_benchmark_withMaxDepth \
  name_exp=large_final_AC_gigapose_IST1e5_AE_noRear_benchmark_withMaxDepth


saved preds: 
gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv


devide per camera:
python -m fine_tuning.split_predictions_by_camera \
  --predictions gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/ \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth


overlay per camera: 
```bash
(front only) -----------------------------------------------
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/front/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/overlays/front \
  --min-score 0.1

python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_old_withRear_last_front=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/front/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/finetune_gt_side_by_side/front \
  --max-images 100
--------------------------------------------------------------

(rear only)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/rear/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/overlays/rear \
  --min-score 0.1


  python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_old_withRear_last_front=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/rear/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/finetune_gt_side_by_side/rear \
  --max-images 100
--------------------------------------------------------------------------------

(stereo_left)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/stereo_left/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/overlays/stereo_left \
  --min-score 0.1


python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_old_withRear_last_front=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/stereo_left/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/finetune_gt_side_by_side/stereo_left \
  --max-images 100
  
------------------------------------------------------------------------------------


(stereo_right)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/stereo_right/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/overlays/stereo_right \
  --min-score 0.1



python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_old_withRear_last_front=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/by_camera/stereo_right/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/finetune_gt_side_by_side/stereo_right \
  --max-images 100
```


Visual GT/model comparison for the whole model:

```bash
python -m fine_tuning.visualize_multi_model_per_car \
  --model IST_only_old_withRear_last=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv\
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --output-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/finetune_gt_side_by_side \
  --max-images 100
```


Use evaluate_gigapose_models with just one --model. It will compare that model against the GT stored in the prepared dataset and write camera_summary.csv
```bash
python -m fine_tuning.evaluate_gigapose_models \
  --model IST_only_old_withRear_last=gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_final_AC_gigapose_IST_only_benchmark_withMaxDepthMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/final_results/metrics/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth

```

If you want plots too
```bash
python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth/metrics/large_final_AC_gigapose_IST_only_benchmark_withMaxDepth
```


if time allowed, try assettocorsa_ist_penultimate_last_ae_corrected_run2 as well. 
gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_corrected_run2/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_ist_penultimate_last_ae_corrected_run2MultiHypothesis.csv

compare this one with the last ckt:
```bash
python -m fine_tuning.evaluate_gigapose_models \
--model IST_AE_run2=gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_corrected_run2/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_ist_penultimate_last_ae_corrected_run2MultiHypothesis.csv \
--model IST_only_old_withRear_last=gigaPose_datasets/results/last_picks/large_assettocorsa_older_corrected_IST_only_benchmark_withMaxDepth_lastckt/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_older_corrected_IST_only_benchmark_withMaxDepth_lastcktMultiHypothesis.csv \
--dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_with_max_depth \
--split test \
--rendered-iou \
--output-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last


  python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last/all_instance_metrics.csv   --comparison-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last  --output-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last/plots

  python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last


  python -m fine_tuning.recommend_model_by_task \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last \
  --plot


```

Now it is time for real world stuff: 


<!--
/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-26-12-19-49/ Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/ Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0 Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2 Done
  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1 Done
   /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4 only front for now:
  -->

Full prep command:
```bash
CUDA_VISIBLE_DEVICES=1  python -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name real_20260518v2v4_front_gsam_v4 \
  --grounded-sam-dir Grounded_Sam_v4 \
  --overwrite

 CUDA_VISIBLE_DEVICES=1  python -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/rear \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name real_20260518v2v4_rear_gsam_v4 \
  --grounded-sam-dir Grounded_Sam_v4 \
  --overwrite


  CUDA_VISIBLE_DEVICES=1  python -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name real_20260518v2v4_stereo_left_gsam_v4 \
  --grounded-sam-dir Grounded_Sam_v4 \
  --overwrite

  ```

Then render templates:
```bash
CUDA_VISIBLE_DEVICES=1 python -m src.scripts.render_custom_templates \
  custom_dataset_name=real_20260518v2v4_front_gsam_v4 \
  machine.num_workers=1


CUDA_VISIBLE_DEVICES=1 python -m src.scripts.render_custom_templates \
  custom_dataset_name=real_20260518v2v4_rear_gsam_v4 \
  machine.num_workers=1


  CUDA_VISIBLE_DEVICES=1 python -m src.scripts.render_custom_templates \
  custom_dataset_name=real_20260518v2v4_stereo_left_gsam_v4 \
  machine.num_workers=1
  ```
Then run inference:
```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  test_dataset_name=real_20260526_front_gsam_v4 \
  run_id=real_20260526_front_gsam_v4_original \
  name_exp=large_real_20260526_front_gsam_v4_original

  ```
  With a fine-tuned checkpoint:
<!-- newest: assettocorsa_ist_penultimate_last_ae_noRear_again  ISTlr1e5
going to use the checkpoint:  gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt
large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth
pred: gigaPose_datasets/results/last_picks/large_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depth/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_with_max_depth-test_assettocorsa_IST_AE_noRear_again_benchmark_with_max_depthMultiHypothesis.csv \ -->


20260526 done
20260518v1v4  done
20260518v0  done
20260505v2  done
20260505v1   done
20260518v2v4  front and stereo left: 
   
```bash
 CUDA_VISIBLE_DEVICES=0 python test.py \
  test_dataset_name=real_20260518v2v4_front_gsam_v4 \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt'" \
  run_id=real_20260518v2v4_front_gsam_v4_finetuned_AE_IST \
  name_exp=large_real_20260518v2v4_front_gsam_v4_finetuned_AE_IST


 CUDA_VISIBLE_DEVICES=0 python test.py \
  test_dataset_name=real_20260518v2v4_stereo_left_gsam_v4 \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt'" \
  run_id=real_20260518v2v4_stereo_left_gsam_v4_finetuned_AE_IST \
  name_exp=large_real_20260518v2v4_stereo_left_gsam_v4_finetuned_AE_IST


   CUDA_VISIBLE_DEVICES=1 python test.py \
  test_dataset_name=real_20260505v2_rear_gsam_v4 \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_noRear_again/checkpoints/epoch=33-step=23000.ckpt'" \
  run_id=real_20260505v2_rear_gsam_v4_finetuned_AE_IST \
  name_exp=large_real_20260505v2_rear_gsam_v4_finetuned_AE_IST
  ```

  we can also visualize it useing: 
  ```bash
  python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260526_front_gsam_v4_finetuned_AE_IST/predictions/large-pbrreal-rgb-mmodel_real_20260526_front_gsam_v4-test_real_20260526_front_gsam_v4_finetuned_AE_ISTMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260526_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260526_front_gsam_v4_finetuned_AE_IST/pred_overlays \
  --min-score 0.01

  python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260526_rear_gsam_v4_finetuned_AE_IST/predictions/large-pbrreal-rgb-mmodel_real_20260526_rear_gsam_v4-test_real_20260526_rear_gsam_v4_finetuned_AE_ISTMultiHypothesis.csv\
  --dataset-dir gigaPose_datasets/datasets/real_20260526_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260526_rear_gsam_v4_finetuned_AE_IST/pred_overlays \
  --min-score 0.01


python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260526_stereo_left_gsam_v4_finetuned_AE_IST/predictions/large-pbrreal-rgb-mmodel_real_20260526_stereo_left_gsam_v4-test_real_20260526_stereo_left_gsam_v4_finetuned_AE_ISTMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260526_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260526_stereo_left_gsam_v4_finetuned_AE_IST/pred_overlays \
  --min-score 0.01


# 20260518v2v4 done
# 20260505v2 done 
# 20260518v0 done
# 20260518v1v4 done
# 20260526 Done
# 20260518v2v4   ????



  ```

gigaPose_datasets/datasets/real_20260526_front_gsam_v4/test/
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/models/
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/frame_map.json
gigaPose_datasets/datasets/cnos-fastsam/cnos-fastsam_real_20260526_front_gsam_v4-test.json