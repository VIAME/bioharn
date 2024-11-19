#!/bin/bash
__doc__="
Trained on toothbrush
"
export CUDA_VISIBLE_DEVICES="1,"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=HABCAM-FISH
EXPERIMENT_NAME="viame2024-train_baseline_regnety_3class_v003"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-3class-v03.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-3class-v03.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-3class-v03.kwcoco.zip
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
mkdir -p "$DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: $EXPERIMENT_NAME
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
base: new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py
init: new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py
cfg:
    dataloader:
        train:
            total_batch_size: 12
            num_workers: 4
            mapper:
                use_instance_mask: false
                recompute_boxes: false
    model:
        roi_heads:
            mask_in_features: null
            mask_pooler: null
            mask_head: null
    optimizer:
        lr: 0.001
    train:
        amp:
            enabled: true
        max_iter: 184375
        eval_period: 5000
        log_period: 20
        checkpointer:
            period: 5000
            max_to_keep: 100
        device: cuda
" > "$DEFAULT_ROOT_DIR"/train_config.yaml
cat "$DEFAULT_ROOT_DIR"/train_config.yaml
python -m geowatch.tasks.detectron2.fit --config "$DEFAULT_ROOT_DIR"/train_config.yaml


# Check which models exist
ls "$DEFAULT_ROOT_DIR"/*/model_*.pth

python -m geowatch.tasks.detectron2.predict \
    --checkpoint_fpath "$HOME"/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0179999.pth \
    --base "auto" \
    --src_fpath "$TEST_FPATH" \
    --dst_fpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --workers=4

    #--base "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py" \


kwcoco eval_detections \
    --true_dataset "$TEST_FPATH" \
    --pred_dataset "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --out_dpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/coco_metrics

python ~/code/kwcoco/dev/poc/detection_confusor_analysis.py \
    --true_fpath "$TEST_FPATH" \
    --pred_fpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --out_dpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/confusion_analysis



# specified models
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=HABCAM-FISH
EXPERIMENT_NAME="viame2024-train_baseline_regnety_3class_v003"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-3class-v03.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-3class-v03.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-3class-v03.kwcoco.zip

DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
EVAL_PATH=$DEFAULT_ROOT_DIR/_habcam_detectron_evals
mkdir -p "$EVAL_PATH"

echo "
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0004999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0009999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0014999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0019999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0024999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0029999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0034999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0039999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0044999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0049999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0054999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0059999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0064999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0069999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0074999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0079999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0084999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0089999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0094999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0099999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0104999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0109999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0114999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0119999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0124999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0129999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0134999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0139999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0144999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0149999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0154999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0159999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0164999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0169999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0174999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_0179999.pth
- /home/joncrall/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_regnety_3class_v003/v_cda8fec7/model_final.pth
" > "$EVAL_PATH"/detectron_models_v3.yaml

#kwcoco info "$VALI_FPATH" -g1


python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'geowatch.tasks.detectron2.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - $EVAL_PATH/detectron_models_v3.yaml
            detectron_pred.src_fpath:
                - $VALI_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: false
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 0
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
EVAL_PATH=$DVC_EXPT_DPATH/_habcam_detectron_evals
python -m geowatch.mlops.aggregate \
    --pipeline='geowatch.tasks.detectron2.pipelines.detectron_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "
