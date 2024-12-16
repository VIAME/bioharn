#!/bin/bash
__doc__="
Trained on namek
"
export CUDA_VISIBLE_DEVICES="0,"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=HABCAM-FISH
EXPERIMENT_NAME="viame2024-train_detectron2_fastercnn_3class_v004"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-3class-v03.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-3class-v03.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-3class-v03.kwcoco.zip
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
mkdir -p "$DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: $EXPERIMENT_NAME
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
base: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
init: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
cfg:
    DATALOADER:
        NUM_WORKERS: 4
    SOLVER:
        IMS_PER_BATCH: 16   # This is the real 'batch size' commonly known to deep learning people
        BASE_LR: 0.02     # pick a good LR
        MAX_ITER: 90000
        STEPS: []            # do not decay learning rate
" > "$DEFAULT_ROOT_DIR"/train_config.yaml
cat "$DEFAULT_ROOT_DIR"/train_config.yaml
python -m geowatch.tasks.detectron2.fit --config "$DEFAULT_ROOT_DIR"/train_config.yaml


# Check which models exist
ls "$DEFAULT_ROOT_DIR"/*/model_*.pth


python -m geowatch.tasks.detectron2.predict \
    --checkpoint_fpath "$HOME"/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_e63998e9/model_0119999.pth \
    --src_fpath "$TEST_FPATH" \
    --base "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" \
    --dst_fpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --workers=4


kwcoco eval_detections \
    --true_dataset "$TEST_FPATH" \
    --pred_dataset "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --out_dpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/coco_metrics

python ~/code/kwcoco/dev/poc/detection_confusor_analysis.py \
    --true_fpath "$TEST_FPATH" \
    --pred_fpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/pred.kwcoco.json \
    --out_dpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate_3class/confusion_analysis