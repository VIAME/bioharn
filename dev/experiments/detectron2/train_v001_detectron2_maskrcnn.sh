#!/bin/bash
__doc__="
FasterRCNN trained on toothbrush

References:
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    https://github.com/facebookresearch/detectron2/issues/2442

SeeAlso:
    ~/code/geowatch/geowatch/tasks/detectron2/fit.py
"
export CUDA_VISIBLE_DEVICES="1,"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=HABCAM-FISH
EXPERIMENT_NAME="viame2024-train_baseline_maskrcnn_v001"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-v01.mscoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v01.mscoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v01.mscoco.json
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

mkdir -p "$DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: $EXPERIMENT_NAME
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
base: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
init: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
" > "$DEFAULT_ROOT_DIR"/train_config_v3.yaml
cat "$DEFAULT_ROOT_DIR"/train_config_v3.yaml
python -m geowatch.tasks.detectron2.fit --config "$DEFAULT_ROOT_DIR"/train_config_v3.yaml

# Check which models exist
ls "$DEFAULT_ROOT_DIR"/*/model_*.pth


python -m geowatch.tasks.detectron2.predict \
    --checkpoint_fpath "$HOME"/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_4e4fb30b/model_0119999.pth \
    --src_fpath "$TEST_FPATH" \
    --base "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" \
    --dst_fpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate/pred.kwcoco.json \
    --workers=4

kwcoco eval_detections \
    --true_dataset "$TEST_FPATH" \
    --pred_dataset "$DEFAULT_ROOT_DIR"/oneoff_evaluate/pred.kwcoco.json \
    --out_dpath "$DEFAULT_ROOT_DIR"/oneoff_evaluate/coco_metrics



export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_habcam_detectron_evals


echo "
- $HOME/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_4e4fb30b/model_0119999.pth
- $HOME/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_4e4fb30b/model_0114999.pth
- $HOME/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_4e4fb30b/model_0004999.pth
- $HOME/data/dvc-repos/viame_dvc/experiments/training/toothbrush/joncrall/HABCAM-FISH/runs/viame2024-train_baseline_maskrcnn_v001/v_4e4fb30b/model_0049999.pth
" > "$HOME"/code/geowatch.tasks.detectron2/experiments/detectron_models.yaml

# specified models
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_EXPT_DPATH=$HOME/data/dvc-repos/viame_dvc/experiments
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v01.mscoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v01.mscoco.json
EVAL_PATH=$DVC_EXPT_DPATH/_habcam_detectron_evals

kwcoco info "$VALI_FPATH" -g1

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'geowatch.tasks.detectron2.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - $HOME/code/geowatch.tasks.detectron2/experiments/detectron_models.yaml
            detectron_pred.src_fpath:
                - $VALI_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
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
