#Also see: ~/code/bioharn/dev/learn_tools/fish_detectors.sh

TRAIN_FPATH=/data/dvc-repos/viame_dvc/Benthic/habcam_2015_2018_2019.kwcoco.json
VALI_FPATH=/data/dvc-repos/viame_dvc/Benthic/US_NE_2017_CFF_HABCAM/data.kwcoco.json

python -m bioharn.detect_fit \
    --name=bioharn-flatfish-rgb-v10 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb" \
    --window_dims=416,416 \
    --input_dims=832,832 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=ReduceLROnPlateau-p15-c15 \
    --max_epoch=10000 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --backbone_init=url \
    --workers=3 \
    --xpu=auto \
    --batch_size=6 \
    --num_batches=2000 \
    --sampler_backend=None \
    --num_vali_batches=500 \
    --with_mask=False \
    --balance=None \
    --bstep=8
