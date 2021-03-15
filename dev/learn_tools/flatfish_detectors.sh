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
    --patience=75 \
    --normalize_inputs=imagenet \
    --backbone_init=url \
    --workers=3 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=2000 \
    --sampler_backend=None \
    --num_vali_batches=500 \
    --with_mask=False \
    --balance=None \
    --bstep=8


python -m bioharn.detect_fit \
    --name=bioharn-flatfish-rgb-v11 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb" \
    --window_dims=512,512 \
    --input_dims=832,832 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=ReduceLROnPlateau-p15-c15 \
    --max_epoch=10000 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --patience=75 \
    --normalize_inputs=imagenet \
    --init=$HOME/remote/viame/work/bioharn/fit/runs/bioharn-flatfish-rgb-v10/svytnbjg/deploy_MM_HRNetV2_w18_MaskRCNN_svytnbjg_016_MYFSVM.zip \
    --workers=3 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=4000 \
    --sampler_backend=None \
    --num_vali_batches=1000 \
    --with_mask=False \
    --balance=None \
    --bstep=4


python -m bioharn.detect_fit \
    --name=bioharn-flatfish-rgb-v12 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb" \
    --window_dims=608,608 \
    --input_dims=832,832 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=ReduceLROnPlateau-p15-c15 \
    --max_epoch=10000 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --patience=75 \
    --normalize_inputs=imagenet \
    --init=$HOME/remote/viame/work/bioharn/fit/runs/bioharn-flatfish-rgb-v11/kqlgozei/deploy_MM_HRNetV2_w18_MaskRCNN_kqlgozei_003_MSOUGL.zip \
    --workers=3 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=4000 \
    --sampler_backend=None \
    --num_vali_batches=1000 \
    --with_mask=False \
    --balance=None \
    --bstep=4
