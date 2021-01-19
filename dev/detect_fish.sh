
TRAIN_FPATH=/data/matt.dawkins/Training3/deep_training/training_truth.json
VALI_FPATH=/data/matt.dawkins/Training3/deep_training/validation_truth.json 

kwcoco stats --src $TRAIN_FPATH $VALI_FPATH

srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=priority --account=noaa --mem 30000 \
python -m bioharn.detect_fit \
    --name=bioharn-fish-hrmask18-rgb-motion-v1 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb,motion" \
    --window_dims=512,512 \
    --input_dims=768,768 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=step-12-22 \
    --max_epoch=400 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=3 \
    --xpu=auto \
    --batch_size=6 \
    --num_batches=2000
    --num_vali_batches=500
    --balance=None \
    --bstep=8

srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=priority --account=noaa --mem 30000 \
python -m bioharn.detect_fit \
    --name=bioharn-fish-hrmask18-rgb-motion-v2 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb,motion" \
    --window_dims=768,768 \
    --input_dims=768,768 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=step-12-22 \
    --max_epoch=400 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=3 \
    --xpu=auto \
    --batch_size=6 \
    --num_batches=2000
    --num_vali_batches=500
    --balance=None \
    --bstep=8


srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=priority --account=noaa --mem 30000 \
python -m bioharn.detect_fit \
    --name=bioharn-fish-hr18-rgb-motion-v3 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --channels="rgb,motion" \
    --window_dims=768,768 \
    --input_dims=768,768 \
    --window_overlap=0.3 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=step-12-22 \
    --max_epoch=400 \
    --augment=complex \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=3 \
    --xpu=auto \
    --batch_size=6 \
    --num_batches=2000
    --num_vali_batches=500
    --with_mask=False \
    --balance=None \
    --bstep=8
