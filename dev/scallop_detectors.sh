python -m bioharn.detect_fit \
    --nice=validate_demo \
    --workdir=$HOME/work/bioharn \
    --train_dataset=special:shapes32 \
    --vali_dataset=special:shapes32 \
    --channels="rgb" \
    --window_dims=768,768 \
    --input_dims=window \
    --window_overlap=0.5 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --optim=AdaBelief \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --backbone_init=url \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --bstep=8


python ~/code/bioharn/dev/coco_cli/coco_add_dummy_segmentations.py \
    --src $HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
    --dst $HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_dummy_sseg.mscoco.json

python ~/code/bioharn/dev/coco_cli/coco_add_dummy_segmentations.py \
    --src $HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
    --dst $HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_dummy_sseg.mscoco.json


python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-coi-v1 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_dummy_sseg.mscoco.json \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
    --channels="rgb" \
    --window_dims=768,768 \
    --input_dims=window \
    --window_overlap=0.5 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --optim=AdaBelief \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --backbone_init=url \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8
