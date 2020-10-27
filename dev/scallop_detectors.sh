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


--train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
--vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \

python ~/code/bioharn/dev/coco_cli/coco_add_dummy_segmentations.py \
    --src $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --dst $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json

python ~/code/bioharn/dev/coco_cli/coco_add_dummy_segmentations.py \
    --src $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --dst $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json

kwcoco stats --src $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json --annot_attrs=True

kwcoco stats --src $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json --annot_attrs=True

python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-disp-habcam-v2 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish" \
    --channels="rgb,disparity" \
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
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=8 \
    --xpu=auto \
    --batch_size=2 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-only-habcam-v3 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish" \
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
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=8 \
    --xpu=auto \
    --batch_size=2 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8


python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-disp-habcam-v4 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish" \
    --channels="rgb,disparity" \
    --window_dims=768,768 \
    --input_dims=window \
    --window_overlap=0.5 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=step-12-22 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=8 \
    --xpu=1 \
    --batch_size=2 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-only-habcam-v5 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish" \
    --channels="rgb" \
    --window_dims=768,768 \
    --input_dims=window \
    --window_overlap=0.5 \
    --arch=MM_HRNetV2_w18_MaskRCNN \
    --schedule=step-12-22 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --optim=sgd \
    --lr=1e-3 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=url \
    --workers=8 \
    --xpu=0 \
    --batch_size=2 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8
