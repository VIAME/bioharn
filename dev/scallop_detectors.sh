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

# Bad runs:
#vim $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5/bcifnsvt/deploy_MM_HRNetV2_w18_MaskRCNN_bcifnsvt_029_KYHMWC.zip
#vim $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v4/sydbgnxj/deploy_MM_HRNetV2_w18_MaskRCNN_sydbgnxj_023_RLQZTH.zip
#$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5/bcifnsvt/deploy_MM_HRNetV2_w18_MaskRCNN_bcifnsvt_029_KYHMWC.zip, \
#$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v4/sydbgnxj/deploy_MM_HRNetV2_w18_MaskRCNN_sydbgnxj_023_RLQZTH.zip, \

python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-only-habcam-v6 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
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
    --xpu=0,1 \
    --batch_size=4 \
    --num_batches=2000 \
    --balance=None \
    --bstep=1

python -m bioharn.detect_fit \
    --nice=bioharn-det-hrmask18-rgb-disp-habcam-v7 \
    --workdir=$HOME/work/bioharn \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
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
    --workers=0 \
    --xpu=1,0 \
    --batch_size=4 \
    --num_batches=2000 \
    --balance=None \
    --bstep=8


python ~/code/bioharn/bioharn/detect_eval.py --xpu=1 --workers=4 --batch_size=8 --draw=0 --verbose=3 --sampler_backend=cog \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    --deployed="[\
        $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6/hldsgogn/deploy_MM_HRNetV2_w18_MaskRCNN_hldsgogn_032_FURHIT.zip, \
        $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/odcvldiy/deploy_MM_HRNetV2_w18_MaskRCNN_odcvldiy_031_EUEOKJ.zip, \
        $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v4/sydbgnxj/deploy_MM_HRNetV2_w18_MaskRCNN_sydbgnxj_023_RLQZTH.zip, \
    ]"



# SANITY CHECK
# download a known "good" model and ensure tha tit predicts correctly
xdoctest -m bioharn.detect_predict DetectPredictor --network
python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=cog \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    --deployed=$HOME/.cache/viame/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=744, --draw=True --workers=0


ls $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6
ls $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/odcvldiy/deploy_MM_HRNetV2_w18_MaskRCNN_odcvldiy_031_EUEOKJ.zip

 $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6/hldsgogn/deploy_MM_HRNetV2_w18_MaskRCNN_hldsgogn_032_FURHIT.zip



 


# NEW MODEL CHECK:
# This should work, but it doesn't seem like it is
python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=cog \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5/bcifnsvt/deploy_MM_HRNetV2_w18_MaskRCNN_bcifnsvt_029_KYHMWC.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=744, --draw=True --workers=0

python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=cog \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6/hldsgogn/deploy_MM_HRNetV2_w18_MaskRCNN_hldsgogn_032_FURHIT.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=744, --draw=True --workers=0

python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=cog \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali_dummy_sseg.mscoco.json \
    --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/odcvldiy/deploy_MM_HRNetV2_w18_MaskRCNN_odcvldiy_031_EUEOKJ.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=744, --draw=True --workers=0

python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=None \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/cog/201503.20150621.005549952.129166_left.cog.tif \
    --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5/bcifnsvt/deploy_MM_HRNetV2_w18_MaskRCNN_bcifnsvt_029_KYHMWC.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --draw=True --workers=0

    #--dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/cog/201503.20150621.005549952.129166_left.cog.tif \

python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=None \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --deployed=$HOME/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5-adapt/udmzrkmb/deploy_MM_HRNetV2_w18_MaskRCNN_udmzrkmb_003_FKJTWB.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=1, --draw=True --workers=0


python -m bioharn.detect_predict \
    --xpu=1 --batch_size=1 --verbose=3 --sampler_backend=None \
    --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train_dummy_sseg.mscoco.json \
    --deployed=$HOME/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6/hldsgogn/deploy_MM_HRNetV2_w18_MaskRCNN_hldsgogn_032_FURHIT.zip \
    --out_dpath="./tmp/tmp-pred" --enable_cache=False --gids=1, --draw=True --workers=0

                                     



zipedit(){
    __heredoc__='
    https://superuser.com/questions/647674/is-there-a-way-to-edit-files-inside-of-a-zip-file-without-explicitly-extracting/647795

    You can just vim a zip file directly
    '
    ARCHIVE_FPATH=$1
    FNAME=$2
    TMP_DPATH=/tmp
    echo "Usage: zipedit archive.zip file.txt"
    unzip "$ARCHIVE_FPATH" "$FNAME" -d $TMP_DPATH
    vim "$TMP_DPATH/$FNAME" && zip -j --update "$ARCHIVE"  "$TMP_DPATH/$FNAME" 
}


test-predict-quality(){
    python -m bioharn.detect_fit \
        --nice=bioharn-det-hrmask18-rgb-only-shapes \
        --workdir=$HOME/work/bioharn \
        --train_dataset=special:shapes256 \
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
        --num_batches=auto \
        --balance=None \
        --bstep=8

    
    cd $HOME/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-shapes/nchuyhmv
    python -m bioharn.detect_predict \
        --xpu=auto --batch_size=1 --verbose=3 --sampler_backend=cog \
        --dataset=$HOME/.cache/kwcoco/toy_dset/shapes_8_cbdda30a08151dc32feab1479be073368c8e325b/images/img_00002.png \
        --deployed=deploy.zip \
        --out_dpath="./tmp/tmp-pred" --enable_cache=False --draw=True --workers=0

    python -m bioharn.detect_fit \
        --nice=bioharn-det-hrmask18-rgb-only-shapes-v2 \
        --workdir=$HOME/work/bioharn \
        --train_dataset=special:shapes256 \
        --channels="rgb" \
        --window_dims=768,768 \
        --input_dims=window \
        --window_overlap=0.5 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=step-12-22 \
        --max_epoch=400 \
        --augment=complex \
        --init=$HOME/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-shapes/nchuyhmv/deploy.zip \
        --optim=sgd \
        --lr=1e-3 \
        --multiscale=False \
        --normalize_inputs=True \
        --backbone_init=url \
        --workers=8 \
        --xpu=0 \
        --batch_size=2 \
        --num_batches=auto \
        --balance=None \
        --bstep=8

    cd $HOME/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-shapes-v2/gycvxcfv
    python -m bioharn.detect_predict \
        --xpu=auto --batch_size=1 --verbose=3 --sampler_backend=cog \
        --dataset=$HOME/.cache/kwcoco/toy_dset/shapes_8_cbdda30a08151dc32feab1479be073368c8e325b/images/img_00002.png \
        --deployed=deploy.zip \
        --out_dpath="./tmp/tmp-pred" --enable_cache=False --draw=True --workers=0


    python -m bioharn.detect_fit \
        --nice=bioharn-det-hrmask18-rgb-disp-shapes-v3 \
        --workdir=$HOME/work/bioharn \
        --train_dataset=special:vidshapes256-aux \
        --channels="rgb,disparity" \
        --window_dims=768,768 \
        --input_dims=window \
        --window_overlap=0.5 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=step-12-22 \
        --max_epoch=400 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-3 \
        --multiscale=False \
        --normalize_inputs=True \
        --backbone_init=url \
        --workers=0 \
        --xpu=0 \
        --batch_size=2 \
        --num_batches=auto \
        --balance=None \
        --bstep=8

}
