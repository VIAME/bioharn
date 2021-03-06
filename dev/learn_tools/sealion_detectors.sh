



__heredoc__ = """

Dataset Prep:

    See Also ../data_tools




    kwcoco modify_categories \
        --src=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json  \
        --dst=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_vali.mscoco.json  \
        --rename='Bull:sealion,Dead Pup:sealion,Fem:sealion,Juv:sealion,Pup:sealion,SAM:sealion'  

    kwcoco modify_categories \
        --src=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json  \
        --dst=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_train.mscoco.json  \
        --rename='Bull:sealion,Dead Pup:sealion,Fem:sealion,Juv:sealion,Pup:sealion,SAM:sealion'  

    kwcoco stats --src $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_vali.mscoco.json
    kwcoco stats --src $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_train.mscoco.json

    # TODO: make sure nothing important is in this directory (use rsync to try and compare if anything is different / non existing)
    mv US_ALASKA_MML_SEALION US_ALASKA_MML_SEALION_old_bad_rsync
    ln -s $HOME/data/noaa/US_ALASKA_MML_SEALION US_ALASKA_MML_SEALION

    rsync -avrPRL --exclude 'detections' --exclude 'BLACKEDOUT' --exclude 'COUNTED' --exclude 'KITWARE' viame:data/./US_ALASKA_MML_SEALION $HOME/data
    rsync -avrPRL --exclude 'detections' --exclude 'BLACKEDOUT' viame:data/./US_ALASKA_MML_SEALION $HOME/data/noaa

    rsync -avcn -delete \
        $HOME/data/US_ALASKA_MML_SEALION_old_bad_rsync/edits/ \
        $HOME/data/US_ALASKA_MML_SEALION/edits  

Sealion Dataset:
    $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_vali.mscoco.json 

    $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json
    $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json

    $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json
    $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json


Dataset Stats:
    kwcoco stats --src $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json

Trained Sealion Models:

    $HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v3/hpjbgxjn/deploy_MM_CascadeRCNN_hpjbgxjn_045_JTZMSY.zip

"""


cp $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json 

cp $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json


### TRAINING ###

python -m bioharn.detect_fit \
    --name=sealion-cascade-v3 \
    --workdir=$HOME/work/sealions \
    --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_train.mscoco.json \
    --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --bstep=3


python -m bioharn.detect_fit \
    --name=sealion-cascade-v5 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json \
    --vali_dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=/home/joncrall/work/sealions/fit/runs/sealion-cascade-v5/hrhgavoc/explit_checkpoints/_epoch_00000001_2020-06-17T184850+5.pt \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=10 \
    --sampler_backend=None \
    --balance=None \
    --num_batches=1000 \
    --bstep=3


python -m bioharn.detect_fit \
    --name=sealion-cascade-manual-coarse-v6 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_train.mscoco.json \
    --vali_dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=10 \
    --sampler_backend=None \
    --balance=None \
    --num_batches=1000 \
    --bstep=3


python -m bioharn.detect_fit \
    --name=sealion-cascade-v7 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9_train.mscoco.json \
    --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --bstep=3

python -m bioharn.detect_fit \
    --name=sealion-cascade-v8 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9_train.mscoco.json \
    --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --bstep=3



### EVALUATION ###

python ~/code/bioharn/bioharn/detect_eval.py \
    --dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_vali.mscoco.json \
    "--deployed=[\
        $HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v5/bzmjrthj/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip,\
    ]" \
    --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.5 --xpu=auto --window_overlap=0.5

python ~/code/bioharn/bioharn/detect_eval.py \
    --dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_vali.mscoco.json \
    "--deployed=[\
        $HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v5/bzmjrthj/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip,\
    ]" \
    --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.5 --xpu=auto --window_overlap=0.5


python ~/code/bioharn/bioharn/detect_eval.py \
    --dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_coarse_vali.mscoco.json \
    "--deployed=[\
        $HOME/work/sealions/fit/name/sealion-cascade-manual-coarse-v6/deploy_MM_CascadeRCNN_igyhuonn_040_GGVZLT.zip,\
        $HOME/work/sealions/fit/name/sealion-cascade-manual-coarse-v6/deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS.zip,\
    ]" \
    --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.5 --xpu=0,1 --window_overlap=0.5


python ~/code/bioharn/bioharn/detect_eval.py \
    --dataset=$HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9_vali.mscoco.json \
    "--deployed=[\
        $HOME/remote/namek/work/sealions/fit/nice/untitled/deploy_MM_CascadeRCNN_jpwjmhhp_023_THXFTS.zip,\
    ]" \
    --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.5 --xpu=auto --window_overlap=0.5
# todo: ignore unknown


#(py38) joncrall@viame:~/code/kwcoco$     
python -m bioharn.detect_fit         --name=sealion-cascade-v9         --workdir=$HOME/work/sealions         --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json         --vali_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json         --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache         --schedule=ReduceLROnPlateau-p5-c5         --max_epoch=400         --augment=complex         --init=noop         --arch=cascade         --channels="rgb"         --optim=sgd         --lr=1e-3         --window_dims=512,512         --input_dims=window         --window_overlap=0.5         --multiscale=False         --normalize_inputs=imagenet         --workers=8         --xpu=auto         --batch_size=4         --sampler_backend=None         --num_batches=1000         --balance=None         --bstep=16

python -m bioharn.detect_fit         --name=sealion-cascade-v11         --workdir=$HOME/work/sealions         --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json         --vali_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json         --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache         --schedule=ReduceLROnPlateau-p5-c5         --max_epoch=400         --augment=complex         --init=noop         --arch=cascade         --channels="rgb"         --optim=AdaBound         --lr=1e-3         --window_dims=512,512         --input_dims=window         --window_overlap=0.5         --multiscale=False         --normalize_inputs=imagenet         --workers=8         --xpu=auto         --batch_size=4         --sampler_backend=None         --num_batches=1000         --balance=None         --bstep=16



#(py38) joncrall@namek:~/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION$     
python -m bioharn.detect_fit         --name=sealion-cascade-v10         --workdir=$HOME/work/sealions         --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache         --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json         --vali_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json         --schedule=ReduceLROnPlateau-p5-c5         --max_epoch=400         --augment=complex         --init=noop         --arch=cascade         --channels="rgb"         --optim=sgd         --lr=1e-3         --window_dims=512,512         --input_dims=window         --window_overlap=0.5         --multiscale=False         --normalize_inputs=imagenet         --workers=8         --xpu=auto         --batch_size=4         --sampler_backend=None         --num_batches=1000         --balance=None         --bstep=16

python -m bioharn.detect_fit         --name=sealion-cascade-v12         --workdir=$HOME/work/sealions         --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache         --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json         --vali_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json         --schedule=ReduceLROnPlateau-p5-c5         --max_epoch=400         --augment=complex         --init=noop         --arch=cascade         --channels="rgb"         --optim=AdaBound         --lr=1e-3         --window_dims=512,512         --input_dims=window         --window_overlap=0.5         --multiscale=False         --normalize_inputs=imagenet         --workers=8         --xpu=auto         --batch_size=4         --sampler_backend=None         --num_batches=1000         --balance=None         --bstep=16


kwcoco stats $HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json --annot_attrs=True --image_attrs=True
kwcoco stats $HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json --annot_attrs=True --image_attrs=True

cd $HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/
kwcoco conform sealions_train_v9.kwcoco.json --dst=sealions_train_v9_.kwcoco.json
kwcoco conform sealions_vali_v9.kwcoco.json --dst=sealions_vali_v9_.kwcoco.json

kwcoco stats sealions_vali_v9_.kwcoco.json --annot_attrs=True --image_attrs=True
kwcoco stats sealions_train_v9_.kwcoco.json --annot_attrs=True --image_attrs=True

mv sealions_vali_v9_.kwcoco.json sealions_vali_v9.kwcoco.json
mv sealions_train_v9_.kwcoco.json sealions_train_v9.kwcoco.json


python -m bioharn.detect_fit \
    --name=sealion-mask-test-v2 \
    --workdir=$HOME/work/sealions\
    --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache\
    --train_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
    --vali_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=adam \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=4 \
    --xpu=auto \
    --batch_size=4 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=16


kwcoco toydata shapes256
kwcoco toydata shapes32

TRAIN_DSET="$HOME/.cache/kwcoco/demodata_bundles/shapes_256_dqikwsaiwlzjup/data.kwcoco.json"
VALI_DSET="$HOME/.cache/kwcoco/demodata_bundles/shapes_32_kahspdeebbfocp/data.kwcoco.json"

python -m bioharn.detect_fit \
    --name=shapes-mask-test-v1 \
    --workdir=$HOME/work/sealions\
    --train_dataset=$TRAIN_DSET \
    --vali_dataset=$VALI_DSET \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=adam \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=0 \
    --xpu=auto \
    --batch_size=4 \
    --sampler_backend=None \
    --num_batches=10 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=16

python -m bioharn.detect_fit \
    --name=sealion-mask-test-v3 \
    --workdir=$HOME/work/sealions\
    --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache\
    --train_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
    --vali_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
    --schedule=ReduceLROnPlateau-p12-c36 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=adam \
    --lr=3e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=10 \
    --xpu=auto \
    --batch_size=11 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=3

python -m bioharn.detect_fit \
    --name=sealion-mask-test-v4 \
    --workdir=$HOME/work/sealions\
    --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache\
    --train_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
    --vali_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
    --schedule=ReduceLROnPlateau-p12-c36 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=adam \
    --lr=2e-3 \
    --window_dims=256,256 \
    --input_dims=window \
    --window_overlap=0.3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=8 \
    --xpu=1 \
    --batch_size=24 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=3

python -m bioharn.detect_fit \
    --name=sealion-mask-test-v5 \
    --workdir=$HOME/work/sealions\
    --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache\
    --train_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
    --vali_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
    --schedule=ReduceLROnPlateau-p12-c36 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=adam \
    --lr=3e-3 \
    --window_dims=416,416 \
    --input_dims=window \
    --window_overlap=0.3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=10 \
    --xpu=auto \
    --batch_size=12 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=3


python -m bioharn.detect_fit \
    --name=sealion-mask-test-v6 \
    --workdir=$HOME/work/sealions\
    --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache\
    --train_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
    --vali_dataset=$HOME/data/dvc-repos/viame_dvc/US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
    --schedule=ReduceLROnPlateau-p12-c36 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --arch=maskrcnn \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=416,416 \
    --input_dims=window \
    --window_overlap=0.3 \
    --multiscale=False \
    --normalize_inputs=imagenet \
    --workers=10 \
    --xpu=0 \
    --batch_size=12 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --sql_cache_view=True \
    --segmentation_bootstrap=ellipse+kpts \
    --bstep=3


# Experimental model folder:
# https://data.kitware.com/#collection/58b747ec8d777f0aef5d0f6a/folder/604f7f8a2fa25629b9567898

source $HOME/internal/secrets
girder-client --api-url https://data.kitware.com/api/v1 upload 604f7f8a2fa25629b9567898 \
    $HOME/remote/namek/work/sealions/fit/name/sealion-mask-test-v6/deploy_sealion-mask-test-v6_rtzduxsw_096_JZMSVD.zip

#girder-client --api-url https://data.kitware.com/api/v1 upload 604f7f8a2fa25629b9567898 \
#    $HOME/remote/namek/work/sealions/fit/name/sealion-mask-test-v5/deploy_sealion-mask-test-v6_rtzduxsw_096_JZMSVD.zip


# Download the sealion test image (sealion_test_img_2010.jpg)
girder-client --api-url https://data.kitware.com/api/v1 download 6011a5ae2fa25629b919fe6a

# Download the sealion mask model (deploy_sealion-mask-test-v6_rtzduxsw_096_JZMSVD.zip)
girder-client --api-url https://data.kitware.com/api/v1 download 604f854a2fa25629b956b486

# Download the sealion test image (20150624_SSLC0013_C.jpg)
girder-client --api-url https://viame.kitware.com/api/v1 download 5f52a390683ae84c2a4f5cc6

python -m bioharn.detect_predict \
    --dataset=20150624_SSLC0013_C.jpg \
    --deployed=./deploy_sealion-mask-test-v6_rtzduxsw_096_JZMSVD.zip \
    --out_dpath=./test_mask_pred \
    --draw=1
