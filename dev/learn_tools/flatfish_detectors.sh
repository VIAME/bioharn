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



# On Numenor

cd $HOME/data/dvc-repos/viame_dvc/public/Benthic

REMOTE_URI=viame.kitware.com
dvc remote modify --local viame url ssh://$REMOTE_URI/data/dvc-caches/viame_dvc 

# Use the local dir
dvc config cache.dir --unset

dvc remote default viame
dvc pull  

find . -iname "*.dvc" -type f

dvc pull ./public/Benthic/US_NE_2018_CFF_HABCAM/annotations.kwcoco.json.dvc \
    ./public/Benthic/US_NE_2019_CFF_HABCAM/annotations.kwcoco.json.dvc \
    ./public/Benthic/US_NE_2015_NEFSC_HABCAM/annotations.kwcoco.json.dvc \
    ./public/Benthic/US_NE_2019_CFF_HABCAM_PART2/annotations.kwcoco.json.dvc \
    ./public/Benthic/US_NE_2017_CFF_HABCAM/annotations.kwcoco.json.dvc

#srun -c 2 --gres=gpu:0 
dvc pull ./public/Benthic/US_NE_2018_CFF_HABCAM/Left.dvc \
 ./public/Benthic/US_NE_2019_CFF_HABCAM/Left.dvc \
 ./public/Benthic/US_NE_2017_CFF_HABCAM/Left.dvc \
 ./public/Benthic/US_NE_2015_NEFSC_HABCAM/Cog.dvc \
 ./public/Benthic/US_NE_2015_NEFSC_HABCAM/Disparities.dvc \
 ./public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected.dvc \
 ./private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/Left.dvc \
 ./private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/Disparity.dvc \
 ./private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/Corrected.dvc \
 ./private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/Raw.dvc


./public/Benthic/US_NE_2018_CFF_HABCAM/Raws.dvc
./public/Benthic/US_NE_2019_CFF_HABCAM/Raws.dvc
./public/Benthic/US_NE_2017_CFF_HABCAM/Raws.dvc


dvc pull \
    ./public/Benthic/US_NE_2018_CFF_HABCAM/annotations.csv.dvc \
    ./public/Benthic/US_NE_2019_CFF_HABCAM/annotations.csv.dvc \
    ./public/Benthic/US_NE_2015_NEFSC_HABCAM/annotations.csv.dvc \
    ./public/Benthic/US_NE_2019_CFF_HABCAM_PART2/annotations.csv.dvc \
    ./public/Benthic/US_NE_2017_CFF_HABCAM/annotations.csv.dvc \
    ./private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/flatfish14.habcam_csv.dvc \


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations.kwcoco.json

kwcoco validate $TRAIN_FPATH
kwcoco validate $VALI_FPATH

srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=community --account=noaa --mem 30000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v13 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
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
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/models/deploy_MM_HRNetV2_w18_MaskRCNN_kqlgozei_003_MSOUGL.zip \
        --workers=3 \
        --xpu=auto \
        --batch_size=8 \
        --num_batches=4000 \
        --sampler_backend=None \
        --num_vali_batches=1000 \
        --with_mask=False \
        --balance=None \
        --bstep=4


# Viame DVC on numenor quickstart
mkdir -p $HOME/tmp
cd $HOME/tmp
git clone git@gitlab.kitware.com:viame/viame_dvc.git
cd viame_dvc
dvc checkout --recursive public/Benthic


cd $DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/
kwcoco subset annotations.kwcoco.json --include_categories=flatfish --dst=annotations_flatfish.kwcoco.json


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019_flatfish.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=community --account=noaa --mem 30000 \
    python -m bioharn.detect_fit \
        --name=bioharn-only-flatfish-rgb-from-v11-v14 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
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
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/models/deploy_MM_HRNetV2_w18_MaskRCNN_kqlgozei_003_MSOUGL.zip \
        --workers=3 \
        --xpu=auto \
        --batch_size=10 \
        --sampler_backend=None \
        --with_mask=False \
        --balance=None \
        --bstep=1

srun --gres=gpu:rtx6000:1 --cpus-per-task=4 --partition=community --account=noaa --mem 30000 \
    python -m bioharn.detect_fit \
        --name=bioharn-only-flatfish-rgb-from-v11-v15 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=608,608 \
        --input_dims=832,832 \
        --window_overlap=0.3 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p5-c5 \
        --max_epoch=10000 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/models/deploy_MM_HRNetV2_w18_MaskRCNN_kqlgozei_003_MSOUGL.zip \
        --workers=3 \
        --xpu=auto \
        --batch_size=10 \
        --sampler_backend=None \
        --with_mask=False \
        --balance=None \
        --bstep=3


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations.kwcoco.json

kwcoco validate $TRAIN_FPATH
kwcoco validate $VALI_FPATH

srun --gres=gpu:rtx6000:1 --cpus-per-task=2 --partition=community --account=noaa --mem 20000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v19 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=832,832 \
        --input_dims=832,832 \
        --window_overlap=0.3 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=10000 \
        --augment=complex \
        --optim=sgd \
        --lr=3e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v13/nryfnjlw/deploy_bioharn-flatfish-rgb-v13_nryfnjlw_001_CSKAGJ.zip \
        --workers=1 \
        --xpu=auto \
        --batch_size=10 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=100 \
        --with_mask=False \
        --balance=None \
        --bstep=3

srun --gres=gpu:rtx6000:2 --cpus-per-task=4 --partition=community --account=noaa --mem 50000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v18-no-warmup \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=1024,1024 \
        --input_dims=1024,1024 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=10000 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v17/pohwrmdi/checkpoints/_epoch_00000008.pt \
        --workers=3 \
        --xpu=0,1 \
        --batch_size=12 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=100 \
        --with_mask=False \
        --balance=None \
        --bstep=4 \
        --warmup_iters=0


dvc pull private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json.dvc

kwcoco validate private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json

kwcoco reroot $DVC_REPO/public/Benthic/habcam_2015_2018_2019_flatfish.kwcoco.json --dst $DVC_REPO/public/Benthic/habcam_2015_2018_2019_flatfish.kwcoco.json.abs --absolute True
kwcoco reroot $DVC_REPO/private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json --dst $DVC_REPO/private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json.abs --absolute True

kwcoco union \
    --src $DVC_REPO/public/Benthic/habcam_2015_2018_2019_flatfish.kwcoco.json.abs \
    $DVC_REPO/private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json.abs \
    --dst $DVC_REPO/habcam_2014_2015_2018_2019_flatfish.kwcoco.json.abs

kwcoco reroot $DVC_REPO/habcam_2014_2015_2018_2019_flatfish.kwcoco.json.abs --absolute False --old_prefix="$DVC_REPO" --new_prefix="" \
    --dst=$DVC_REPO/habcam_2014_2015_2018_2019_flatfish.kwcoco.json
dvc add habcam_2014_2015_2018_2019_flatfish.kwcoco.json

DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/habcam_2014_2015_2018_2019_flatfish.kwcoco.json.abs
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json

cd $DVC_REPO
#TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json

kwcoco validate --corrupted=True $TRAIN_FPATH

srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 15000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v19-warmup-0 \
        --warmup_iters=0 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=832,832 \
        --input_dims=832,832 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v16/goqjyouc/deploy_bioharn-flatfish-rgb-v16_goqjyouc_001_MINKUB.zip \
        --workers=2 \
        --xpu=0 \
        --batch_size=8 \
        --num_batches=auto \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=4 \
        --timeout=86400

srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 15000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v19-warmup-30 \
        --warmup_iters=30 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=832,832 \
        --input_dims=832,832 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v16/goqjyouc/deploy_bioharn-flatfish-rgb-v16_goqjyouc_001_MINKUB.zip \
        --workers=2 \
        --xpu=0 \
        --batch_size=8 \
        --num_batches=auto \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=4 \
        --timeout=86400


srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 15000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v19-warmup-100 \
        --warmup_iters=100 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=832,832 \
        --input_dims=832,832 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v16/goqjyouc/deploy_bioharn-flatfish-rgb-v16_goqjyouc_001_MINKUB.zip \
        --workers=2 \
        --xpu=0 \
        --batch_size=8 \
        --num_batches=auto \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=4 \
        --timeout=86400

srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 15000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-rgb-v19-warmup-800 \
        --warmup_iters=800 \
        --workdir=$HOME/data/dvc-repos/viame_dvc/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=832,832 \
        --input_dims=832,832 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$HOME/remote/numenor/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/bioharn-flatfish-rgb-v16/goqjyouc/deploy_bioharn-flatfish-rgb-v16_goqjyouc_001_MINKUB.zip \
        --workers=2 \
        --xpu=0 \
        --batch_size=8 \
        --num_batches=auto \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=4 \
        --timeout=86400


# What can be evaluated:
ls $HOME/data/dvc-repos/viame_dvc/work/bioharn

ls /home/khq.kitware.com/jon.crall/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/eval
ls /home/khq.kitware.com/jon.crall/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/*/checkpoints/*
ls /home/khq.kitware.com/jon.crall/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/*/deploy_*

ls /home/khq.kitware.com/jon.crall/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/*/eval

DVC_REPO=$HOME/data/dvc-repos/viame_dvc
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_eval \
        --workers=2 \
        --dataset=$VALI_FPATH \
        "--deployed=[
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000003.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000003.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000003.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000004.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000040.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000040.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000040.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000070.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000070.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000070.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000075.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-30/fkcvtwxr/checkpoints/_epoch_00000075.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-800/nabysaeb/checkpoints/_epoch_00000075.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-100/cuvszthu/checkpoints/_epoch_00000075.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints
        ]"


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
kwcoco reroot \
    --src $DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json \
    --dst $DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json.abs \
    --absolute True

kwcoco union --src \
    $DVC_REPO/public/Benthic/habcam_2015_2018_2019.kwcoco.json.abs \
    $DVC_REPO/private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/annotations.kwcoco.json.abs \
    --dst $DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json.abs

jq .images[0] $DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json.abs
kwcoco reroot $DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json.abs --absolute=False --dst $DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json
jq .images[10000] $DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json
dvc add habcam_2014_2015_2018_2019.kwcoco.json

kwcoco stats habcam_2014_2015_2018_2019.kwcoco.json

DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/habcam_2014_2015_2018_2019.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations.kwcoco.json

srun --gres=gpu:rtx6000:2 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_fit \
        --name=bioharn-allclass-rgb-v20\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=rmsprop \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-rgb-v19-warmup-0/udquckjh/checkpoints/_epoch_00000045.pt \
        --workers=2 \
        --xpu=0,1 \
        --batch_size=16 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=100 \
        --with_mask=False \
        --balance=None \
        --bstep=4


sacct -o "Account,User,ReqMem,JobID,JobName,ExitCode,ReqTRES,State" -j 1603
sacct -o "Account,User,ReqMem,JobID,JobName,ExitCode,ReqTRES,State" -j 1606


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_eval \
        --workers=2 \
        --draw=0 \
        --dataset=$VALI_FPATH \
        "--deployed=[
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000004.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000005.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000006.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000011.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000016.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000017.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000021.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000023.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000040.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000070.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000071.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000072.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000073.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000074.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000075.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000076.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000077.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000078.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000079.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000080.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000081.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000082.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000083.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000084.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000085.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000086.pt,\
        ]"


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/habcam_2014_2015_2018_2019_flatfish.kwcoco.json.abs
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:2 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_fit \
        --name=bioharn-flatfish-finetune-rgb-v21\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=sgd \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=imagenet \
        --init=$DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000081.pt \
        --workers=2 \
        --xpu=0,1 \
        --batch_size=16 \
        --num_batches=auto \
        --sampler_backend=None \
        --num_vali_batches=auto \
        --with_mask=False \
        --balance=None \
        --bstep=4


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_eval \
        --workers=2 \
        --draw=0 \
        --dataset=$VALI_FPATH \
        "--deployed=[
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000003.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000004.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000005.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000006.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000008.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000012.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000040.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000045.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000050.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000055.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000060.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000061.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000062.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000063.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000064.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000065.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000066.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000067.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000068.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000069.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000070.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000071.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000072.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000073.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000074.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000075.pt,\
        ]"


python -m torch_liberator \
    --model MM_HRNetV2_w18_MaskRCNN_9aeb83.py \
    --weights $DVC_REPO/work/bioharn/fit/runs/bioharn-flatfish-finetune-rgb-v21/uffjlobk/checkpoints/_epoch_00000006.pt \
    --info train_info.json \
    --dst deploy_bioharn-flatfish-finetune-rgb-v21_uffjlobk_006_custom.zip

dvc add deploy_bioharn-flatfish-finetune-rgb-v21_uffjlobk_006_custom.zip

cd $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww

python -m torch_liberator \
    --model MM_HRNetV2_w18_MaskRCNN_9aeb83.py \
    --weights $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-rgb-v20/vitklgww/checkpoints/_epoch_00000081.pt \
    --info train_info.json \
    --dst deploy_bioharn-allclass-rgb-v20_vitklgww_081_custom.zip

dvc add deploy_bioharn-allclass-rgb-v20_vitklgww_081_custom.zip




############

# WORK ON FUSION


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
kwcoco union --src \
    $DVC_REPO/public/Benthic/US_NE_2015_NEFSC_HABCAM/annotations_disp.kwcoco.json \
    $DVC_REPO/public/Benthic/US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json \
    $DVC_REPO/public/Benthic/US_NE_2019_CFF_HABCAM/annotations_disp.kwcoco.json \
    $DVC_REPO/public/Benthic/US_NE_2019_CFF_HABCAM_PART2/annotations_disp.kwcoco.json \
    --dst $DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json 


jq .images[0] $DVC_REPO/public/Benthic/US_NE_2015_NEFSC_HABCAM/annotations_disp.kwcoco.json
jq .images[0] $DVC_REPO/public/Benthic/US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json

jq .images[30000] $DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json
jq .images[0] $DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json

kwcoco validate $DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json

cd $DVC_REPO/public/Benthic
kwcoco subset habcam_2015_2018_2019_disp.kwcoco.json --include_categories=flatfish --dst=habcam_2015_2018_2019_disp_flatfish.kwcoco.json
dvc add habcam_2015_2018_2019_disp.kwcoco.json habcam_2015_2018_2019_disp_flatfish.kwcoco.json

cd $DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/
kwcoco subset annotations_disp.kwcoco.json --include_categories=flatfish --dst=annotations_disp_flatfish.kwcoco.json
kwcoco stats annotations_disp_flatfish.kwcoco.json
dvc add annotations_disp_flatfish.kwcoco.json


dvc pull public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp_flatfish.kwcoco.json.dvc \
    public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json.dvc \
    public/Benthic/habcam_2015_2018_2019_disp_flatfish.kwcoco.json.dvc

DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp.kwcoco.json

#srun --gres=gpu:rtx6000:2 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
# ON NAMEK
    python -m bioharn.detect_fit \
        --name=bioharn-allclass-rgb-v23\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb,disparity" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=adam \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=True \
        --workers=2 \
        --xpu=1 \
        --batch_size=2 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=100 \
        --with_mask=False \
        --balance=None \
        --bstep=4


# ON NUMENOR
DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_fit \
        --name=bioharn-allclass-rgb-v24\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb,disparity" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=adam \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=True \
        --workers=2 \
        --xpu=0 \
        --batch_size=4 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=100 \
        --with_mask=False \
        --balance=None \
        --bstep=4


VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_eval \
        --workers=2 \
        --draw=0 \
        --dataset=$VALI_FPATH \
        "--deployed=[
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000000.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000001.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000002.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000003.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000004.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000005.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000007.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000009.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000010.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000012.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000014.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000015.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000016.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000020.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000025.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000030.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000031.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000032.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000033.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000034.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000035.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000036.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000037.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000038.pt,\
            $DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000039.pt,\
        ]"


# ON NUMENOR
DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:1 --cpus-per-task=3 --partition=priority --account=noaa --mem 20000 \
    python -m bioharn.detect_fit \
        --name=bioharn-allclass-rgb-v25\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb,disparity" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=rmsprop \
        --init=$DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000039.pt \
        --lr=1e-4 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=True \
        --workers=2 \
        --xpu=0 \
        --batch_size=4 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=4


DVC_REPO=$HOME/data/dvc-repos/viame_dvc
TRAIN_FPATH=$DVC_REPO/public/Benthic/habcam_2015_2018_2019_disp.kwcoco.json
VALI_FPATH=$DVC_REPO/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp_flatfish.kwcoco.json
srun --gres=gpu:rtx6000:2 --cpus-per-task=3 --partition=priority --account=noaa --mem 30000 \
    python -m bioharn.detect_fit \
        --name=bioharn-allclass-rgb-v26\
        --warmup_iters=0 \
        --workdir=$DVC_REPO/work/bioharn \
        --train_dataset=$TRAIN_FPATH \
        --vali_dataset=$VALI_FPATH \
        --channels="rgb,disparity" \
        --window_dims=928,928 \
        --input_dims=928,928 \
        --window_overlap=0.0 \
        --arch=MM_HRNetV2_w18_MaskRCNN \
        --schedule=ReduceLROnPlateau-p15-c15 \
        --max_epoch=100 \
        --augment=complex \
        --optim=adam \
        --init=$DVC_REPO/work/bioharn/fit/runs/bioharn-allclass-fusion-hrnet18-habcam-v2/lyxlmrfz/checkpoints/_epoch_00000039.pt \
        --lr=1e-3 \
        --multiscale=False \
        --patience=75 \
        --normalize_inputs=True \
        --workers=4 \
        --xpu=0,1 \
        --batch_size=8 \
        --num_batches=1000 \
        --sampler_backend=None \
        --num_vali_batches=10 \
        --with_mask=False \
        --balance=None \
        --bstep=1
