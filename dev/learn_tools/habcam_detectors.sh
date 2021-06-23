python -m bioharn.clf_fit \
    --name=bioharn-clf-rgb-v001 \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --workdir=$HOME/work/bioharn \
    --arch=resnext101 \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=256,256 \
    --normalize_inputs=True \
    --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x8d-8ba56ff5.pth \
    --workers=8 \
    --xpu=auto \
    --batch_size=32 \
    --num_batches=2000 \
    --balance=classes



python -m bioharn.clf_fit \
    --name=bioharn-clf-rgb-v002 \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --workdir=$HOME/work/bioharn \
    --arch=resnext101 \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=256,256 \
    --normalize_inputs=True \
    --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x8d-8ba56ff5.pth \
    --workers=8 \
    --xpu=auto \
    --batch_size=32 \
    --balance=None

python -m bioharn.clf_fit \
    --name=bioharn-clf-rgb-v003 \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=simple \
    --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-clf-rgb-v001/nrorbmcb/deploy_ClfModel_nrorbmcb_051_UFCIUU.zip \
    --workdir=$HOME/work/bioharn \
    --arch=resnext101 \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=256,256 \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=32 \
    --balance=None


python -m bioharn.clf_fit \
    --name=bioharn-clf-rgb-hard-v004 \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_hardbg1.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_hardbg1.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=simple \
    --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip \
    --workdir=$HOME/work/bioharn \
    --arch=resnext101 \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=256,256 \
    --normalize_inputs=True \
    --workers=8 \
    --xpu=auto \
    --batch_size=32 \
    --balance=classes
