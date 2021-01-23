
        python -m bioharn.detect_fit \
            --nice=bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune \
            --train_dataset=$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_trainval.mscoco.json \
            --schedule=step-1-2 \
            --max_epoch=5 \
            --patience=20 \
            --augment=simple \
            --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44/gvizryca/torch_snapshots/_epoch_00000016.pt \
            --workdir=/home/joncrall/work/bioharn \
            "--classes_of_interest=[flatfish,]" \
            --arch=cascade \
            --channels="rgb|disparity" \
            --optim=sgd \
            --lr=3e-4 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=False \
            --normalize_inputs=True \
            --workers=4 \
            --xpu=auto \
            --batch_size=4 \
            --num_batches=100 \
            --balance=tfidf \
            --sampler_backend=None \
            --bstep=8


        python -m bioharn.detect_fit \
            --nice=bioharn-det-mc-cascade-rgb-flatfish-only-v45 \
            --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
            --schedule=step-10-40 \
            --max_epoch=50 \
            --augment=complex \
            --pretrained=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/torch_snapshots/_epoch_00000017.pt \
            --workdir=/home/joncrall/work/bioharn \
            "--classes_of_interest=[flatfish,]" \
            --arch=cascade \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=False \
            --normalize_inputs=True \
            --workers=4 \
            --xpu=auto \
            --batch_size=4 \
            --num_batches=600 \
            --balance=tfidf \
            --bstep=8

        python -m bioharn.detect_fit \
            --nice=bioharn-det-mc-cascade-rgbd-flatfish-only-v44 \
            --train_dataset=$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
            --vali_dataset=$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
            --schedule=step-10-40 \
            --max_epoch=50 \
            --patience=20 \
            --augment=complex \
            --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v41/ufkqjjuk/torch_snapshots/_epoch_00000016.pt \
            --workdir=/home/joncrall/work/bioharn \
            "--classes_of_interest=[flatfish,]" \
            --arch=cascade \
            --channels="rgb|disparity" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=False \
            --normalize_inputs=True \
            --workers=4 \
            --xpu=auto \
            --batch_size=4 \
            --num_batches=100 \
            --balance=tfidf \
            --sampler_backend=None \
            --bstep=8


        # flatfish only models
        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json \
            "--deployed=[\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44/gvizryca/torch_snapshots/_epoch_00000016.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/torch_snapshots/_epoch_00000001.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/torch_snapshots/_epoch_00000002.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/torch_snapshots/_epoch_00000003.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/torch_snapshots/_epoch_00000004.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/torch_snapshots/_epoch_00000005.pt]" \
            "--classes_of_interest=[flatfish,]" \
            --sampler_backend=cog --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.8 --xpu=auto







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


TRAIN_FPATH=/data/matt.dawkins/Training3/deep_training/training_truth.json
VALI_FPATH=/data/matt.dawkins/Training3/deep_training/validation_truth.json 
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
