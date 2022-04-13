# Uses defaults with demo data
python ~/code/netharn/examples/object_detection.py
python ~/code/netharn/examples/grab_voc.py
python ~/code/netharn/examples/object_detection.py --datasets=special:voc

python ~/code/ndsampler/ndsampler/make_demo_coco.py

python ~/code/bioharn/bioharn/detect_eval.py \
    --deployed=$HOME/work/bioharn/fit/nice/bioharn_shapes_example/best_snapshot.pt \
    --dataset=/home/joncrall/.cache/coco-demo/shapes256.mscoco.json

python -m bioharn.detect_fit \
    --nice=detect-singleclass-cascade-v4 \
    --workdir=$HOME/work/sealions \
    --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v3.mscoco.json \
    --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v3.mscoco.json \
    --schedule=ReduceLROnPlateau-p2-c2 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --optim=sgd --lr=1e-2 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --min_lr=1e-6 \
    --workers=4 --xpu=1,0 --batch_size=8 --bstep=1

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-v21 \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb|disparity" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=4 \
    --xpu=0 \
    --batch_size=4 \
    --bstep=4

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-v23 \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb|disparity" \
    --optim=DiffGrad \
    --lr=2e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=4 \
    --xpu=1 \
    --batch_size=4 \
    --bstep=4

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v22 \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=4 \
    --xpu=0 \
    --batch_size=4 \
    --bstep=4

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v24 \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=1024,1024 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=4 \
    --xpu=0 \
    --batch_size=2 \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-v25 \
    --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json \
    --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb|disparity" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=1024,1024 \
    --window_overlap=0.0 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=2 \
    --xpu=1,0 \
    --batch_size=4 \
    --bstep=8


python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v29-balanced \
    --train_dataset=/home/joncrall/data/private/_combo_cfarm/cfarm_train.mscoco.json \
    --vali_dataset=/home/joncrall/data/private/_combo_cfarm/cfarm_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=simple \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.5 \
    --multiscale=True \
    --normalize_inputs=True \
    --workers=0 \
    --xpu=1 \
    --batch_size=4 \
    --balance=tfidf \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v30-bigger-balanced \
    --train_dataset=$HOME/data/private/_combos/train_cfarm_habcam_v1.mscoco.json \
    --vali_dataset=$HOME/data/private/_combos/vali_cfarm_habcam_v1.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=0 \
    --xpu=1 \
    --batch_size=3 \
    --balance=tfidf \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v31-bigger-balanced \
    --train_dataset=$HOME/data/private/_combos/train_cfarm_habcam_v2.mscoco.json \
    --vali_dataset=$HOME/data/private/_combos/vali_cfarm_habcam_v2.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=0 \
    --xpu=0 \
    --batch_size=3 \
    --balance=tfidf \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=detect-sealion-cascade-v6 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6_train.mscoco.json \
    --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p2-c2 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --optim=sgd --lr=1e-2 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.5 \
    --multiscale=True \
    --normalize_inputs=True \
    --min_lr=1e-6 \
    --workers=4 --xpu=0 --batch_size=8 --bstep=1

python -m bioharn.detect_fit \
    --nice=detect-sealion-retina-v10 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p2-c2 \
    --augment=complex \
    --init=noop \
    --arch=retinanet \
    --optim=sgd --lr=1e-2 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --balance=False \
    --multiscale=False \
    --normalize_inputs=True \
    --min_lr=1e-6 \
    --workers=4 --xpu=0 --batch_size=22 --bstep=1

python -m bioharn.detect_fit \
    --nice=detect-sealion-cascade-v11 \
    --workdir=$HOME/work/sealions \
    --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p2-c2 \
    --augment=complex \
    --init=noop \
    --arch=cascade \
    --optim=sgd --lr=1e-3 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --balance=False \
    --multiscale=False \
    --normalize_inputs=True \
    --min_lr=1e-6 \
    --workers=4 --xpu=0 --batch_size=8 --bstep=1



coco_stats --src=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json
coco_stats --src=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json

    --train_dataset=/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v5_train.mscoco.json \

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-v35-bigger-balanced \
    --train_dataset=/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v5_train.mscoco.json \
    --vali_dataset=/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v5_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=simple \
    --workdir=/home/joncrall/work/bioharn_vhack \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-4 \
    --input_dims=window \
    --window_dims=512,512 \
    --window_overlap=0.0 \
    --multiscale=False \
    --normalize_inputs=True \
    --workers=0 \
    --xpu=1 \
    --batch_size=3 \
    --balance=tfidf \
    --sampler_backend=None \
    --bstep=8 \
    --init=noop

/home/joncrall/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/deploy.zip

    # --init=noop

python -m bioharn.detect_fit \
--nice=bioharn-det-mc-cascade-rgb-v32-bigger-balanced \
--schedule=step-10-20 \
--augment=complex \
--workdir=/home/joncrall/work/bioharn \
--channels="rgb" \
--optim=sgd \
--lr=1e-3 \
--input_dims=window \
--window_dims=512,512 \
--window_overlap=0.0 \
--multiscale=False \
--normalize_inputs=True \
--workers=0 \
--xpu=auto \
--batch_size=3 \
--balance=tfidf \
--sampler_backend=cog \
--bstep=8 \
--arch=cascade \
--backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
--init=noop

python -m bioharn.detect_fit \
--nice=bioharn-det-mc-cascade-rgbd-v36 \
--train_dataset=/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v6_train.mscoco.json \
--vali_dataset=/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v6_vali.mscoco.json \
--schedule=step-10-20 \
--augment=complex \
--workdir=/home/joncrall/work/bioharn \
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
--batch_size=3 \
--balance=tfidf \
--sampler_backend=cog \
--bstep=8 \
--arch=cascade \
--backbone_init=/home/joncrall/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
--init=noop

--init=/home/joncrall/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/deploy.zip \


python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-fine-coi-v40 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v6_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v6_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v31-bigger-balanced/moskmhld/deploy_MM_CascadeRCNN_moskmhld_015_SVBZIV.zip \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
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
    --xpu=0 \
    --batch_size=6 \
    --balance=tfidf \
    --bstep=8


python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-fine-coi-v41 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v6_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v6_vali.mscoco.json \
    --schedule=step-10-20 \
    --augment=complex \
    --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/torch_snapshots/_epoch_00000015.pt \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
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
    --xpu=0 \
    --batch_size=3 \
    --balance=tfidf \
    --bstep=8

### --- RUN ON FIXED SHIFTED 39 pixel BBOXES

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-fine-coi-v43 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=step-10-40 \
    --max_epoch=50 \
    --augment=complex \
    --pretrained=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/torch_snapshots/_epoch_00000017.pt \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
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
    --nice=bioharn-det-mc-cascade-rgbd-fine-coi-v42 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=step-10-40 \
    --max_epoch=50 \
    --augment=complex \
    --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v41/ufkqjjuk/torch_snapshots/_epoch_00000016.pt \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
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
    --num_batches=4172 \
    --balance=tfidf \
    --bstep=8

#######

# --- vali fine tune

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-coi-v42_valitune \
    --train_dataset=$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=step-1-2 \
    --max_epoch=5 \
    --patience=20 \
    --augment=simple \
    --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v42/nfmnvqwq/torch_snapshots/_epoch_00000027.pt \
    --workdir=$HOME/work/bioharn \
    "--classes_of_interest=[live sea scallop,swimming sea scallop,flatfish,clapper]" \
    --arch=cascade \
    --channels="rgb|disparity" \
    --optim=sgd \
    --lr=5e-4 \
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
    --nice=bioharn-det-mc-cascade-rgb-coi-v43_valitune \
    --train_dataset=$HOME/remote/viame/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=step-1-2 \
    --max_epoch=5 \
    --patience=20 \
    --augment=simple \
    --pretrained=$HOME/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-fine-coi-v43/bvbvdplp/torch_snapshots/_epoch_00000006.pt \
    --workdir=$HOME/work/bioharn \
    "--classes_of_interest=[live sea scallop,swimming sea scallop,flatfish,clapper]" \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=5e-4 \
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

    kwcoco union --src $HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json $HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json --dst $HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_trainval.mscoco.json


python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-fine-coi-v44 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p4-c2 \
    --max_epoch=200 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=512,512 \
    --window_dims=full \
    --window_overlap=0.0 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=600 \
    --balance=tfidf \
    --bstep=8

python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgbd-fine-coi-v45 \
    --train_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p4-c2 \
    --max_epoch=200 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
    --arch=cascade \
    --channels="rgb|disparity" \
    --optim=sgd \
    --lr=1e-3 \
    --input_dims=512,512 \
    --window_dims=full \
    --window_overlap=0.0 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=600 \
    --balance=tfidf \
    --bstep=8


python -m bioharn.detect_fit \
    --nice=bioharn-det-mc-cascade-rgb-coi-v46 \
    --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
    --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=noop \
    --workdir=/home/joncrall/work/bioharn \
    --arch=cascade \
    --channels="rgb" \
    --optim=sgd \
    --lr=1e-3 \
    --window_dims=512,512 \
    --input_dims=window \
    --window_overlap=0.5 \
    --multiscale=False \
    --normalize_inputs=True \
    --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x4d-a5af3160.pth \
    --workers=8 \
    --xpu=auto \
    --batch_size=4 \
    --num_batches=2000 \
    --balance=tfidf \
    --bstep=8

girder-client --api-url https://data.kitware.com/api/v1 download 5ee6a3ef9014a6d84ec02c36 $HOME/work/bioharn/_cache/checkpoint_VOC_efficientdet-d0_268.pth

python -m bioharn.detect_fit \
    --nice=sealion-efficientdet-v1 \
    --workdir=$HOME/work/sealions \
    --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_train.mscoco.json \
    --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --pretrained=$HOME/work/bioharn/_cache/checkpoint_VOC_efficientdet-d0_268.pth \
    --arch=efficientdet \
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
    --batch_size=7 \
    --sampler_backend=None \
    --num_batches=1000 \
    --balance=None \
    --bstep=3

python -m bioharn.detect_fit \
    --nice=sealion-cascade-v3 \
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
