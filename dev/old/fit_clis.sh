        python -m bioharn.detect_fit \
            --nice=bioharn-test-yolo \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --pretrained=imagenet \
            --schedule=step90 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=16 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-test-yolo-v5 \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --pretrained=/home/joncrall/work/bioharn/fit/nice/bioharn-test-yolo/torch_snapshots/_epoch_00000011.pt \
            --schedule=ReduceLROnPlateau \
            --optim=adamw --lr=3e-4 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=16 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v12-test-retinanet \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau \
            --arch=retinanet \
            --augment=medium \
            --init=noop \
            --optim=sgd --lr=1e-3 \
            --input_dims=512,512 \
            --workers=6 --xpu=1 --batch_size=12 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v9-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --arch=cascade \
            --init=noop \
            --optim=sgd --lr=1e-2 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v11-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v9-test-cascade/zjolejwz/deploy_MM_CascadeRCNN_zjolejwz_010_LUAKQJ.zip \
            --augment=medium \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v11-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p1-c2 \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip \
            --augment=medium \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=1024,1024 \
            --workers=4 --xpu=1 --batch_size=1 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v12-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p1-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-4 \
            --input_dims=512,512 \
            --window_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v13-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-4 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v14-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v16-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --use_disparity=True \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v16-cascade/ozeaiwmm/explit_checkpoints/_epoch_00000000_2019-12-12T185656+5.pt \
            --normalize_inputs=False \
            --workers=4 --xpu=1 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v17-cascade-mc-disp \
            --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --workdir=$HOME/work/bioharn \
            --arch=cascade \
            --use_disparity=True \
            --optim=adamw --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=False \
            --workers=4 --xpu=0 --batch_size=16 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v18-cascade-mc-disp \
            --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --workdir=$HOME/work/bioharn \
            --arch=cascade \
            --use_disparity=True \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=1024,1024 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=False \
            --workers=4 --xpu=0 --batch_size=4 --bstep=4



        $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json

        # --pretrained='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth' \


        python -m bioharn.detect_fit \
            --nice=demo_shapes_gpu10 \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=1,0 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=demo_shapes_gpu1 \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 G
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=1 --batch_size=4 --bstep=1


        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v2 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v2.mscoco.json \
            --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v2.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --pretrained=/home/joncrall/work/sealions/fit/runs/detect-singleclass-cascade-v2/nkdvpjss/explit_checkpoints/_epoch_00000000_2020-01-16T182834+5.pt \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=8 --bstep=1


        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v3 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v3.mscoco.json \
            --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v3.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=0 --batch_size=16 --bstep=1

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
            --nice=test-pyrosome2 \
            --train_dataset=/data/projects/GOOD/pyrosome-train/deep_training/training_truth.json \
            --vali_dataset=/data/projects/GOOD/pyrosome-train/deep_training/validation_truth.json \
            --xpu=0,1,2,3 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --sampler_backend=None \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=1024,1024 \
            --window_overlap=0.0 \
            --workers=6 --batch_size=12 --bstep=1

