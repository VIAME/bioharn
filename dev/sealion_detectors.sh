



__heredoc__ = """

Sealion Dataset:
    $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7_vali.mscoco.json 

    /home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json
    /home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json


Trained Sealion Models:

    $HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v3/hpjbgxjn/deploy_MM_CascadeRCNN_hpjbgxjn_045_JTZMSY.zip

"""


### TRAINING ###

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


python -m bioharn.detect_fit \
    --nice=sealion-cascade-v4 \
    --workdir=$HOME/work/sealions \
    --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_train.mscoco.json \
    --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json \
    --schedule=ReduceLROnPlateau-p5-c5 \
    --max_epoch=400 \
    --augment=complex \
    --init=$HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v4/erhbywip/explit_checkpoints/_epoch_00000003_2020-06-17T134405+5.pt \
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
        $HOME/remote/namek/work/sealions/fit/runs/sealion-cascade-v3/hpjbgxjn/deploy_MM_CascadeRCNN_hpjbgxjn_045_JTZMSY.zip,\
    ]" \
    --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.8 --xpu=auto --window_overlap=0.5
