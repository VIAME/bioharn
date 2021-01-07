cd /data/matt.dawkins/Training-PenguinBoogaloo/deep_training

kwcoco stats --src training_truth.json
kwcoco stats --src validation_truth.json

__doc__="
                  training_truth.json
blasted                           115
krill                             121
surf                              540
water_open                        768
water_pengfriend                   13

                  validation_truth.json
blasted                               6
krill                                 7
surf                                 13
water_open                           54
water_pengfriend                      1
"


export QT_DEBUG_PLUGINS=1
kwcoco show --src training_truth.json --gid 1 --dst $HOME/foo.jpg

cd /home/khq.kitware.com/jon.crall/.local/conda/envs/py38/lib/python3.8/site-packages/PyQt5/Qt/plugins/platforms/
ldd libqxcb.so | grep "not found"


# Super hack, does not work
cd $HOME/tmp
curl http://archive.ubuntu.com/ubuntu/pool/main/libx/libxcb/libxcb-xinerama0_1.14-2_amd64.deb --output libxcb-xinerama0_1.14-2_amd64.deb
# ar x libxcb-xinerama0_1.14-2_amd64.deb
mkdir -p extract-libxcb-xinerama0_1.14-2_amd64
dpkg-deb -xv libxcb-xinerama0_1.14-2_amd64.deb extract-libxcb-xinerama0_1.14-2_amd64
chmod 755 $HOME/tmp/extract-libxcb-xinerama0_1.14-2_amd64/usr/lib/x86_64-linux-gnu/libxcb-xinerama.so.0
cp $HOME/tmp/extract-libxcb-xinerama0_1.14-2_amd64/usr/lib/x86_64-linux-gnu/libxcb-xinerama.so.0 /home/khq.kitware.com/jon.crall/.local/conda/envs/py38/lib/python3.8/site-packages/PyQt5/Qt/plugins/platforms/



TRAIN_FPATH=/data/matt.dawkins/Training-PenguinBoogaloo/deep_training/training_truth.json
VALI_FPATH=/data/matt.dawkins/Training-PenguinBoogaloo/deep_training/validation_truth.json 


srun --gres=gpu:rtx6000:1 --cpus-per-task=11 --partition=priority --account=noaa --mem 20000 \
python -m bioharn.clf_fit \
    --name=test-basic-fullframe-clf \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --workdir=$HOME/work/bioharn \
    --arch=resnet50 \
    --channels="rgb" \
    --optim=AdaBelief \
    --augmenter=complex \
    --input_dims=256,256 \
    --normalize_inputs=imagenet \
    --workers=10 \
    --xpu=auto \
    --schedule=ReduceLROnPlateau-p10-c10 \
    --sampler_backend=None \
    --eager_dump_tensorboard=False \
    --dump_tensorboard=False \
    --balance=classes \
    --num_batches=200 \
    --batch_size=224 --lr=0.0005

    --pretrained=/home/joncrall/.cache/torch/checkpoints/resnet50-19c8e357.pth \
