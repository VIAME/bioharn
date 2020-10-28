Training harness for biology related problems

Uses `netharn` (https://gitlab.kitware.com/computer-vision/netharn) to write
the boilerplate for training loops. 

Scripts take `kwcoco` datasets as inputs. See
https://gitlab.kitware.com/computer-vision/kwcoco for how to format in the
extended-COCO format (regular MS-COCO files will also work).

To train a detection model see `bioharn/detect_fit.py`. 

To train a classification model see `bioharn/clf_fit.py`. 

To predict with a pretrained detection model see `bioharn/detect_predict.py`. 

To predict with a pretrained classification model see `bioharn/clf_predict.py`. 

To evaluate ROC / PR-curves with a pretrained detection model and truth see `bioharn/detect_eval.py`. 

To evaluate ROC / PR-curves with a pretrained classification model and truth see `bioharn/clf_eval.py`. 


Current supported detection models include

* YOLO-v2
* EfficientDet
* MaskRCNN - Requires mmdet 
* CascadeRCNN - Requires mmdet 
* RetinaNet - Requires mmdet 

Older versions of bioharn were previously targeting mmdet 1.0 revision
4c94f10d0ebb566701fb5319f5da6808df0ebf6a but we are now targeting v2.0.


This repo is a component of the VIAME project: https://github.com/VIAME/VIAME

some of the data for this project can be found here

https://data.kitware.com/#collection/58b747ec8d777f0aef5d0f6a


Notes for mmcv install on cuda 10.1 with torch 1.5:

See: https://github.com/open-mmlab/mmcv

```python
 pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
```



### Training a simple detector

To train a simple detector let use the kwcoco toy data to make sure we can fit
to a small dummy dataset. Lets use the kwcoco CLI to generate toy training. For
this test we will forgo a validation set. 

```bash
kwcoco toydata --key=shapes1024 --dst=toydata.kwcoco.json
```


For our notable hyperparameters we are going to use:

* `--optim=adam` - we will use the ADAM optimizer for faster convergence (may also want to try `sgd`).

* `--lr=1e-4` - we will start with a small learning rate

* `--decay=1e-4` - we will use weight decay regularization of `1e-4` to encourage smaller network weights.

* `--window_dims=full` - which means that each batch item will sample a full image.

* `--input_dims=512,512` - which means we are going to resize each image to H=512, W=512 (using letterboxing to preserve aspect ratio) before inputting the item to the network.

* `--schedule=step-16-22` - will divide the learning rate by 10 at epoch 16 and 22.

* `--augment=medium` - will do random flips, crops, and color jitter for augmentation (`--augment=complex` will do much more and `--augment=simple` will only do flips and crops).

* `--num_batches=auto` - determines the number of batches per epoch. If auto it will use the entire dataset. If you set it to a number if will use that many batches per epoch with random sampling with replacement. This is useful if you are going to use over/undersampling via the `--balance` CLI arg.


`--batch_size=8` will use 8 items (sampled windows) per batch.

`--bstep=8` will run 8 batches before backpropagating (approximates a larger batch size)


See `python -m bioharn.detect_fit --help` for help on all available options.


```bash
python -m bioharn.detect_fit \
    --name=det_baseline_toydata \
    --workdir=$HOME/work/bioharn \
    --train_dataset=./toydata.kwcoco.json \
    --arch=retinanet \
    --optim=adam \
    --lr=1e-4 \
    --schedule=step-16-22 \
    --augment=medium \
    --input_dims=512,512 \
    --window_dims=full \
    --window_overlap=0.0 \
    --normalize_inputs=imagenet \
    --num_batches=auto \
    --workers=4 --xpu=auto --batch_size=8 --bstep=8 \
    --sampler_backend=cog 
```

This should start producing reasonable training-set bounding boxes after a few
minutes of training.

Because we are using netharn, training this detection model will write a
"training directory" in your work directory. This directory will be a function
of your "name" and the hash of the learning-relevant hyperparameters. 

In this case the training directory will be in:
`$HOME/work/bioharn/fit/runs/det_baseline_toydata/qxvodtak` and for convenience there will
be a symlink to this directory in
`$HOME/work/bioharn/fit/name/det_baseline_toydata`. Example training (and
validation if specified) images will be written to the `monitor` directory. If
tensorboard and matplotlib are installed the `monitor` directory will also
contain a `tensorboard` subdirectory with loss curves as they are produced.

