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

