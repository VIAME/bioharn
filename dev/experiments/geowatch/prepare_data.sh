#!/bin/bash
__doc__="
See Also:
    ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py
"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
FULL_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip
kwcoco conform "$FULL_DSET" --inplace=True --workers=16

# TODO: add CLI that can infer / assume channel specs
python -c "if 1:
    fpath = '$FULL_DSET'
    import kwcoco
    dset = kwcoco.CocoDataset(fpath)
    for img in dset.images().objs_iter():
        img['sensor'] = 'cam'
        img['channels'] = 'red|green|blue'
    dset.dump()
"


FULL_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-v01.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v01.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v01.kwcoco.zip

if [ -f "$TRAIN_FPATH" ] ; then
    echo "Splits already exist"
else
    kwcoco split "$FULL_DSET" --rng 10621406 --dst1 "$KWCOCO_BUNDLE_DPATH/learn-v01.kwcoco.zip" --dst2 "$TEST_FPATH"
    kwcoco split "$KWCOCO_BUNDLE_DPATH/learn-v01.kwcoco.zip" --rng 10631407 --dst1 "$TRAIN_FPATH" --dst2 "$VALI_FPATH"

    kwcoco conform "$TRAIN_FPATH" --inplace=True --workers=16
    kwcoco conform "$VALI_FPATH" --inplace=True --workers=16
    kwcoco conform "$TEST_FPATH" --inplace=True --workers=16

    kwcoco conform "$VALI_FPATH" --inplace=True --workers=16
    kwcoco tables "$VALI_FPATH" -g1

    # TODO: add CLI that can infer / assume channel specs
    python -c "if 1:
        fpath = '$VALI_FPATH'
        import kwcoco
        dset = kwcoco.CocoDataset(fpath)
        to_remove = []
        for img in dset.images().objs_iter():
            if 'width' not in img:
                to_remove.append(img['id'])
        dset.remove_images(to_remove)
        dset.dump()
    "
    python -c "if 1:
        fpath = '$TRAIN_FPATH'
        import kwcoco
        dset = kwcoco.CocoDataset(fpath)
        to_remove = []
        for img in dset.images().objs_iter():
            if 'width' not in img:
                to_remove.append(img['id'])
        dset.remove_images(to_remove)
        dset.dump()
    "
fi

KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/train-v01.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/train-v01.mscoco.json --legacy=True
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/vali-v01.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/vali-v01.mscoco.json --legacy=True
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/test-v01.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/test-v01.mscoco.json --legacy=True
