#!/bin/bash
__doc__="
See Also:
    ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py
"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
FULL_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip

python ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py \
    --in_fpath "$DVC_DATA_DPATH"/habcam-2020-2021.csv \
    --out_fpath "$DVC_DATA_DPATH"/data.kwcoco.zip

# TODO: add CLI that can infer / assume channel specs
kwcoco conform "$FULL_DSET" --inplace=True --legacy=True --mmlab=True --workers=16

kwcoco fixup "$FULL_DSET" --inplace=True --corrupted_assets=only_shape --workers=16

FULL_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-v02.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v02.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v02.kwcoco.zip

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
fi

KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/train-v02.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/train-v02.mscoco.json --legacy=True
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/vali-v02.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/vali-v02.mscoco.json --legacy=True
kwcoco conform "$KWCOCO_BUNDLE_DPATH"/test-v02.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/test-v02.mscoco.json --legacy=True

KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
kwcoco stats "$KWCOCO_BUNDLE_DPATH"/data.kwcoco.zip
kwcoco modify_categories --remove_empty_images=True --keep "
- Jonah or rock crab
- red hake
- winter-little skate
" --src "$KWCOCO_BUNDLE_DPATH"/data.kwcoco.zip --dst "$KWCOCO_BUNDLE_DPATH"/data-3class.kwcoco.zip
kwcoco stats "$KWCOCO_BUNDLE_DPATH"/data-3class.kwcoco.zip

FULL_DSET=$KWCOCO_BUNDLE_DPATH/data-3class.kwcoco.zip
LEARN_FPATH=$KWCOCO_BUNDLE_DPATH/learn-3class-v03.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-3class-v03.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-3class-v03.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-3class-v03.kwcoco.zip
kwcoco split "$FULL_DSET" --rng 10621406 --dst1 "$LEARN_FPATH" --dst2 "$TEST_FPATH"
kwcoco split "$LEARN_FPATH" --rng 10631407 --dst1 "$TRAIN_FPATH" --dst2 "$VALI_FPATH"

cd "$KWCOCO_BUNDLE_DPATH"
kwcoco stats train-3class-v03.kwcoco.zip vali-3class-v03.kwcoco.zip test-3class-v03.kwcoco.zip
