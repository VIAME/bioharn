#!/bin/bash
__doc__="
See Also:
    ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py
"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
RAW_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip

if [ ! -f "$RAW_DSET" ]; then

    python ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py \
        --in_fpath "$DVC_DATA_DPATH"/habcam-2020-2021.csv \
        --out_fpath "$DVC_DATA_DPATH"/data.kwcoco.zip

    # TODO: add CLI that can infer / assume channel specs
    kwcoco conform "$RAW_DSET" --inplace=True --legacy=True --mmlab=True --workers=16

    kwcoco fixup "$RAW_DSET" --inplace=True --corrupted_assets=only_shape --workers=0

fi

cd "$KWCOCO_BUNDLE_DPATH"
FULL_DSET=$KWCOCO_BUNDLE_DPATH/data-with-polys.kwcoco.zip

show_stats(){
    kwcoco stats "$FULL_DSET"
}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-v04.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v04.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v04.kwcoco.zip

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

#KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
#kwcoco conform "$KWCOCO_BUNDLE_DPATH"/train-v04.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/train-v04.mscoco.json --legacy=True
#kwcoco conform "$KWCOCO_BUNDLE_DPATH"/vali-v04.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/vali-v04.mscoco.json --legacy=True
#kwcoco conform "$KWCOCO_BUNDLE_DPATH"/test-v04.kwcoco.zip "$KWCOCO_BUNDLE_DPATH"/test-v04.mscoco.json --legacy=True

KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
DVC_DATA_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH

#INPUT_DSET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip
NEWSTYLE_DSET=$KWCOCO_BUNDLE_DPATH/data-with-polys.kwcoco.zip
OLDSTYLE_DSET=$KWCOCO_BUNDLE_DPATH/data-with-closed-polys.kwcoco.zip
kwcoco conform "$NEWSTYLE_DSET" "$OLDSTYLE_DSET" --legacy=True

kwcoco validate "$OLDSTYLE_DSET" --workers 10

#INPUT_DSET=$RAW_DSET
INPUT_DSET=$OLDSTYLE_DSET

FULL_DSET=$KWCOCO_BUNDLE_DPATH/data-v05-noscallop.kwcoco.zip
LEARN_FPATH=$KWCOCO_BUNDLE_DPATH/learn-v05-noscallop.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-v05-noscallop.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-v05-noscallop.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test-v05-noscallop.kwcoco.zip

#kwcoco stats "$KWCOCO_BUNDLE_DPATH"/data-with-polys.kwcoco.zip


kwcoco modify_categories --remove_empty_images=True \
   --keep "
- Jonah or rock crab
- red hake
- winter-little skate
- unknown/other cephalopod
- unknown/other shark
- yellowtail flounder
- American lobster
- windowpane flounder
- barndoor skate
- ocean pout
- spiny dogfish
- monkfish
- gadid
- sculpin/grubby/sea raven
- madmade/trash
- squid
- unknown/other skate
- unknown/other crab
- fourspot flounder
- unknown/other decapod
- winter-little skate
- moon snail
- unknown/other fish
- unknown/other gastropod
- unknown/other invert
- unknown/other flatfish
- unknown/other bivalve
- whelk
- red hake
- Jonah or rock crab
- unknown/other hake
- hermit crab
- eel-like fish
" --src "$INPUT_DSET" --dst "$FULL_DSET"
kwcoco stats "$FULL_DSET"
#kwcoco show "$FULL_DSET"

kwcoco split "$FULL_DSET" --rng 10621406 --dst1 "$LEARN_FPATH" --dst2 "$TEST_FPATH"
kwcoco split "$LEARN_FPATH" --rng 10631407 --dst1 "$TRAIN_FPATH" --dst2 "$VALI_FPATH"

cd "$KWCOCO_BUNDLE_DPATH"
kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"


python -c "if 1:
import kwcoco
import kwimage
dset = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH/data-with-polys.kwcoco.zip')
for ann in dset.dataset['annotations']:
    ann.pop('poly', None)
dset.dump()

for ann in dset.annots().objs_iter():
    sseg = kwimage.Segmentation.coerce(ann['segmentation'])
    sseg.to_coco('orig')

tmp = dset.subset([3]).copy()
tmp.conform(legacy=True)
from kwcoco.compat_dataset import COCO

coco = tmp._aspycoco()
coco.annToMask(coco.dataset['annotations'][1])

import kwimage
from kwcoco import coco_schema
poly = kwimage.Polygon.random().to_coco(style='orig')
mpoly = kwimage.MultiPolygon.random().to_coco(style='orig')

coco_schema.MSCOCO_POLYGON.validate(poly)
coco_schema.MSCOCO_MULTIPOLYGON.validate(mpoly)


    ...
"