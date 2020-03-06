"""
Sync data from the VIAME server to local machine


# on local
mkdir -p $HOME/data/raid/private
ln -s $HOME/data/raid/private $HOME/data/private

# on viame
ln -s /data/private $HOME/data/private

rsync -avrRP viame:data/private/./US_NE_2017_CFARM_HABCAM $HOME/data/private/
rsync -avrRP viame:data/private/./US_NE_2018_CFARM_HABCAM $HOME/data/private/
rsync -avrRP viame:data/private/./US_NE_2019_CFARM_HABCAM $HOME/data/private/
rsync -avrRP viame:data/private/./US_NE_2019_CFARM_HABCAM_PART2 $HOME/data/private/
"""
import numpy as np
from os.path import exists
import pandas as pd
import ubelt as ub


# Simplify the categories
catname_map = {
    'American Lobster': 'lobster',
    'squid': 'squid',

    'probably didemnum': 'didemnum',
    'probable scallop-like rock': 'rock',

    'dust cloud': 'dust cloud',

    'unidentified skate (less than half)': 'skate',
    'winter-little skate': 'skate',
    'unidentified skate': 'skate',
    'unknown skate': 'skate',

    'jonah or rock crab': 'crab',
    'Jonah crab': 'crab',
    'Rock crab': 'crab',
    'unknown crab': 'crab',

    'dead scallop (width)': 'dead sea scallop',
    'dead sea scallop inexact': 'dead sea scallop',
    'dead sea scallop': 'dead sea scallop',
    'probable dead sea scallop inexact': 'dead sea scallop',
    'probable dead sea scallop width': 'dead sea scallop',
    'probable dead sea scallop': 'dead sea scallop',
    'probable dead sea scallop': 'dead sea scallop',
    'sea scallop clapper inexact': 'dead sea scallop',
    'sea scallop clapper width': 'dead sea scallop',
    'sea scallop clapper': 'dead sea scallop',

    'probable swimming sea scallop inexact': 'swimming sea scallop',
    'probable swimming sea scallop': 'swimming sea scallop',
    'swimming scallop width': 'swimming sea scallop',
    'swimming scallop': 'swimming sea scallop',
    'swimming sea scallop inexact':  'swimming sea scallop',
    'swimming sea scallop width': 'swimming sea scallop',
    'swimming sea scallop': 'swimming sea scallop',

    'live sea scallop inexact': 'live sea scallop',
    'live sea scallop width': 'live sea scallop',
    'live sea scallop': 'live sea scallop',
    'probable live sea scallop inexact': 'live sea scallop',
    'probable live sea scallop width': 'live sea scallop',
    'probable live sea scallop': 'live sea scallop',
    'scallop (width)': 'live sea scallop',
    'white sea scallop width': 'live sea scallop',
    'white scallop': 'live sea scallop',

    'unidentified flatfish (less than half)': 'flatfish',
    'unidentified flatfish': 'flatfish',
    'unknown flounder': 'flatfish',
    'winter flounder': 'flatfish',
    'windowpane flounder': 'flatfish',
    'fourspot flounder': 'flatfish',
    'yellowtail flounder': 'flatfish',
    'grey sole': 'flatfish',

    'silver hake': 'roundfish',
    'sculpin/grubby': 'roundfish',
    'longhorn sculpin': 'roundfish',
    'Hake spp.': 'roundfish',
    'unknown fish': 'roundfish',
    'monkfish': 'roundfish',
    'red hake': 'roundfish',
    'unidentified roundfish (less than half)': 'roundfish',
    'unidentified roundfish': 'roundfish',
    'unidentified fish (less than half)': 'roundfish',
    'unidentified fish': 'roundfish',

    'Henricia': 'seastar',
    'Astropecten': 'seastar',
    'Asterias rubens': 'seastar',
    'any white seastar': 'seastar',
    'any white seastar': 'seastar',
    'unknown seastar': 'seastar',
    'red cushion star': 'seastar',

    'unknown cerianthid': 'cerianthid',

    'snake eel': 'eel',
    'convict worm': 'eel',
    'blackrim cusk-eel': 'eel',

    'unknown mollusk': 'mollusk',

    'moon snail': 'snail',
    'waved whelk': 'snail',
    'moon snail-like': 'snail',
}


def convert_cfarm(df, img_root):
    records = df.to_dict(orient='records')

    cathist = ub.ddict(lambda: 0)
    objname_to_objid = {}
    for row in ub.ProgIter(records):
        object_name = row['Name'].strip()
        cathist[object_name] += 1
        objname_to_objid[object_name] = row['Name']
    print('Raw categories:')
    print(ub.repr2(ub.odict(sorted(list(cathist.items()), key=lambda t: t[1]))))

    for old_cat in objname_to_objid.keys():
        if old_cat not in catname_map:
            print('NEED TO REGISTER: old_cat = {!r}'.format(old_cat))

    import ndsampler
    coco_dset = ndsampler.CocoDataset(img_root=img_root)

    for row in ub.ProgIter(records):
        image_name = row['Imagename']
        # image_name = row['TIFImagename']

        row['Name'] = row['Name'].strip()
        object_name = row['Name'].strip()

        # add category modifiers
        weight = 1.0
        if 'probable' in object_name:
            weight *= 0.5
        if 'inexact' in object_name:
            weight *= 0.5

        import kwimage
        x1, y1, x2, y2 = list(ub.take(row, ['X1', 'Y1', 'X2', 'Y2']))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        tlbr = [x1, y1, x2, y2]

        sf = np.sqrt(row['Altitude'])
        sf = 1.8
        xywh = kwimage.Boxes([tlbr], 'tlbr').to_xywh().scale(sf).data[0].round(3).tolist()

        gid = coco_dset.ensure_image(file_name=image_name)

        cat_name = catname_map[object_name]
        if cat_name is None:
            raise KeyError(cat_name)
            # cat_name = 'other'
            # print('cat_name = {!r}'.format(cat_name))

        cid = coco_dset.ensure_category(cat_name)

        ann = {
            'category_id': cid,
            'image_id': gid,
            'bbox': xywh,
            'weight': weight,
            'meta': row,
        }
        coco_dset.add_annotation(**ann)

    # Remove hyper-small annotations, they are probably bad
    weird_anns = []
    for ann in coco_dset.anns.values():
        if np.sqrt(ann['bbox'][2] * ann['bbox'][3]) < 10:
            weird_anns.append(ann)
    coco_dset.remove_annotations(weird_anns)

    if 0:
        import kwplot
        kwplot.autompl()
        gid = weird_anns[-1]['image_id']
        coco_dset.show_image(gid)

        coco_dset.show_image(15)
        im = kwimage.imread(coco_dset.get_image_fpath(gid))

        for gid in xdev.InteractiveIter(list(coco_dset.imgs.keys())):
            coco_dset.show_image(gid)
            xdev.InteractiveIter.draw()


def convert_cfarm_2017():
    csv_fpath =  ub.expandpath('~/data/private/US_NE_2017_CFARM_HABCAM/HabCam 2017 dataset1 annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    img_root = dirname(csv_fpath)


def convert_cfarm_2018():
    csv_fpath =  ub.expandpath('~/data/private/US_NE_2018_CFARM_HABCAM/annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    img_root = dirname(csv_fpath)
    # csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')


def convert_cfarm_2019():
    csv_fpath =  ub.expandpath('~/data/private/US_NE_2019_CFARM_HABCAM/raws/annotations-corrected.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    # csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')


def convert_cfarm_2019_part2():
    csv_fpath =  ub.expandpath('~/data/private/US_NE_2019_CFARM_HABCAM_PART2/HabCam 2019 dataset2 annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    # csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')
