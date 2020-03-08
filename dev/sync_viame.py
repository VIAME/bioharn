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
from os.path import basename
from os.path import join
from os.path import normpath
from os.path import dirname
import numpy as np
from os.path import exists
import pandas as pd
import kwimage
import kwarray
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
    import multiprocessing
    from ndsampler.utils import util_futures
    import ndsampler
    import kwimage

    records = df.to_dict(orient='records')

    for row in ub.ProgIter(records, desc='fix formatting'):
        for k, v in list(row.items()):
            if isinstance(v, str):
                row[k] = v.strip()

    cathist = ub.ddict(lambda: 0)
    objname_to_objid = {}
    for row in ub.ProgIter(records):
        object_name = row['Name']
        cathist[object_name] += 1
        objname_to_objid[object_name] = row['Name']
    print('Raw categories:')
    print(ub.repr2(ub.odict(sorted(list(cathist.items()), key=lambda t: t[1]))))

    for old_cat in objname_to_objid.keys():
        if old_cat not in catname_map:
            print('NEED TO REGISTER: old_cat = {!r}'.format(old_cat))

    coco_dset = ndsampler.CocoDataset(img_root=img_root)

    dev_root = ub.ensuredir((img_root, '_dev'))
    cog_root = ub.ensuredir((dev_root, 'cog_rgb'))

    workers = min(10, multiprocessing.cpu_count())
    jobs = util_futures.JobPool(mode='thread', max_workers=workers)
    for row in ub.ProgIter(records):
        image_name = row['Imagename']

        # TODO: de-bayer and preprocess
        # image_name = row['TIFImagename']

        # Handle Image
        gid = coco_dset.ensure_image(file_name=image_name)
        img = coco_dset.imgs[gid]

        if img.get('is_bad', False):
            continue
        if 'width' not in img:
            gpath = coco_dset.get_image_fpath(gid)
            try:
                if not exists(gpath):
                    raise Exception
                shape  = kwimage.load_image_shape(gpath)
            except Exception:
                img['is_bad'] = True
                print('Bad image gpath = {!r}'.format(gpath))
                continue

            height, width = shape[0:2]
            img['height'] = height
            img['width'] = width

        if 'in_queue' not in img:
            # Convert to COG in the background
            job = jobs.submit(_ensure_rgb_cog, coco_dset, gid, cog_root)
            job.img = img
            img['in_queue'] = True  # hack

        # Handle Category
        object_name = row['Name']
        cat_name = catname_map[object_name]
        if cat_name is None:
            raise KeyError(cat_name)
        cid = coco_dset.ensure_category(cat_name)

        # Handle Annotations
        # add category modifiers
        weight = 1.0
        if 'probable' in object_name:
            weight *= 0.5
        if 'inexact' in object_name:
            weight *= 0.5

        # Assume these are line annotations
        x1, y1, x2, y2 = list(ub.take(row, ['X1', 'Y1', 'X2', 'Y2']))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        pt1, pt2 = np.array([x1, y1]), np.array([x2, y2])
        cx, cy = (pt1 + pt2) / 2
        diameter = np.linalg.norm(pt1 - pt2)
        cxywh = [cx, cy, diameter, diameter]

        sf = img['height'] / row['Image_Height']
        xywh = kwimage.Boxes([cxywh], 'cxywh').to_xywh().scale(sf).data[0].round(3).tolist()

        ann = {
            'category_id': cid,
            'image_id': gid,
            'bbox': xywh,
            'weight': weight,
            'meta': row,
        }
        coco_dset.add_annotation(**ann)

    for job in ub.ProgIter(jobs, desc='redirect to cog images'):
        img = job.img
        img.pop('in_queue', None)
        cog_fpath = job.result()
        img['file_name'] = cog_fpath

    # Remove hyper-small annotations, they are probably bad
    weird_anns = []
    for ann in coco_dset.anns.values():
        if np.sqrt(ann['bbox'][2] * ann['bbox'][3]) < 10:
            weird_anns.append(ann)
    coco_dset.remove_annotations(weird_anns)

    coco_dset.dataset.pop('img_root', None)
    coco_dset.img_root = dev_root
    bad_images = coco_dset._ensure_imgsize(workers=16, fail=False)
    coco_dset.remove_images(bad_images)

    # Add special tag indicating a stereo image
    dset_name = basename(img_root)
    for img in coco_dset.imgs.values():
        img['source'] = dset_name

    stats = coco_dset.basic_stats()
    suffix = 'g{n_imgs:06d}_a{n_anns:08d}_c{n_cats:04d}'.format(**stats)

    coco_dset.fpath = ub.augpath(
        '', dpath=dev_root, ext='',
        base=dset_name + '_{}_v3.mscoco.json'.format(suffix))

    coco_dset.rebase(dev_root)
    coco_dset.img_root = dev_root

    if 0:
        import kwplot
        import xdev
        kwplot.autompl()
        for gid in xdev.InteractiveIter(list(coco_dset.imgs.keys())):
            coco_dset.show_image(gid)
            xdev.InteractiveIter.draw()

    datasets = train_vali_split(coco_dset)
    print('datasets = {!r}'.format(datasets))

    coco_dset.dump(coco_dset.fpath, newlines=True)
    for tag, tag_dset in datasets.items():
        print('{} fpath = {!r}'.format(tag, tag_dset.fpath))
        print(ub.repr2(tag_dset.category_annotation_frequency()))
        tag_dset.dump(tag_dset.fpath, newlines=True)


def _ensure_rgb_cog(dset, gid, cog_root):
    img = dset.imgs[gid]
    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='', ext='.cog.tif')
    cog_fpath = join(cog_root, cog_fname)
    ub.ensuredir(dirname(cog_fpath))

    if not exists(cog_fpath):
        # Note: probably should be atomic
        img3 = dset.load_image(gid)
        imgL = img3[:, 0:img3.shape[1] // 2]
        kwimage.imwrite(cog_fpath, imgL, backend='gdal', compress='DEFLATE')
    return cog_fpath


def train_vali_split(coco_dset):

    split_gids = _split_train_vali_test_gids(coco_dset)
    datasets = {}
    for tag, gids in split_gids.items():
        tag_dset = coco_dset.subset(gids)
        tag_dset.fpath = ub.augpath(coco_dset.fpath, suffix='_' + tag, multidot=True)
        datasets[tag] = tag_dset

    return datasets


def _split_train_vali_test_gids(coco_dset, factor=2):
    def _stratified_split(gids, cids, n_splits=2, rng=None):
        """ helper to split while trying to maintain class balance within images """
        rng = kwarray.ensure_rng(rng)
        from ndsampler.utils.util_sklearn import StratifiedGroupKFold
        selector = StratifiedGroupKFold(n_splits=n_splits, random_state=rng,
                                        shuffle=True)
        skf_list = list(selector.split(X=gids, y=cids, groups=gids))
        trainx, testx = skf_list[0]
        return trainx, testx

    # Create flat table of image-ids and category-ids
    gids, cids = [], []
    images = coco_dset.images()
    for gid_, cids_ in zip(images, images.annots.cids):
        cids.extend(cids_)
        gids.extend([gid_] * len(cids_))

    # Split into learn/test then split learn into train/vali
    rng = kwarray.ensure_rng(1617402282)
    # FIXME: make train bigger with 2
    learnx, testx = _stratified_split(gids, cids, rng=rng,
                                      n_splits=factor)
    learn_gids = list(ub.take(gids, learnx))
    learn_cids = list(ub.take(cids, learnx))
    _trainx, _valix = _stratified_split(learn_gids, learn_cids, rng=rng,
                                        n_splits=factor)
    trainx = learnx[_trainx]
    valix = learnx[_valix]

    split_gids = {
        'train': sorted(set(ub.take(gids, trainx))),
        'vali': sorted(set(ub.take(gids, valix))),
        'test': sorted(set(ub.take(gids, testx))),
    }
    print('splits = {}'.format(ub.repr2(ub.map_vals(len, split_gids))))
    return split_gids



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



