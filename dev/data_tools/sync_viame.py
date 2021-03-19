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

rsync -avrRP $HOME/data/private/./_combos viame:data/private
rsync -avrRP $HOME/data/private/US_NE_2018_CFARM_HABCAM/./_dev viame:data/private/US_NE_2018_CFARM_HABCAM
rsync -avrRP $HOME/data/private/US_NE_2017_CFARM_HABCAM/./_dev viame:data/private/US_NE_2017_CFARM_HABCAM
rsync -avrP $HOME/data/private/US_NE_2019_CFARM_HABCAM/raws/./_dev viame:data/private/US_NE_2019_CFARM_HABCAM/raws


rsync -avrLP $HOME/data/./noaa_habcam viame:data


# Move data FROM viame
# TODO: can we use $HOME/data/ as the dest and not have it overwrite the symlink that lives there.
rsync -avrPRLK --exclude detections viame:data/./US_ALASKA_MML_SEALION $HOME/data
rsync -avrPR --exclude detections viame:data/./US_ALASKA_MML_SEALION $HOME/data


# Move data TO viame
rsync -vrPRLK \
    --exclude detections \
    --exclude coco-wip \
    --exclude edits \
    --exclude KITWARE \
    --exclude COUNTED \
    --exclude raw \
    --exclude Images_2017-2018 \
    $HOME/data/./US_ALASKA_MML_SEALION viame:data


/media/joncrall/raid/home/joncrall/data/noaa/
rsync -avrPRLK --exclude detections viame:data/./US_ALASKA_MML_SEALION $HOME/raid/home/joncrall/data/noaa/

$HOME/data/

]$HOME/data/./noaa_habcam viame:data


kwcoco stats --src /home/joncrall/data/private/_combo_cfarm/cfarm_test.mscoco.json
kwcoco stats --src /home/joncrall/data/private/_combo_cfarm/cfarm_vali.mscoco.json
kwcoco stats --src /home/joncrall/data/private/_combo_cfarm/cfarm_train.mscoco.json

kwcoco stats --src ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3.mscoco.json
kwcoco stats --src ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json
kwcoco stats --src ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json


kwcoco union --dst ~/data/private/_combos/train_cfarm_habcam_v1.mscoco.json \
        --src ~/data/private/_combo_cfarm/cfarm_train.mscoco.json \
        ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json

kwcoco union --dst ~/data/private/_combos/vali_cfarm_habcam_v1.mscoco.json \
        --src ~/data/private/_combo_cfarm/cfarm_vali.mscoco.json \
        ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json

kwcoco union --dst ~/data/private/_combos/test_cfarm_habcam_v1.mscoco.json \
        --src ~/data/private/_combo_cfarm/cfarm_test.mscoco.json \
        ~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json

kwcoco stats --src /home/joncrall/data/private/_combos/train_cfarm_habcam_v1.mscoco.json
kwcoco stats --src /home/joncrall/data/private/_combos/vali_cfarm_habcam_v1.mscoco.json
kwcoco stats --src /home/joncrall/data/private/_combos/test_cfarm_habcam_v1.mscoco.json

kwcoco stats --src /home/joncrall/data/private/_combo_cfarm/cfarm_vali.mscoco.json
kwcoco stats --src /home/joncrall/data/private/_combo_cfarm/cfarm_train.mscoco.json
"""
from os.path import basename
from os.path import relpath
from os.path import join
from os.path import normpath
from os.path import dirname
import numpy as np
from os.path import exists
import pandas as pd
import kwimage
import kwarray
import ubelt as ub


def hack_rebase_dset():
    import kwcoco
    self = kwcoco.CocoDataset('/home/joncrall/remote/namek/data/private/_combos/test_cfarm_habcam_v1.mscoco.json')

    new_root = self.img_root

    # HACK
    for gid, img in ub.ProgIter(self.imgs.items(), desc='relativize'):
        path = img['file_name']
        if 'private/_combo_cfarm' in path:
            new_rel_path = normpath(img['file_name']).replace('/home/joncrall/data/private', '..')
        elif 'public/Benthic' in path:
            new_rel_path = path.replace('/home/joncrall/data', '../..')
        else:
            assert False
        # new_path = relpath(realpath(normpath(path)), realpath(new_root))
        assert exists(join(new_root, new_rel_path))
        img['file_name'] = new_rel_path

    self.dump(self.fpath.replace('_v1', '_v2'))

    # prefixes = {}
    # suffixes = {}
    # idx = None

    # import itertools as it
    # worked_idxs = ub.oset()

    # for gid, img in ub.ProgIter(self.imgs.items(), desc='relativize'):
    #     import os
    #     path = img['file_name']
    #     parts = path.split(os.path.sep)
    #     found = None
    #     for idx in it.chain(worked_idxs, range(len(parts))):
    #         suffix = join(*parts[idx:])
    #         cand = join(new_root, suffix)
    #         if exists(cand):
    #             worked_idxs.add(idx)
    #             prefixes[gid] = parts[:idx]
    #             suffixes[gid] = suffix
    #             found = idx
    #             break
    #     if not found:
    #         raise Exception
    #     if gid > 100:
    #         break


# Simplify the categories
catname_map = {
    'American Lobster': 'lobster',
    'squid': 'squid',

    'probably didemnum': 'didemnum',
    'probable scallop-like rock': 'rock',
    'misc manmade objects': 'misc',
    'waved whelk egg mass': 'misc',

    'dust cloud': 'dust cloud',

    'unidentified skate (less than half)': 'skate',
    'winter-little skate': 'skate',
    'unidentified skate': 'skate',
    'unknown skate': 'skate',

    'jonah or rock crab': 'crab',
    'Jonah crab': 'crab',
    'Rock crab': 'crab',
    'unknown crab': 'crab',

    'dead scallop': 'dead sea scallop',
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
    'probable clapper (width)': 'dead sea scallop',
    'clapper':  'dead sea scallop',

    'probable swimming sea scallop inexact': 'swimming sea scallop',
    'probable swimming sea scallop': 'swimming sea scallop',
    'swimming scallop width': 'swimming sea scallop',
    'swimming scallop': 'swimming sea scallop',
    'swimming sea scallop inexact':  'swimming sea scallop',
    'swimming sea scallop width': 'swimming sea scallop',
    'swimming sea scallop': 'swimming sea scallop',
    'probable swimming scallop': 'swimming sea scallop',
    'swimming scallop (width)': 'swimming sea scallop',
    'probable swimming scallop (width)': 'swimming sea scallop',

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

    'atlantic hagfish': 'roundfish',
    'spiny dogfish': 'roundfish',
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
    'Asterias spp': 'seastar',

    'unknown cerianthid': 'cerianthid',

    'snake eel': 'eel',
    'convict worm': 'eel',
    'blackrim cusk-eel': 'eel',

    'unknown mollusk': 'mollusk',

    'hermit crab': 'snail',
    'waved whelk or hermit crab': 'snail',
    'moon snail': 'snail',
    'waved whelk': 'snail',
    'moon snail-like': 'snail',
    'unknown whelk': 'snail',
}


def assign_files_to_assets(asset_dpath, registered_paths):
    """
    Assigns file names in a CSV to an existing asset.
    """
    import os
    fpaths = []
    for root, ds, fs in os.walk(asset_dpath):
        for f in fs:
            fpath = join(root, f)
            fpaths.append(fpath)

    fname_to_regi = ub.group_items(registered_paths, key=basename)
    fname_to_asset = ub.group_items(fpaths, key=basename)

    regi_fname_dups = {
        fname: cands for fname, cands in fname_to_regi.items()
        if len(cands) > 1
    }
    assert not regi_fname_dups, 'registry should not have dups!'

    unhandled_dups = []

    asset_mapping = {}
    for fname, regis in fname_to_regi.items():
        regi = regis[0]
        candidates = fname_to_asset[fname]
        if len(candidates) > 1:
            unhandled_dups.append(candidates)
            print('Duplicate fname = {!r}'.format(fname))
            print('candidates = {}'.format(ub.repr2(candidates, nl=1)))
        else:
            asset_mapping[regi] = candidates[0]

    if unhandled_dups:
        raise Exception('unhandled duplicates')

    return asset_mapping


def convert_cfarm(df, bundle_dpath):
    import multiprocessing
    from kwcoco.util import util_futures
    import kwcoco
    import kwimage

    records = df.to_dict(orient='records')

    for row in ub.ProgIter(records, desc='fix formatting'):
        for k, v in list(row.items()):
            if isinstance(v, str):
                row[k] = v.strip()

    # Scan the records to find what is registered
    cathist = ub.ddict(lambda: 0)
    objname_to_objid = {}
    registered_paths = set()
    for row in ub.ProgIter(records, desc='first pass'):
        object_name = row['Name']
        cathist[object_name] += 1
        objname_to_objid[object_name] = row['Name']
        registered_paths.add(row['Imagename'])
    print('Raw categories:')
    print(ub.repr2(ub.odict(sorted(list(cathist.items()), key=lambda t: t[1]))))

    unknown_cats = False
    for old_cat in objname_to_objid.keys():
        if old_cat not in catname_map:
            print('NEED TO REGISTER: old_cat = {!r}'.format(old_cat))
            unknown_cats = True

    if unknown_cats:
        raise Exception('need to register cats')

    asset_dpath = join(bundle_dpath, '_assets')
    if not exists(asset_dpath):
        raise Exception('Expected _assets to exist')

    asset_mapping = assign_files_to_assets(asset_dpath, registered_paths)

    AUTOCOG = True
    cog_root = ub.ensuredir((asset_dpath, 'cog_rgb'))

    dset_name = basename(bundle_dpath)
    coco_dset = kwcoco.CocoDataset(img_root=bundle_dpath, tag=dset_name)

    workers = min(10, multiprocessing.cpu_count())
    jobs = util_futures.JobPool(mode='thread', max_workers=workers)
    for row in ub.ProgIter(records):
        orig_image_name = row['Imagename']

        if orig_image_name not in asset_mapping:
            raise Exception('Cannot associate image name with an asset')
        else:
            asset_fpath = asset_mapping[orig_image_name]

        file_name = relpath(asset_fpath, bundle_dpath)

        # TODO: de-bayer and preprocess if the TIFImagename exists
        # (this should point to a left-right "processed" image?)
        # image_name = row['TIFImagename']

        # Handle Image
        gid = coco_dset.ensure_image(file_name=file_name)
        img = coco_dset.imgs[gid]

        if img.get('is_bad', False):
            continue
        if 'width' not in img:
            gpath = coco_dset.get_image_fpath(gid)
            try:
                if not exists(gpath):
                    raise Exception
                shape = kwimage.load_image_shape(gpath)
            except Exception:
                img['is_bad'] = True
                print('Bad image gpath = {!r}'.format(gpath))
                continue

            height, width = shape[0:2]
            img['height'] = height
            img['width'] = width

        if AUTOCOG:
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

    if AUTOCOG:
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
    # coco_dset.img_root = dev_root
    bad_images = coco_dset._ensure_imgsize(workers=16, fail=False)
    coco_dset.remove_images(bad_images)

    # Add special tag indicating a stereo image
    for img in coco_dset.imgs.values():
        img['source'] = dset_name

    coco_dset.fpath = join(bundle_dpath, dset_name + '_v4.kwcoco.json')
    coco_dset.dump(coco_dset.fpath, newlines=True)

    # if 0:
    #     import kwplot
    #     import xdev
    #     kwplot.autompl()
    #     for gid in xdev.InteractiveIter(list(coco_dset.imgs.keys())):
    #         coco_dset.show_image(gid)
    #         xdev.InteractiveIter.draw()
    # datasets = train_vali_split(coco_dset)
    # print('datasets = {!r}'.format(datasets))
    # def _split_annot_freq_table(datasets):
    #     tag_to_freq = {}
    #     for tag, tag_dset in datasets.items():
    #         freq = tag_dset.category_annotation_frequency()
    #         tag_to_freq[tag] = freq
    #     df = pd.DataFrame.from_dict(tag_to_freq)
    #     return df
    # print(_split_annot_freq_table(datasets))
    # for tag, tag_dset in datasets.items():
    #     print('{} fpath = {!r}'.format(tag, tag_dset.fpath))
    #     tag_dset.dump(tag_dset.fpath, newlines=True)


def _ensure_rgb_cog(coco_dset, gid, cog_root):
    img = coco_dset.imgs[gid]
    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='', ext='.cog.tif')
    cog_fpath = join(cog_root, cog_fname)

    if not exists(cog_fpath):
        ub.ensuredir(dirname(cog_fpath))
        # Note: probably should be atomic
        imgL = coco_dset.load_image(gid)
        kwimage.imwrite(cog_fpath, imgL, backend='gdal', compress='DEFLATE')
    return cog_fpath


# def train_vali_split(coco_dset):
#     split_gids = _split_train_vali_test_gids(coco_dset)
#     datasets = {}
#     for tag, gids in split_gids.items():
#         tag_dset = coco_dset.subset(gids)
#         img_pcnt = int(round(tag_dset.n_images / coco_dset.n_images, 2) * 100)
#         # ann_pcnt = int(round(tag_dset.n_annots / coco_dset.n_annots, 2) * 100)
#         # suffix = '_{:02d}_{:02d}_{}'.format(img_pcnt, ann_pcnt, tag)
#         suffix = '_{:02d}_{}'.format(img_pcnt, tag)
#         tag_dset.fpath = ub.augpath(coco_dset.fpath, suffix=suffix,
#                                     multidot=True)
#         datasets[tag] = tag_dset

#     return datasets


# def _split_train_vali_test_gids(coco_dset, factor=3):

#     def _stratified_split(gids, cids, n_splits=2, rng=None):
#         """ helper to split while trying to maintain class balance within images """
#         rng = kwarray.ensure_rng(rng)
#         from ndsampler.utils.util_sklearn import StratifiedGroupKFold
#         selector = StratifiedGroupKFold(n_splits=n_splits, random_state=rng,
#                                         shuffle=True)
#         skf_list = list(selector.split(X=gids, y=cids, groups=gids))
#         trainx, testx = skf_list[0]
#         return trainx, testx

#     # Create flat table of image-ids and category-ids
#     gids, cids = [], []
#     images = coco_dset.images()
#     for gid_, cids_ in zip(images, images.annots.cids):
#         cids.extend(cids_)
#         gids.extend([gid_] * len(cids_))

#     # Split into learn/test then split learn into train/vali
#     rng = kwarray.ensure_rng(1617402282)
#     # FIXME: make train bigger with 2
#     test_factor = factor
#     vali_factor = factor
#     learnx, testx = _stratified_split(gids, cids, rng=rng,
#                                       n_splits=test_factor)

#     print('* learn = {}, {}'.format(len(learnx), len(learnx) / len(gids)))
#     print('* test = {}, {}'.format(len(testx), len(testx) / len(gids)))

#     learn_gids = list(ub.take(gids, learnx))
#     learn_cids = list(ub.take(cids, learnx))
#     _trainx, _valix = _stratified_split(learn_gids, learn_cids, rng=rng,
#                                         n_splits=vali_factor)
#     trainx = learnx[_trainx]
#     valix = learnx[_valix]

#     print('* trainx = {}, {}'.format(len(trainx), len(trainx) / len(gids)))
#     print('* valix = {}, {}'.format(len(valix), len(valix) / len(gids)))

#     split_gids = {
#         'train': sorted(set(ub.take(gids, trainx))),
#         'vali': sorted(set(ub.take(gids, valix))),
#         'test': sorted(set(ub.take(gids, testx))),
#     }
#     print('splits = {}'.format(ub.repr2(ub.map_vals(len, split_gids))))
#     return split_gids


def convert_cfarm_2017():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/bioharn/dev'))
    from sync_viame import *  # NOQA

    HACK:
        rsync -avrRP \
            /home/joncrall/data/private/US_NE_2017_CFARM_HABCAM/_dev/./cog_rgb \
            viame:/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM/_assets/
    """
    # csv_fpath = ub.expandpath('~/data/private/US_NE_2017_CFARM_HABCAM/HabCam 2017 dataset1 annotations.csv')
    csv_fpath = ub.expandpath('~/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM/HabCam 2017 dataset1 annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    bundle_dpath = dirname(csv_fpath)
    convert_cfarm(df, bundle_dpath)


def convert_cfarm_2018():
    """
    HACK:
        rsync -avrRP \
            /home/joncrall/data/private/US_NE_2018_CFARM_HABCAM/_dev/./cog_rgb \
            viame:/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2018_CFARM_HABCAM/_assets/
    """

    # csv_fpath =  ub.expandpath('~/data/private/US_NE_2018_CFARM_HABCAM/annotations.csv')
    csv_fpath = ub.expandpath('~/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2018_CFARM_HABCAM/annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    bundle_dpath = dirname(csv_fpath)
    # csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')
    convert_cfarm(df, bundle_dpath)


def convert_cfarm_2019():
    """
    Notes:
        This has several assets:
            * Processed,
            * Left_Old (which used to be called raw), and
            * sample-3d-results

        Processed - contains stiched left / right image pairs
        Left_Old - contains the left half of the data
        3d-sample-results - looks like it contains modified versions of
            processed images

    HACK:
        rsync -avrRP \
            /home/joncrall/data/private/US_NE_2019_CFARM_HABCAM/raws/_dev/./cog_rgb \
            viame:/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM/_assets/
    """
    # csv_fpath =  ub.expandpath('~/data/private/US_NE_2019_CFARM_HABCAM/raws/annotations-corrected.csv')
    csv_fpath = ub.expandpath('~/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM/annotations-corrected.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    bundle_dpath = dirname(csv_fpath)
    convert_cfarm(df, bundle_dpath)


def convert_cfarm_2019_part2():
    """
    HACK:
        rsync -avrRP \
            /home/joncrall/data/private/US_NE_2019_CFARM_HABCAM_PART2/raws/_dev/./cog_rgb \
            viame:/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM/_assets/

    """
    # csv_fpath =  ub.expandpath('~/data/private/US_NE_2019_CFARM_HABCAM_PART2/HabCam 2019 dataset2 annotations.csv')
    csv_fpath = ub.expandpath('~/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM_PART2/HabCam 2019 dataset2 annotations.csv')
    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)
    print('df.columns = {!r}'.format(df.columns))
    bundle_dpath = dirname(csv_fpath)
    convert_cfarm(df, bundle_dpath)


def convert_US_NE_NEFSC_2014_HABCAM_FLATFISH():
    """
    ls US_NE_NEFSC_2014_HABCAM_FLATFISH
    """
    csv_fpath = ub.expandpath('~/data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/flatfish14.habcam_csv')
    assert exists(csv_fpath)
    # Need to know classid to imagename mappings
    df = pd.read_csv(csv_fpath)
    df['class_id']
    # TODO: Unfinished!
    print('df.columns = {!r}'.format(df.columns))
    bundle_dpath = dirname(csv_fpath)
    convert_cfarm(df, bundle_dpath)


def merge():
    import ndsampler
    split_fpaths = {
        'train':  [
            '/home/joncrall/data/private/US_NE_2017_CFARM_HABCAM/_dev/US_NE_2017_CFARM_HABCAM_g001921_a00024144_c0010_v3_44_train.mscoco.json',
            '/home/joncrall/data/private/US_NE_2018_CFARM_HABCAM/_dev/US_NE_2018_CFARM_HABCAM_g001412_a00012452_c0013_v3_44_train.mscoco.json',
            '/home/joncrall/data/private/US_NE_2019_CFARM_HABCAM/raws/_dev/raws_g003795_a00018894_c0012_v3_44_train.mscoco.json',
        ],
        'vali':  [
            '/home/joncrall/data/private/US_NE_2017_CFARM_HABCAM/_dev/US_NE_2017_CFARM_HABCAM_g001921_a00024144_c0010_v3_22_vali.mscoco.json',
            '/home/joncrall/data/private/US_NE_2018_CFARM_HABCAM/_dev/US_NE_2018_CFARM_HABCAM_g001412_a00012452_c0013_v3_22_vali.mscoco.json',
            '/home/joncrall/data/private/US_NE_2019_CFARM_HABCAM/raws/_dev/raws_g003795_a00018894_c0012_v3_22_vali.mscoco.json',
        ],
        'test': [
            '/home/joncrall/data/private/US_NE_2017_CFARM_HABCAM/_dev/US_NE_2017_CFARM_HABCAM_g001921_a00024144_c0010_v3_33_test.mscoco.json',
            '/home/joncrall/data/private/US_NE_2018_CFARM_HABCAM/_dev/US_NE_2018_CFARM_HABCAM_g001412_a00012452_c0013_v3_33_test.mscoco.json',
            '/home/joncrall/data/private/US_NE_2019_CFARM_HABCAM/raws/_dev/raws_g003795_a00018894_c0012_v3_33_test.mscoco.json',
        ],
    }

    splits = {}
    for tag, paths in split_fpaths.items():
        print('tag = {!r}'.format(tag))
        dsets = []
        for fpath in ub.ProgIter(paths, desc='read datasets'):
            dset = ndsampler.CocoDataset(fpath)
            dset.rebase(absolute=True)
            dsets.append(dset)
        splits[tag] = dsets

    out_dpath = ub.ensuredir('/home/joncrall/data/private/_combo_cfarm')

    combo_dsets = {}
    for tag, dsets in splits.items():
        print('merging')
        combo_dset = ndsampler.CocoDataset.union(*dsets, tag=tag)
        combo_dset.fpath = join(out_dpath, 'cfarm_{}.mscoco.json'.format(tag))
        print('{!r}'.format(combo_dset.fpath))
        combo_dset.rebase(out_dpath)
        combo_dsets[tag] = combo_dset

    for tag, combo_dset in combo_dsets.items():
        combo_dset.dump(combo_dset.fpath, newlines=True)

    for tag, combo_dset in combo_dsets.items():

        combo_dset = ndsampler.CocoDataset(combo_dset.fpath)

        for gid, img in ub.ProgIter(list(combo_dset.imgs.items()),
                                    desc='test load gids'):
            imdata = combo_dset.load_image(gid)
            shape = imdata.shape
            assert img['width'] == shape[1]
            assert img['height'] == shape[0]


def convert_public_CFF():
    """
    """
    csv_fpaths = [
        ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2017_CFF_HABCAM/annotations.csv'),
        ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2018_CFF_HABCAM/annotations.csv'),
        ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM/annotations.csv'),
        ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM_PART2/annotations.csv'),
        ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2015_NEFSC_HABCAM/annotations.csv'),
    ]
    for csv_fpath in csv_fpaths:
        print('csv_fpath = {!r}'.format(csv_fpath))
        assert exists(csv_fpath)
        import kwcoco
        from bioharn.io.viame_csv import ViameCSV

        dset = kwcoco.CocoDataset()
        csv = ViameCSV(csv_fpath)
        csv.extend_coco(dset=dset)

        old_catnames = [cat['name'] for cat in dset.cats.values()]

        def map_categories(old_catnames):
            def normalize_catname(name):
                name = name.lower()
                name = name.replace(' ', '_')
                name = name.replace('-', '_')
                name = name.replace('(', '')
                name = name.replace(')', '')
                import re
                name = re.sub('__*', '_', name)
                return name

            valid_catnames = set(catname_map.values())
            catname_map_ = catname_map.copy()
            for key, value in catname_map.items():
                key_ = normalize_catname(key)
                catname_map_[key_] = value
            final_mapping = {}
            for catname in old_catnames:
                catname_ = normalize_catname(catname)
                if catname in valid_catnames:
                    new_catname = catname_
                elif catname_ in catname_map_:
                    new_catname = catname_map_[catname_]
                else:
                    raise Exception(catname_)
                final_mapping[catname] = new_catname
            return final_mapping

        catname_map_ = map_categories(old_catnames)
        dset.rename_categories(catname_map_)

        registered_paths = dset.images().lookup('file_name')
        bundle_dpath = dirname(csv_fpath)
        asset_dpath = bundle_dpath
        mapping = assign_files_to_assets(asset_dpath, registered_paths)

        # Fix image names
        tasks = []
        for key, val in mapping.items():
            img = dset.index.file_name_to_img[key]
            tasks.append((img, val))
        for img, val in tasks:
            img['file_name'] = relpath(val, bundle_dpath)
        dset._build_index()

        dset.img_root = bundle_dpath
        dset.fpath = ub.augpath(
            csv_fpath, ext='.kwcoco.json', multidot=True)
        assert not dset.missing_images()

        print('write dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)


def hack_for_2015_habcam():
    fpath = ub.expandpath('/data/dvc-repos/viame_dvc/Benthic/US_NE_2015_NEFSC_HABCAM/annotations.kwcoco.json')
    import kwcoco
    dset = kwcoco.CocoDataset(fpath)
    corrupted = dset.corrupted_images(check_aux=True, verbose=1)
    assert not corrupted

    cog_dpath = join(dset.img_root, 'Cog')
    missing = []
    assert exists(cog_dpath)
    for img in dset.imgs.values():
        if img['file_name'].startswith('Corrected'):
            cog_fpath = ub.augpath(img['file_name'], dpath=cog_dpath, suffix='_left', ext='.cog.tif')
            if not exists(cog_fpath):
                missing.append(img)
            else:
                fname = relpath(cog_fpath, dset.bundle_dpath)
                img['file_name'] = fname
                img.pop('width', None)
                img.pop('height', None)

    dset.remove_images(missing)
    dset._ensure_imgsize(workers=8, verbose=1)
    dset.dump(dset.fpath)
