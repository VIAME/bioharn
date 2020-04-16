"""
notes for preprocessing cfarm data

https://github.com/VIAME/VIAME


https://data.kitware.com/api/v1/item/5e76a11baf2e2eed355b5228/download

pip install girder-client
eval "$(_GIRDER_CLI_COMPLETE=source girder-client)"


mkdir -p /home/joncrall/data/raid/viame_install
cd /home/joncrall/data/raid/viame_install
girder-client --api-url https://data.kitware.com/api/v1 download 5e76a11baf2e2eed355b5228
tar -xvf VIAME-v0.10.8-Ubuntu18.04-64Bit.tar.gz

cd /home/joncrall/data/raid/viame_install/viame
source /home/joncrall/data/raid/viame_install/viame/setup_viame.sh


CURRENT DATA LAYOUT 2020-04-14:

    private
    ├── _combo_cfarm
    ├── _combos
    ├── US_NE_2017_CFARM_HABCAM
    │   └── _dev
    │       └── cog_rgb
    ├── US_NE_2018_CFARM_HABCAM
    │   └── _dev
    │       ├── cog_rgb
    │       └── images
    │           └── cog
    ├── US_NE_2019_CFARM_HABCAM
    │   ├── processed
    │   │   └── _dev
    │   │       └── cog_rgb
    │   ├── raws
    │   │   └── _dev
    │   │       └── cog_rgb
    │   └── sample-3d-results
    │       ├── flounder
    │       └── swimmers
    └── US_NE_2019_CFARM_HABCAM_PART2
    public
    ├── Benthic
    │   └── US_NE_2015_NEFSC_HABCAM
    │       ├── cog
    │       ├── Corrected
    │       ├── _dev
    │       └── disparities
    └── _dev



https://github.com/VIAME/VIAME/tree/master/examples/image_enhancement
debayer_and_enhance.sh
consumes debayer_and_enhance.sh
input_list_raw_images.txt
by default though that can be changed in script
outputs in current directory
one these days I should adjust so it runs it on both camera sides independently instead of jointly


├── <DATASET_NAME_1>
│   ├── raw
│   │   ├── ...
│   ├── processed
│   │   ├── left
│   │   │   ├── ...
│   │   ├── right
│   │   │   ├── ...
│   │   └── disparity
│   │   │   ├── ...
│   ├── annotations.csv
│   ├── annotations.mscoco.json
│   └── _developer_stuff
│  
├── <DATASET_NAME_2>
...


+ <DATASET_NAME>
|
+-- images
|
+-- images

"""
import numpy as np
import kwimage
import kwarray
import pandas as pd
from os.path import basename
from os.path import dirname
from os.path import exists
import ubelt as ub
from os.path import join
import glob
from ndsampler.utils import util_futures


def preproc_cfarm():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/bioharn/dev'))
    from cfarm_preproc import *  # NOQA
    # vimtk encountered a syntax error
    """
    root = ub.expandpath('$HOME/remote/namek/')
    workdir = ub.ensuredir((root, 'data/noaa_habcam'))
    viame_install = join(root, 'data/raid/viame_install/viame')
    dpath = join(root, 'data/private')

    raw_dpaths = {
        '2017_CFARM': join(dpath, 'US_NE_2017_CFARM_HABCAM'),
        '2018_CFARM': join(dpath, 'US_NE_2018_CFARM_HABCAM'),
        '2019_CFARM_P1': join(dpath, 'US_NE_2019_CFARM_HABCAM/raws'),
        '2019_CFARM_P2': join(dpath, 'US_NE_2019_CFARM_HABCAM_PART2'),
    }

    gpaths = {}
    for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
        print('raw_dpath = {!r}'.format(raw_dpath))
        raw_gpaths = sorted(glob.glob(join(raw_dpath, '*.tif')))
        gpaths[key] = raw_gpaths
        print('#raw_gpaths = {!r}'.format(len(raw_gpaths)))

    if 1:
        for key, raw_gpaths in ub.ProgIter(gpaths.items()):
            jobs = util_futures.JobPool('thread', max_workers=8)
            dset_dir = ub.ensuredir((workdir, key))
            left_dpath = ub.ensuredir((dset_dir, 'raw', 'left'))
            right_dpath = ub.ensuredir((dset_dir, 'raw', 'right'))
            for raw_gpath in raw_gpaths:
                jobs.submit(split_raws, raw_gpath, left_dpath, right_dpath)
            left_paths = []
            right_paths = []
            for job in ub.ProgIter(jobs.as_completed(), total=len(jobs),
                                   desc='collect split jobs'):
                left_gpath, right_gpath = job.result()
                left_paths.append(left_gpath)
                right_paths.append(right_gpath)
            do_debayer(left_dpath, left_paths, viame_install)
            do_debayer(right_dpath, right_paths, viame_install)

        for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
            dset_dir = ub.ensuredir((workdir, key))
            left_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'left', '*.png')))
            right_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'right', '*.png')))

            left_dpath = ub.ensuredir((dset_dir, 'images', 'left'))
            right_dpath = ub.ensuredir((dset_dir, 'images', 'right'))

            jobs = util_futures.JobPool('thread', max_workers=8)
            for src_fpath in left_png_gpaths:
                dst_fpath = ub.augpath(src_fpath, dpath=left_dpath, ext='.cog.tif')
                jobs.submit(convert_to_cog, src_fpath, dst_fpath)

            for src_fpath in right_png_gpaths:
                dst_fpath = ub.augpath(src_fpath, dpath=right_dpath, ext='.cog.tif')
                jobs.submit(convert_to_cog, src_fpath, dst_fpath)
            for job in ub.ProgIter(jobs.as_completed(), total=len(jobs),
                                   desc='collect convert-to-cog jobs'):
                job.result()

    csv_fpaths = {
        '2017_CFARM': join(dpath, 'US_NE_2017_CFARM_HABCAM/HabCam 2017 dataset1 annotations.csv'),
        '2018_CFARM': join(dpath, 'US_NE_2018_CFARM_HABCAM/annotations.csv'),
        '2019_CFARM_P1': join(dpath, 'US_NE_2019_CFARM_HABCAM/raws/annotations-corrected.csv'),
        '2019_CFARM_P2': join(dpath, 'US_NE_2019_CFARM_HABCAM_PART2/HabCam 2019 dataset2 annotations.csv'),
    }
    for key, csv_fpath in csv_fpaths.items():
        assert exists(csv_fpath)
        df = pd.read_csv(csv_fpath)
        print('df.columns = {!r}'.format(df.columns))
        img_root = dset_dir = ub.ensuredir((workdir, key))
        coco_dset = convert_cfarm(df, img_root)

    split_fpaths = ub.ddict(list)
    for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
        dset_dir = ub.ensuredir((workdir, key))
        train_fpath = list(glob.glob(join(dset_dir, '*_v5*train*.mscoco.json')))[0]
        vali_fpath = list(glob.glob(join(dset_dir, '*_v5*vali*.mscoco.json')))[0]
        test_fpath = list(glob.glob(join(dset_dir, '*_v5*test*.mscoco.json')))[0]
        split_fpaths['train'].append(train_fpath)
        split_fpaths['vali'].append(vali_fpath)
        split_fpaths['test'].append(test_fpath)

    split_fpaths['train'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json'))
    split_fpaths['vali'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json'))
    split_fpaths['test'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json'))

    splits = {}
    import kwcoco
    out_dpath = ub.ensuredir((workdir, 'combos'))
    for tag, paths in split_fpaths.items():
        dsets = []
        for fpath in ub.ProgIter(paths, desc='read datasets'):
            assert exists(fpath)
            dset = kwcoco.CocoDataset(fpath)
            dset.dataset.pop('img_root')
            dset.img_root = normpath(dset.img_root)
            dset.reroot(out_dpath, absolute=False)
            dset.missing_images()
            dsets.append(dset)
        splits[tag] = dsets

    combo_dsets = {}
    for tag, dsets in splits.items():
        print('merging')
        combo_dset = kwcoco.CocoDataset.union(*dsets, tag=tag)
        combo_dset.fpath = join(out_dpath, 'cfarm_{}.mscoco.json'.format(tag))
        print('{!r}'.format(combo_dset.fpath))
        combo_dset.rebase(out_dpath)
        combo_dsets[tag] = combo_dset

    for tag, combo_dset in combo_dsets.items():
        combo_dset.dump(combo_dset.fpath, newlines=True)

    for tag, combo_dset in combo_dsets.items():

        combo_dset = kwcoco.CocoDataset(combo_dset.fpath)

        for gid, img in ub.ProgIter(list(combo_dset.imgs.items()),
                                    desc='test load gids'):
            imdata = combo_dset.load_image(gid)
            shape = imdata.shape
            assert img['width'] == shape[1]
            assert img['height'] == shape[0]

    if 0:
        import kwplot
        import xdev
        kwplot.autompl()
        gids = coco_dset.find_representative_images()
        for gid in xdev.InteractiveIter(list(gids)):
            coco_dset.show_image(gid)
            xdev.InteractiveIter.draw()


def convert_to_cog(src_fpath, dst_fpath):
    from ndsampler.utils.util_gdal import _cli_convert_cloud_optimized_geotiff
    if not exists(dst_fpath):
        _cli_convert_cloud_optimized_geotiff(
            src_fpath, dst_fpath, compress='LZW', blocksize=256)


def do_debayer(dpath, fpaths, viame_install):
    debayer_input_fpath = join(dpath, 'input_list_raw_images.txt')
    with open(debayer_input_fpath, 'w') as file:
        file.write('\n'.join(fpaths))
    sh_text = ub.codeblock(
        r'''
        #!/bin/sh
        # Setup VIAME Paths (no need to run multiple times if you already ran it)
        export VIAME_INSTALL="{viame_install}"
        source $VIAME_INSTALL/setup_viame.sh
        # Run pipeline
        kwiver runner $VIAME_INSTALL/configs/pipelines/filter_debayer_and_enhance.pipe \
                      -s input:video_filename={debayer_input_fpath}

        ''').format(viame_install=viame_install, debayer_input_fpath=debayer_input_fpath)
    sh_fpath = join(dpath, 'debayer.sh')
    ub.writeto(sh_fpath, sh_text)
    ub.cmd('chmod +x ' + sh_fpath)
    ub.cmd('bash ' + sh_fpath, cwd=dpath, shell=0, verbose=3)


def split_raws(raw_gpath, left_dpath, right_dpath):
    import kwimage
    left_gpath = ub.augpath(raw_gpath, dpath=left_dpath)
    right_gpath = ub.augpath(raw_gpath, dpath=right_dpath)
    if not exists(right_gpath) or not exists(right_gpath):
        raw_img = kwimage.imread(raw_gpath)
        h, w = raw_img.shape[0:2]
        half_w = w // 2
        left_img = raw_img[:, :half_w]
        right_img = raw_img[:, half_w:]
        kwimage.imwrite(right_gpath, right_img)
        kwimage.imwrite(left_gpath, left_img)
    return left_gpath, right_gpath

# Simplify the categories
catname_map = {
    'American Lobster': 'lobster',
    'squid': 'squid',

    'probably didemnum': 'didemnum',
    'probable scallop-like rock': 'rock',
    'misc manmade objects': 'misc',
    'waved whelk egg mass': 'misc',
    'whelk': 'misc',
    'unknown whelk': 'skate',

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
    'dead scallop': 'dead sea scallop',
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
}


def convert_cfarm(df, img_root):
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

    left_img_root = join('images', 'left')
    right_img_root = join('images', 'right')

    for row in ub.ProgIter(records):
        # image_name = row['Imagename']

        # ASSUME WE HAVE de-bayered and preprocessed
        image_name = row['TIFImagename']

        left_cog_name = ub.augpath(image_name, dpath=left_img_root, ext='.cog.tif')
        right_cog_name = ub.augpath(image_name, dpath=right_img_root, ext='.cog.tif')

        left_gpath = join(img_root, left_cog_name)
        right_gpath = join(img_root, right_cog_name)
        assert exists(right_gpath), f'{right_gpath}'
        assert exists(left_gpath), f'{left_gpath}'

        # Handle Image
        gid = coco_dset.ensure_image(file_name=left_cog_name, aux={
            'right': right_cog_name,
        })
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
            weight *= 0.2
        if 'inexact' in object_name:
            weight *= 0.2

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

    # Remove hyper-small annotations, they are probably bad
    weird_anns = []
    for ann in coco_dset.anns.values():
        if np.sqrt(ann['bbox'][2] * ann['bbox'][3]) < 10:
            weird_anns.append(ann)
    coco_dset.remove_annotations(weird_anns)

    coco_dset.dataset.pop('img_root', None)
    coco_dset.img_root = img_root
    bad_images = coco_dset._ensure_imgsize(workers=16, fail=False)
    coco_dset.remove_images(bad_images)

    # Add special tag indicating a stereo image
    dset_name = basename(img_root)
    for img in coco_dset.imgs.values():
        img['source'] = dset_name

    # stats = coco_dset.basic_stats()
    # suffix = 'g{n_imgs:06d}_a{n_anns:08d}_c{n_cats:04d}'.format(**stats)
    coco_dset.fpath = ub.augpath(
        '', dpath=img_root, ext='',
        base=dset_name + '_v5.mscoco.json'.format())

    coco_dset.rebase(img_root)
    coco_dset.img_root = img_root

    if 0:
        import kwplot
        import xdev
        kwplot.autompl()
        gids = coco_dset.find_representative_images()
        for gid in xdev.InteractiveIter(list(gids)):
            coco_dset.show_image(gid)
            xdev.InteractiveIter.draw()

    datasets = train_vali_split(coco_dset, vali_factor=5)
    print('datasets = {!r}'.format(datasets))

    def _split_annot_freq_table(datasets):
        tag_to_freq = {}
        for tag, tag_dset in datasets.items():
            freq = tag_dset.category_annotation_frequency()
            tag_to_freq[tag] = freq
        df = pd.DataFrame.from_dict(tag_to_freq)
        return df
    print(_split_annot_freq_table(datasets))

    coco_dset.dump(coco_dset.fpath, newlines=True)
    for tag, tag_dset in datasets.items():
        print('{} fpath = {!r}'.format(tag, tag_dset.fpath))
        tag_dset.dump(tag_dset.fpath, newlines=True)

    return coco_dset


def _ensure_rgb_cog(dset, gid, cog_root):
    img = dset.imgs[gid]
    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='', ext='.cog.tif')
    cog_fpath = join(cog_root, cog_fname)
    ub.ensuredir(dirname(cog_fpath))

    if not exists(cog_fpath):
        # Note: probably should be atomic
        imgL = dset.load_image(gid)
        kwimage.imwrite(cog_fpath, imgL, backend='gdal', compress='DEFLATE')
    return cog_fpath


def train_vali_split(coco_dset, **kw):
    split_gids = _split_train_vali_test_gids(coco_dset, **kw)
    datasets = {}
    for tag, gids in split_gids.items():
        tag_dset = coco_dset.subset(gids)
        img_pcnt = int(round(tag_dset.n_images / coco_dset.n_images, 2) * 100)
        # ann_pcnt = int(round(tag_dset.n_annots / coco_dset.n_annots, 2) * 100)
        # suffix = '_{:02d}_{:02d}_{}'.format(img_pcnt, ann_pcnt, tag)
        suffix = '_{:02d}_{}'.format(img_pcnt, tag)
        tag_dset.fpath = ub.augpath(coco_dset.fpath, suffix=suffix,
                                    multidot=True)
        datasets[tag] = tag_dset
    return datasets


def _split_train_vali_test_gids(coco_dset, test_factor=3, vali_factor=6):

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
                                      n_splits=test_factor)

    print('* learn = {}, {}'.format(len(learnx), len(learnx) / len(gids)))
    print('* test = {}, {}'.format(len(testx), len(testx) / len(gids)))

    learn_gids = list(ub.take(gids, learnx))
    learn_cids = list(ub.take(cids, learnx))
    _trainx, _valix = _stratified_split(learn_gids, learn_cids, rng=rng,
                                        n_splits=vali_factor)
    trainx = learnx[_trainx]
    valix = learnx[_valix]

    print('* trainx = {}, {}'.format(len(trainx), len(trainx) / len(gids)))
    print('* valix = {}, {}'.format(len(valix), len(valix) / len(gids)))

    split_gids = {
        'train': sorted(set(ub.take(gids, trainx))),
        'vali': sorted(set(ub.take(gids, valix))),
        'test': sorted(set(ub.take(gids, testx))),
    }
    print('splits = {}'.format(ub.repr2(ub.map_vals(len, split_gids))))
    return split_gids


def merge():
    split_fpaths['train'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json'))
    split_fpaths['vali'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json'))
    split_fpaths['test'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json')

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