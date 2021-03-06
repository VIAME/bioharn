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
from os.path import relpath
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
import kwcoco
from os.path import normpath, realpath, abspath


def preproc_cfarm():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/bioharn/dev'))
    from cfarm_preproc import *  # NOQA
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
        # Debayer raw images
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

    if 1:
        # Convert raw images to COG
        src_fpaths = []
        dst_fpaths = []
        for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
            dset_dir = ub.ensuredir((workdir, key))
            left_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'left', '*.png')))
            right_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'right', '*.png')))

            left_dpath = ub.ensuredir((dset_dir, 'images', 'left'))
            right_dpath = ub.ensuredir((dset_dir, 'images', 'right'))

            for src_fpath in left_png_gpaths:
                dst_fpath = ub.augpath(src_fpath, dpath=left_dpath, ext='.cog.tif')
                src_fpaths.append(src_fpath)
                dst_fpaths.append(dst_fpath)

            for src_fpath in right_png_gpaths:
                dst_fpath = ub.augpath(src_fpath, dpath=right_dpath, ext='.cog.tif')
                src_fpaths.append(src_fpath)
                dst_fpaths.append(dst_fpath)

        if 0:
            from ndsampler.utils.util_gdal import batch_validate_cog
            existing_fpaths = [fpath for fpath in dst_fpaths if exists(fpath)]
            infos = list(batch_validate_cog(existing_fpaths, mode='process', max_workers=2))
            bad_infos = [info for info in infos if info['errors']]
            for info in bad_infos:
                ub.delete(info['fpath'])
            missing_fpaths = [fpath for fpath in dst_fpaths if not exists(fpath)]
            print('missing_fpaths = {!r}'.format(missing_fpaths))

        cog_config = {
            'compress': 'DEFLATE', 'blocksize': 256,
        }
        from ndsampler.utils.util_gdal import batch_validate_cog
        from ndsampler.utils.util_gdal import batch_convert_to_cog
        batch_convert_to_cog(
            src_fpaths, dst_fpaths, cog_config=cog_config, mode='process',
            max_workers=4)

    # Create the COCO datasets (and compute disparities)
    csv_fpaths = {
        '2017_CFARM': join(dpath, 'US_NE_2017_CFARM_HABCAM/HabCam 2017 dataset1 annotations.csv'),
        '2018_CFARM': join(dpath, 'US_NE_2018_CFARM_HABCAM/annotations.csv'),
        '2019_CFARM_P1': join(dpath, 'US_NE_2019_CFARM_HABCAM/raws/annotations-corrected.csv'),
        '2019_CFARM_P2': join(dpath, 'US_NE_2019_CFARM_HABCAM_PART2/HabCam 2019 dataset2 annotations.csv'),
    }
    for key, csv_fpath in csv_fpaths.items():
        print('key = {!r}'.format(key))
        print('csv_fpath = {!r}'.format(csv_fpath))
        assert exists(csv_fpath)
        df = pd.read_csv(csv_fpath)
        img_root = dset_dir = ub.ensuredir((workdir, key))
        print('img_root = {!r}'.format(img_root))
        coco_dset = convert_cfarm(df, img_root)

    # ---
    # Combine into a big dataset

    split_fpaths = ub.ddict(list)
    for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
        dset_dir = ub.ensuredir((workdir, key))
        all_fpath = list(glob.glob(join(dset_dir, '*_v6.mscoco.json')))[0]
        train_fpath = list(glob.glob(join(dset_dir, '*_v6*train*.mscoco.json')))[0]
        vali_fpath = list(glob.glob(join(dset_dir, '*_v6*vali*.mscoco.json')))[0]
        test_fpath = list(glob.glob(join(dset_dir, '*_v6*test*.mscoco.json')))[0]
        split_fpaths['all'].append(all_fpath)
        split_fpaths['train'].append(train_fpath)
        split_fpaths['vali'].append(vali_fpath)
        split_fpaths['test'].append(test_fpath)

    # Include habcam
    split_fpaths.pop('all', None)
    split_fpaths['train'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_train.mscoco.json'))
    split_fpaths['vali'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json'))
    split_fpaths['test'].append(ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json'))

    splits = {}
    out_dpath = ub.ensuredir((workdir, 'combos'))
    for tag, paths in split_fpaths.items():
        dsets = []
        for fpath in ub.ProgIter(paths, desc='read datasets', verbose=3):
            assert exists(fpath)
            print('fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset(fpath)
            dset.dataset.get('img_root', None)
            # try:
            #     # dset.img_root = normpath(dset.img_root)
            #     dset.reroot(realpath(out_dpath), absolute=False)
            # except Exception:
            for gid, img in dset.imgs.items():
                gpath = dset.get_image_fpath(gid)
                gpath = normpath(gpath)
                assert exists(gpath)
                img['file_name'] = abspath(gpath)
                for aux in img['auxillary']:
                    gpath = dset.get_auxillary_fpath(gid, aux['channels'])
                    aux['file_name'] = abspath(gpath)

            dset.dataset.pop('img_root', None)
            assert not dset.missing_images(check_aux=True)
            dsets.append(dset)
        splits[tag] = dsets

    combo_dsets = {}
    for tag, dsets in splits.items():
        print('merging')
        print(ub.repr2(ub.peek(dsets[0].imgs.values())))
        combo_dset = kwcoco.CocoDataset.union(*dsets, tag=tag, img_root='/')

        # standardize image prefixes
        combo_dset.img_root = out_dpath
        real_base = realpath(out_dpath)
        for img in combo_dset.imgs.values():
            img['file_name'] = relpath(realpath(img['file_name']), real_base)
            for aux in img['auxillary']:
                aux['file_name'] = relpath(realpath(aux['file_name']), real_base)

        print(ub.repr2(ub.peek(combo_dset.imgs.values())))
        missing = combo_dset.missing_images(check_aux=True)
        assert not missing
        combo_dset.fpath = join(out_dpath, 'habcam_cfarm_v6_{}.mscoco.json'.format(tag))
        print('{!r}'.format(combo_dset.fpath))
        # combo_dset.rebase(out_dpath)
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


def hack():
    fpath = '/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v5_train.mscoco.json'
    for fpath in glob.glob('/home/joncrall/remote/namek/data/noaa_habcam/combos/*v5*.mscoco.json'):
        from ndsampler.utils import validate_cog
        dset = kwcoco.CocoDataset(fpath)
        for gid, img in dset.imgs.items():
            img['file_name'] = realpath(img['file_name'])
            img.pop('aux', None)
            img.pop('auxillary', None)

        bad_gids = []
        for gid, img in ub.ProgIter(dset.imgs.items(), total=len(dset.imgs)):
            gpath = img['file_name']
            try:
                warn, err, details = validate_cog.validate(gpath)
            except Exception:
                err = 1
            if err:
                print('err = {!r}'.format(err))
                bad_gids.append(gid)
        dset._build_index()
        dset.remove_images(bad_gids)

        print('dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)

    # '/home/joncrall/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v5_vali.mscoco.json'


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
    import kwcoco
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

    coco_dset = kwcoco.CocoDataset(img_root=img_root)

    left_img_root = join('images', 'left')
    right_img_root = join('images', 'right')

    # Two loop, handle images first
    # ASSUME WE HAVE de-bayered and preprocessed
    image_names = list(ub.unique([row['TIFImagename'] for row in records]))
    for image_name in image_names:
        left_cog_name = ub.augpath(image_name, dpath=left_img_root, ext='.cog.tif')
        right_cog_name = ub.augpath(image_name, dpath=right_img_root, ext='.cog.tif')
        left_gpath = join(img_root, left_cog_name)
        right_gpath = join(img_root, right_cog_name)
        assert exists(right_gpath), f'{right_gpath}'
        assert exists(left_gpath), f'{left_gpath}'
        # Handle Image
        gid = coco_dset.ensure_image(file_name=left_cog_name)
        coco_dset.imgs[gid]['right_cog_name'] = right_cog_name

    coco_dset._ensure_imgsize(workers=4)

    for row in ub.ProgIter(records, desc='convert annotations'):
        # image_name = row['Imagename']
        # ASSUME WE HAVE de-bayered and preprocessed
        image_name = row['TIFImagename']
        left_cog_name = ub.augpath(image_name, dpath=left_img_root, ext='.cog.tif')
        img = coco_dset.index.file_name_to_img[left_cog_name]
        gid = img['id']

        # Handle Category
        object_name = row['Name']
        cat_name = catname_map[object_name]
        assert cat_name is not None
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
        base=dset_name + '_v6.mscoco.json'.format())

    coco_dset.rebase(img_root)
    coco_dset.img_root = img_root

    if 1:
        # Compute dispartiy maps
        img = coco_dset.imgs[1]
        img_dsize = (img['width'], img['height'])
        from bioharn.stereo import StereoCalibration
        cali_root = ub.expandpath('~/remote/namek/data/noaa_habcam/extras/calibration_habcam_2019_leotta')
        extrinsics_fpath = join(cali_root, 'extrinsics.yml')
        intrinsics_fpath = join(cali_root, 'intrinsics.yml')
        cali = StereoCalibration.from_cv2_yaml(intrinsics_fpath, extrinsics_fpath)
        camera1 = cali.cameras[1]
        camera2 = cali.cameras[2]
        camera1._precache(img_dsize)
        camera2._precache(img_dsize)

        jobs = util_futures.JobPool('thread', max_workers=4)

        for gid in ub.ProgIter(list(coco_dset.imgs.keys()), desc='disparity'):
            job = jobs.submit(_ensure_cfarm_disparity_frame, coco_dset, gid, cali)
            job.gid = gid

        for job in ub.ProgIter(jobs.as_completed(), desc='disparity', total=len(jobs)):
            disp_unrect_fpath1 = job.result()
            disp_unrect_fname1 = relpath(disp_unrect_fpath1, coco_dset.img_root)
            img = coco_dset.imgs[job.gid]
            img['auxillary'] = [{
                'channels': 'disparity',
                'file_name': disp_unrect_fname1,
            }]

    if 0:
        import kwplot
        import xdev
        kwplot.autompl()
        gids = sorted(coco_dset.find_representative_images())
        gid = gids[0]
        for gid in xdev.InteractiveIter(list(gids)):
            disp_fpath = coco_dset.get_auxillary_fpath(gid, 'disparity')
            coco_dset.show_image(gid)
            disp_img = kwimage.imread(disp_fpath)
            disp_heat = kwimage.make_heatmask(disp_img)
            kwplot.imshow(disp_heat)
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


def _ensure_cfarm_disparity_frame(coco_dset, gid, cali):
    """
    import kwplot
    kwplot.autompl()
    import xdev
    aids = coco_dset.index.cid_to_aids[coco_dset._alias_to_cat('flatfish')['id']]
    gids = list(coco_dset.annots(aids).gids)

    from bioharn.stereo import StereoCalibration
    cali_root = ub.expandpath('~/remote/namek/data/noaa_habcam/extras/calibration_habcam_2019_leotta')
    extrinsics_fpath = join(cali_root, 'extrinsics.yml')
    intrinsics_fpath = join(cali_root, 'intrinsics.yml')
    cali = StereoCalibration.from_cv2_yaml(intrinsics_fpath, extrinsics_fpath)

    for gid in xdev.InteractiveIter(gids):
        img = coco_dset.imgs[gid]
        gpath2 = join(coco_dset.img_root, img['right_cog_name'])
        gpath1 = join(coco_dset.img_root, img['file_name'])

        info = _compute_disparity(gpath1, gpath2, cali)

        _disp1_rect = kwimage.make_heatmask(info['disp1_rect'], 'magma')[..., 0:3]
        _disp1_unrect = kwimage.make_heatmask(info['disp1_unrect'], 'magma')[..., 0:3]
        canvas_rect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_rect, 0.6), info['img1_rect']])
        canvas_unrect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_unrect, 0.6), info['img1']])

        _, ax1 = kwplot.imshow(info['img1_rect'], pnum=(2, 3, 1), fnum=1, title='left rectified')
        _, ax2 = kwplot.imshow(info['disp1_rect'], pnum=(2, 3, 2), fnum=1, title='left rectified disparity')
        _, ax3 = kwplot.imshow(info['img2_rect'], pnum=(2, 3, 3), fnum=1, title='right rectified')
        _, ax4 = kwplot.imshow(canvas_rect, pnum=(2, 3, 4), fnum=1, title='left rectified')
        _, ax5 = kwplot.imshow(canvas_unrect, pnum=(2, 3, 5), fnum=1, title='left unrectified')
        _, ax6 = kwplot.imshow(info['img2'], pnum=(2, 3, 6), fnum=1, title='right unrectified')

        if coco_dset is not None:
            annots = coco_dset.annots(gid=gid)
            unrect_dets1 = annots.detections
            rect_dets1 = unrect_dets1.warp(camera1.rectify_points)
            rect_dets1.draw(ax=ax1)
            rect_dets1.boxes.draw(ax=ax2)
            unrect_dets1.boxes.draw(ax=ax5)
        xdev.InteractiveIter.draw()

    """
    import kwimage
    img = coco_dset.imgs[gid]
    gpath1 = join(coco_dset.img_root, img['file_name'])
    gpath2 = join(coco_dset.img_root, img['right_cog_name'])

    disp_unrect_dpath1 = ub.ensuredir((
        coco_dset.img_root, 'images', 'left_disparity_unrect'))
    disp_unrect_fpath1 = join(disp_unrect_dpath1, basename(img['file_name']))

    if not exists(disp_unrect_fpath1):
        info = _compute_disparity(gpath1, gpath2, cali)
        # Note: probably should be atomic
        kwimage.imwrite(disp_unrect_fpath1, info['disp1_unrect'],
                        backend='gdal', compress='DEFLATE')
    return disp_unrect_fpath1


def _compute_disparity(gpath1, gpath2, cali, coco_dset=None, gid=None):
    import kwimage
    from bioharn.disparity import multipass_disparity
    img1 = kwimage.imread(gpath1)
    img2 = kwimage.imread(gpath2)

    camera1 = cali.cameras[1]
    camera2 = cali.cameras[2]

    img1_rect = camera1.rectify_image(img1)
    img2_rect = camera2.rectify_image(img2)
    disp1_rect = multipass_disparity(
        img1_rect, img2_rect, scale=0.5, as01=True)
    disp1_rect = disp1_rect.astype(np.float32)

    disp1_unrect = camera1.unrectify_image(
        disp1_rect, interpolation='linear')

    info = {
        'img1': img1,
        'img2': img2,
        'img1_rect': img1_rect,
        'img2_rect': img2_rect,
        'disp1_rect': disp1_rect,
        'disp1_unrect': disp1_unrect,
    }

    if 0:
        import kwplot
        kwplot.autompl()
        _disp1_rect = kwimage.make_heatmask(info['disp1_rect'], 'magma')[..., 0:3]
        _disp1_unrect = kwimage.make_heatmask(info['disp1_unrect'], 'magma')[..., 0:3]
        canvas_rect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_rect, 0.6), info['img1_rect']])
        canvas_unrect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_unrect, 0.6), info['img1']])

        _, ax1 = kwplot.imshow(info['img1_rect'], pnum=(2, 3, 1), fnum=1, title='left rectified')
        _, ax2 = kwplot.imshow(info['disp1_rect'], pnum=(2, 3, 2), fnum=1, title='left rectified disparity')
        _, ax3 = kwplot.imshow(info['img2_rect'], pnum=(2, 3, 3), fnum=1, title='right rectified')
        _, ax4 = kwplot.imshow(canvas_rect, pnum=(2, 3, 4), fnum=1, title='left rectified')
        _, ax5 = kwplot.imshow(canvas_unrect, pnum=(2, 3, 5), fnum=1, title='left unrectified')
        _, ax6 = kwplot.imshow(info['img2'], pnum=(2, 3, 6), fnum=1, title='right unrectified')

        if coco_dset is not None:
            annots = coco_dset.annots(gid=gid)
            unrect_dets1 = annots.detections
            rect_dets1 = unrect_dets1.warp(camera1.rectify_points)
            rect_dets1.draw(ax=ax1)
            rect_dets1.boxes.draw(ax=ax2)
            unrect_dets1.boxes.draw(ax=ax5)

            if 0:
                import cv2
                # Is there a way to aprox map cam1 to cam2?
                pts1 = unrect_dets1.boxes.corners()
                K1, D1 = ub.take(camera1, ['K', 'D'])
                K2, D2 = ub.take(camera2, ['K', 'D'])
                # Remove dependence on camera1 intrinsics / distortion
                pts1_norm = cv2.undistortPoints(pts1, K1, D1)[:, 0, :]
                R, T = ub.take(cali.extrinsics, ['R', 'T'])
                pts1_xyz = kwimage.add_homog(pts1_norm)
                pts1_xyz[:, 2] = 0
                # info['disp1_rect']
                # pts2_xyz = kwimage.warp_points(R, pts1_xyz) + T.T
                rvec = cv2.Rodrigues(R)[0]
                tvec = T.ravel()
                pts2, _ = cv2.projectPoints(pts1_xyz, rvec, tvec, cameraMatrix=K2, distCoeffs=D2)
                pts2 = pts2.reshape(-1, 2)

    return info


def _ensure_rgb_cog(dset, gid, cog_root):
    img = dset.imgs[gid]
    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='', ext='.cog.tif')
    cog_fpath = join(cog_root, cog_fname)
    ub.ensuredir(dirname(cog_fpath))

    if not exists(cog_fpath):
        # Note: probably should be atomic
        img1 = dset.load_image(gid)
        kwimage.imwrite(cog_fpath, img1, backend='gdal', compress='DEFLATE')
    return cog_fpath


def train_vali_split(coco_dset, **kw):
    split_gids = _split_train_vali_test_gids(coco_dset, **kw)
    datasets = {}
    for tag, gids in split_gids.items():
        tag_dset = coco_dset.subset(gids)
        img_pcnt = int(round(tag_dset.n_images / coco_dset.n_images, 2) * 100)
        # ann_pcnt = int(round(tag_dset.n_annots / coco_dset.n_annots, 2) * 100)
        # suffix = '_{:02d}_{:02d}_{}'.format(img_pcnt, ann_pcnt, tag)
        # suffix = '_{:02d}_{}'.format(img_pcnt, tag)
        suffix = '_{}'.format(tag)
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


def rework_cats_may_priority():
    """
    Rework such that certain classes are given priority and other are given a
    weight of zero.

    rsync may_priority_habcam* viame:data/noaa_habcam/combos/
    """
    import kwcoco
    root = ub.expandpath('~/data/noaa_habcam/combos')
    dsets = [
        kwcoco.CocoDataset(join(root, 'habcam_cfarm_v6_train.mscoco.json')),
        kwcoco.CocoDataset(join(root, 'habcam_cfarm_v6_vali.mscoco.json')),
        kwcoco.CocoDataset(join(root, 'habcam_cfarm_v6_test.mscoco.json')),
    ]

    clapper = {
        'sea scallop clapper inexact': 'clapper sea scallop',
        'sea scallop clapper width': 'clapper sea scallop',
        'sea scallop clapper': 'clapper sea scallop',
        'probable clapper (width)': 'clapper sea scallop',
    }

    for dset in dsets:
        clapper_cid = dset.ensure_category('clapper')
        dset.rename_categories({'fish': 'roundfish'})
        for ann in ub.ProgIter(dset.anns.values(), total=len(dset.anns)):
            meta = ann['meta']
            if 'object_name' in meta:
                raw_cat = meta['object_name']
            elif 'Name' in meta:
                raw_cat = meta['Name']
            else:
                assert 0
            if raw_cat in clapper:
                ann['category_id'] = clapper_cid
        dset._build_index()
        dset.fpath = ub.augpath(dset.fpath, prefix='may_priority_')

    for dset in dsets:
        dset.dump(dset.fpath, newlines=True)


def fix_cats_offset():
    """
    Rework such that certain classes are given priority and other are given a
    weight of zero.

    cd ~/data/noaa_habcam/combos
    rsync may_priority_habcam* viame:data/noaa_habcam/combos/


    """
    import kwcoco
    root = ub.expandpath('~/data/noaa_habcam/combos')
    dsets = [
        kwcoco.CocoDataset(join(root, 'may_priority_habcam_cfarm_v6_train.mscoco.json')),
        kwcoco.CocoDataset(join(root, 'may_priority_habcam_cfarm_v6_vali.mscoco.json')),
        kwcoco.CocoDataset(join(root, 'may_priority_habcam_cfarm_v6_test.mscoco.json')),
    ]

    for dset in dsets:
        for img in dset.imgs.values():
            if img['source'] in {'2017_CFARM', '2018_CFARM'}:
                for aid in dset.gid_to_aids[img['id']]:
                    ann = dset.anns[aid]
                    ann['bbox'][0] += 39  # right shift of 39 pixels
        dset.fpath = dset.fpath.replace('v6', 'v7')

    for dset in dsets:
        dset.dump(dset.fpath, newlines=True)


def _reduce_test_size():
    """

    kwcoco split --src may_priority_habcam_cfarm_v7_test.mscoco.json --dst1 tmp1.mscoco.json --dst2 tmp2.mscoco.json --factor=4
    kwcoco stats --src tmp1.mscoco.json tmp2.mscoco.json

    kwcoco union --src may_priority_habcam_cfarm_v7_test.mscoco.json tmp1.mscoco.json --dst habcam_cfarm_v8_train.mscoco.json
    mv tmp2.mscoco.json habcam_cfarm_v8_test.mscoco.json
    cp may_priority_habcam_cfarm_v7_vali.mscoco.json habcam_cfarm_v8_vali.mscoco.json


    kwcoco stats --src habcam_cfarm_v8_*.mscoco.json


    """
