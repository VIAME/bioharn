"""

mkdir -p ~/data/raid/public/Benthic/US_NE_2015_NEFSC_HABCAM
ln -s ~/data/raid/public ~/data/public


rsync -vrltD /tmp/software /nas10 | pv -lep -s 42

rsync -avrP --stats --human-readable --info=progress2  viame:data/public/Benthic/US_NE_2015_NEFSC_HABCAM/./Corrected $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM
rsync -avrP --stats --human-readable --info=progress2  $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/./disparities viame:data/public/Benthic/US_NE_2015_NEFSC_HABCAM
rsync -avrP --stats --human-readable --info=progress2  $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/./cog viame:data/public/Benthic/US_NE_2015_NEFSC_HABCAM
rsync -avrP --stats --human-readable --info=progress2  $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/./_dev viame:data/public/Benthic/US_NE_2015_NEFSC_HABCAM

rsync -avP --stats --human-readable  viame:data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected/annotations.habcam_csv $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected/annotations.habcam_csv

"""
from os.path import normpath
from os.path import basename
from os.path import join
import numpy as np
import ubelt as ub
from os.path import dirname
from os.path import exists


def main():
    """
    Convert the habcam CSV format to an MS-COCO like format
    """
    import pandas as pd
    csv_fpath =  ub.expandpath('~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected/annotations.habcam_csv')
    # csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')

    with open(csv_fpath, 'r') as file:
        print(file.readline())
        print(file.readline())
        print(file.readline())
        print(file.readline())

    assert exists(csv_fpath)
    df = pd.read_csv(csv_fpath)

    records = df.to_dict(orient='records')

    cathist = ub.ddict(lambda: 0)
    objname_to_objid = {}
    for row in ub.ProgIter(records):
        object_name = row['Object_Name']
        cathist[object_name] += 1
        objname_to_objid[object_name] = row['Object_Id']
    print('Raw categories:')
    print(ub.repr2(ub.odict(sorted(list(cathist.items()), key=lambda t: t[1]))))

    if 0:
        old_mapping = ub.codeblock('''
            live_scallop 185 197 207 208 213 215 515 523 525 531 537 907 912 915 916 919 920
            skate 340 342 343 345 346 347 348 524 533 1016 1036 1037 1038 1044 1045 1046 1047 1048 1049 1081 1082
            roundfish 377 1001 1002 1014 358 374 375 389 1040 1041 1042 1043 1066 1067 1068 353 397 1052 1053 368 398 360 1064 1065 355 1058 1059 356 415 1039 1050 1051 337 338 339 1060 1061 357 1054 1055 390 1056 1057 359 384 401 410 1062 1063 386 397 1071 1072 404
            flatfish 351 362 363 366 367 370 376 1011 1012 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1003 1004 1015 418
            eels 352 361 380 415 1069 1070 1100 336
            crab 258
            whelk 158
        ''')

        objid_to_objname = ub.invert_dict(objname_to_objid)
        for line in old_mapping.split('\n'):
            parts = line.split(' ')
            group = parts[0]
            members = [objid_to_objname.get(int(p), p) for p in parts[1:]]
            print('group = {!r}'.format(group))
            print('members = {!r}'.format(members))
            print('')

    # Note: Clappers are dead. They differ from just shells because the hinge
    # is intact. They are generally open more widely than a live scallop

    # Simplify the categories
    catname_map = {

        'dead sea scallop inexact': 'dead sea scallop',
        'probable swimming sea scallop inexact': 'swimming sea scallop',
        'probable dead sea scallop width': 'dead sea scallop',
        'sea scallop clapper inexact': 'dead sea scallop',
        'probable dead sea scallop inexact': 'dead sea scallop',
        'dead sea scallop': 'dead sea scallop',

        'American Lobster': 'lobster',
        'monkfish': 'monkfish',
        'squid': 'squid',

        'probably didemnum': 'didemnum',
        'probable scallop-like rock': 'rock',

        'probable swimming sea scallop': 'swimming sea scallop',
        'sea scallop clapper width': 'dead sea scallop',
        'sea scallop clapper': 'dead sea scallop',

        'unidentified fish (less than half)': 'fish',
        'unidentified fish': 'fish',

        'unidentified roundfish (less than half)': 'roundfish',
        'unidentified roundfish': 'roundfish',

        'unidentified flatfish (less than half)': 'flatfish',
        'unidentified flatfish': 'flatfish',

        'convict worm': 'convict worm',

        'dust cloud': 'dust cloud',
        'waved whelk': 'waved whelk',

        'unidentified skate (less than half)': 'skate',
        'unidentified skate': 'skate',

        'jonah or rock crab': 'crab',

        'swimming sea scallop inexact':  'swimming sea scallop',
        'probable dead sea scallop': 'dead sea scallop',
        'swimming sea scallop width': 'swimming sea scallop',
        'probable live sea scallop inexact': 'live sea scallop',
        'probable live sea scallop': 'live sea scallop',
        'live sea scallop inexact': 'live sea scallop',
        'swimming sea scallop': 'swimming sea scallop',
        'live sea scallop width': 'live sea scallop',
        'live sea scallop': 'live sea scallop',
    }

    import ndsampler
    coco_dset = ndsampler.CocoDataset()

    def _decode_geom(code, value, weight):
        if code == 'line':
            pt1, pt2 = map(np.array, eval(value.strip(' '), {}))
            cxy = (pt2 + pt1) / 2
            w = h = np.sqrt(((pt2 - pt1) ** 2).sum())
            weight = weight * 0.9
            tl = cxy - (w / 2, h / 2)
            xywh = list(map(float, [tl[0], tl[1], w, h]))
        elif code == 'boundingBox':
            tl, br = map(np.array, eval(value.strip(' '), {}))
            w, h = (br - tl)
            xywh = list(map(float, [tl[0], tl[1], w, h]))
        elif code == 'point':
            pt0, = map(np.array, eval(value.strip(' '), {}))
            weight = weight * 0.1
            # SKIP THESE
            raise NotImplementedError('point')
        elif code == 'circle':
            pt1, pt2 = map(np.array, eval(value.strip(' '), {}))
            weight = weight * 0.9
            cxy = (pt2 + pt1) / 2
            w = h = np.sqrt(((pt2 - pt1) ** 2).sum())
            weight = weight * 0.9
            tl = cxy - (w / 2, h / 2)
            xywh = list(map(float, [tl[0], tl[1], w, h]))
            raise NotImplementedError('circle')
        else:
            raise KeyError(code)
        return xywh, weight

    for row in ub.ProgIter(records):
        geom_text = row['Geometry_Text']
        image_name = row['Image']
        object_id = row['Object_Id']
        object_name = row['Object_Name']

        # add category modifiers
        weight = 1.0
        if 'probable' in object_name:
            weight *= 0.5
        if 'inexact' in object_name:
            weight *= 0.5

        if not isinstance(geom_text, str) and np.isnan(geom_text):
            print('Bad geom: ' + str(row))
            continue

        code, value = geom_text.split(':')
        code = code.strip('"')
        try:
            xywh, weight = _decode_geom(code, value, weight)
        except NotImplementedError:
            continue

        img = coco_dset.index.file_name_to_img.get(image_name, None)
        if img is None:
            gid = coco_dset.add_image(file_name=image_name)
        else:
            gid = img['id']

        cat_name = catname_map[object_name]
        if cat_name is None:
            cat_name = 'other'
            print('cat_name = {!r}'.format(cat_name))

        try:
            cid = coco_dset._resolve_to_cid(cat_name)
        except KeyError:
            cid = coco_dset.add_category(cat_name)

        ann = {
            'category_id': cid,
            'image_id': gid,
            'bbox': xywh,
            'weight': weight,
            'meta': {
                'geom_code': code,
                'geom_data': value,
                'object_id': object_id,
                'object_name': object_name,
            }
        }
        coco_dset.add_annotation(**ann)

    # raw_dset = coco_dset.copy()
    # Other has some weird (bad?) anns in it, lets just remove it
    coco_dset.remove_categories(['other'])

    # Remove hyper-small annotations, they are probably bad
    weird_anns = []
    for ann in coco_dset.anns.values():
        if np.sqrt(ann['bbox'][2] * ann['bbox'][3]) < 10:
            weird_anns.append(ann)
    coco_dset.remove_annotations(weird_anns)

    # populate image size / remove bad images
    coco_dset.dataset.pop('img_root', None)
    coco_dset.img_root = dirname(csv_fpath)
    bad_images = coco_dset._ensure_imgsize(workers=16, fail=False)

    coco_dset.remove_images(bad_images)

    # Add special tag indicating a stereo image
    for img in coco_dset.imgs.values():
        img['source'] = 'habcam_2015_stereo'

    stats = coco_dset.basic_stats()
    suffix = 'g{n_imgs:06d}_a{n_anns:08d}_c{n_cats:04d}'.format(**stats)

    dset_root = dirname(dirname(csv_fpath))
    annot_dpath = ub.ensuredir((dset_root, '_dev'))
    coco_dset.fpath = ub.augpath('', dpath=annot_dpath, ext='', base='Habcam_2015_{}_v3.mscoco.json'.format(suffix))

    coco_dset.rebase(dset_root)
    coco_dset.dataset['img_root'] = '..'
    coco_dset.img_root = normpath(join(dirname(coco_dset.fpath), '..'))

    if True:
        from ndsampler.utils import util_futures
        jobs = util_futures.JobPool(mode='thread', max_workers=10)
        # hack in precomputed disparities
        for img in coco_dset.imgs.values():
            gid = img['id']
            job = jobs.submit(_ensure_habcam_disparity_frame, coco_dset, gid)
            job.gid = gid
            # assert False

        for job in ub.ProgIter(jobs, desc='collect results', verbose=3):
            gid = job.gid
            disp_fname = job.result()
            img = coco_dset.imgs[gid]
            data_dims = ((img['width'] // 2), img['height'])
            # Add auxillary channel information
            img['aux'] = [
                {
                    'channels': ['disparity'],
                    'file_name': disp_fname,
                    'dims': data_dims,
                }
            ]

        from ndsampler.utils import util_futures
        jobs = util_futures.JobPool(mode='thread', max_workers=10)
        # hack in precomputed disparities
        for img in coco_dset.imgs.values():
            gid = img['id']
            job = jobs.submit(_ensure_habcam_rgb_cogs, coco_dset, gid)
            job.gid = gid
            # assert False

        for job in ub.ProgIter(jobs, desc='collect results', verbose=3):
            gid = job.gid
            cog_fname = job.result()
            img = coco_dset.imgs[gid]
            # Hack so the a the main image is a cog file.
            if cog_fname != img['file_name']:
                img['file_name'] = cog_fname
                img['width'] = img['width'] // 2

    datasets = train_vali_split(coco_dset)
    print('datasets = {!r}'.format(datasets))

    coco_dset.dump(coco_dset.fpath, newlines=True)
    for tag, tag_dset in datasets.items():
        print('{} fpath = {!r}'.format(tag, tag_dset.fpath))
        print(ub.repr2(tag_dset.category_annotation_frequency()))
        tag_dset.dump(tag_dset.fpath, newlines=True)

    """
    # To Inspect

    weird_anns = []
    for ann in coco_dset.anns.values():

    dlens = [np.sqrt(ann['bbox'][2] * ann['bbox'][3]) for ann in coco_dset.anns.values()]
    dlens = np.array(dlens)
    draw_data_pmf(dlens)

    if ann['bbox'][2] > 300:
        weird_anns.append(ann)

    weird_anns = []
    for ann in coco_dset.anns.values():
        if np.sqrt(ann['bbox'][2] * ann['bbox'][3]) < 10:
            weird_anns.append(ann)

    import xdev
    for ann in xdev.InteractiveIter(weird_anns):
        print('ann = {!r}'.format(ann))
        coco_dset.show_image(aid=ann['id'])
        xdev.InteractiveIter.draw()

    for ann in coco_dset.anns.values():
        if ann['meta']['geom_code'] == 'line':
            pass
        if 'clapper' in ann['meta']['object_name']:
            break
            pass

        pass

    coco_dset.show_image(aid=107976)

    coco_dset.show_image(aid=ann['id'])

    import kwplot
    kwplot.autompl()
    """


def _ensure_habcam_rgb_cogs(dset, gid):
    import kwimage

    img = dset.imgs[gid]
    # image_fname = img['file_name']
    # cog_dpath = ub.ensuredir((dset.img_root))
    # cog_fname = join('cogarities', ub.augpath(image_fname, suffix='_left_cog_v7', ext='.cog.tif'))
    # cog_fpath = join(cog_dpath, cog_fname)

    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='cog', suffix='_left', ext='.cog.tif')
    cog_fpath = join(dset.img_root, cog_fname)
    ub.ensuredir(dirname(cog_fpath))

    if not exists(cog_fpath):
        # Note: probably should be atomic
        img3 = dset.load_image(gid)
        imgL = img3[:, 0:img3.shape[1] // 2]
        kwimage.imwrite(cog_fpath, imgL, backend='gdal',
                        compress='DEFLATE')
    return cog_fname


def _ensure_habcam_disparity_frame(dset, gid):
    from bioharn.detect_dataset import multipass_disparity
    import kwimage

    img = dset.imgs[gid]
    # image_fname = img['file_name']
    # disp_dpath = ub.ensuredir((dset.img_root))
    # disp_fname = join('disparities', ub.augpath(image_fname, suffix='_left_disp_v7', ext='.cog.tif'))
    # disp_fpath = join(disp_dpath, disp_fname)

    fname = basename(img['file_name'])
    disp_fname = ub.augpath(fname, dpath='disparities', suffix='_left_disp_v7', ext='.cog.tif')
    disp_fpath = join(dset.img_root, disp_fname)
    ub.ensuredir(dirname(disp_fpath))

    if not exists(disp_fpath):
        # Note: probably should be atomic
        img3 = dset.load_image(gid)
        imgL = img3[:, 0:img3.shape[1] // 2]
        imgR = img3[:, img3.shape[1] // 2:]
        img_disparity = multipass_disparity(
            imgL, imgR, scale=0.5, as01=True)
        img_disparity = img_disparity.astype(np.float32)

        kwimage.imwrite(disp_fpath, img_disparity, backend='gdal',
                        compress='DEFLATE')

    return disp_fname


def find_anchors3(train_dset, reduction=32, num_anchors=5):
    from sklearn import cluster
    all_wh = np.array([ann['bbox'][2:4] for ann in train_dset.anns.values()])
    all_imgwh = np.array([
        (train_dset.imgs[ann['image_id']]['width'] // 2, train_dset.imgs[ann['image_id']]['height'])
        for ann in train_dset.anns.values()
    ])
    all_norm_wh = all_wh / all_imgwh
    ogrid_wh = 18
    all_ogrid_wh = all_norm_wh * (ogrid_wh / reduction)
    algo = cluster.KMeans(
        n_clusters=num_anchors, n_init=20, max_iter=10000, tol=1e-6,
        algorithm='elkan', verbose=0)
    algo.fit(all_ogrid_wh)
    anchors = algo.cluster_centers_
    return anchors


def draw_data_pmf(data, bw_factor=0.05, color='red', nbins=500):
    import kwplot
    import scipy

    data_pdf = scipy.stats.gaussian_kde(data, bw_factor)
    data_pdf.covariance_factor = bw_factor

    xmin = data.min()
    xmax = data.max()

    counts, edges = np.histogram(data, bins=np.linspace(xmin, xmax, nbins))
    percents = counts / counts.sum()
    centers = (edges[0:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    xdata = np.linspace(xmin, xmax, nbins)
    ydata = data_pdf(xdata)
    normalizer = percents.max() / ydata.max()
    ydata = ydata * normalizer

    color01 = kwplot.Color(color).as01()
    ax = kwplot.multi_plot(xdata=xdata, ydata=ydata, color=color01, marker='',
                           fnum=1, doclf=0)
    ax.bar(centers, height=percents, width=widths, color=color, alpha=.5)


def train_vali_split(coco_dset):

    split_gids = _split_train_vali_test_gids(coco_dset)
    datasets = {}
    for tag, gids in split_gids.items():
        tag_dset = coco_dset.subset(gids)
        tag_dset.fpath = ub.augpath(coco_dset.fpath, suffix='_' + tag, multidot=True)
        datasets[tag] = tag_dset

    return datasets


def _split_train_vali_test_gids(coco_dset, factor=2):
    import kwarray

    def _stratified_split(gids, cids, n_splits=2, rng=None):
        """ helper to split while trying to maintain class balance within images """
        rng = kwarray.ensure_rng(rng)
        from ndsampler.utils.util_sklearn import StratifiedGroupKFold
        selector = StratifiedGroupKFold(n_splits=n_splits, random_state=rng)
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

"""

/home/joncrall/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v1_test.mscoco.json

"""
if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/habcam_to_coco.py
    """
    main()
