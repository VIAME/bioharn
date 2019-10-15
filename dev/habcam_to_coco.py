from os.path import join
import numpy as np
import ubelt as ub


def main():
    """
    Convert the habcam CSV format to an MS-COCO like format
    """
    import pandas as pd
    csv_fpath =  ub.expandpath('~/raid/data/noaa/2015_Habcam_photos/Habcam_2015_AnnotatedObjects.csv')
    df = pd.read_csv(csv_fpath)

    records = df.to_dict(orient='records')

    cathist = ub.ddict(lambda: 0)
    for row in ub.ProgIter(records):
        object_name = row['Object_Name']
        cathist[object_name] += 1
    print('Raw categories:')
    print(ub.repr2(ub.odict(sorted(list(cathist.items()), key=lambda t: t[1]))))

    # Note: Clappers are dead. They differ from just shells because the hinge
    # is intact. They are generally open more widely than a live scallop

    # Simplify the categories
    catname_map = {
        'American Lobster': None,
        'monkfish': None,
        'squid': None,
        'probably didemnum': None,
        'dead sea scallop inexact': 'dead sea scallop',
        'unidentified fish (less than half)': None,
        'unidentified flatfish (less than half)': None,
        'probable scallop-like rock': None,
        'probable swimming sea scallop inexact': 'swimming sea scallop',
        'probable dead sea scallop width': 'dead sea scallop',
        'sea scallop clapper inexact': 'dead sea scallop',
        'probable dead sea scallop inexact': 'dead sea scallop',
        'unidentified fish': None,
        'convict worm': None,
        'dead sea scallop': 'dead sea scallop',
        'unidentified skate (less than half)': None,
        'probable swimming sea scallop': 'swimming sea scallop',
        'sea scallop clapper width': 'dead sea scallop',
        'sea scallop clapper': 'dead sea scallop',
        'unidentified roundfish (less than half)': None,
        'swimming sea scallop inexact':  'swimming sea scallop',
        'probable dead sea scallop': 'dead sea scallop',
        'dust cloud': None,
        'waved whelk': None,
        'unidentified flatfish': None,
        'unidentified skate': None,
        'swimming sea scallop width': 'swimming sea scallop',
        'probable live sea scallop inexact': 'live sea scallop',
        'jonah or rock crab': None,
        'probable live sea scallop': 'live sea scallop',
        'live sea scallop inexact': 'live sea scallop',
        'unidentified roundfish': None,
        'swimming sea scallop': 'swimming sea scallop',
        'live sea scallop width': 'live sea scallop',
        'live sea scallop': 'live sea scallop',
    }

    if True:
        # do one category
        for key in catname_map:
            if catname_map[key] is not None:
                catname_map[key] = 'scallop'

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
    from PIL import Image
    coco_dset.dataset['img_root'] = '2015_Habcam_photos'
    coco_dset.img_root = ub.expandpath('~/raid/data/noaa/2015_Habcam_photos')
    bad_images = []
    for img in ub.ProgIter(coco_dset.dataset['images'],
                           verbose=1):
        gpath = join(coco_dset.img_root, img['file_name'])
        if 'width' not in img:
            try:
                pil_img = Image.open(gpath)
                w, h = pil_img.size
                pil_img.close()
            except OSError:
                bad_images.append(img)
            else:
                img['width'] = w
                img['height'] = h
    coco_dset.remove_images(bad_images)

    # Add special tag indicating a stereo image
    for img in coco_dset.imgs.values():
        img['source'] = 'habcam_2015_stereo'

    stats = coco_dset.basic_stats()
    suffix = 'g{n_imgs:06d}_a{n_anns:08d}_c{n_cats:04d}'.format(**stats)

    coco_dset.dataset['img_root'] = '2015_Habcam_photos'
    coco_dset.fpath = ub.expandpath('~/raid/data/noaa/Habcam_2015_{}_v2.mscoco.json'.format(suffix))

    datasets = train_vali_split(coco_dset)
    print('datasets = {!r}'.format(datasets))

    coco_dset.dump(coco_dset.fpath, newlines=True)
    for tag, tag_dset in datasets.items():
        print('{} fpath = {!r}'.format(tag, tag_dset.fpath))
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
    import kwil
    import kwarray

    def _stratified_split(gids, cids, n_splits=2, rng=None):
        """ helper to split while trying to maintain class balance within images """
        rng = kwarray.ensure_rng(rng)
        selector = kwil.StratifiedGroupKFold(n_splits=n_splits, random_state=rng)
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
    rng = kwil.ensure_rng(1617402282)
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
