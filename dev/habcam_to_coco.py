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
            'source': 'habcam_2015',
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
    coco_fpath =  ub.expandpath('~/raid/data/noaa/Habcam_2015_AnnotatedObjects_all.mscoco.json')
    coco_dset.dataset['img_root'] = '2015_Habcam_photos'
    coco_dset.img_root = ub.expandpath('~/raid/data/noaa/2015_Habcam_photos')
    from PIL import Image
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

    coco_dset.dataset['img_root'] = '2015_Habcam_photos'
    coco_dset.fpath = coco_fpath
    coco_dset.dump(coco_fpath, newlines=True)
    train_dset, vali_dset = train_vali_split(coco_dset)

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
    images = coco_dset.images()
    cids_per_image = images.annots.cids
    gid_to_cids = ub.odict(zip(images.gids, cids_per_image))

    # Note: this removes images with no annotations, which is what we want here
    gids = [gid for gid, cids_ in gid_to_cids.items() for cid in cids_]
    cids = [cid for gid, cids_ in gid_to_cids.items() for cid in cids_]

    groups = gids

    import kwil
    rng = kwil.ensure_rng(1617402282)
    factor = 4
    skf = kwil.StratifiedGroupKFold(n_splits=factor, random_state=rng)
    skf_list = list(skf.split(X=gids, y=cids, groups=groups))
    trainx, valix = skf_list[0]

    train_gids = sorted(ub.unique(ub.take(gids, trainx)))
    vali_gids = sorted(ub.unique(ub.take(gids, valix)))

    train_dset = coco_dset.subset(train_gids)
    vali_dset = coco_dset.subset(vali_gids)
    print('train_dset = {!r}'.format(train_dset))
    print('vali_dset = {!r}'.format(vali_dset))

    train_fpath = ub.augpath(coco_dset.fpath, suffix='_train', multidot=True)
    vali_fpath = ub.augpath(coco_dset.fpath, suffix='_vali', multidot=True)

    train_dset.dump(train_fpath, newlines=True)
    vali_dset.dump(vali_fpath, newlines=True)
    return train_dset, vali_dset
