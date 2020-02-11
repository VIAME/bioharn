from os.path import join
import numpy as np
import ubelt as ub
import warnings
from sklearn.utils.validation import check_array
from sklearn.externals.six.moves import zip
from sklearn.model_selection._split import (_BaseKFold,)


def main():
    """
    Convert the habcam CSV format to an MS-COCO like format
    """
    import pandas as pd
    csv_fpath =  ub.expandpath('~/raid/data/noaa/2015_Habcam_photos/Habcam_2015_AnnotatedObjects.csv')
    csv_fpath =  ub.expandpath('~/remote/viame/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_AnnotatedObjects.csv')
    assert exists(csv_fpath)
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

    coco_dset.fpath = ub.augpath(csv_fpath, ext='', base='Habcam_2015_{}_v3.mscoco.json'.format(suffix))

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


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds cross-validator with Grouping

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of GroupKFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedGroupKFold, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        """
        Args:
            X (ndarray): data
            y (ndarray): labels
            groups (ndarray): groupids for items. Items with the same groupid
                must be placed in the same group.

        Returns:
            list: test_folds

        Example:
            >>> groups = [1, 1, 3, 4, 2, 2, 7, 8, 8]
            >>> y      = [1, 1, 1, 1, 2, 2, 2, 3, 3]
            >>> X = np.empty((len(y), 0))
            >>> rng = kwarray.ensure_rng(0)
            >>> self = StratifiedGroupKFold(random_state=rng)
            >>> skf_list = list(self.split(X=X, y=y, groups=groups))
            ...
            >>> import ubelt as ub
            >>> print(ub.repr2(skf_list, nl=1, with_dtype=False))
            [
                (np.array([2, 3, 4, 5, 6]), np.array([0, 1, 7, 8])),
                (np.array([0, 1, 2, 7, 8]), np.array([3, 4, 5, 6])),
                (np.array([0, 1, 3, 4, 5, 6, 7, 8]), np.array([2])),
            ]
        """
        import kwarray
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value')
            n_splits = self.n_splits
            y = np.asarray(y)
            n_samples = y.shape[0]

            unique_y, y_inversed = np.unique(y, return_inverse=True)
            n_classes = max(unique_y) + 1
            unique_groups, group_idxs = kwarray.group_indices(groups)
            grouped_y = kwarray.apply_grouping(y, group_idxs)
            grouped_y_counts = np.array([
                np.bincount(y_, minlength=n_classes) for y_ in grouped_y])

            target_freq = grouped_y_counts.sum(axis=0)
            target_freq = target_freq.astype(np.float)
            target_ratio = target_freq / float(target_freq.sum())

            # Greedilly choose the split assignment that minimizes the local
            # * squared differences in target from actual frequencies
            # * and best equalizes the number of items per fold
            # Distribute groups with most members first
            split_freq = np.zeros((n_splits, n_classes))
            # split_ratios = split_freq / split_freq.sum(axis=1)
            split_ratios = np.ones(split_freq.shape) / split_freq.shape[1]
            split_diffs = ((split_freq - target_ratio) ** 2).sum(axis=1)
            sortx = np.argsort(grouped_y_counts.sum(axis=1))[::-1]
            grouped_splitx = []

            # import ubelt as ub
            # print(ub.repr2(grouped_y_counts, nl=-1))
            # print('target_ratio = {!r}'.format(target_ratio))

            for count, group_idx in enumerate(sortx):
                # print('---------\n')
                group_freq = grouped_y_counts[group_idx]
                cand_freq = (split_freq + group_freq)
                cand_freq = cand_freq.astype(np.float)
                cand_ratio = cand_freq / cand_freq.sum(axis=1)[:, None]
                cand_diffs = ((cand_ratio - target_ratio) ** 2).sum(axis=1)
                # Compute loss
                losses = []
                # others = np.nan_to_num(split_diffs)
                other_diffs = np.array([
                    sum(split_diffs[x + 1:]) + sum(split_diffs[:x])
                    for x in range(n_splits)
                ])
                # penalize unbalanced splits
                ratio_loss = other_diffs + cand_diffs
                # penalize heavy splits
                freq_loss = split_freq.sum(axis=1)
                freq_loss = freq_loss.astype(np.float)
                freq_loss = freq_loss / freq_loss.sum()
                losses = ratio_loss + freq_loss
                #-------
                splitx = np.argmin(losses)
                # print('losses = %r, splitx=%r' % (losses, splitx))
                split_freq[splitx] = cand_freq[splitx]
                split_ratios[splitx] = cand_ratio[splitx]
                split_diffs[splitx] = cand_diffs[splitx]
                grouped_splitx.append(splitx)

            test_folds = np.empty(n_samples, dtype=np.int)
            for group_idx, splitx in zip(sortx, grouped_splitx):
                idxs = group_idxs[group_idx]
                test_folds[idxs] = splitx

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedGroupKFold, self).split(X, y, groups)

"""

/home/joncrall/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v1_test.mscoco.json

"""
