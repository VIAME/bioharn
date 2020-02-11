from os.path import realpath
from os.path import relpath
from os.path import dirname
from os.path import exists
from os.path import splitext
from os.path import isfile
import ubelt as ub
from os.path import basename
from os.path import isdir
from os.path import join
import kwimage
import cv2
import numpy as np


def process_dots(orig_fpath, dot_fpath, debug=False, verbose=0):

    """
    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/bioharn/dev'))
        from process_dots import *  # NOQA
        orig_fpath = '/home/joncrall/raid/data/noaa/KITWARE/CountedImgs/2011_Original+CountedLoRes/Original/20110627_SSLC0066_Orig.JPG'
        dot_fpath = '/home/joncrall/raid/data/noaa/KITWARE/CountedImgs/2011_Original+CountedLoRes/Counted_Processed/20110627_SSLC0066_C.jpg'
        dot_data = kwimage.imread(dot_fpath)[1500:-1700, 4900:]
        orig_data = kwimage.imread(orig_fpath)[1500:-1700, 4900:]
    """
    import kwplot

    if verbose:
        print('Read data')
    dot_data = kwimage.imread(dot_fpath)
    orig_data = kwimage.imread(orig_fpath)

    # dot_data = kwimage.imread(dot_fpath)[:1500, :2000]
    # orig_data = kwimage.imread(orig_fpath)[:1500, :2000]

    dot_mag = np.sqrt((dot_data.astype(np.float32) ** 2).sum(axis=2))

    if verbose:
        print('Computing censor mask')
    # Part of the dot image is censored with a black brush stroke.
    censor_mask = dot_mag < 15
    # Remove small regions from being detected as part of the black censor

    # Clean up any remaining small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        censor_mask.astype(np.uint8), 8, cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]
    chosen_labels = np.where(areas < 1000)[0][1:]
    for i in chosen_labels:
        x, y, w, h = stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                               cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
        region = (slice(y, y + h), slice(x, x + w))
        censor_mask[region] = False

    # kernel = np.ones(15)
    # censor_mask = cv2.morphologyEx(
    #     censor_mask.astype(np.uint8),
    #     cv2.MORPH_CLOSE, kernel, iterations=5).astype(np.bool)

    kernel = np.ones(15)
    censor_mask = cv2.morphologyEx(
        censor_mask.astype(np.uint8),
        cv2.MORPH_DILATE, kernel, iterations=5).astype(np.bool)

    if verbose:
        print('Computing residual')
    signed_residual = orig_data.astype(np.int32) - dot_data.astype(np.int32)

    orig_residual = np.abs(signed_residual).astype(np.uint8)
    residual = orig_residual.copy()
    # orig_isresidual = (orig_residual > 0).astype(np.uint8) * 255

    if verbose:
        print('Supress residual noise')
    residual[(residual.sum(axis=2) != 0) & (residual.max(axis=2) < 27)]
    noise_mask = residual.max(axis=2) < 20

    residual[noise_mask] = 0
    residual[censor_mask] = 0

    kernel = np.ones(3)
    smoothed_residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel)
    smoothed_residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel)

    if verbose:
        print('Detecting dots in residual')
    isdot = (smoothed_residual.max(axis=2) > 29)
    colored_dots = dot_data * isdot[..., None]

    gray = kwimage.convert_colorspace(colored_dots, 'rgb', 'gray')
    isdot2 = (gray > 0).astype(np.uint8) * 255

    if verbose:
        print('Detecting circles in dots')

    # Clean up any remaining small connected components
    METHOD = 'HOUGH'
    METHOD = 'CC'
    if METHOD == 'HOUGH':
        # circles = cv2.HoughCircles(isdot2, method=cv2.HOUGH_GRADIENT,
        #                            dp=1,
        #                            minDist=7,
        #                            param1=10,
        #                            param2=5,
        #                            minRadius=4,
        #                            maxRadius=6)

        circles = cv2.HoughCircles(isdot2, method=cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=5,
                                   param1=10,  # edge detection threshold
                                   param2=4,   # smaller = more detections
                                   minRadius=3, maxRadius=6)

        if circles is None:
            circles = np.empty((1, 0, 3))[0]
            if debug:
                kwplot.autompl()
                kwplot.imshow(dot_data)
            raise Exception('no hough circles found')
        else:
            # All the radii should be similar
            import kwarray
            circles = circles[0]

            # Remove circles with radii that deviate from the median
            radii = circles[..., 2]
            stats = kwarray.stats_dict(radii, median=True)
            flags = np.abs(radii - stats['med']) < 1
            circles = circles[flags]

    elif METHOD == 'CC':
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            isdot2.astype(np.uint8), 8, cv2.CV_32S)
        areas = stats[:, cv2.CC_STAT_AREA]
        chosen_labels = np.where((areas < 1000))[0]
        ave = np.median(areas[chosen_labels])
        isdot3 = isdot2.copy()
        circles = []
        for i in chosen_labels:
            x, y, w, h = stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
            area = areas[i]
            box_area = w * h
            if box_area > area * 3:
                # Skip long skinny boxes
                if verbose:
                    print('box_area = {!r}'.format(box_area))
                    print('area = {!r}'.format(area))
                continue

            if np.abs(area - ave) > 25:
                continue

            # Mark the area as processed
            isdot3[y:y + h, x: x + w] = 0
            circles.append([x + w / 2, y + w / 2, (w + h) / 4])

        # The CC method is pretty good at getting lone sea lions, but we need
        # hough for the ones that are touching

        circles = np.array(circles)

        if np.any(isdot3):
            if len(circles) == 0:
                minRadius = 3
                maxRadius = 6
            else:
                minRadius = int(np.floor(circles.T[2].min() - .1))
                maxRadius = int(np.ceil(circles.T[2].max() + .1))
            extra_circles = cv2.HoughCircles(
                isdot3,
                method=cv2.HOUGH_GRADIENT,
                dp=1, minDist=5,
                param1=10,  # edge detection threshold
                param2=4,   # smaller = more detections
                minRadius=minRadius,
                maxRadius=maxRadius,
            )

            if extra_circles is not None:
                if verbose:
                    print('Found {} extra circles'.format(len(extra_circles[0])))
                circles = circles.tolist() + extra_circles[0].tolist()
            circles = np.array(circles)
    else:
        raise KeyError(METHOD)

    if len(circles) == 0:
        circles = np.empty((0, 3))

    xs = circles[..., 0].ravel()
    ys = circles[..., 1].ravel()
    radii = circles[..., 2].ravel()
    if verbose:
        print('Found {} circles'.format(len(xs)))

    # radii = np.median(circles[:, :, 2])
    # print('radii = {!r}'.format(radii))
    # print('circles = {!r}'.format(circles))

    # Colors indicate the classes
    if verbose:
        print('Determining circle colors')
    colors = []
    for x, y, r in zip(xs.ravel(), ys.ravel(), radii.ravel()):
        low_x = x - r / 2
        high_x = x + r / 2
        low_y = y - r / 2
        high_y = y + r / 2
        # subpxl_sl = (slice(low_y, high_y), slice(low_x, high_x))
        # kwimage.subpixel_slice(colored_dots, subpxl_sl)

        low_x = int(np.floor(x - r / 4))
        high_x = int(np.ceil(x + r / 4))
        low_y = int(np.floor(y - r / 4))
        high_y = int(np.ceil(y + r / 4))
        dscrt_sl = (slice(low_y, high_y), slice(low_x, high_x))

        sample_colors = colored_dots[dscrt_sl].reshape(-1, 3)
        sample_colors = np.array([c for c in sample_colors if not np.all(c == 0)])

        med_color = np.median(sample_colors, axis=0)
        # mean_color = np.mean(colors, axis=0)
        # pts = np.array([[y, x]])
        # subpxl_color = kwimage.subpixel_getvalue(colored_dots, pts)
        # print('mean_color = {!r}'.format(mean_color))
        # print('subpxl_color = {!r}'.format(subpxl_color))

        colors.append(med_color)

    if verbose:
        print('debug = {!r}'.format(debug))
    if debug:
        print('building debug canvas')
        plt = kwplot.autoplt()
        canvas = dot_data.copy()
        canvas = kwimage.overlay_alpha_images(
            kwimage.ensure_alpha_channel(isdot2, alpha=0.8),
            kwimage.ensure_alpha_channel(canvas, alpha=1.0),
        )
        canvas = kwimage.ensure_uint255(canvas[..., 0:3])
        circles_2 = np.uint16(np.around(circles))
        for i in circles_2:
            # draw the outer circle
            # cv2.circle(canvas, (i[0], i[1]), i[2], (0, 255, 0), 1)
            cv2.circle(canvas, (i[0], i[1]), i[2], (255, 0, 0), 1)
            # draw the center of the circle
            # ccolored_dotsv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), 3)

    if debug == 1:
        print('show debug')
        # Crop to relevant region
        pad = 50
        min_x = max(int(np.floor(circles_2.T[0].min())) - pad, 0)
        max_x = int(np.ceil(circles_2.T[0].max())) + pad
        min_y = max(int(np.floor(circles_2.T[1].min())) - pad, 0)
        max_y = int(np.ceil(circles_2.T[1].max())) + pad

        sl = (slice(min_y, max_y), slice(min_x, max_x))
        sub_canvas = canvas[sl]
        kwplot.imshow(sub_canvas, fnum=2, doclf=True)

        if len(circles_2) <= 3 or len(circles_2) > 1000:
            debug = 2

    if debug == 2:
        print('show debug2')
        s = 0.5
        kwplot.imshow(kwimage.imresize(orig_data, scale=s), fnum=1, pnum=(3, 3, 1), title='orig')
        kwplot.imshow(kwimage.imresize(dot_data, scale=s), fnum=1, pnum=(3, 3, 2), title='dots')
        kwplot.imshow(kwimage.imresize(censor_mask.astype(np.float32), scale=s).clip(0, 1), fnum=1, pnum=(3, 3, 3), title='censor')

        kwplot.imshow(kwimage.imresize(noise_mask.astype(np.uint8) * 255, scale=s), fnum=1, pnum=(3, 3, 4), title='noise_mask')

        kwplot.imshow(kwimage.imresize(orig_residual, scale=s), fnum=1, pnum=(3, 3, 5), title='residual')
        kwplot.imshow(kwimage.imresize(smoothed_residual, scale=s), fnum=1, pnum=(3, 3, 6), title='opened residual')

        kwplot.imshow(kwimage.imresize(isdot.astype(np.float32), scale=s).clip(0, 1), fnum=1, pnum=(3, 3, 7), title='is dot')
        # kwplot.imshow(kwimage.imresize(colored_dots, scale=s), fnum=1, pnum=(3, 3, 7), title='colored dots')

        kwplot.imshow(kwimage.imresize(canvas, scale=s), fnum=1, pnum=(3, 3, 8), title='circles')

        kwplot.imshow(dot_data, fnum=1, pnum=(3, 3, 9))
        for x, y, c in zip(xs.ravel(), ys.ravel(), colors):
            c = c / 255.0
            plt.plot(x, y, 'x', c=c, markersize=10, markeredgewidth=3)

        # kwplot.imshow(dot_data, doclf=True, fnum=4, pnum=()
        # kwplot.imshow(orig_isresidual, doclf=True, pnum=(1, 3, 2))
        # kwplot.imshow(orig_residual, doclf=True)
        # kwplot.imshow(orig_data)

        # kwplot.autompl()
        # kwplot.imshow(residual)

        # x = (residual.sum(axis=2) > 0).astype(np.uint8) * 255
        # kwplot.imshow(kwimage.atleast_3channels(x))

    if verbose:
        print('Finishing up')

    return xs, ys, colors


def set_overlaps(set1, set2, s1='s1', s2='s2'):
    """
    return info about set overlaps
    """
    set1 = set(set1)
    set2 = set(set2)
    overlaps = ub.odict([
        (s1, len(set1)),
        (s2, len(set2)),
        ('isect', len(set1.intersection(set2))),
        ('union', len(set1.union(set2))),
        ('%s - %s' % (s1, s2), len(set1.difference(set2))),
        ('%s - %s' % (s2, s1), len(set2.difference(set1))),
    ])
    return overlaps


def _build_dot_original_fpath_association():
    """
    Builds association between jpeg images with dots and the original images

    Returns:
        List[Dict]: where each dictionary indicates what the dot image
            and the original image are.
    """
    dpath = '/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted'
    rows = []
    import glob
    for sub_dpath in sorted(glob.glob(join(dpath, '*'))):
        collect_id = basename(sub_dpath)
        if collect_id.startswith('coco'):
            continue
        if int(collect_id[0:4]) <= 2011:
            print('---------------')
            print('sub_dpath = {!r}'.format(sub_dpath))

            contents = list(glob.glob(join(sub_dpath, '*')))

            subfolders = [c for c in contents if isdir(c)]
            for sub in subfolders:
                if basename(sub) == 'Original':
                    orig_images = list(glob.glob(join(sub, '*')))
                    orig_images = [c for c in orig_images if isfile(c) and c.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    break

            dot_images = [c for c in contents if isfile(c) and c.lower().endswith(('.jpg', '.png', '.jpeg'))]

            def normalize_image_id(p):
                k = splitext(basename(p).lower())[0]
                if k.endswith('_c'):
                    k = k[:-2]
                elif k.endswith('_orig'):
                    k = k[:-5]
                return k

            basename_to_dot = {normalize_image_id(p): p for p in dot_images}
            basename_to_orig = {normalize_image_id(p): p for p in orig_images}

            print(ub.repr2(set_overlaps(basename_to_dot, basename_to_orig, 'dot', 'orig')))

            common = set(basename_to_dot) & set(basename_to_orig)

            rows.extend([
                {
                    'key': key,
                    'collect_id': collect_id,
                    'dot': basename_to_dot[key],
                    'orig': basename_to_orig[key]
                }
                for key in common
            ])

    # These are not aligned correctly
    bad = {
        '20070609_slap6113',  # not aligned
        '20110701_sslc2391',  # weird residual
    }
    good_rows = [row for row in rows if row['key'] not in bad]

    needs_fix = [
        '20070609_slap6073',  # missing one
        '20080608_slap0499',  # missing a few
        '20070616_slap7071',  # missing one
        '20110704_ssls3345',
        '20080607_slap0367',

        '20090624_sslc0117',  # false positives
        '20110704_sslc3177',  # double detect
        '20080612_slap1755',  # missed one
        '20100607_sslc0237',
    ]
    return good_rows


def gather_raw_biologist_data():
    """
    Ok, so there are these folders with years on them. Folders from 2011
    and before contain images with dots physically on the pixels. These are
    what we want to process.

    Each of these folders contains:
        * a list of images with blacked out regions and dots.
        * An [Originals] folder with images that also contains dots, text, but
            no blacked out region
        * An Original folder that actually has original images in it.
    """

    # First find the pairs of images that we will use to extract dots
    good_rows = _build_dot_original_fpath_association()

    import ndsampler
    dset = ndsampler.CocoDataset()

    from ndsampler import util_futures

    coco_dpath = ub.ensuredir((dpath, 'coco-wip_v2'))

    # pool = util_futures.JobPool(mode='thread', max_workers=4)
    pool = util_futures.JobPool(mode='process', max_workers=0)
    with pool:

        for row in ub.ProgIter(good_rows, desc='submit dot detect jobs'):
            orig_fpath = row['orig']
            dot_fpath = row['dot']
            single_fpath = ub.augpath(orig_fpath, ext='.mscoco.json', dpath=coco_dpath)
            if not exists(single_fpath):
                job = pool.submit(process_dots, orig_fpath, dot_fpath, debug=0, verbose=0)
                job.row = row
            else:
                print('already have single_fpath = {!r}'.format(single_fpath))

        for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect jobs', verbose=3):
            row = job.row
            orig_fpath = row['orig']
            dot_fpath = row['dot']
            try:
                xs, ys, colors = job.result()
            except Exception:
                print('SKIP: due to exception')
                continue

            if len(xs) > 1000:
                print('SKIP: {} because too many poitns'.format(row, len(xs)))
                continue

            gid = dset.add_image(file_name=orig_fpath)

            single_dset = ndsampler.CocoDataset()
            gid = single_dset.add_image(**dset.imgs[gid])
            for x, y, c in zip(xs, ys, colors):
                bbox = kwimage.Boxes([[x, y, 10, 10]], 'cxywh').to_xywh().data[0].tolist()
                keypoints = [{
                    'xy': (x, y),
                    'category': 'dot',
                }]
                single_dset.add_annotation(bbox=bbox, image_id=gid, keypoints=keypoints, color=c.tolist())
            single_dset.fpath = ub.augpath(orig_fpath, ext='.mscoco.json', dpath=coco_dpath)
            print('Dump single_dset.fpath = {!r}'.format(single_dset.fpath))
            single_dset.dump(single_dset.fpath, newlines=True)

    if False:

        for row in ub.ProgIter(good_rows, desc='building coco dset', nl=True):
            orig_fpath = row['orig']
            dot_fpath = row['dot']

            xs, ys, colors = process_dots(orig_fpath, dot_fpath, debug=0)

        print(ub.repr2(rows[0:5], nl=3))
        import xdev
        import kwarray

        collect_id_to_rows = ub.group_items(good_rows, lambda x: x['collect_id'])
        arrs = []
        for colid, colrows in collect_id_to_rows.items():
            a = kwarray.shuffle(colrows)
            arrs.append(a)

        rand_checks = list(ub.flatten(zip(*arrs)))

        for row in xdev.InteractiveIter(rand_checks):
            print('row = {!r}'.format(row))
            orig_fpath = row['orig']
            dot_fpath = row['dot']
            xs, ys, colors = process_dots(orig_fpath, dot_fpath, debug=1)
            xdev.InteractiveIter.draw()


def _dev_verify():
    import glob
    import xdev
    import ndsampler
    dpath = '/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted/coco-wip'
    fpaths = sorted(glob.glob(join(dpath, '*.json')))
    for fpath in xdev.InteractiveIter(fpaths):
        dset = ndsampler.CocoDataset(fpath)
        gid = ub.peek(dset.imgs)
        dset.show_image(gid)
        xdev.InteractiveIter.draw()


def _postprocess_merged_dot_coco(merged_dots):
    """
    Hueristically cluster the colors of the jpeg-dots and convert them into
    sealion categories. Then modify the bounding box size based on the
    category.
    """
    import kwplot
    # Categories in email:
    # https://mail.google.com/mail/u/1/#inbox/FMfcgxwGCknfwcRkVfJzzkjBnHLxlbKh
    # Bulls/Adult Males (red)
    # Sub-Adult Males (SAM) (bright pink)
    # Adult females (turquoise):
    # Juveniles (royal blue or purple):
    # Pups (chartreuse i.e. greenish):
    known_categories = [
        {'name': 'Bull', 'colors': [kwplot.Color('red').as01()]},
        {'name': 'SAM', 'colors': [kwplot.Color('hotpink').as01()]},
        {'name': 'Fem', 'colors': [kwplot.Color('turquoise').as01(), kwplot.Color('brown').as01()]},
        {'name': 'Juv', 'colors': [kwplot.Color('royalblue').as01(), kwplot.Color('purple').as01()]},
        {'name': 'Pup', 'colors': [kwplot.Color('chartreuse').as01()]},
        {'name': 'Dead Pup', 'colors': [kwplot.Color((250, 200, 0)).as01()]},
        {'name': 'unknown', 'colors': [kwplot.Color('white').as01()]},
    ]
    cluster_rgbs = []
    cluster_labs = []
    cluster_labels = []
    for cat in known_categories:
        for rgb in cat['colors']:
            lab = kwimage.Color(rgb).as01(space='lab')
            cluster_rgbs.append(rgb)
            cluster_labs.append(lab)
            cluster_labels.append(cat['name'])
    cluster_rgbs = np.array(cluster_rgbs)
    cluster_labs = np.array(cluster_labs)

    import sklearn.neighbors
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    neighbors.fit(cluster_labs)

    # hack, add some metadata back to the dataset
    good_rows = _build_dot_original_fpath_association()
    orig_to_row = {row['orig']: row for row in good_rows}

    for img in merged_dots.imgs.values():
        row = orig_to_row[img['file_name']]
        img['dot_fpath'] = row['dot']

    ### Read raw data
    colors_rgb = []
    colors_lab = []
    aids = []
    nan_gids = set()
    for ann in ub.ProgIter(list(merged_dots.anns.values())):
        if np.any(np.isnan(ann['color'])):
            nan_gids.add(ann['image_id'])
        else:
            rgb = np.array(ann['color']) / 255.0
            lab = kwimage.Color(rgb).as01(space='lab')
            colors_rgb.append(rgb)
            colors_lab.append(lab)
            aids.append(ann['id'])
    colors_rgb = np.array(colors_rgb)
    colors_lab = np.array(colors_lab)
    # import kwimage
    # colors_lab = kwimage.convert_colorspace(colors_rgb[None, :], 'rgb', 'lab')[0]

    nn_dist, nn_idx = neighbors.kneighbors(colors_lab)
    annot_cnames = list(ub.take(cluster_labels, nn_idx.T[0]))

    # apply category names to the annotations
    for aid, cname in zip(aids, annot_cnames):
        ann = merged_dots.anns[aid]
        ann['category_name'] = cname
        ann['category_id'] = merged_dots.ensure_category(cname)
        if ann['category_name'] == 'Bull':
            w = h = 128
        elif ann['category_name'] == 'Fem':
            w = h = 96
        else:
            w = h = 64
        cx, cy = kwimage.Boxes([ann['bbox']], 'xywh').to_cxywh().data[0][0:2]
        bbox = kwimage.Boxes([[cx, cy, w, h]], 'cxywh')
        ann['bbox'] = bbox.to_xywh().data[0].round(1).tolist()

    merged_dots._build_index()

    if 0:
        from sklearn import cluster
        n_clusters = 7
        kmeans = cluster.KMeans(
            n_clusters=n_clusters, n_init=20, max_iter=10000, tol=1e-6,
            algorithm='elkan', verbose=0)
        kmeans.fit(colors_lab)
        centers_lab = kmeans.cluster_centers_
        centers_rgb = kwimage.convert_colorspace(centers_lab[None, :], 'lab', 'rgb')[0]
        # Assign each annotation to a cluster
        cluster_idx = kmeans.predict(colors_lab)

    ###
    # Find out what the brown color means
    if False:
        # from sklearn import metrics
        # dist = metrics.euclidean_distances(
        #     centers_lab,
        #     [kwplot.Color('white').as01('lab')])
        # nn_idx = dist[:, 0].argmin()
        # is_query = (cluster_idx == nn_idx)
        # query_aids = list(ub.compress(aids, is_query))

        query_aids = merged_dots.cid_to_aids[merged_dots._resolve_to_cid('unknown')]
        query_gids = merged_dots.annots(query_aids).gids
        query_gid_to_aids = ub.group_items(query_aids, query_gids)

        gids = ub.argsort(query_gid_to_aids, key=len)
        gid = gids[len(gids) // 2]
        img = merged_dots.imgs[gid]

        dot_fpath = img['dot_fpath']
        orig_fpath = img['file_name']
        print('dot_fpath = {!r}'.format(dot_fpath))
        print('orig_fpath = {!r}'.format(orig_fpath))

        canvas = kwimage.imread(dot_fpath)

        annots = merged_dots.annots(gid=gid)
        boxes = annots._lookup('bbox')
        colors = list(map(lambda x: tuple(x) if ub.iterable(x) else tuple([x]), annots._lookup('color')))
        for color, sub_boxes in ub.group_items(boxes, colors).items():
            if len(color) == 1:
                color = 'silver'
            canvas = kwimage.Boxes(sub_boxes, 'xywh').draw_on(canvas, color=color)

        kwplot.imshow(canvas)

        annots.boxes

        canvas = kwimage.imread(dot_fpath)

        merged_dots.show_image(gid=gid)

    ###

    SHOW = False
    if SHOW:
        # visualize the kmeans reasults
        from sklearn import decomposition
        # from sklearn import manifold
        # tsne = manifold.TSNE(verbose=1)
        # tsne.fit(colors_lab[::100])
        # xy = tsne.transform(colors_lab)

        pca = decomposition.PCA(n_components=2)
        pca.fit(colors_lab[::10])
        xy = pca.transform(colors_lab)

        xy_centers = pca.transform(centers_lab)

        import kwplot
        kwplot.autoplt()
        fig = kwplot.figure(fnum=2, doclf=True)
        ax = fig.gca()

        stride = 10
        x, y = xy[::stride].T
        c = colors_rgb[::stride]
        ax.scatter(x, y, c=c)

        x, y = xy_centers.T
        c = centers_rgb
        ax.scatter(x, y, c=c, s=500, marker='P', edgecolors='black')

        for cat in known_categories:
            rgb = cat['color']
            lab = kwimage.Color(cat['color']).as01(space='lab')
            x, y = pca.transform(np.array([lab]))[0]
            ax.scatter([x], [y], c=[rgb], s=600, marker='*',
                       edgecolors='black')


def _prepare_master_coco():
    import glob
    import ndsampler
    dpath = '/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted/coco-wip'
    fpaths = sorted(glob.glob(join(dpath, '*.json')))

    # Load all the single-image jpeg-dot processed coco files
    datasets = []
    for fpath in ub.ProgIter(fpaths):
        dset = ndsampler.CocoDataset(fpath)
        datasets.append(dset)
    # Merge the jpeg-dot coco files into a single coco file
    merged_dots = ndsampler.CocoDataset.union(*datasets)

    # POSTPROCESS THE JPEG DOT COCO FILES. FIX BBOX SIZES AND CATEGORIES
    _postprocess_merged_dot_coco(merged_dots)

    # Load the photoshop-processed annotations
    fpath = '/home/joncrall/data/noaa/sealions/sealions_photoshop_annots_v1.mscoco.json'
    ps_dset = ndsampler.CocoDataset(fpath)
    ps_img_root = '/home/joncrall/data/noaa/sealions/BLACKEDOUT/extracted'

    real_img_root = '/home/joncrall/data/noaa/sealions'
    for img in ps_dset.imgs.values():
        gpath = join(ps_img_root, img['file_name'])
        img['file_name'] = gpath

    merged = ndsampler.CocoDataset.union(merged_dots, ps_dset)

    for img in merged.imgs.values():
        dpath = dirname(img['file_name'])
        base_folder = basename(dpath)
        if base_folder == 'Original':
            base_folder = basename(dirname(dpath))
        print('base_folder = {!r}'.format(base_folder))
        img['year_code'] = base_folder

    merged.rebase(real_img_root)

    # for img in merged.imgs.values():
    #     assert exists(join(img['file_name']))
    #     rel_file_name = relpath(realpath(img['file_name']), realpath(real_img_root))
    #     img['file_name'] = rel_file_name
    #     assert exists(join(real_img_root, img['file_name']))

    if False:
        # Remove all categories
        merged.remove_categories(merged.cats, keep_annots=True)
        # add one category
        cid = merged.add_category('sealion')
        for ann in merged.anns.values():
            ann['category_id'] = cid
    else:
        for ann in merged.anns.values():
            if ann['category_id'] is None:
                ann['category_id'] = merged._resolve_to_cid('unknown')
    merged.index.clear()
    merged._build_index()

    merged.img_root = real_img_root
    merged._ensure_imgsize()
    merged.fpath = '/home/joncrall/data/noaa/sealions/sealions_all_v3.mscoco.json'
    merged.dump(merged.fpath, newlines=True)

    year_to_imgs = ub.group_items(merged.imgs.values(), lambda x: x['year_code'])
    print(ub.map_vals(len, year_to_imgs))

    vali_years = ['2010', '2016']

    split_gids = {}

    split_gids['vali'] = [img['id'] for img in ub.flatten(ub.dict_subset(year_to_imgs, vali_years).values())]
    split_gids['train'] = [img['id'] for img in ub.flatten(ub.dict_diff(year_to_imgs, vali_years).values())]

    print(ub.map_vals(len, split_gids))

    for tag, gids in split_gids.items():
        subset = merged.subset(gids)
        subset.fpath = ub.augpath(merged.fpath, base='sealions_{}_v3'.format(tag), multidot=1)
        print('subset.fpath = {!r}'.format(subset.fpath))
        subset.dump(subset.fpath, newlines=True)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/process_dots.py

        rsync -avPR /home/joncrall/data/noaa/./US_ALASKA_MML_SEALION/ viame:/data/projects/viame/.

        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v1 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/noaa/sealions/sealions_train_v1.mscoco.json \
            --vali_dataset=/home/joncrall/data/noaa/sealions/sealions_vali_v1.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=8 --bstep=4
    """
    gather_raw_biologist_data()
