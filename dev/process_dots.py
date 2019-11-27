from os.path import splitext
from os.path import isfile
import ubelt as ub
from os.path import basename
from os.path import isdir
from os.path import join
import kwimage
import cv2
import numpy as np


def process_dots(orig_fpath, dot_fpath, debug=False):
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

    print('Read data')
    dot_data = kwimage.imread(dot_fpath)
    orig_data = kwimage.imread(orig_fpath)

    dot_mag = np.sqrt((dot_data.astype(np.float32) ** 2).sum(axis=2))

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

    print('Computing residual')
    signed_residual = orig_data.astype(np.int32) - dot_data.astype(np.int32)

    orig_residual = np.abs(signed_residual).astype(np.uint8)
    residual = orig_residual.copy()
    # orig_isresidual = (orig_residual > 0).astype(np.uint8) * 255

    print('Supress residual noise')
    residual[(residual.sum(axis=2) != 0) & (residual.max(axis=2) < 27)]
    noise_mask = residual.max(axis=2) < 20

    residual[noise_mask] = 0
    residual[censor_mask] = 0

    kernel = np.ones(7)
    opened_residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel)

    print('Detecting dots in residual')
    isdot = (opened_residual.max(axis=2) > 30)
    colored_dots = dot_data * isdot[..., None]

    gray = kwimage.convert_colorspace(colored_dots, 'rgb', 'gray')
    isdot2 = (gray > 0).astype(np.uint8) * 255

    print('Detecting circles in dots')

    # Clean up any remaining small connected components
    METHOD = 'HOUGH'
    if METHOD == 'CC':
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            isdot2.astype(np.uint8), 8, cv2.CV_32S)
        areas = stats[:, cv2.CC_STAT_AREA]
        chosen_labels = np.where(areas < 1000 & areas > 1)[0][1:]
        for i in chosen_labels:
            x, y, w, h = stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
            region = (slice(y, y + h), slice(x, x + w))

    elif METHOD == 'HOUGH':
        circles = cv2.HoughCircles(isdot2, method=cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=7, param1=10, param2=5,
                                   minRadius=5, maxRadius=7)

        if circles is None:
            circles = cv2.HoughCircles(isdot2, method=cv2.HOUGH_GRADIENT,
                                       dp=1, minDist=5,
                                       param1=10,  # edge detection threshold
                                       param2=3,   # smaller = more detections
                                       minRadius=3, maxRadius=9)

        if circles is None:
            circles = np.empty((1, 0, 3))
            raise Exception('no hough circles found')

        print('Determining circle colors')
        xs = circles[:, :, 0].ravel()
        ys = circles[:, :, 1].ravel()
        radii = circles[:, :, 2].ravel()

    # radii = np.median(circles[:, :, 2])
    # print('radii = {!r}'.format(radii))
    # print('circles = {!r}'.format(circles))

    # Colors indicate the classes
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

    print('Finishing up')

    SHOW_PROCESS = debug
    if SHOW_PROCESS:
        import kwplot
        plt = kwplot.autoplt()

        s = 1.0
        kwplot.imshow(kwimage.imresize(orig_data, scale=s), fnum=1, pnum=(3, 3, 1), title='orig')
        kwplot.imshow(kwimage.imresize(dot_data, scale=s), fnum=1, pnum=(3, 3, 2), title='dots')
        kwplot.imshow(kwimage.imresize(censor_mask.astype(np.float32), scale=s).clip(0, 1), fnum=1, pnum=(3, 3, 3), title='censor')

        kwplot.imshow(kwimage.imresize(noise_mask.astype(np.uint8) * 255, scale=s), fnum=1, pnum=(3, 3, 4), title='noise_mask')
        kwplot.imshow(kwimage.imresize(opened_residual, scale=s), fnum=1, pnum=(3, 3, 5), title='opened residual')

        kwplot.imshow(kwimage.imresize(isdot.astype(np.float32), scale=s).clip(0, 1), fnum=1, pnum=(3, 3, 6), title='is dot')
        kwplot.imshow(kwimage.imresize(colored_dots, scale=s), fnum=1, pnum=(3, 3, 7), title='colored dots')

        canvas = colored_dots.copy()
        circles_2 = np.uint16(np.around(circles))
        for i in circles_2[0, :]:
            # draw the outer circle
            cv2.circle(canvas, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            # ccolored_dotsv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), 3)

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

    return xs, ys, colors


def gather_raw_biologist_data():
    """
    Ok, so there are these folders with years on them. Folders from 2011
    and before contain images with dots physically on the pixels. These are
    what we want to process.

    Each of these folders contains:
        * a list of imges with blacked out regions and dots.
        * An [Originals] folder with images that also contains dots, text, but
            no blacked out region
        * An Original folder that actually has original images in it.
    """

    dpath = '/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted'

    rows = []

    import glob
    for sub_dpath in sorted(glob.glob(join(dpath, '*'))):
        collect_id = basename(sub_dpath)
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

            def set_overlaps(set1, set2, s1='s1', s2='s2'):
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

            basename_to_dot = {normalize_image_id(p): p for p in dot_images}
            basename_to_orig = {normalize_image_id(p): p for p in orig_images}

            print(ub.repr2(set_overlaps(basename_to_dot, basename_to_orig, 'dot', 'orig')))

            common = set(basename_to_dot) & set(basename_to_orig)

            rows.extend([
                {
                    'collect_id': collect_id,
                    'dot': basename_to_dot[key],
                    'orig': basename_to_orig[key]
                }
                for key in common
            ])

        for row in rows:
            orig_fpath = row['orig']
            dot_fpath = row['dot']
            xs, ys, colors = process_dots(orig_fpath, dot_fpath)

        if False:
            import xdev

            for row in xdev.InteractiveIter(rows):
                orig_fpath = row['orig']
                dot_fpath = row['dot']
                xs, ys, colors = process_dots(orig_fpath, dot_fpath, debug=True)
                xdev.InteractiveIter.draw()
