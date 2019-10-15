
def process_dots():
    orig_fpath = '/home/joncrall/raid/data/noaa/KITWARE/CountedImgs/2011_Original+CountedLoRes/Original/20110627_SSLC0066_Orig.JPG'
    dot_fpath = '/home/joncrall/raid/data/noaa/KITWARE/CountedImgs/2011_Original+CountedLoRes/Counted_Processed/20110627_SSLC0066_C.jpg'

    import kwimage
    import cv2
    import numpy as np
    import kwplot
    dot_data = kwimage.imread(dot_fpath)[1500:-1700, 4900:]
    orig_data = kwimage.imread(orig_fpath)[1500:-1700, 4900:]

    dot_mag = np.sqrt((dot_data.astype(np.float32) ** 2).sum(axis=2))

    censor_mask = dot_mag < 20
    kernel = np.ones(15)
    censor_mask = cv2.morphologyEx(censor_mask.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=5).astype(np.bool)

    signed_residual = orig_data.astype(np.int32) - dot_data.astype(np.int32)

    residual = np.abs(signed_residual).astype(np.uint8)
    orig_residual = residual.copy()
    orig_isresidual = (orig_residual > 0).astype(np.uint8) * 255

    noise_mask = residual.sum(axis=2) < 20

    residual[noise_mask] = 0
    residual[censor_mask] = 0

    kernel = np.ones(7)
    opened_residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel)

    kwplot.imshow(opened_residual)

    isdot = (opened_residual.sum(axis=2) > 50)

    kwplot.imshow(dot_data)
    kwplot.imshow(censor_mask)
    kwplot.imshow(isdot.astype(np.float32))

    colored_dots = dot_data * isdot[..., None]
    kwplot.imshow(colored_dots)

    img = colored_dots
    kwplot.imshow(img, doclf=True)
    kwplot.imshow(dot_data, doclf=True)

    gray = kwimage.convert_colorspace(img, 'rgb', 'gray')
    isdot2 = (gray > 0).astype(np.uint8) * 255
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(isdot2, method=cv2.HOUGH_GRADIENT, dp=1, minDist=7, param1=10,
                               param2=5, minRadius=5, maxRadius=7)

    radii = np.median(circles[:, :, 2])
    print('radii = {!r}'.format(radii))
    print('circles = {!r}'.format(circles))

    canvas = colored_dots.copy()
    circles_2 = np.uint16(np.around(circles))
    for i in circles_2[0, :]:
        # draw the outer circle
        cv2.circle(canvas, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        # cv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), 3)
    kwplot.imshow(canvas)
    xs = circles[:, :, 0]
    ys = circles[:, :, 1]
    plt = kwplot.autoplt()
    plt.plot(xs, ys, 'rx', markersize=10, markeredgewidth=3)

    kwplot.imshow(dot_data, doclf=True)
    kwplot.imshow(orig_isresidual, doclf=True, pnum=(1, 3, 2))
    kwplot.imshow(orig_residual, doclf=True)
    kwplot.imshow(orig_data)

    cv2.CL

    residual

    kwplot.autompl()
    kwplot.imshow(residual)

    x = (residual.sum(axis=2) > 0).astype(np.uint8) * 255
    kwplot.imshow(kwimage.atleast_3channels(x))
    residual

    pass
