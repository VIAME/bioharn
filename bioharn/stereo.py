import cv2
import kwimage
import ubelt as ub
import numpy as np


def _calibrate_single_camera(img_points, object_points, img_dsize):
    K = np.array([[1000, 0, img_dsize[1] / 2],
                  [0, 1000, img_dsize[0] / 2],
                  [0, 0, 1]], dtype=np.float32)
    d = np.array([0, 0, 0, 0], dtype=np.float32)

    objectPoints = [object_points[:, None, :]]
    imgPoints = [img_points[:, None, :]]

    flags = 0
    cal_result = cv2.calibrateCamera(
        objectPoints=objectPoints, imagePoints=imgPoints, imageSize=img_dsize,
        cameraMatrix=K, distCoeffs=d, flags=flags)
    print("initial calibration error: ", cal_result[0])

    # per frame analysis
    #frames, imgPoints, objectPoints = evaluate_error(imgPoints, objectPoints, frames, cal_result)
    ret, mtx, dist, rvecs, tvecs = cal_result
    aspect_ratio = mtx[0, 0] / mtx[1, 1]
    print("aspect ratio: ", aspect_ratio)
    if 1.0 - min(aspect_ratio, 1.0 / aspect_ratio) < 0.01:
        print("fixing aspect ratio at 1.0")
        flags += cv2.CALIB_FIX_ASPECT_RATIO
        cal_result = cv2.calibrateCamera(
            objectPoints, imgPoints, img_dsize, K, d, flags=flags)
        ret, mtx, dist, rvecs, tvecs = cal_result
        print("Fixed aspect ratio error: ", cal_result[0])

    pp = np.array([mtx[0, 2], mtx[1, 2]])
    print("principal point: ", pp)
    rel_pp_diff = (pp - np.array(img_dsize) / 2) / np.array(img_dsize)
    print("rel_pp_diff", max(abs(rel_pp_diff)))
    if max(abs(rel_pp_diff)) < 0.05:
        print("fixed principal point to image center")
        flags += cv2.CALIB_FIX_PRINCIPAL_POINT
        cal_result = cv2.calibrateCamera(
            objectPoints, imgPoints, img_dsize, K, d, flags=flags)
        print("Fixed principal point error: ", cal_result[0])

    # set a threshold 25% more than the baseline error
    error_threshold = 1.25 * cal_result[0]

    last_result = (cal_result, flags)

    # Ignore tangential distortion
    flags += cv2.CALIB_ZERO_TANGENT_DIST
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No tangential error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K3
    flags += cv2.CALIB_FIX_K3
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No K3 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K2
    flags += cv2.CALIB_FIX_K2
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No K2 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K1
    flags += cv2.CALIB_FIX_K1
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No distortion error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    return (cal_result, flags)


def _detect_grid_image(image, grid_dsize):
    """Detect a grid in a grayscale image"""
    min_len = min(image.shape)
    scale = 1.0
    while scale * min_len > 1000:
        scale /= 2.0

    if scale < 1.0:
        small = kwimage.imresize(image, scale=scale)
        return _detect_grid_image(small, grid_dsize)
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Find the chess board corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH
        ret, corners = cv2.findChessboardCorners(image, grid_dsize, flags=flags)
        if ret:
            # refine the location of the corners
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            return corners[:, 0, :]
        else:
            raise Exception('Failed to localize grid')


def _make_object_points(grid_size=(6, 5)):
    """construct the array of object points for camera calibration"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    return objp


def demo_calibrate():
    """
    References:
        https://programtalk.com/vs2/python/8176/opencv-python-blueprints/chapter4/scene3D.py/
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    """
    img_left_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg')
    img_right_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg')
    img_left = kwimage.imread(img_left_fpath)
    img_right = kwimage.imread(img_right_fpath)
    grid_dsize = (6, 9)  # columns x rows
    square_width = 3  # centimeters?

    left_corners = _detect_grid_image(img_left, grid_dsize)
    right_corners = _detect_grid_image(img_right, grid_dsize)
    object_points = _make_object_points(grid_dsize) * square_width

    img_dsize = img_left.shape[0:2][::-1]

    # Intrinstic camera matrix (K) and distortion coefficients (D)
    (_, K1, D1, _, _), _ = _calibrate_single_camera(left_corners, object_points, img_dsize)
    (_, K2, D2, _, _), _ = _calibrate_single_camera(right_corners, object_points, img_dsize)

    objectPoints = [object_points[:, None, :]]
    leftPoints = [left_corners[:, None, :]]
    rightPoints = [right_corners[:, None, :]]
    ret = cv2.stereoCalibrate(objectPoints, leftPoints, rightPoints, K1,
                              D1, K2, D2, img_dsize,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    # extrinsic rotation (R) and translation (T) from the left to right camera
    R, T = ret[5:7]

    # Rectification (R1, R2) matrix (R1 and R2 are homographies, todo: mapping between which spaces?),
    # New camera projection matrix (P1, P2),
    # Disparity-to-depth mapping matrix (Q).
    ret2 = cv2.stereoRectify(K1, D1, K2, D2, img_dsize, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]

    # NOTE: using cv2.CV_16SC2 is more efficient because it uses a fixed-point
    # encoding of subpixel coordinates, but needs to be decoded to preform the
    # inverse mapping. Using cv2.CV_32FC1 returns a simpler floating-point based
    # representation, which can be directly inverted.
    map11, map12 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_dsize, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_dsize, cv2.CV_16SC2)
    map11f, map12f = cv2.convertMaps(map11, map12, cv2.CV_32FC1)
    map21f, map22f = cv2.convertMaps(map21, map22, cv2.CV_32FC1)
    # map11f, map12f = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_dsize, cv2.CV_32FC1)
    # map21f, map22f = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_dsize, cv2.CV_32FC1)

    left_points = np.array(left_corners.tolist() + [
        # hacked extra points
        [0, 0], [3, 3], [5, 5], [10, 10], [15, 15], [19, 19], [31, 31],
        [50, 50], [90, 90], [100, 100], [110, 110],
        [123, 167], [147, 299], [46, 393],
    ])
    right_points = right_corners

    # Map points and images from camera space to rectified space.
    left_rect = cv2.remap(img_left, map11, map12, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(img_right, map21, map22, cv2.INTER_LANCZOS4)
    left_points_rect = cv2.undistortPoints(left_points, K1, D1, R=R1, P=P1)[:, 0, :]
    right_points_rect = cv2.undistortPoints(right_points, K2, D2, R=R2, P=P2)[:, 0, :]

    if 1:
        def invert_remap(map11f, map12f):
            """
            References:
                https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid
            """
            h, w = map11f.shape[0:2]
            inv_map12f, inv_map11f = np.mgrid[0:h, 0:w].astype(np.float32)
            dx = inv_map11f - map11f
            dy = inv_map12f - map12f
            # inv_map11f + inv_map11f - map11f
            inv_map11f += dx
            inv_map12f += dy
            inv_map11f, inv_map12f
            return inv_map11f, inv_map12f
        # Invert the rectification
        # FIXME: This appears to have an issue
        inv_map11f, inv_map12f = invert_remap(map11f, map12f)
        inv_map21f, inv_map22f = invert_remap(map21f, map22f)
        left_unrect_v1 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
        right_unrect_v1 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)

    if 1:
        # https://stackoverflow.com/questions/21615298/opencv-distort-back
        # Note negating the distortion coefficients is only a first order approximation
        inv_map11f, inv_map12f = cv2.initUndistortRectifyMap(P1[:, 0:3], -D1, np.linalg.inv(R1), K1, img_dsize, cv2.CV_32FC1)
        inv_map21f, inv_map22f = cv2.initUndistortRectifyMap(P2[:, 0:3], -D2, np.linalg.inv(R2), K2, img_dsize, cv2.CV_32FC1)
        left_unrect_v2 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
        right_unrect_v2 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)

        # H1_inv = np.linalg.inv(P1[:, 0:3]) @ np.linalg.inv(R1)  # @ np.linalg.inv(P1[:, 0:3])
        # H2_inv = np.linalg.inv(R2)  # @ np.linalg.inv(P2[:, 0:3])
        # left_unrect = cv2.warpPerspective(left_rect, H1_inv, img_dsize)
        # right_unrect = cv2.warpPerspective(right_rect, H2_inv, img_dsize)

    import kwplot
    kwplot.autompl()

    nrows = 4

    kwplot.figure(fnum=1, doclf=True)
    _, ax1 = kwplot.imshow(img_left, pnum=(nrows, 2, 1), title='raw')
    _, ax2 = kwplot.imshow(img_right, pnum=(nrows, 2, 2))
    kwplot.draw_points(left_points, color='red', radius=7, ax=ax1)
    kwplot.draw_points(right_points, color='red', radius=7, ax=ax2)

    _, ax3 = kwplot.imshow(left_rect, pnum=(nrows, 2, 3), title='rectified')
    _, ax4 = kwplot.imshow(right_rect, pnum=(nrows, 2, 4))
    kwplot.draw_points(left_points_rect, color='red', radius=7, ax=ax3)
    kwplot.draw_points(right_points_rect, color='red', radius=7, ax=ax4)

    # v1

    left_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(left_unrect_v1, 0.65),
        img_left,
    ])
    right_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(right_unrect_v1, 0.65),
        img_right,
    ])
    _, ax5 = kwplot.imshow(left_unrect2, pnum=(nrows, 2, 5), title='un-rectified V1 (with overlay)')
    _, ax6 = kwplot.imshow(right_unrect2, pnum=(nrows, 2, 6))

    # V2
    left_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(left_unrect_v2, 0.65),
        img_left,
    ])
    right_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(right_unrect_v2, 0.65),
        img_right,
    ])
    _, ax7 = kwplot.imshow(left_unrect2, pnum=(nrows, 2, 7), title='un-rectified V2 (with overlay)')
    _, ax8 = kwplot.imshow(right_unrect2, pnum=(nrows, 2, 8))

    if 1:
        # This seems to work very well
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        rect = cv2.convertPointsToHomogeneous(left_points_rect)[:, 0, :]
        rect = kwimage.warp_points(np.linalg.inv(R1) @ np.linalg.inv(P1[:, 0:3]), rect)
        left_points_unrect, _ = cv2.projectPoints(rect, rvec, tvec, cameraMatrix=K1, distCoeffs=D1)
        left_points_unrect = left_points_unrect.reshape(-1, 2)

        err = left_points_unrect - left_points
        med_err = np.median(err)
        print('med_err = {!r}'.format(med_err))
        kwplot.draw_points(left_points_unrect, color='orange', radius=7, ax=ax5)

        kwplot.draw_points(left_points_unrect, color='orange', radius=7, ax=ax7)

    if 0:
        # This works ok, but fails on out-of-bounds points
        left_points_unrect = np.hstack([
            kwimage.subpixel_getvalue(map11, left_points_rect[:, ::-1])[:, None],
            kwimage.subpixel_getvalue(map12, left_points_rect[:, ::-1])[:, None]
        ])
        kwplot.draw_points(left_points_unrect, color='purple', radius=7, ax=ax5)
        err = left_points_unrect - left_points
        med_err = np.median(err)
        print('med_err = {!r}'.format(med_err))


def _notes():
    """
    pip install plottool_ibeis
    pip install vtool_ibeis

    # img_left = kwimage.grab_test_image('tsukuba_l')
    # img_right = kwimage.grab_test_image('tsukuba_r')
    import pyhesaff
    kpts1, desc1 = pyhesaff.detect_feats_in_image(img_left)
    kpts2, desc2 = pyhesaff.detect_feats_in_image(img_right)

    from vtool_ibeis import matching
    annot1 = {'kpts': kpts1, 'vecs': desc1, 'rchip': img_left}
    annot2 = {'kpts': kpts2, 'vecs': desc2, 'rchip': img_right}
    match = matching.PairwiseMatch(annot1, annot2)
    match.assign()

    idx1, idx2 = match.fm.T
    xy1_m = kpts1[idx1, 0:2]
    xy2_m = kpts2[idx2, 0:2]

    # TODO: need to better understand R1, and P1 and what
    # initUndistortRectifyMap does wrt to K and D.
    # cv2.initUndistortRectifyMap(K1, D1, np.linalg.inv(R1), np.linalg.pinv(P1)[0:3], img_dsize, cv2.CV_32FC1)
    # cv2.initUndistortRectifyMap(np.eye(3), None, np.linalg.inv(R1), np.eye(3), img_dsize, cv2.CV_32FC1)

    # Invert rectification?
    # https://groups.google.com/forum/#!topic/pupil-discuss/8eSuYYNEaIQ
    # https://stackoverflow.com/questions/35060164/reverse-undistortrectifymap
    # https://answers.opencv.org/question/129425/difference-between-undistortpoints-and-projectpoints-in-opencv/

    # K1_inv = np.linalg.inv(K1)
    # left_points_unrect = cv2.undistortPoints(left_points_rect, K1, D1, R=R1, P=P1)[:, 0, :]
    # left_points_unrect = cv2.undistortPoints(left_points_rect, P1[:, 0:3], D1, R=R1, P=K1)[:, 0, :]
    # left_points_unrect = cv2.undistortPoints(left_points_rect, P1[:, 0:3], D1, R=None, P=K1)[:, 0, :]
    # M = np.linalg.inv(P1[:, 0:3]) @ R1 @ K1
    # M = K1 @ R1 @ np.linalg.inv(P1[:, 0:3])
    # left_points_unrect = kwimage.warp_points(M, left_points_rect)
    # kwplot.draw_points(left_points_unrect, color='red', radius=7, ax=ax1)
    """
    if 0:
        E, Emask = cv2.findEssentialMat(left_corners, right_corners, K1)  # NOQA
        F, Fmask = cv2.findFundamentalMat(left_corners, right_corners,  # NOQA
                                          cv2.FM_RANSAC, 0.1, 0.99)  # NOQA
        E = K1.T.dot(F).dot(K1)  # NOQA


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/stereo.py
    """
    import kwplot
    plt = kwplot.autoplt()
    demo_calibrate()
    plt.show()
