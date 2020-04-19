import cv2
import numpy as np


def calibrate_single_camera(img_points, object_points, img_dsize):
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


def demo_calibrate():
    """
    https://programtalk.com/vs2/python/8176/opencv-python-blueprints/chapter4/scene3D.py/
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

    ret = cv2.stereoCalibrate(object_points, left_points, right_points,
                              K1, D1, K2, D2, img_dsize,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    R, T = ret[5:7]
    ret2 = cv2.stereoRectify(K1, D1, K2, D2, img_dsize, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]
    """
    import kwimage
    import cv2
    import ubelt as ub
    img_left_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg')
    img_right_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg')
    img_left = kwimage.imread(img_left_fpath)
    img_right = kwimage.imread(img_right_fpath)
    grid_dims = {
        'rows': 6,
        'cols': 9,
    }
    square_width = 3  # centimeters?

    def detect_grid_image(image, grid_dsize):
        """Detect a grid in a grayscale image"""
        min_len = min(image.shape)
        scale = 1.0
        while scale * min_len > 1000:
            scale /= 2.0

        if scale < 1.0:
            small = kwimage.imresize(image, scale=scale)
            return detect_grid_image(small, grid_dsize)
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

    def make_object_points(grid_size=(6, 5)):
        """construct the array of object points for camera calibration"""
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        import numpy as np
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
        return objp

    grid_dsize = (grid_dims['cols'], grid_dims['rows'])
    left_corners = detect_grid_image(img_left, grid_dsize)
    right_corners = detect_grid_image(img_right, grid_dsize)
    object_points = make_object_points(grid_dsize) * square_width

    img_dsize = img_left.shape[0:2][::-1]
    (_, K1, D1, _, _), _ = calibrate_single_camera(left_corners, object_points, img_dsize)
    (_, K2, D2, _, _), _ = calibrate_single_camera(right_corners, object_points, img_dsize)

    objectPoints = [object_points[:, None, :]]
    leftPoints = [left_corners[:, None, :]]
    rightPoints = [right_corners[:, None, :]]
    ret = cv2.stereoCalibrate(objectPoints, leftPoints, rightPoints, K1,
                              D1, K2, D2, img_dsize,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    R, T = ret[5:7]
    ret2 = cv2.stereoRectify(K1, D1, K2, D2, img_dsize, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]

    if 0:
        E, Emask = cv2.findEssentialMat(left_corners, right_corners, K1)
        print('E = {!r}'.format(E))
        F, Fmask = cv2.findFundamentalMat(left_corners, right_corners,
                                          cv2.FM_RANSAC, 0.1, 0.99)
        E = K1.T.dot(F).dot(K1)
        print('F = {!r}'.format(F))
        print('E = {!r}'.format(E))

    map11, map12 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_dsize, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_dsize, cv2.CV_16SC2)
    left_rect = cv2.remap(img_left, map11, map12, cv2.INTER_CUBIC)
    right_rect = cv2.remap(img_right, map21, map22, cv2.INTER_CUBIC)

    left_corners_rect = cv2.undistortPoints(left_corners, K1, D1, R=R1, P=P1)[:, 0, :]
    right_corners_rect = cv2.undistortPoints(right_corners, K2, D2, R=R2, P=P2)[:, 0, :]

    # left_corners_rect = kwimage.warp_points(R1 @ K1, left_corners_norm)
    # right_corners_rect = kwimage.warp_points(R2 @ K2, right_corners_norm)

    import kwplot
    kwplot.autompl()

    kwplot.figure(fnum=1, doclf=True)
    _, ax1 = kwplot.imshow(img_left, pnum=(2, 2, 1), title='raw')
    _, ax2 = kwplot.imshow(img_right, pnum=(2, 2, 2))
    kwplot.draw_points(left_corners, color='red', radius=7, ax=ax1)
    kwplot.draw_points(right_corners, color='red', radius=7, ax=ax2)

    _, ax1 = kwplot.imshow(left_rect, pnum=(2, 2, 3), title='rectified')
    _, ax2 = kwplot.imshow(right_rect, pnum=(2, 2, 4))
    kwplot.draw_points(left_corners_rect, color='red', radius=7, ax=ax1)
    kwplot.draw_points(right_corners_rect, color='red', radius=7, ax=ax2)


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
    """
