import numpy as np
import ubelt as ub
import kwimage
import kwplot


def _devcheck_stereo():
    """
    pip install opencv-contrib-python
    """
    import ndsampler
    fpath = ub.expandpath('~/remote/namek/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json')
    dset = ndsampler.CocoDataset(fpath)

    import cv2
    img3 = dset.load_image(1)

    imgL = img3[:, 0:img3.shape[1] // 2]
    imgR = img3[:, img3.shape[1] // 2:]

    imgL = np.ascontiguousarray(imgL)
    imgR = np.ascontiguousarray(imgR)

    imgL = kwimage.imresize(imgL, scale=0.5)
    imgR = kwimage.imresize(imgR, scale=0.5)

    DEBUG_MATCHES = 0
    if DEBUG_MATCHES:
        import vtool_ibeis as vt
        annot1 = {'rchip': imgR}
        annot2 = {'rchip': imgL}
        match = vt.PairwiseMatch(annot1, annot2)
        if 0:
            import guitool_ibeis as gt
            gt.ensure_qapp()
            match.ishow()
        match.apply_all({'refine_method': 'affine', 'affine_invariance': False, 'rotation_invariance': False})
        dsize = imgR.shape[0:2][::-1]
        imgR_warp = vt.warpHomog(imgR, match.H_12, dsize)

        if 0:
            kwplot.imshow(imgL, pnum=(2, 1, 1))
            kwplot.imshow(imgR_warp, pnum=(2, 1, 2))
            kwplot.imshow(imgL, pnum=(2, 1, 1))
            kwplot.imshow(imgR_warp, pnum=(2, 1, 2))
        imgR = imgR_warp

    # window_size = 3
    # left_matcher = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=5,
    #     P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #     P2=32 * 3 * window_size ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )

    disp_alg = cv2.StereoBM_create(numDisparities=112, blockSize=9)
    disparity = disp_alg.compute(kwimage.convert_colorspace(imgL, 'rgb', 'gray'), kwimage.convert_colorspace(imgR, 'rgb', 'gray'))
    disparity = disparity - disparity.min()
    disparity = disparity / disparity.max()
    kwplot.imshow(disparity, pnum=(2, 1, 1), title='BM Disparity', fnum=1)
    kwplot.imshow(imgL, pnum=(2, 2, 3), fnum=1)
    kwplot.imshow(imgR, pnum=(2, 2, 4), fnum=1)

    disp_alg = cv2.StereoSGBM_create(numDisparities=256, minDisparity=-64, uniquenessRatio=1, blockSize=20, speckleWindowSize=150, speckleRange=2, P1=600, P2=2400)
    disp_alg = cv2.StereoSGBM_create(numDisparities=256, minDisparity=-64, uniquenessRatio=1, blockSize=20, speckleWindowSize=150, speckleRange=2, P1=600, P2=2400)

    window_size = 9
    min_disp = 16
    num_disp = 112 - min_disp
    disp_alg = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16, P1=8 * 3 * window_size**2, P2=32 * 3 * window_size**2,
        disp12MaxDiff=10000,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disp_alg = cv2.StereoSGBM_create(numDisparities=256, blockSize=5, disp12MaxDiff=512)
    disp_alg = cv2.StereoSGBM_create(numDisparities=256, minDisparity=-64, uniquenessRatio=1, blockSize=20, speckleWindowSize=150, speckleRange=2, P1=600, P2=2400)

    # THIS WORKS ON (512, 680)
    disp_alg = cv2.StereoSGBM_create(uniquenessRatio=1, numDisparities=64, blockSize=14, speckleRange=32, P1=600, P2=2400, disp12MaxDiff=10)
    disp_alg = cv2.StereoSGBM_create(uniquenessRatio=8, numDisparities=16, blockSize=14)

    # disparity = disp_alg.compute(kwimage.convert_colorspace(imgL, 'rgb', 'gray'), kwimage.convert_colorspace(imgR, 'rgb', 'gray'))
    disparity = disp_alg.compute(kwimage.convert_colorspace(imgL, 'rgb', 'bgr'), kwimage.convert_colorspace(imgR, 'rgb', 'bgr'))
    disparity = disparity - disparity.min()
    disparity = disparity / disparity.max()
    kwplot.imshow(disparity, pnum=(2, 1, 1), title='SGBM Disparity', fnum=2)
    kwplot.imshow(imgL, pnum=(2, 2, 3), fnum=2)
    kwplot.imshow(imgR, pnum=(2, 2, 4), fnum=2)
