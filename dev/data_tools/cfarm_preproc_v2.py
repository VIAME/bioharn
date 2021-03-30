"""
For updating disparity into the DVC repo.
"""

from os.path import relpath
import numpy as np
import kwimage
from bioharn.disparity import multipass_disparity
from os.path import basename
from os.path import dirname
from os.path import exists
import ubelt as ub
from os.path import join
from ndsampler.utils import util_futures
import kwcoco
]



def update_cfarm_datasets_with_disparity():
    r"""
    xdoctest ~/code/bioharn/dev/data_tools/cfarm_preproc_v2.py update_cfarm_datasets_with_disparity

    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/bioharn/dev/data_tools'))
    from cfarm_preproc_v2 import *  # NOQA

    cd $HOME/data/dvc-repos/viame_dvc

    find . -iname "*.json.dvc" -exec dvc pull {} \;
    find . -iname "Left.dvc" -exec dvc pull {} \;
    find . -iname "Raws.dvc" -exec dvc pull {} \;

    dvc pull -r viame \
        public/Benthic/US_NE_2017_CFF_HABCAM/Raws.dvc \
        public/Benthic/US_NE_2018_CFF_HABCAM/Raws.dvc \
        public/Benthic/US_NE_2019_CFF_HABCAM/Raws.dvc \
        public/Benthic/US_NE_2019_CFF_HABCAM_PART2/Raws.dvc

    ls US_NE_2017_CFF_HABCAM/images/disparity_unrect_left
    dvc add US_NE_2018_CFF_HABCAM/images/disparity_unrect_left US_NE_2019_CFF_HABCAM/images/disparity_unrect_left

    dvc add US_NE_2017_CFF_HABCAM/annotations_disp.kwcoco.json \
            US_NE_2019_CFF_HABCAM/annotations_disp.kwcoco.json \
            US_NE_2019_CFF_HABCAM_PART2/annotations_disp.kwcoco.json \
            US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json

    ls US_NE_2019_CFF_HABCAM_PART2/images/

    find . -iname "annotations_disp.kwcoco.json.dvc"
    dvc pull ./Benthic/US_NE_2019_CFF_HABCAM/annotations_disp.kwcoco.json.dvc ./Benthic/US_NE_2017_CFF_HABCAM/annotations_disp.kwcoco.json.dvc ./Benthic/US_NE_2018_CFF_HABCAM/annotations_disp.kwcoco.json.dvc ./Benthic/US_NE_2019_CFF_HABCAM_PART2/annotations_disp.kwcoco.json.dvc

    """
    # workdir = ub.ensuredir((root, 'data/noaa_habcam'))
    dvc_repo = ub.expandpath('$HOME/data/dvc-repos/viame_dvc/')

    extrinsics_fpath = join(dvc_repo, 'public/Benthic/US_NE_CFF_HABCAM_V3_CALIBRATION/calibration-kw-produced/extrinsics.yml')
    intrinsics_fpath = join(dvc_repo, 'public/Benthic/US_NE_CFF_HABCAM_V3_CALIBRATION/calibration-kw-produced/intrinsics.yml')

    import gdal
    print('gdal.Open = {!r}'.format(gdal.Open))

    dset_names = [
        'US_NE_2017_CFF_HABCAM',
        'US_NE_2018_CFF_HABCAM',
        'US_NE_2019_CFF_HABCAM',
        'US_NE_2019_CFF_HABCAM_PART2',
    ]
    dset_infos = []
    for name in dset_names:
        dset_infos.append({
            'name': name,
            'coco_fpath': join(dvc_repo, 'public/Benthic', name, 'annotations.kwcoco.json'),
            'raws_dpath': join(dvc_repo, 'public/Benthic', name, 'Raws'),
            'extrinsics_fpath': extrinsics_fpath,
            'intrinsics_fpath': intrinsics_fpath,
        })
    info = dset_infos[-2]

    for info in dset_infos:
        raws_dpath = info['raws_dpath']
        coco_fpath = info['coco_fpath']
        intrinsics_fpath = info['intrinsics_fpath']
        extrinsics_fpath = info['extrinsics_fpath']
        print('coco_fpath = {!r}'.format(coco_fpath))
        update_dataset_with_disparity(coco_fpath, raws_dpath, extrinsics_fpath,
                                      intrinsics_fpath)

    if 0:
        for info in dset_infos:
            coco_fpath = info['coco_fpath']
            dpath = dirname(coco_fpath)
            print('dpath = {!r}'.format(dpath))
            disp_fpath = join(dpath, 'images/disparity_unrect_left')
            assert exists(disp_fpath)
            import glob
            for fpath in ub.ProgIter(glob.glob(join(disp_fpath, '*.tif')), desc='check', verbose=1):
                kwimage.imread(fpath)


def update_dataset_with_disparity(coco_fpath, raws_dpath, extrinsics_fpath,
                                  intrinsics_fpath):
    """
    Disparity process for a single dataset
    """
    from bioharn.stereo import StereoCalibration
    cali = StereoCalibration.from_cv2_yaml(intrinsics_fpath, extrinsics_fpath)

    dset = kwcoco.CocoDataset(coco_fpath)
    dset._ensure_imgsize(workers=4)

    # raw_gpaths = sorted(glob.glob(join(raws_dpath, '*.tif')))
    # print('#raw_gpaths = {!r}'.format(len(raw_gpaths)))

    raw_gpaths = []
    for img in dset.imgs.values():
        fname = basename(img['file_name'])
        raw_fpath = ub.augpath(fname, dpath=raws_dpath, ext='.tif')
        assert exists(raw_fpath)
        raw_gpaths.append(raw_fpath)

    assert len(dset.imgs) == len(raw_gpaths)

    # NEED TO POINT TO A BUILD OF VIAME.
    # HACKING THIS IN FOR NOW.
    viame_install = ub.expandpath('$HOME/remote/namek/data/raid/viame_install/viame')
    assert exists(viame_install)

    workers = 0

    split_jobs = util_futures.JobPool('thread', max_workers=workers)
    dset_dir = dset.bundle_dpath
    raw_left_dpath = ub.ensuredir((dset_dir, 'images', 'raw-left'))
    raw_right_dpath = ub.ensuredir((dset_dir, 'images', 'raw-right'))

    for raw_gpath in raw_gpaths:
        split_jobs.submit(split_raws, raw_gpath, raw_left_dpath, raw_right_dpath)

    left_raw_paths = []
    right_raw_paths = []
    for job in ub.ProgIter(split_jobs.as_completed(), total=len(split_jobs),
                           desc='collect split jobs'):
        left_gpath, right_gpath = job.result()
        left_raw_paths.append(left_gpath)
        right_raw_paths.append(right_gpath)

    assert len(dset.imgs) == len(right_raw_paths)
    assert len(dset.imgs) == len(left_raw_paths)

    # TODO: could ensure images are output as cog here.

    # Produces the same images (with same sha1) that were already in
    # the root Left folder

    rgb_left_dpath = ub.ensuredir((dset_dir, 'images', 'rgb-left'))
    rgb_right_dpath = ub.ensuredir((dset_dir, 'images', 'rgb-right'))

    # raw_dpath = raw_left_dpath
    # raw_fpaths = left_raw_paths
    # rgb_dpath = rgb_left_dpath
    """
    mkdir -p rgb-left
    mkdir -p rgb-right
    mv raw-left/*.png rgb-left
    mv raw-right/*.png rgb-right

    tree --filelimit 100
    """
    left_rgb_fpaths = do_debayer(raw_left_dpath, left_raw_paths, rgb_left_dpath, viame_install)
    right_rgb_fpaths = do_debayer(raw_right_dpath, right_raw_paths, rgb_right_dpath, viame_install)

    if __debug__:
        for img in dset.imgs.values():
            fname = basename(img['file_name'])
            fpath = dset.get_image_fpath(img['id'])
            assert exists(fpath)

            raw_gpath = ub.augpath(fname, dpath=raws_dpath, ext='.tif')

            raw_dpath = dirname(raw_gpath)
            dset_dir = dirname(raw_dpath)
            right_dpath = join(dset_dir, 'images')
            left_dpath = join(dset_dir, 'images', 'raw-left')
            right_dpath = join(dset_dir, 'images', 'raw-right')
            assert exists(right_dpath)
            assert exists(left_dpath)
            assert raw_dpath.endswith('Raws')

            tif_fname = basename(raw_gpath)
            png_fname = ub.augpath(tif_fname, ext='.png')

            tif_fpath1 = join(left_dpath, tif_fname)
            tif_fpath2 = join(right_dpath, tif_fname)
            assert exists(tif_fpath1)
            assert exists(tif_fpath2)

            left_dpath2 = join(dset_dir, 'images', 'rgb-left')
            right_dpath2 = join(dset_dir, 'images', 'rgb-right')
            png_fpath1 = join(left_dpath2, png_fname)
            png_fpath2 = join(right_dpath2, png_fname)
            assert exists(png_fpath1)
            assert exists(png_fpath2)

            if 0:
                assert exists(fpath)
                assert ub.hash_file(fpath) == ub.hash_file(png_fpath1)

    # Prepopulate disparity maps, assuming a constant image size
    images = dset.images()
    assert ub.allsame(images.width)
    assert ub.allsame(images.height)
    img = ub.peek(dset.imgs.values())
    img_dsize = (img['width'], img['height'])
    camera1 = cali.cameras[1]
    camera2 = cali.cameras[2]
    camera1._precache(img_dsize)
    camera2._precache(img_dsize)

    # Make a directory where we can store disparity images that corresond
    # to the debayered raw left images.
    disp_unrect_dpath1 = ub.ensuredir((
        dset.bundle_dpath, 'images', 'disparity_unrect_left'))

    rgb_left_dpath = join(dset.bundle_dpath, 'images', 'rgb-left')
    rgb_right_dpath = join(dset.bundle_dpath, 'images', 'rgb-right')

    disp_tasks = []
    disp_jobs = util_futures.JobPool('thread', max_workers=workers)
    for gid, img in dset.imgs.items():
        fname = basename(img['file_name'])

        gpath1 = join(rgb_left_dpath, fname)
        gpath2 = join(rgb_right_dpath, fname)
        assert exists(gpath1)
        assert exists(gpath2)

        disp_unrect_fpath1 = ub.augpath(
            fname, dpath=disp_unrect_dpath1, ext='.tif')

        disp_tasks.append({
            'gpath1': gpath1,
            'gpath2': gpath2,
            'disp_unrect_fpath1': disp_unrect_fpath1,
            'gid': gid,
        })

        if not exists(disp_unrect_fpath1):
            job = disp_jobs.submit(
                compute_disparity_worker,
                gpath1, gpath2, disp_unrect_fpath1, cali
            )

    for job in ub.ProgIter(disp_jobs.as_completed(), total=len(disp_jobs),
                           desc='collect disparity jobs'):
        job.result()

    for task in disp_tasks:
        gid = task['gid']
        # right_fpath = relpath(task['gpath2'], dset.bundle_dpath)
        disp_unrect_fpath1 = task['disp_unrect_fpath1']
        disp_unrect_fname1 = relpath(disp_unrect_fpath1,
                                     dset.bundle_dpath)
        img = dset.imgs[gid]
        img['channels'] = 'rgb'
        img['auxillary'] = [
            {
                'channels': 'disparity',
                'file_name': disp_unrect_fname1,
                'width': img['width'],
                'height': img['height']
            },
            # {'channels': 'right',
            #  'file_name': right_fpath},
        ]

    coco_fpath2 = ub.augpath(dset.fpath, suffix='_disp', multidot=True)
    dset.fpath = coco_fpath2
    dset.dump(coco_fpath2, newlines=True)


def _devcheck_insect_final(dset, cali):
    import kwplot
    kwplot.autompl()
    import xdev

    gid = ub.peek(dset.imgs.keys())
    gids = list(dset.imgs.keys())

    camera1 = cali.cameras[1]

    for gid in xdev.InteractiveIter(gids):

        gpath1 = dset.get_image_fpath(gid)
        gpath2 = dset.get_image_fpath(gid, channels='right')

        disp_fpath = dset.get_image_fpath(gid, channels='disparity')
        disp = kwimage.imread(disp_fpath)

        # gpath2 = join(dset.bundle_dpath, img['right_cog_name'])
        # gpath1 = join(dset.bundle_dpath, img['file_name'])
        info = _compute_disparity(gpath1, gpath2, cali)

        if 0:
            assert np.allclose(info['disp1_unrect'], disp)

        _disp1_rect = kwimage.make_heatmask(info['disp1_rect'], 'magma')[..., 0:3]
        _disp1_unrect = kwimage.make_heatmask(info['disp1_unrect'], 'magma')[..., 0:3]
        canvas_rect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_rect, 0.6), info['img1_rect']])
        canvas_unrect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_unrect, 0.6), info['img1']])

        _, ax1 = kwplot.imshow(info['img1_rect'], pnum=(2, 3, 1), fnum=1, title='left rectified')
        _, ax2 = kwplot.imshow(info['disp1_rect'], pnum=(2, 3, 2), fnum=1, title='left rectified disparity')
        _, ax3 = kwplot.imshow(info['img2_rect'], pnum=(2, 3, 3), fnum=1, title='right rectified')
        _, ax4 = kwplot.imshow(canvas_rect, pnum=(2, 3, 4), fnum=1, title='left rectified')
        _, ax5 = kwplot.imshow(canvas_unrect, pnum=(2, 3, 5), fnum=1, title='left unrectified')
        _, ax6 = kwplot.imshow(info['img2'], pnum=(2, 3, 6), fnum=1, title='right unrectified')

        if dset is not None:
            annots = dset.annots(gid=gid)
            unrect_dets1 = annots.detections
            rect_dets1 = unrect_dets1.warp(camera1.rectify_points)
            rect_dets1.draw(ax=ax1)
            rect_dets1.boxes.draw(ax=ax2)
            unrect_dets1.boxes.draw(ax=ax5)
        xdev.InteractiveIter.draw()


def compute_disparity_worker(gpath1, gpath2, disp_unrect_fpath1, cali):
    info = _compute_disparity(gpath1, gpath2, cali)
    # Note: probably should be atomic
    kwimage.imwrite(disp_unrect_fpath1, info['disp1_unrect'],
                    backend='gdal', compress='DEFLATE')
    return disp_unrect_fpath1


def do_debayer(raw_dpath, raw_fpaths, rgb_dpath, viame_install):
    """
    raw_dpath = raw_left_dpath
    raw_fpaths = left_raw_paths[0:3]
    rgb_dpath = rgb_left_dpath
    """
    rgb_fpaths = []
    convert_fpaths = []
    for raw_fpath in raw_fpaths:
        rgb_fpath = ub.augpath(raw_fpath, ext='.png', dpath=rgb_dpath)
        rgb_fpaths.append(rgb_fpath)
        if not exists(rgb_fpath):
            convert_fpaths.append(raw_fpath)

    debayer_input_fpath = join(raw_dpath, 'input_list_raw_images.txt')
    sh_fpath = join(raw_dpath, 'debayer.sh')

    if convert_fpaths:
        with open(debayer_input_fpath, 'w') as file:
            file.write('\n'.join(raw_fpaths))
        sh_text = ub.codeblock(
            r'''
            #!/bin/sh
            # Setup VIAME Paths (no need to run multiple times if you already ran it)
            export VIAME_INSTALL="{viame_install}"
            source $VIAME_INSTALL/setup_viame.sh
            # Run pipeline
            kwiver runner $VIAME_INSTALL/configs/pipelines/filter_debayer_and_enhance.pipe \
                          -s input:video_filename={debayer_input_fpath}

            ''').format(
                viame_install=viame_install,
                debayer_input_fpath=debayer_input_fpath
            )
        ub.writeto(sh_fpath, sh_text)
        ub.cmd('chmod +x ' + sh_fpath)
        ub.cmd('bash ' + sh_fpath, cwd=rgb_dpath, shell=0, verbose=3)

    ub.delete(sh_fpath, verbose=1)
    ub.delete(debayer_input_fpath, verbose=1)
    return rgb_fpaths


def split_raws(raw_gpath, left_dpath, right_dpath):
    import kwimage
    left_gpath = ub.augpath(raw_gpath, dpath=left_dpath)
    right_gpath = ub.augpath(raw_gpath, dpath=right_dpath)
    if not exists(right_gpath) or not exists(left_gpath):
        raw_img = kwimage.imread(raw_gpath)
        h, w = raw_img.shape[0:2]
        half_w = w // 2
        left_img = raw_img[:, :half_w]
        right_img = raw_img[:, half_w:]
        kwimage.imwrite(right_gpath, right_img)
        kwimage.imwrite(left_gpath, left_img)
    return left_gpath, right_gpath


def _compute_disparity(gpath1, gpath2, cali, coco_dset=None, gid=None):
    img1 = kwimage.imread(gpath1)
    img2 = kwimage.imread(gpath2)

    camera1 = cali.cameras[1]
    camera2 = cali.cameras[2]

    img1_rect = camera1.rectify_image(img1)
    img2_rect = camera2.rectify_image(img2)
    disp1_rect = multipass_disparity(
        img1_rect, img2_rect, scale=0.5, as01=True)
    disp1_rect = disp1_rect.astype(np.float32)

    disp1_unrect = camera1.unrectify_image(
        disp1_rect, interpolation='linear')

    info = {
        'img1': img1,
        'img2': img2,
        'img1_rect': img1_rect,
        'img2_rect': img2_rect,
        'disp1_rect': disp1_rect,
        'disp1_unrect': disp1_unrect,
    }

    if 0:
        import kwplot
        kwplot.autompl()
        _disp1_rect = kwimage.make_heatmask(info['disp1_rect'], 'magma')[..., 0:3]
        _disp1_unrect = kwimage.make_heatmask(info['disp1_unrect'], 'magma')[..., 0:3]
        canvas_rect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_rect, 0.6), info['img1_rect']])
        canvas_unrect = kwimage.overlay_alpha_layers([
            kwimage.ensure_alpha_channel(_disp1_unrect, 0.6), info['img1']])

        _, ax1 = kwplot.imshow(info['img1_rect'], pnum=(2, 3, 1), fnum=1, title='left rectified')
        _, ax2 = kwplot.imshow(info['disp1_rect'], pnum=(2, 3, 2), fnum=1, title='left rectified disparity')
        _, ax3 = kwplot.imshow(info['img2_rect'], pnum=(2, 3, 3), fnum=1, title='right rectified')
        _, ax4 = kwplot.imshow(canvas_rect, pnum=(2, 3, 4), fnum=1, title='left rectified')
        _, ax5 = kwplot.imshow(canvas_unrect, pnum=(2, 3, 5), fnum=1, title='left unrectified')
        _, ax6 = kwplot.imshow(info['img2'], pnum=(2, 3, 6), fnum=1, title='right unrectified')

        if coco_dset is not None:
            annots = coco_dset.annots(gid=gid)
            unrect_dets1 = annots.detections
            rect_dets1 = unrect_dets1.warp(camera1.rectify_points)
            rect_dets1.draw(ax=ax1)
            rect_dets1.boxes.draw(ax=ax2)
            unrect_dets1.boxes.draw(ax=ax5)

            if 0:
                import cv2
                # Is there a way to aprox map cam1 to cam2?
                pts1 = unrect_dets1.boxes.corners()
                K1, D1 = ub.take(camera1, ['K', 'D'])
                K2, D2 = ub.take(camera2, ['K', 'D'])
                # Remove dependence on camera1 intrinsics / distortion
                pts1_norm = cv2.undistortPoints(pts1, K1, D1)[:, 0, :]
                R, T = ub.take(cali.extrinsics, ['R', 'T'])
                pts1_xyz = kwimage.add_homog(pts1_norm)
                pts1_xyz[:, 2] = 0
                # info['disp1_rect']
                # pts2_xyz = kwimage.warp_points(R, pts1_xyz) + T.T
                rvec = cv2.Rodrigues(R)[0]
                tvec = T.ravel()
                pts2, _ = cv2.projectPoints(pts1_xyz, rvec, tvec, cameraMatrix=K2, distCoeffs=D2)
                pts2 = pts2.reshape(-1, 2)

    return info


def _ensure_rgb_cog(dset, gid, cog_root):
    img = dset.imgs[gid]
    fname = basename(img['file_name'])
    cog_fname = ub.augpath(fname, dpath='', ext='.cog.tif')
    cog_fpath = join(cog_root, cog_fname)
    ub.ensuredir(dirname(cog_fpath))

    if not exists(cog_fpath):
        # Note: probably should be atomic
        img1 = dset.load_image(gid)
        kwimage.imwrite(cog_fpath, img1, backend='gdal', compress='DEFLATE')
    return cog_fpath
