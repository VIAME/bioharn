import numpy as np
from os.path import dirname
from os.path import exists
from os.path import join
import ubelt as ub


def main():
    import kwcoco
    from ndsampler.utils import util_futures

    dset = kwcoco.CocoDataset(ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v3_train.mscoco.json'))

    dset = kwcoco.CocoDataset(ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v3_test.mscoco.json'))

    dset = kwcoco.CocoDataset(ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v3_vali.mscoco.json'))

    jobs = util_futures.JobPool(mode='thread', max_workers=0)

    for gid, img in ub.ProgIter(list(dset.imgs.items())):
        if img.get('source', '') in ['habcam_2015_stereo', 'habcam_stereo']:
            job = jobs.submit(_ensure_habcam_disparity_frame, dset, gid)
            job.gid = gid

    for job in ub.ProgIter(jobs, desc='collect results', verbose=3):
        gid = job.gid
        disp_dpath, disp_fname = job.result()
        img = dset.imgs[gid]
        data_dims = ((img['width'] // 2), img['height'])
        # Add auxillary channel information
        img['aux'] = [
            {
                'bands': ['disparity'],
                'file_name': disp_fname,
                'dims': data_dims,
            }
        ]

    dset.fpath = dset.fpath.replace('_v3_', '_v4_')
    dset.dump(dset.fpath, newlines=True)


def _ensure_habcam_disparity_frame(dset, gid):
    from bioharn.disparity import multipass_disparity
    import kwimage

    img = dset.imgs[gid]
    image_fname = img['file_name']

    disp_dpath = ub.ensuredir((dset.img_root))
    disp_fname = join('disparities', ub.augpath(image_fname, suffix='_left_disp_v7', ext='.cog.tif'))
    disp_fpath = join(disp_dpath, disp_fname)
    ub.ensuredir(dirname(disp_fpath))

    if not exists(disp_fpath):
        # Note: probably should be atomic
        img3 = dset.load_image(gid)
        imgL = img3[:, 0:img3.shape[1] // 2]
        imgR = img3[:, img3.shape[1] // 2:]
        img_disparity = multipass_disparity(
            imgL, imgR, scale=0.5, as01=True)
        img_disparity = img_disparity.astype(np.float32)

        kwimage.imwrite(disp_fpath, img_disparity, backend='gdal',
                        compress='DEFLATE')

    return disp_dpath, disp_fname
