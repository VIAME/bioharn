"""
notes for preprocessing cfarm data

https://github.com/VIAME/VIAME


https://data.kitware.com/api/v1/item/5e76a11baf2e2eed355b5228/download

pip install girder-client
eval "$(_GIRDER_CLI_COMPLETE=source girder-client)"


mkdir -p /home/joncrall/data/raid/viame_install
cd /home/joncrall/data/raid/viame_install
girder-client --api-url https://data.kitware.com/api/v1 download 5e76a11baf2e2eed355b5228
tar -xvf VIAME-v0.10.8-Ubuntu18.04-64Bit.tar.gz

cd /home/joncrall/data/raid/viame_install/viame
source /home/joncrall/data/raid/viame_install/viame/setup_viame.sh


CURRENT DATA LAYOUT 2020-04-14:

    private
    ├── _combo_cfarm
    ├── _combos
    ├── US_NE_2017_CFARM_HABCAM
    │   └── _dev
    │       └── cog_rgb
    ├── US_NE_2018_CFARM_HABCAM
    │   └── _dev
    │       ├── cog_rgb
    │       └── images
    │           └── cog
    ├── US_NE_2019_CFARM_HABCAM
    │   ├── processed
    │   │   └── _dev
    │   │       └── cog_rgb
    │   ├── raws
    │   │   └── _dev
    │   │       └── cog_rgb
    │   └── sample-3d-results
    │       ├── flounder
    │       └── swimmers
    └── US_NE_2019_CFARM_HABCAM_PART2
    public
    ├── Benthic
    │   └── US_NE_2015_NEFSC_HABCAM
    │       ├── cog
    │       ├── Corrected
    │       ├── _dev
    │       └── disparities
    └── _dev



https://github.com/VIAME/VIAME/tree/master/examples/image_enhancement
debayer_and_enhance.sh
consumes debayer_and_enhance.sh
input_list_raw_images.txt
by default though that can be changed in script
outputs in current directory
one these days I should adjust so it runs it on both camera sides independently instead of jointly


├── <DATASET_NAME_1>
│   ├── raw
│   │   ├── ...
│   ├── processed
│   │   ├── left
│   │   │   ├── ...
│   │   ├── right
│   │   │   ├── ...
│   │   └── disparity
│   │   │   ├── ...
│   ├── annotations.csv
│   ├── annotations.mscoco.json
│   └── _developer_stuff
│  
├── <DATASET_NAME_2>
...


+ <DATASET_NAME>
|
+-- images
|
+-- images

"""
from os.path import exists
import ubelt as ub
from os.path import join
import glob


def preproc_cfarm():

    root = ub.expandpath('$HOME/remote/namek/')
    dpath = join(root, 'data/private')

    raw_dpaths = {
        '2017_CFARM': join(dpath, 'US_NE_2017_CFARM_HABCAM'),
        '2018_CFARM': join(dpath, 'US_NE_2018_CFARM_HABCAM'),
        '2019_CFARM_P1': join(dpath, 'US_NE_2019_CFARM_HABCAM/raws'),
        '2019_CFARM_P2': join(dpath, 'US_NE_2019_CFARM_HABCAM_PART2'),
    }

    gpaths = {}
    for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
        print('raw_dpath = {!r}'.format(raw_dpath))
        raw_gpaths = sorted(glob.glob(join(raw_dpath, '*.tif')))
        gpaths[key] = raw_gpaths
        print('#raw_gpaths = {!r}'.format(len(raw_gpaths)))

    workdir = ub.ensuredir((root, 'data/noaa_habcam'))
    viame_install = join(root, 'data/raid/viame_install/viame')

    from ndsampler.utils import util_futures
    for key, raw_gpaths in ub.ProgIter(gpaths.items()):

        jobs = util_futures.JobPool('thread', max_workers=8)

        dset_dir = ub.ensuredir((workdir, key))
        left_dpath = ub.ensuredir((dset_dir, 'raw', 'left'))
        right_dpath = ub.ensuredir((dset_dir, 'raw', 'right'))
        for raw_gpath in raw_gpaths:
            jobs.submit(split_raws, raw_gpath, left_dpath, right_dpath)

        left_paths = []
        right_paths = []
        for job in ub.ProgIter(jobs.as_completed(), total=len(jobs),
                               desc='collect split jobs'):
            left_gpath, right_gpath = job.result()
            left_paths.append(left_gpath)
            right_paths.append(right_gpath)

        do_debayer(left_dpath, left_paths, viame_install)
        do_debayer(right_dpath, right_paths, viame_install)

    jobs = util_futures.JobPool('thread', max_workers=8)

    for key, raw_dpath in ub.ProgIter(raw_dpaths.items()):
        dset_dir = ub.ensuredir((workdir, key))
        left_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'left', '*.png')))
        right_png_gpaths = sorted(glob.glob(join(dset_dir, 'raw', 'right', '*.png')))

        left_dpath = ub.ensuredir((dset_dir, 'images', 'left'))
        right_dpath = ub.ensuredir((dset_dir, 'images', 'right'))

        for src_fpath in left_png_gpaths:
            dst_fpath = ub.augpath(src_fpath, dpath=left_dpath, ext='.cog.tif')
            jobs.submit(convert_to_cog, src_fpath, dst_fpath)

        for src_fpath in right_png_gpaths:
            dst_fpath = ub.augpath(src_fpath, dpath=right_dpath, ext='.cog.tif')
            jobs.submit(convert_to_cog, src_fpath, dst_fpath)

    for job in ub.ProgIter(jobs.as_completed(), total=len(jobs),
                           desc='collect convert-to-cog jobs'):
        job.result()


def convert_to_cog(src_fpath, dst_fpath):
    from ndsampler.utils.util_gdal import _cli_convert_cloud_optimized_geotiff
    if not exists(dst_fpath):
        _cli_convert_cloud_optimized_geotiff(
            src_fpath, dst_fpath, compress='LZW', blocksize=256)


def do_debayer(dpath, fpaths, viame_install):
    debayer_input_fpath = join(dpath, 'input_list_raw_images.txt')
    with open(debayer_input_fpath, 'w') as file:
        file.write('\n'.join(fpaths))
    sh_text = ub.codeblock(
        r'''
        #!/bin/sh
        # Setup VIAME Paths (no need to run multiple times if you already ran it)
        export VIAME_INSTALL="{viame_install}"
        source $VIAME_INSTALL/setup_viame.sh
        # Run pipeline
        kwiver runner $VIAME_INSTALL/configs/pipelines/filter_debayer_and_enhance.pipe \
                      -s input:video_filename={debayer_input_fpath}

        ''').format(viame_install=viame_install, debayer_input_fpath=debayer_input_fpath)
    sh_fpath = join(dpath, 'debayer.sh')
    ub.writeto(sh_fpath, sh_text)
    ub.cmd('chmod +x ' + sh_fpath)
    ub.cmd('bash ' + sh_fpath, cwd=dpath, shell=0, verbose=3)


def split_raws(raw_gpath, left_dpath, right_dpath):
    import kwimage
    left_gpath = ub.augpath(raw_gpath, dpath=left_dpath)
    right_gpath = ub.augpath(raw_gpath, dpath=right_dpath)
    if not exists(right_gpath) or not exists(right_gpath):
        raw_img = kwimage.imread(raw_gpath)
        h, w = raw_img.shape[0:2]
        half_w = w // 2
        left_img = raw_img[:, :half_w]
        right_img = raw_img[:, half_w:]
        kwimage.imwrite(right_gpath, right_img)
        kwimage.imwrite(left_gpath, left_img)
    return left_gpath, right_gpath
