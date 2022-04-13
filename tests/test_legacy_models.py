import ubelt as ub
from os.path import join


def test_legacy_models():
    """
    Tests to make sure our old models still work.

    Various detection models are hosted here:
        https://data.kitware.com/#collection/58b747ec8d777f0aef5d0f6a

    CommandLine:
        xdoctest -m tests/test_legacy_models.py test_legacy_models --slow
    """

    if not ub.argflag('--slow'):
        import pytest
        pytest.skip()

    # import kwimage
    # img_fpath = kwimage.grab_test_image_fpath('airport')
    from bioharn.util.util_girder import grabdata_girder
    api_url = 'https://data.kitware.com/api/v1'

    legacy_models = [
        # Sealion models
        {
            'model': {
                'file_id': '5f172bce9014a6d84e2f4863',
                'hash_prefix': '698e9f85b60eb3a92acfcbde802f5e0bcf',
                'fname': 'deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS_sealion_coarse.zip'
            },
            'image': {
                'file_id': '6011a5ae2fa25629b919fe6c',
                'hash_prefix': 'f016550faa2c96ef4fdca0f5723a6',
                'fname': 'sealion_test_img_2010.jpg'
            }
        }
    ]

    for info in legacy_models:

        deployed_fpath = grabdata_girder(
            api_url, info['model']['file_id'],
            hash_prefix=info['model']['hash_prefix'])

        img_fpath = grabdata_girder(
            api_url, info['image']['file_id'],
            hash_prefix=info['image']['hash_prefix'])

        out_dpath = ub.ensure_app_cache_dir(
            'bioharn/test-legacy/', info['model']['file_id'])

        command = ub.codeblock(
            '''
            python -m bioharn.detect_predict \
                --dataset={img_fpath} \
                --deployed={deployed_fpath} \
                --out_dpath={out_dpath} \
                --xpu=auto --batch_size=1
            ''').format(
                img_fpath=img_fpath, deployed_fpath=deployed_fpath,
                out_dpath=out_dpath)

        print(command)
        info = ub.cmd(command, verbose=3)

        pred_fpath = join(out_dpath, 'pred', 'detections.mscoco.json')

        import kwcoco
        pred_dset = kwcoco.CocoDataset(pred_fpath)

        if len(pred_dset.anns) == 0:
            raise AssertionError('Should have detected something')

        if 0:
            import kwplot
            kwplot.autompl()
            pred_dset.show_image(gid=1)
            # canvas = pred_dset.draw_image(gid=1)
            # kwplot.imshow(canvas)
