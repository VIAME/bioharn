import ubelt as ub


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

    from bioharn.util.util_girder import grabdata_girder
    api_url = 'https://data.kitware.com/api/v1'

    # Sealion models
    # deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS_sealion_coarse.zip
    file_id = '5f172bce9014a6d84e2f4863'
    hash_prefix = '698e9f85b60eb3a92acfcbde802f5e0bcf'
    deployed_fpath = grabdata_girder(api_url, file_id, hash_prefix=hash_prefix)

    out_dpath = ub.ensure_app_cache_dir('bioharn/test-legacy/', file_id)

    import kwimage
    img_fpath = kwimage.grab_test_image_fpath('airport')

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
