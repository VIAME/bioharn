from os.path import dirname
import ubelt as ub
from os.path import join


def _test_train_and_eval_model(aux):
    """
    aux = True
    """
    try:
        import imgaug  # NOQA
    except ImportError:
        import pytest
        pytest.skip()

    from bioharn import detect_fit
    from bioharn import detect_eval
    import kwcoco

    dset = kwcoco.CocoDataset.demo('shapes8', aux=aux)
    dpath = ub.ensure_app_cache_dir('bioharn/tests')
    ub.delete(dpath)  # start fresh

    workdir = ub.ensuredir((dpath, 'work'))

    dset.fpath = join(dpath, 'shapes_train.mscoco')
    dset.remove_categories(['background'])
    dset.dump(dset.fpath)

    channels = 'rgb|disparity' if aux else 'rgb'

    deploy_fpath = detect_fit.fit(
        cmdline=False,
        # arch='cascade',
        arch='yolo2',
        normalize_inputs=10,
        train_dataset=dset.fpath,
        channels=channels,
        workers=0,
        workdir=workdir,
        batch_size=2,
        window_dims=(256, 256),
        timeout=10,
    )

    train_dpath = dirname(deploy_fpath)
    out_dpath = ub.ensuredir(train_dpath, 'out_eval')

    detect_eval.evaluate_models(
        cmdline=False,
        deployed=deploy_fpath,
        dataset=dset.fpath,
        workdir=workdir,
        out_dpath=out_dpath,
    )


def test_train_and_eval_model_rgb():
    """
    xdoctest tests/test_end_to_end.py test_train_and_eval_model_rgb
    """
    _test_train_and_eval_model(aux=False)


def test_train_and_eval_model_rgbd():
    """
    xdoctest tests/test_end_to_end.py test_train_and_eval_model_rgbd
    """
    _test_train_and_eval_model(aux=True)


if __name__ == '__main__':
    """
    CommandLine:
        python tests/test_end_to_end.py
    """
    test_train_and_eval_model_rgbd()
