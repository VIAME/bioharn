import pytest


def test_num_batches_overflow():
    """
    Test what happens when we specify num_batches larger than the number of
    possible batches within an epoch.
    """
    import ubelt as ub
    pytest.skip()
    command = ub.codeblock(
        r"""
        python -m bioharn.detect_fit \
            --name=test_num_batches_overflow \
            --train_dataset=special:shapes8 \
            --arch=retinanet50 \
            --input_dims=256,256 \
            --workers=4 --xpu=auto --batch_size=3 \
            --num_batches=100000
        """)
    info = ub.cmd(command, verbose=3, check=False)
    assert 'IndexError: num_samples=300000' in info['err']
