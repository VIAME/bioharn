

def test_toydata():
    import ubelt as ub
    command = ub.codeblock(
        r"""
        python -m bioharn.detect_fit \
            --nice=bioharn_shapes_example \
            --datasets=special:shapes256 \
            --schedule=step-10-30 \
            --augment=complex \
            --init=noop \
            --arch=retinanet \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=128,128 \
            --window_overlap=0.0 \
            --normalize_inputs=True \
            --workers=4 --xpu=0 --batch_size=8 --bstep=1 \
            --balance=tfidf
            --sampler_backend=cog
        """)
    info = ub.cmd(command, verbose=3, check=True)
