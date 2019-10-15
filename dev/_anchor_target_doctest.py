    """
    For ~/code/mmdetection/mmdet/core/anchor/anchor_target.py

    Example:
        flat_anchors = torch.Tensor([
            [-21.,  -9.,  24.,  12.],
            [294.,  -6., 325.,  25.],
            [  6.,  90.,  37., 121.],
            [445., 215., 490., 304.],
            [251., 413., 340., 594.],
            [299., 118., 660., 841.]
        ])
        valid_flags = torch.ones(len(flat_anchors), dtype=torch.uint8)
        gt_bboxes = torch.FloatTensor([[132.6667, 104.8757, 238.6326, 151.8874]])
        gt_labels = torch.LongTensor([[1]])
        img_meta = {
            'img_shape': (1024, 1024, 3),
        }
        target_means = [0., 0., 0., 0.]
        target_stds = [1., 1., 1., 1.]
        import mmcv
        cfg = mmcv.Config({
            'allowed_border': 0,
            'assigner': {
                'ignore_iof_thr': -1,
                'min_pos_iou': 0.3,
                'neg_iou_thr': 0.3,
                'pos_iou_thr': 0.7,
                'type': 'MaxIoUAssigner',
            },
            'debug': False,
            'pos_weight': -1,
            'sampler': {
                'add_gt_as_proposals': False,
                'neg_pos_ub': -1,
                'num': 256,
                'pos_fraction': 0.5,
                'type': 'RandomSampler',
            },
        })
        label_channels = 1
        sampling = True
        unmap_outputs = True

    """
