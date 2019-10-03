"""
Notes:
    https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md
"""
import numpy as np
import netharn as nh
import torch
import kwimage
from distutils.version import LooseVersion


def _hack_mm_backbone_in_channels(backbone_cfg):
    """
    Verify the backbone supports input channels
    """
    if 'in_channels' not in backbone_cfg:
        return
    import mmdet
    _NEEDS_CHECK = LooseVersion(mmdet.__version__) < LooseVersion('1.1')
    _NEEDS_CHECK = True
    if _NEEDS_CHECK:
        import inspect
        from mmdet import models
        backbone_key = backbone_cfg['type']
        if backbone_key == 'ResNeXt':
            backbone_key = 'ResNet'
        backbone_cls = models.registry.BACKBONES.get(backbone_key)
        cls_kw = inspect.signature(backbone_cls).parameters
        # import xdev
        # cls_kw = xdev.get_func_kwargs(backbone_cls)
        if 'in_channels' not in cls_kw:
            if backbone_cfg['in_channels'] == 3:
                backbone_cfg.pop('in_channels')
            else:
                raise ValueError((
                    'mmdet.__version__={!r} does not support in_channels'
                ).format(mmdet.__version__))


def _batch_to_mm_inputs(batch):
    """
    Convert our netharn style batch to mmdet style
    """
    B, C, H, W = batch['im'].shape
    label = batch['label']

    # hack in img meta
    img_metas = [
        {
            'img_shape': (H, W, C),
            'ori_shape': (H, W, C),
            'pad_shape': (H, W, C),
            'filename': '<memory>.png',
            'scale_factor': 1.0,
            'flip': False,
        }
        for _ in range(B)
    ]

    gt_bboxes = []
    gt_labels = []

    batch_flags = label['class_idxs'] > -1
    for bx, flags in enumerate(batch_flags):
        flags = flags.view(-1)
        if 'tlbr' in label:
            gt_bboxes.append(label['tlbr'][bx][flags])
        elif 'cxywh' in label:
            gt_bboxes.append(kwimage.Boxes(label['cxywh'][bx][flags], 'cxywh').to_tlbr().data)
        gt_labels.append(label['class_idxs'][bx][flags].view(-1))
        # gt_bboxes_ignore = weight < 0.5
        # weight = label.get('weight', None)

    mm_inputs = {
        'imgs': batch['im'],
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
    }
    return mm_inputs


def _demo_batch(bsize=1, in_channels=3, h=256, w=256, classes=3):
    """
    Input data for testing this detector

    Example:
        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> globals().update(**xdev.get_func_kwargs(_demo_batch))
        >>> batch = _demo_batch()
        >>> mm_inputs = _batch_to_mm_inputs(batch)
    """
    input_shape = B, C, H, W = (bsize, in_channels, h, w)
    imgs = torch.rand(*input_shape)

    batch_items = []
    for bx in range(B):
        dets = kwimage.Detections.random(num=(10 - bx) % 11, classes=classes)
        dets = dets.scale((W, H)).tensor()

        label = {
            'tlbr': dets.boxes.to_tlbr().data.float(),
            'class_idxs': dets.class_idxs,
        }
        item = {
            'im': imgs[bx],
            'label': label,
        }
        batch_items.append(item)

    import netharn as nh
    batch = nh.data.collate.padded_collate(batch_items)
    return batch


class MM_Detector(nh.layers.Module):
    """
    """
    __BUILTIN_CRITERION__ = True

    def __init__(self, mm_config, train_cfg=None, test_cfg=None, classes=None,
                 input_stats=None):
        super(MM_Detector, self).__init__()
        import mmcv
        from mmdet.models import build_detector

        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes

        if input_stats is None:
            input_stats = {}

        self.input_norm = nh.layers.InputNorm(**input_stats)

        self.classes = classes

        if train_cfg is not None:
            train_cfg = mmcv.utils.config.ConfigDict(train_cfg)

        if test_cfg is not None:
            test_cfg = mmcv.utils.config.ConfigDict(test_cfg)

        self.detector = build_detector(mm_config, train_cfg=train_cfg,
                                       test_cfg=test_cfg)

    def demo_batch(self, bsize=3, in_channels=3, h=256, w=256):
        """
        Input data for testing this detector

        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> self = MM_RetinaNet(classes).mm_detector
        >>> globals().update(**xdev.get_func_kwargs(MM_Detector.demo_batch))
        >>> self.demo_batch()
        """
        batch = _demo_batch(bsize, in_channels, h, w)
        return batch

    def forward(self, batch, return_loss=True, return_result=True):
        """
        Wraps the mm-detection interface with something that plays nicer with
        netharn.

        Args:
            batch (Dict): containing:
                - im (tensor):
                - label (None | Dict): optional if loss is needed. Contains:
                    tlbr: bounding boxes in tlbr space
                    class_idxs: bounding box class indices
                    weight: bounding box class weights (only used to set ignore
                        flags)
            return_loss (bool): compute the loss
            return_result (bool): compute the result
                TODO: make this more efficient if loss was computed as well

        Returns:
            Dict: containing results and losses depending on if return_loss and
                return_result were specified.
        """
        mm_inputs = _batch_to_mm_inputs(batch)

        imgs = mm_inputs.pop('imgs')
        img_metas = mm_inputs.pop('img_metas')

        outputs = {}
        if return_loss:
            gt_bboxes = mm_inputs['gt_bboxes']
            gt_labels = mm_inputs['gt_labels']
            gt_bboxes_ignore = mm_inputs['gt_bboxes_ignore']
            losses = self.detector.forward(imgs, img_metas,
                                           gt_bboxes=gt_bboxes,
                                           gt_labels=gt_labels,
                                           gt_bboxes_ignore=gt_bboxes_ignore,
                                           return_loss=True)
            loss_parts = {}
            for k, vals in losses.items():
                for sx, val in enumerate(vals):
                    loss_parts[k + '_scale{}'.format(sx)] = val
            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                hack_imgs = [g[None, :] for g in imgs]
                # For whaver reason we cant run more than one test image at the
                # same time.
                import kwimage
                batch_dets = []
                batch_results = []

                for one_img, one_meta in zip(hack_imgs, img_metas):
                    result = self.detector.forward([one_img], [one_meta],
                                                   return_loss=False)
                    batch_results.append(result)

                    # Stack the results into a detections object
                    pred_cidxs = []
                    for cidx, cls_results in enumerate(result):
                        pred_cidxs.extend([cidx] * len(cls_results))
                    pred_tlbr = np.vstack([r[:, 0:4] for r in result])
                    pred_score = np.hstack([r[:, 4] for r in result])
                    det = kwimage.Detections(
                        boxes=kwimage.Boxes(pred_tlbr, 'tlbr'),
                        scores=pred_score,
                        class_idxs=pred_cidxs,
                        classes=self.classes
                    )
                    batch_dets.append(det)

                outputs['batch_dets'] = batch_dets
                outputs['batch_results'] = batch_results
        return outputs


class MM_RetinaNet(MM_Detector):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_RetinaNet

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> import mmcv
        >>> classes = ['class_{:0d}'.format(i) for i in range(80)]
        >>> xpu = nh.XPU(0)
        >>> self = MM_RetinaNet(classes).to(xpu.main_device)
        >>> filename = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        >>> _ = mmcv.runner.checkpoint.load_checkpoint(self.detector, filename)
        >>> batch = self.demo_batch()
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)

        import kwplot
        kwplot.autompl()

        kwplot.imshow(batch['im'][0])
        det = outputs['batch_dets'][0]
        det.draw()

        # filename = '/home/joncrall/Downloads/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        ours = self.detector.state_dict()

        print(ub.repr2(list(state_dict.keys()),nl=1))
        print(ub.repr2(list(ours.keys()),nl=1))
    """

    def __init__(self, classes, in_channels=3, input_stats=None):

        # from mmcv.runner.checkpoint import load_url_dist
        # url =
        # checkpoint = load_url_dist(url)
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        # pretrained = '/home/joncrall/Downloads/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        # pretrained = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth'

        # NOTE: mmdetect is weird how it determines which category is
        # background. When use_sigmoid_cls is True, there is physically one
        # less class that is evaluated. When softmax is True the first output
        # is the background, but this is obfuscated by slicing, which makes it
        # seem as if your foreground class idxs do start at zero (even though
        # they don't in this case).
        #
        # Either way I think we can effectively treat these detectors as if the
        # bacground class is at the end of the list.
        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes
        if 'background' in classes:
            assert classes.node_to_idx['background'] == len(classes) - 1
            num_classes = len(classes)
        else:
            num_classes = len(classes) + 1

        # model settings
        mm_config = dict(
            type='RetinaNet',
            pretrained='torchvision://resnet50',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                style='pytorch'),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs=True,
                num_outs=5),
            bbox_head=dict(
                type='RetinaHead',
                num_classes=num_classes,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                octave_base_scale=4,
                scales_per_octave=3,
                anchor_ratios=[0.5, 1.0, 2.0],
                anchor_strides=[8, 16, 32, 64, 128],
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
        # training and testing settings
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
        test_cfg = dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100)

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)
        super(MM_RetinaNet, self).__init__(mm_config, train_cfg=train_cfg,
                                           test_cfg=test_cfg, classes=classes,
                                           input_stats=input_stats)


class MM_CascadeRCNN(nh.layers.Module):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_CascadeRCNN

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> self = MM_CascadeRCNN(classes)
        >>> mm_inputs = self.detector.demo_batch()
        >>> gt_bboxes = mm_inputs['gt_bboxes']
        >>> gt_labels = mm_inputs['gt_labels']
        >>> mm_inputs['img'] = mm_inputs.pop('imgs')
        >>> mm_inputs['img_meta'] = mm_inputs.pop('img_metas')
        >>> img = mm_inputs['img']
        >>> img_meta = mm_inputs['img_meta']

        >>> imgs = [g[None, :] for g in img]
        >>> outputs = self.detector.forward(imgs, img_meta, return_loss=False)

        >>> losses = self.detector.forward(**mm_inputs, return_loss=True)

        >>> results = self.detector.simple_test(imgs[0], img_meta)
        >>> proposals=None
        >>> rescale=False
        >>> self = self.detector

        outputs = self.detector.extract_feat(inputs)
    """
    def __init__(self, classes, in_channels=3, input_stats=None):
        super(MM_CascadeRCNN, self).__init__()

        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes

        if input_stats is None:
            input_stats = {}
        self.input_norm = nh.layers.InputNorm(**input_stats)

        pretrained = 'open-mmlab://resnext101_32x4d'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth'

        num_classes = len(classes)
        mm_config =  dict(
            type='CascadeRCNN',
            num_stages=3,
            # pretrained='open-mmlab://resnext101_32x4d',
            pretrained=pretrained,
            backbone=dict(
                type='ResNeXt',
                depth=101,
                groups=32,
                base_width=4,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                style='pytorch',
                in_channels=in_channels
            ),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_scales=[8],
                anchor_ratios=[0.5, 1.0, 2.0],
                anchor_strides=[4, 8, 16, 32, 64],
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2, use_torchvision=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type='SharedFCBBoxHead',
                    num_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='SharedFCBBoxHead',
                    num_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='SharedFCBBoxHead',
                    num_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ])

        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=2000,
                max_num=2000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False)
            ],
            stage_loss_weights=[1, 0.5, 0.25])

        test_cfg = dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_thr=0.5),
                max_per_img=100,
                mask_thr_binary=0.5),
            keep_all_stages=False)

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)

        self.mm_detector = MM_Detector(mm_config, train_cfg=train_cfg,
                                       test_cfg=test_cfg, classes=self.classes)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.mm_detector(x)
        return x
