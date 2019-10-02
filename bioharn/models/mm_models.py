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
        backbone_key = backbone_cfg['type']
        if backbone_key == 'ResNeXt':
            backbone_key = 'ResNet'
        backbone_cls = mmdet.models.registry.BACKBONES.get(backbone_key)
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


class MM_Detector(nh.layers.Module):
    """

    """
    def __init__(self, mm_config, train_cfg=None, test_cfg=None, classes=None):
        super(MM_Detector, self).__init__()
        import mmcv
        from mmdet.models import build_detector

        self.classes = classes

        if train_cfg is not None:
            train_cfg = mmcv.utils.config.ConfigDict(train_cfg)

        if test_cfg is not None:
            test_cfg = mmcv.utils.config.ConfigDict(test_cfg)

        self.detector = build_detector(mm_config, train_cfg=train_cfg,
                                       test_cfg=test_cfg)

    def demo_batch(self, bsize=1, in_channels=3, h=256, w=256):
        """
        Input data for testing this detector
        """
        input_shape = B, C, H, W = (bsize, in_channels, h, w)
        imgs = torch.rand(*input_shape)

        batch_dets = []
        for _ in range(B):
            dets = kwimage.Detections.random(num=10, classes=self.classes)
            dets = dets.scale((W, H)).tensor()
            batch_dets.append(dets)
        gt_bboxes = [dets.boxes.to_tlbr().data.float() for dets in batch_dets]
        gt_labels = [dets.class_idxs for dets in batch_dets]

        label = {
            'tlbr': [b.to(self.main_device) for b in gt_bboxes],
            'class_idxs': [g.to(self.main_device) for g in gt_labels],
            'weights': None,
        }

        if self.detector.with_mask:
            label['gt_masks'] = NotImplemented  # TODO

        batch = {
            'im': imgs.to(self.main_device),
            'label': label,
        }
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
        imgs = batch['im']
        B, C, H, W = imgs.shape

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

        outputs = {}
        if return_loss:
            label = batch['label']
            gt_bboxes = label['tlbr']
            gt_labels = label['class_idxs']
            weight = label.get('weight', None)
            if weight is None:
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = weight < 0.5

            losses = self.mm_detector.detector.forward(imgs, img_metas,
                                                       gt_bboxes=gt_bboxes,
                                                       gt_labels=gt_labels,
                                                       gt_bboxes_ignore=gt_bboxes_ignore,
                                                       return_loss=True)
            loss_parts = losses
            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                hack_imgs = [g[None, :] for g in imgs]
                result = self.mm_detector.detector.forward(hack_imgs, img_metas,
                                                            return_loss=False)
                outputs['result'] = result
        return outputs


class MM_RetinaNet(nh.layers.Module):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_CascadeRCNN

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> self = MM_RetinaNet(classes).to(0)
        >>> inputs = self.demo_batch()
        >>> outputs = self.forward(inputs)

    """

    def __init__(self, classes, in_channels=3, input_stats=None):
        super(MM_RetinaNet, self).__init__()

        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes

        if input_stats is None:
            input_stats = {}
        self.input_norm = nh.layers.InputNorm(**input_stats)

        num_classes = len(classes)

        # from mmcv.runner.checkpoint import load_url_dist
        # url =
        # checkpoint = load_url_dist(url)
        pretrained = 'torchvision://resnet50'
        pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'

        # model settings
        mm_config = dict(
            type='RetinaNet',
            pretrained=pretrained,
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

        self.mm_detector = MM_Detector(mm_config, train_cfg=train_cfg,
                                       test_cfg=test_cfg, classes=self.classes)


class MM_CascadeRCNN(nh.layers.Module):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_CascadeRCNN

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> self = MM_CascadeRCNN(classes)
        >>> mm_inputs = self.demo_inputs()
        >>> gt_bboxes = mm_inputs['gt_bboxes']
        >>> gt_labels = mm_inputs['gt_labels']
        >>> mm_inputs['img'] = mm_inputs.pop('imgs')
        >>> mm_inputs['img_meta'] = mm_inputs.pop('img_metas')
        >>> img = mm_inputs['img']
        >>> img_meta = mm_inputs['img_meta']

        >>> imgs = [g[None, :] for g in img]
        >>> outputs = self.mm_detector.detector.forward(imgs, img_meta, return_loss=False)

        >>> losses = self.mm_detector.detector.forward(**mm_inputs, return_loss=True)

        >>> results = self.mm_detector.detector.simple_test(imgs[0], img_meta)
        >>> proposals=None
        >>> rescale=False
        >>> self = self.mm_detector.detector

        outputs = self.mm_detector.detector.extract_feat(inputs)
    """
    def __init__(self, classes, in_channels=3, input_stats=None):
        super(MM_CascadeRCNN, self).__init__()

        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes

        if input_stats is None:
            input_stats = {}
        self.input_norm = nh.layers.InputNorm(**input_stats)

        num_classes = len(classes)
        mm_config =  dict(
            type='CascadeRCNN',
            num_stages=3,
            pretrained='open-mmlab://resnext101_32x4d',
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
