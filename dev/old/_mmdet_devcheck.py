def _devel():
    from mmdet import models

    models.detectors.CascadeRCNN

    from mmdet.models import build_detector
    build_detector({
        'type': 'CascadeRCNN',
        'num_stages': 1,
        'backbone': {
            'type': 'resnet',
            'depth': 50,
        }
    })

    import ndsampler
    classes = ndsampler.CategoryTree.coerce(3)

    # https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn_x101_32x4d_fpn_1x.py

    # model settings
    model_config = dict(
        type='CascadeRCNN',
        num_stages=3,
        pretrained='pytorch_resnext101.pth',
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            style='pytorch'),
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
            # use_sigmoid_cls=True
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='SharedFCBBoxHead',
                num_fcs=3,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(classes),
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                reg_class_agnostic=True),
            dict(
                type='SharedFCBBoxHead',
                num_fcs=3,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(classes),
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1],
                reg_class_agnostic=True),
            dict(
                type='SharedFCBBoxHead',
                num_fcs=3,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(classes),
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067],
                reg_class_agnostic=True)
        ])

    num_classes = len(classes)

    model_config = dict(
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
            style='pytorch'),
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
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
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
    import mmdet
    import mmdet.core

    from netharn.export.closer import Closer
    closer = Closer()
    closer.add_dynamic(mmdet.models.ConvFCBBoxHead)
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())

    closer = Closer()
    closer.add_dynamic(mmdet.models.RPNHead)
    print(closer.current_sourcecode())
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())
    print(chr(10).join(closer.logs))

    closer = Closer()
    closer.add_dynamic(mmdet.ops.nms)
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())
    print(chr(10).join(closer.logs))

    closer = Closer()
    closer.add_dynamic(mmdet.core.anchor_target)
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())
    print(chr(10).join(closer.logs))

    # FIXME: the build_assigner function directly references the assigners
    # module, which is something that is targeted for closure. I dont think we
    # can work around this.
    closer = Closer()
    closer.add_dynamic(mmdet.core.bbox.assign_and_sample)
    print(closer.current_sourcecode())
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())
    print(chr(10).join(closer.logs))

    closer = Closer()
    closer.add_dynamic(mmdet.models.anchor_heads.anchor_head.AnchorHead)
    print(closer.current_sourcecode())
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())
    print(chr(10).join(closer.logs))

    closer = Closer()
    closer.add_dynamic(mmdet.core.bbox.assign_sampling.build_assigner)
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())

    closer = Closer()
    closer.add_dynamic(mmdet.models.RPNHead)
    closer.expand(['mmdet'])
    print(closer.current_sourcecode())

    model = build_detector(model_config)

    closer = Closer()
    closer.add_dynamic(model.__class__)
    closer.expand(['mmdet'])

    print(closer.current_sourcecode())
