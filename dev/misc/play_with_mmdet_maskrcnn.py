def main():
    import mmcv
    norm_cfg = dict(type='BN', requires_grad=False)
    config = dict(
        backbone=dict(
            type='ResNet', depth=50, num_stages=3,
            strides=(1, 2, 2), dilations=(1, 1, 1), out_indices=(2, ),
            frozen_stages=1, norm_cfg=norm_cfg, norm_eval=True,
            style='caffe'),
        rpn_head=dict(
            type='RPNHead', in_channels=1024, feat_channels=1024,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            shared_head=dict(
                type='ResLayer',
                depth=50,
                stage=3,
                stride=2,
                dilation=1,
                style='caffe',
                norm_cfg=norm_cfg,
                norm_eval=True),
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=1024,
                featmap_strides=[16]),
            bbox_head=dict(
                type='BBoxHead',
                with_avg_pool=True,
                roi_feat_size=7,
                in_channels=2048,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=None,
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=0,
                in_channels=2048,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ),
        # model training and testing settings
        train_cfg=mmcv.Config(dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
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
                nms_pre=12000,
                nms_post=2000,
                max_num=2000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=14,
                pos_weight=-1,
                debug=False)
        )),
        test_cfg=mmcv.Config(dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=6000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5),
        )),
    )

    #     if isinstance(val, dict):
    #         walker[path] = mmcv.Config(val)

    from mmdet.models.detectors import MaskRCNN
    self = MaskRCNN(**config)

    import mmcv
    import kwimage
    import torch

    from bioharn.models import mm_models

    bioharn_model = mm_models.MM_MaskRCNN(classes=11)

    if 1:
        self = self.to(0)

        batch = bioharn_model.demo_batch()
        mm_batch = mm_models._batch_to_mm_inputs(batch)
        imgs = mm_batch.pop('imgs').to(0).data[0].to(0)
        img_metas = mm_batch.pop('img_metas').to(0).data[0]
        gt_bboxes = mm_batch.pop('gt_bboxes').to(0).data[0]
        gt_labels = mm_batch.pop('gt_labels').to(0).data[0]
        gt_masks = mm_batch.pop('gt_masks').to(0).data[0]
        outputs = self.forward_train(imgs, img_metas, gt_bboxes, gt_labels,
                                     gt_masks=gt_masks)

    else:
        C, W, H = 3, 224, 224
        B = 1
        img_metas = [{
            'img_shape': (H, W, C),
            'ori_shape': (H, W, C),
            'pad_shape': (H, W, C),
            'filename': '<memory>.png',
            'scale_factor': 1.0,
            'flip': False,
        } for _ in range(B)]
        gt_bboxes = mmcv.parallel.DataContainer([
            kwimage.Boxes.random(3).scale((W, H)).toformat('ltrb').tensor().data,
        ], stack=False)

        gt_labels = mmcv.parallel.DataContainer([
            torch.LongTensor([1, 1, 3]),
        ], stack=False)

        mmdet_batch = {
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
        }
        inputs = torch.rand(B, C, H, W)

