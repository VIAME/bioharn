def main():
    """
    SeeAlso:
        ~/code/mmdetection/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py

        # Mask Tools:
        ~/code/mmdetection/mmdet/core/mask/mask_target.py
        ~/code/mmdetection/mmdet/core/mask/structures.py


        # Forward Train Entry Point
        self.forward_train
        ~/code/mmdetection/mmdet/models/detectors/two_stage.py

            # RPN Proposal
            self.rpn_head.forward_train
            ~/code/mmdetection/mmdet/models/dense_heads/base_dense_head.py

            # ROI Stage
            self.roi_head.forward_train
            self.roi_head._mask_forward_train
            ~/code/mmdetection/mmdet/models/roi_heads/standard_roi_head.py

            # Mask Head
            ~/code/mmdetection/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py
    """
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


    if 1:
        from bioharn.models import mm_models
        bioharn_model = mm_models.MM_MaskRCNN(classes=11)

        self = self.to(0)

        batch = bioharn_model.demo_batch(h=224, w=224)
        mm_batch = mm_models._batch_to_mm_inputs(batch)
        imgs = mm_batch.pop('imgs').to(0).data[0].to(0)
        img_metas = mm_batch.pop('img_metas').to(0).data[0]
        gt_bboxes = mm_batch.pop('gt_bboxes').to(0).data[0]
        gt_labels = mm_batch.pop('gt_labels').to(0).data[0]
        gt_masks = mm_batch.pop('gt_masks').to(0).data[0]

        gt_btmp = gt_masks

        # gt_masks_ = [g.float() * 0.25 + 0.1 for g in gt_masks]
        # gt_btmp = mm_models._hack_numpy_gt_masks(gt_masks)

        gt_bboxes_ignore = None
        proposals = None
        kwargs = {}

        gt_btmp
        if 0:
            outputs = self.forward_train(imgs, img_metas, gt_bboxes, gt_labels,
                                         gt_masks=gt_btmp)

        else:
            # Unfolded

            img = imgs
            x = self.extract_feat(img)
            losses = dict()

            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            if 0:
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels,
                                                         gt_bboxes_ignore, gt_masks,
                                                         **kwargs)
            else:
                roi_head = self.roi_head
                # assign gts and sample proposals
                if roi_head.with_bbox or roi_head.with_mask:
                    num_imgs = len(img_metas)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    sampling_results = []
                    for i in range(num_imgs):
                        assign_result = roi_head.bbox_assigner.assign(
                            proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                            gt_labels[i])
                        sampling_result = roi_head.bbox_sampler.sample(
                            assign_result,
                            proposal_list[i],
                            gt_bboxes[i],
                            gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                        sampling_results.append(sampling_result)

                losses = dict()
                # bbox head forward and loss
                if roi_head.with_bbox:
                    bbox_results = roi_head._bbox_forward_train(
                        x, sampling_results, gt_bboxes, gt_labels, img_metas)
                    losses.update(bbox_results['loss_bbox'])

                # mask head forward and loss
                if roi_head.with_mask:

                    from mmdet.core import bbox2roi
                    bbox_feats = bbox_results['bbox_feats']
                    if 0:
                        mask_results = roi_head._mask_forward_train(
                            x, sampling_results, bbox_feats, gt_masks,
                            img_metas)
                    else:
                        if not roi_head.share_roi_extractor:
                            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                            mask_results = roi_head._mask_forward(x, pos_rois)
                        else:
                            pos_inds = []
                            device = bbox_feats.device
                            for res in sampling_results:
                                pos_inds.append(
                                    torch.ones(
                                        res.pos_bboxes.shape[0],
                                        device=device,
                                        dtype=torch.uint8))
                                pos_inds.append(
                                    torch.zeros(
                                        res.neg_bboxes.shape[0],
                                        device=device,
                                        dtype=torch.uint8))
                            pos_inds = torch.cat(pos_inds)

                            mask_results = roi_head._mask_forward(
                                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

                        import numpy as np
                        for m in gt_masks:
                            m.masks = gt_masks[0].masks.astype(np.float32)
                            m.masks += .5
                            m.masks *= 0.1

                        roi_head.train_cfg['mask_soft'] = True
                        mask_targets = roi_head.mask_head.get_targets(
                            sampling_results, gt_masks, roi_head.train_cfg)
                        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
                        loss_mask = roi_head.mask_head.loss(mask_results['mask_pred'],
                                                        mask_targets, pos_labels)

                        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
                    losses.update(mask_results['loss_mask'])
                    losses.update(roi_losses)


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

    if 0:
        import kwimage
        dets = kwimage.Detections.random(num=3, classes=3, segmentations=True)
        W, H = 224, 224
        dets = dets.scale((W, H))

        import kwplot
        kwplot.autompl()

        dets.draw(setlim=True)

        for sseg in dets.data['segmentations']:
            naive1 = sseg.to_mask(dims=(H, W))
            print('naive1.shape = {!r}'.format(naive1.shape))

            mask = sseg.to_relative_mask()
            print('mask.shape = {!r}'.format(mask.shape))
