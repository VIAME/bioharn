"""
Ignore:
    >>> from bioharn.detect_fit import *  # NOQA
    >>> harn = setup_harn(bsize=2, datasets='special:shapes256',
    >>>     arch='MM_HRNetV2_w18_MaskRCNN', xpu='auto',
    >>>     workers=0, normalize_inputs='imagenet', sampler_backend=None)
    >>> harn.initialize()
    >>> batch = harn._demo_batch(1, 'vali')
    >>> #del batch['label']['has_mask']
    >>> #del batch['label']['class_masks']
    >>> from bioharn.models.mm_models import _batch_to_mm_inputs
    >>> mm_batch = _batch_to_mm_inputs(batch)

    >>> outputs, loss = harn.run_batch(batch)

Ignore:
    import mmdet
    import liberator
    closer = liberator.closer.Closer()
    # closer.add_dynamic(mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead_V2)
    # closer.add_dynamic(mmdet.models.detectors.MaskRCNN)
    # closer.add_dynamic(mmdet.models.detectors.TwoStageDetector)
    # closer.add_dynamic(mmdet.models.roi_heads.StandardRoIHead)
    #closer.add_dynamic(mmdet.models.roi_heads.StandardRoIHead)
    # closer.add_dynamic(mmdet.models.necks.HRFPN)
    # closer.add_dynamic(mmdet.models.roi_heads.test_mixins.BBoxTestMixin)
    # closer.add_dynamic(mmdet.models.roi_heads.test_mixins.MaskTestMixin)
    # closer.add_dynamic(mmdet.models.backbones.HRNet)
    # closer.add_dynamic(mmdet.models.roi_heads_V2.Shared2FCBBoxHead)
    closer.add_dynamic(mmdet.models.roi_heads.BBoxHead)

    # closer.expand(['mmdet'])
    print(closer.current_sourcecode())
"""

import ubelt as ub
import warnings  # NOQA
from netharn.data.channel_spec import ChannelSpec
from bioharn.models.mm_models import MM_Detector
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import build_backbone
from mmdet.models.builder import build_head
from mmdet.models.builder import build_neck
import torch.nn as nn
import torch

from bioharn.models.new_backbone import HRNet_V2
from bioharn.models.new_neck import HRFPN_V2
from bioharn.models.new_head import Shared2FCBBoxHead_V2
from bioharn.models.new_head import StandardRoIHead_V2
from bioharn.models.new_head import FCNMaskHead_V2
from bioharn.models.new_detector import MaskRCNN_V2

# import torch

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


class MM_HRNetV2_w18_MaskRCNN(MM_Detector):
    """
    SeeAlso:
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/detectors/base.py
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/detectors/two_stage.py
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py

    CommandLine:
        xdoctest -m /home/joncrall/code/bioharn/bioharn/models/new_models_v1.py MM_HRNetV2_w18_MaskRCNN

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from bioharn.models.new_models_v1 import *  # NOQA
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3)
        >>> import xdev
        >>> xdev.make_warnings_print_tracebacks()
        >>> import netharn as nh
        >>> print(nh.util.number_of_parameters(self))
        >>> self.to(0)
        >>> batch = self.demo_batch()
        >>> print('batch = {!r}'.format(batch))
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
        >>> print('batch_dets = {!r}'.format(batch_dets))
    """
    pretrained_url = 'open-mmlab://msra/hrnetv2_w18'

    def __init__(self, classes=None, input_stats=None, channels='rgb'):
        import mmcv
        import kwcoco
        classes = kwcoco.CategoryTree.coerce(classes)
        channels = ChannelSpec.coerce(channels)
        chann_norm = channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))

        mm_cfg = mmcv.Config({
            'model': {
                'backbone': {
                    'extra': {
                        'stage1': {
                            'block': 'BOTTLENECK',
                            'num_blocks': (4,),
                            'num_branches': 1,
                            'num_channels': (64,),
                            'num_modules': 1
                        },
                        'stage2': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4),
                            'num_branches': 2,
                            'num_channels': (18, 36),
                            'num_modules': 1
                        },
                        'stage3': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4),
                            'num_branches': 3,
                            'num_channels': (18, 36, 72),
                            'num_modules': 4
                        },
                        'stage4': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4, 4),
                            'num_branches': 4,
                            'num_channels': (18, 36, 72, 144),
                            'num_modules': 3
                        }
                    },
                    'in_channels': in_channels,
                    'type': HRNet_V2
                },
                'neck': {
                    'in_channels': [18, 36, 72, 144],
                    'out_channels': 256,
                    'type': HRFPN_V2,
                },
                'rpn_head': {
                    'anchor_generator': {
                        'ratios': [0.5, 1.0, 2.0],
                        'scales': [8],
                        'strides': [4, 8, 16, 32, 64],
                        'type': 'AnchorGenerator'
                    },
                    'bbox_coder': {
                        'target_means': [0.0, 0.0, 0.0, 0.0],
                        'target_stds': [1.0, 1.0, 1.0, 1.0],
                        'type': 'DeltaXYWHBBoxCoder'
                    },
                    'feat_channels': 256,
                    'in_channels': 256,
                    'loss_bbox': {'loss_weight': 1.0, 'type': 'L1Loss'},
                    'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': True},
                    'type': 'RPNHead'
                },
                'roi_head': {
                    'bbox_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 7, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'mask_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 14, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'bbox_head': {
                        'bbox_coder': {
                            'target_means': [0.0, 0.0, 0.0, 0.0],
                            'target_stds': [0.1, 0.1, 0.2, 0.2],
                            'type': 'DeltaXYWHBBoxCoder'
                        },
                        'fc_out_channels': 1024,
                        'in_channels': 256,
                        'loss_bbox': {'loss_weight': 1.0, 'type': 'L1Loss'},
                        'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
                        'num_classes': len(classes),
                        'reg_class_agnostic': False,
                        'roi_feat_size': 7,
                        'type': Shared2FCBBoxHead_V2
                    },
                    'mask_head': {
                        'conv_out_channels': 256,
                        'in_channels': 256,
                        'loss_mask': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_mask': True},
                        'num_classes': len(classes),
                        'num_convs': 4,
                        'type': FCNMaskHead_V2,
                    },
                    'type': StandardRoIHead_V2,
                },
                'pretrained': None,
                'type': MaskRCNN_V2,
            },
            'test_cfg': {
                'rcnn': {
                    'mask_thr_binary': 0.5,
                    'max_per_img': 100,
                    'nms': {'iou_threshold': 0.5, 'type': 'nms'},
                    'score_thr': 0.05
                },
                'rpn': {'max_num': 1000, 'min_bbox_size': 0, 'nms_across_levels': False, 'nms_post': 1000, 'nms_pre': 1000, 'nms_thr': 0.7}
            },
            'train_cfg': {
                'rcnn': {
                    'assigner': {'ignore_iof_thr': -1, 'match_low_quality': True, 'min_pos_iou': 0.5, 'neg_iou_thr': 0.5, 'pos_iou_thr': 0.5, 'type': 'MaxIoUAssigner'},
                    'debug': False,
                    'mask_size': 28,
                    'pos_weight': -1,
                    'sampler': {'add_gt_as_proposals': True, 'neg_pos_ub': -1, 'num': 512, 'pos_fraction': 0.25, 'type': 'RandomSampler'}
                },
                'rpn': {
                    'allowed_border': -1,
                    'assigner': {'ignore_iof_thr': -1, 'match_low_quality': True, 'min_pos_iou': 0.3, 'neg_iou_thr': 0.3, 'pos_iou_thr': 0.7, 'type': 'MaxIoUAssigner'},
                    'debug': False,
                    'pos_weight': -1,
                    'sampler': {'add_gt_as_proposals': False, 'neg_pos_ub': -1, 'num': 256, 'pos_fraction': 0.5, 'type': 'RandomSampler'}
                },
                'rpn_proposal': {'max_num': 1000, 'min_bbox_size': 0, 'nms_across_levels': False, 'nms_post': 1000, 'nms_pre': 2000, 'nms_thr': 0.7}
            }
        })

        super().__init__(
                mm_cfg['model'], train_cfg=mm_cfg['train_cfg'],
                test_cfg=mm_cfg['test_cfg'],
                classes=classes, input_stats=input_stats,
                channels=channels)


class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
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

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
