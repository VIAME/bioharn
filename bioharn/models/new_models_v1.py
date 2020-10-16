import ubelt as ub
import warnings  # NOQA
from netharn.data.channel_spec import ChannelSpec
from bioharn.models.mm_models import MM_Detector


class MM_HRNetV2_w18_MaskRCNN(MM_Detector):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from bioharn.models.new_models_v1 import *  # NOQA
        >>> self = MM_HRNet_MaskRCNN(classes=3)
        >>> print(nh.util.number_of_parameters(self))
        >>> self.to(0)
        >>> batch = self.demo_batch()
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
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
                    'type': 'HRNet'
                },
                'neck': {
                    'in_channels': [18, 36, 72, 144],
                    'out_channels': 256,
                    'type': 'HRFPN'
                },
                'pretrained': None,
                'roi_head': {
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
                        'type': 'Shared2FCBBoxHead'
                    },
                    'bbox_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 7, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'mask_head': {
                        'conv_out_channels': 256,
                        'in_channels': 256,
                        'loss_mask': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_mask': True},
                        'num_classes': len(classes),
                        'num_convs': 4,
                        'type': 'FCNMaskHead'
                    },
                    'mask_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 14, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'type': 'StandardRoIHead'
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
                'type': 'MaskRCNN'
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
