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
# from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import build_backbone
# from mmdet.models.builder import build_head
# from mmdet.models.builder import build_neck
import torch.nn as nn
import torch

from bioharn.models.new_backbone import HRNet_V2
from bioharn.models.new_neck import HRFPN_V2
from bioharn.models.new_head import Shared2FCBBoxHead_V2
from bioharn.models.new_head import StandardRoIHead_V2
from bioharn.models.new_head import FCNMaskHead_V2
from bioharn.models.new_detector import MaskRCNN_V2

import mmcv
import kwcoco
import netharn as nh
from collections import OrderedDict
import warnings  # NOQA
from netharn.data import data_containers

# from bioharn.models.mm_models import MM_Detector
from bioharn.models.mm_models import MM_Coder
from bioharn.models.mm_models import _demo_batch
from bioharn.models.mm_models import _batch_to_mm_inputs
from bioharn.models.mm_models import _load_mmcv_weights
from bioharn.models.mm_models import _hack_numpy_gt_masks
from bioharn.models.mm_models import _ensure_unwrapped_and_mounted

# import torch

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


class MM_Detector_V3(nh.layers.Module):
    """
    Wraps mm detectors. Attempt to include logic for late fusion.
    """
    __BUILTIN_CRITERION__ = True

    def __init__(self, detector=None, classes=None, channels=None):
        super().__init__()
        self.detector = detector
        self.channels = ChannelSpec.coerce(channels)
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.coder = MM_Coder(self.classes)

    def demo_batch(self, bsize=3, h=256, w=256, with_mask=None):
        """
        Input data for testing this detector

        Example:
            >>> # xdoctest: +REQUIRES(module:mmdet)
            >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
            >>> self = MM_RetinaNet(classes)
            >>> #globals().update(**xdev.get_func_kwargs(MM_Detector.demo_batch))
            >>> self.demo_batch()
        """
        if with_mask is None:
            with_mask = getattr(self.detector, 'with_mask', False)
        channels = self.channels
        batch = _demo_batch(bsize, channels, h, w, with_mask=with_mask)
        return batch

    def forward(self, batch, return_loss=True, return_result=True):
        """
        Wraps the mm-detection interface with something that plays nicer with
        netharn.

        Args:
            batch (Dict): containing:
                - inputs (Dict[str, Tensor]):
                    mapping of input streams (e.g. rgb or motion) to
                    corresponding tensors.
                - label (None | Dict): optional if loss is needed. Contains:
                    tlbr: bounding boxes in tlbr space
                    class_idxs: bounding box class indices
                    weight: bounding box class weights (only used to set ignore
                        flags)

                OR an mmdet style batch containing:
                    imgs
                    img_metas
                    gt_bboxes
                    gt_labels
                    etc...

                    # OR new auxillary information
                    auxs
                    main_key
                    <subject to change>

            return_loss (bool): compute the loss
            return_result (bool): compute the result
                TODO: make this more efficient if loss was computed as well

        Returns:
            Dict: containing results and losses depending on if return_loss and
                return_result were specified.
        """
        if 'img_metas' in batch and ('inputs' in batch or 'imgs' in batch):
            # already in mm_inputs format
            orig_mm_inputs = batch
        else:
            orig_mm_inputs = _batch_to_mm_inputs(batch)

        mm_inputs = orig_mm_inputs.copy()

        # Hack: remove data containers if it hasn't been done already
        import netharn as nh
        xpu = nh.XPU.from_data(self)
        mm_inputs = _ensure_unwrapped_and_mounted(mm_inputs, xpu)

        if 'inputs' not in mm_inputs:
            raise Exception('Experimental MMDet stuff requires an inputs dict')

        inputs = mm_inputs.pop('inputs')
        img_metas = mm_inputs.pop('img_metas')

        if not isinstance(inputs, dict):
            raise ValueError('expected dict mapping channel names to tensors')

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', 'indexing with dtype')

        outputs = {}
        if return_loss:
            gt_bboxes = mm_inputs['gt_bboxes']
            gt_labels = mm_inputs['gt_labels']

            # _report_data_shape(mm_inputs)
            gt_bboxes_ignore = mm_inputs.get('gt_bboxes_ignore', None)

            trainkw = {}
            try:
                with_mask = self.detector.with_mask
            except AttributeError:
                with_mask = False
            if with_mask:
                if 'gt_masks' in mm_inputs:
                    # mmdet only allows numpy inputs
                    trainkw['gt_masks'] = _hack_numpy_gt_masks(mm_inputs['gt_masks'])

            # Compute input normalization
            losses = self.detector.forward(inputs, img_metas,
                                           gt_bboxes=gt_bboxes,
                                           gt_labels=gt_labels,
                                           gt_bboxes_ignore=gt_bboxes_ignore,
                                           return_loss=True, **trainkw)
            loss_parts = OrderedDict()
            for loss_name, loss_value in losses.items():
                if 'loss' in loss_name:
                    # Ensure these are tensors and not scalars for
                    # DataParallel
                    if isinstance(loss_value, torch.Tensor):
                        loss_parts[loss_name] = loss_value.mean().unsqueeze(0)
                    elif isinstance(loss_value, list):
                        loss_parts[loss_name] = sum(_loss.mean().unsqueeze(0) for _loss in loss_value)
                    else:
                        raise TypeError(
                            '{} is not a tensor or list of tensors'.format(loss_name))

            if hasattr(self, '_fix_loss_parts'):
                self._fix_loss_parts(loss_parts)

            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                an_input = ub.peek(inputs.values())
                bsize = an_input.shape[0]

                hack_inputs = [
                    {k: v[b:b + 1] for k, v in inputs.items()}
                    for b in range(bsize)
                ]
                # For whaver reason we cant run more than one test image at the
                # same time.
                batch_results = []
                for one_input, one_meta in zip(hack_inputs, img_metas):
                    result = self.detector.forward([one_input], [[one_meta]],
                                                   return_loss=False)
                    batch_results.append(result)
                outputs['batch_results'] = data_containers.BatchContainer(
                    batch_results, stack=False, cpu_only=True)
        return outputs

    def _init_backbone_from_pretrained(self, filename):
        """
        Loads pretrained backbone weights
        """
        import netharn as nh
        model_state = _load_mmcv_weights(filename)

        # HACK TO ONLY INIT THE RGB PART
        if 1:
            print('hacked off init backbone from pretrained')
        else:
            print('init backbone from pretrained')
            info = nh.initializers.functional.load_partial_state(
                self.detector.backbone.chan_backbones.rgb,
                model_state, verbose=1,
                mangle=False,
                association='embedding',
                leftover='kaiming_normal',
            )
            # print('info = {}'.format(ub.repr2(info, nl=True)))
            return info


class LateFusionPyramidBackbone(nn.Module):
    """
    Wraps another backbone to perform late fusion

    Ignore:
        >>> from bioharn.models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb,mx|my')
        >>> self = LateFusionPyramidBackbone(channels=channels)
        >>> batch = _demo_batch(3, channels, 256, 256, packed=True)
        >>> inputs = batch['inputs']
        >>> outputs = self.forward(inputs)
        >>> print(nh.data.data_containers.nestshape(fused_outputs))
        [torch.Size([4, 18, 64, 64]), torch.Size([4, 36, 32, 32]),
         torch.Size([4, 72, 16, 16]), torch.Size([4, 144, 8, 8])]

        >>> nh.util.number_of_parameters(self)
    """
    def __init__(self, channels='rgb', input_stats=None):
        super().__init__()
        channels = ChannelSpec.coerce(channels)
        chann_norm = channels.normalize()
        if input_stats is not None:
            assert set(input_stats.keys()) == set(chann_norm.keys())
        chan_backbones = {}
        for chan_key, chan_labels in chann_norm.items():
            if input_stats is None:
                chan_input_stats = None
            else:
                chan_input_stats = input_stats[chan_key]

            # TODO: generalize so different channels can use different
            # backbones
            hrnet_backbone_config = {
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
                'in_channels': len(chan_labels),
                'input_stats': chan_input_stats,
                'type': HRNet_V2
            }
            chan_backbone = build_backbone(hrnet_backbone_config)
            chan_backbones[chan_key] = chan_backbone
        self.chan_backbones = torch.nn.ModuleDict(chan_backbones)

    def forward(self, inputs):
        prefused_outputs = ub.ddict(dict)
        for chan_key in inputs.keys():
            chan_imgs = inputs[chan_key]
            chan_backbone = self.chan_backbones[chan_key]
            chan_outputs = chan_backbone.forward(chan_imgs)
            # chan_outputs is a list for each pyramid level
            for level, lvl_out in enumerate(chan_outputs):
                prefused_outputs[level][chan_key] = lvl_out

        fused_outputs = []
        for level, prefused in prefused_outputs.items():
            # Fuse by summing.
            # TODO: if the input streams are not aligned we should do that
            # here.
            # TODO: allow alternate late fusion schemes other than sum?
            fused = sum(prefused.values())
            fused_outputs.append(fused)

        return fused_outputs

    def init_weights(self, pretrained=None):
        for chan_key, chan_backbone in self.chan_backbones.items():
            chan_backbone.init_weights(pretrained=pretrained)


class MM_HRNetV2_w18_MaskRCNN(MM_Detector_V3):
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
        >>> channels = ChannelSpec.coerce('rgb,mx|my')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels)
        >>> batch = self.demo_batch()
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

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from bioharn.models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels)
        >>> batch = self.demo_batch()
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
        classes = kwcoco.CategoryTree.coerce(classes)
        channels = ChannelSpec.coerce(channels)

        mm_cfg = mmcv.Config({
            'model': {
                'backbone': {
                    'channels': channels,
                    'type': LateFusionPyramidBackbone
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
                        'classes': classes,
                        'reg_class_agnostic': False,
                        'roi_feat_size': 7,
                        'type': Shared2FCBBoxHead_V2
                    },
                    'mask_head': {
                        'conv_out_channels': 256,
                        'in_channels': 256,
                        'loss_mask': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_mask': True},
                        'classes': classes,
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
                'rpn': {'max_num': 1000, 'min_bbox_size': 0,
                        'nms_across_levels': False, 'nms_post': 1000,
                        'nms_pre': 1000, 'nms_thr': 0.7}
            },
            'train_cfg': {
                'rcnn': {
                    'assigner': {
                        'ignore_iof_thr': -1,
                        'match_low_quality': True,
                        'min_pos_iou': 0.5,
                        'neg_iou_thr': 0.5,
                        'pos_iou_thr': 0.5,
                        'type': 'MaxIoUAssigner'},
                    'debug': False,
                    'mask_size': 28,
                    'pos_weight': -1,
                    'sampler': {
                        'add_gt_as_proposals': True,
                        'neg_pos_ub': -1,
                        'num': 512,
                        'pos_fraction': 0.25,
                        'type': 'RandomSampler'}
                },
                'rpn': {
                    'allowed_border': -1,
                    'assigner': {
                        'ignore_iof_thr': -1,
                        'match_low_quality': True,
                        'min_pos_iou': 0.3,
                        'neg_iou_thr': 0.3,
                        'pos_iou_thr': 0.7,
                        'type': 'MaxIoUAssigner'},
                    'debug': False,
                    'pos_weight': -1,
                    'sampler': {
                        'add_gt_as_proposals': False,
                        'neg_pos_ub': -1,
                        'num': 256,
                        'pos_fraction': 0.5,
                        'type': 'RandomSampler'}
                },
                'rpn_proposal': {
                    'max_num': 1000, 'min_bbox_size': 0,
                    'nms_across_levels': False, 'nms_post': 1000,
                    'nms_pre': 2000, 'nms_thr': 0.7}
            }
        })

        from mmdet.models import build_detector
        detector = build_detector(
            mm_cfg['model'], train_cfg=mm_cfg['train_cfg'],
            test_cfg=mm_cfg['test_cfg'])

        super().__init__(detector, classes=classes, channels=channels)
