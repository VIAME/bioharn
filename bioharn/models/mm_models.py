"""
Notes:
    https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md
"""
import numpy as np
import netharn as nh
import torch
import kwimage
from collections import OrderedDict
from distutils.version import LooseVersion
import warnings


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

    Example:
        >>> # Test batch with empty item
        >>> bsize = [2, 0, 1, 1]
        >>> batch = _demo_batch(bsize)
        >>> mm_inputs = _batch_to_mm_inputs(batch)

        >>> # Test empty batch
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> bsize = [0, 0, 0, 0]
        >>> batch = _demo_batch(bsize)
        >>> mm_inputs = _batch_to_mm_inputs(batch)
    """
    ignore_thresh = 0.1

    if type(batch['im']).__name__ in ['BatchContainer']:
        # Things are already in data containers

        # Get the number of batch items for each GPU / group
        groupsizes = [item.shape[0] for item in batch['im'].data]

        B = len(batch['im'].data)
        C, H, W = batch['im'].data[0].shape[1:]

        DC = type(batch['im'])

        # hack in img meta
        img_metas = DC([
            [
                {
                    'img_shape': (H, W, C),
                    'ori_shape': (H, W, C),
                    'pad_shape': (H, W, C),
                    'filename': '<memory>.png',
                    'scale_factor': 1.0,
                    'flip': False,
                }
                for _ in range(num)
            ]
            for num in groupsizes
        ], stack=False, cpu_only=True)

        mm_inputs = {
            'imgs': batch['im'],
            'img_metas': img_metas,
        }

        # Handled pad collated batches. Ensure shapes are correct.
        if 'label' in batch:
            gt_bboxes = []
            gt_labels = []
            gt_masks = []

            label = batch['label']
            mm_inputs['gt_labels'] = label['class_idxs']

            if 'cxywh' in label:
                mm_inputs['gt_bboxes'] = DC(
                    [[kwimage.Boxes(b, 'cxywh').to_tlbr().data for b in bbs]
                     for bbs in label['cxywh'].data],
                    label['cxywh'].stack,
                    label['cxywh'].padding_value)

            if 'tlbr' in label:
                assert 'gt_bboxes' not in mm_inputs, 'already have boxes'
                mm_inputs['gt_bboxes'] = DC(
                    [[kwimage.Boxes(b, 'tlbr').to_tlbr().data for b in bbs]
                     for bbs in label['tlbr'].data],
                    label['cxywh'].stack,
                    label['cxywh'].padding_value)

            if 'has_mask' in label:
                # TODO
                if gt_masks:
                    mm_inputs['gt_masks'] = gt_masks

            if 'weight' in label:
                ignore_flags = DC(
                    [[w < ignore_thresh for w in ws]
                     for ws in label['weight'].data], label['weight'].stack)

                # filter ignore boxes
                outer_bboxes_ignore = []
                for outer_bx in range(len(ignore_flags.data)):
                    inner_bboxes_ignore = []
                    for inner_bx in range(len(ignore_flags.data[outer_bx])):
                        flags = ignore_flags.data[outer_bx][inner_bx]
                        ignore_bboxes = mm_inputs['gt_bboxes'].data[outer_bx][inner_bx][flags]
                        mm_inputs['gt_labels'].data[outer_bx][inner_bx] = mm_inputs['gt_labels'].data[outer_bx][inner_bx][~flags]
                        mm_inputs['gt_bboxes'].data[outer_bx][inner_bx] = mm_inputs['gt_bboxes'].data[outer_bx][inner_bx][~flags]
                        inner_bboxes_ignore.append(ignore_bboxes)
                    outer_bboxes_ignore.append(inner_bboxes_ignore)

                mm_inputs['gt_bboxes_ignore'] = DC(outer_bboxes_ignore,
                                                   label['weight'].stack)

    else:
        B, C, H, W = batch['im'].shape

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

        mm_inputs = {
            'imgs': batch['im'],
            'img_metas': img_metas,
        }

        # Handled pad collated batches. Ensure shapes are correct.
        if 'label' in batch:

            label = batch['label']

            if isinstance(label['class_idxs'], list):
                # Data was already collated as a list

                mm_inputs['gt_labels'] = label['class_idxs']

                if 'cxywh' in label:
                    mm_inputs['gt_bboxes'] = [
                        kwimage.Boxes(b, 'cxywh').to_tlbr().data
                        for b in label['cxywh']
                    ]
                if 'tlbr' in label:
                    assert 'gt_bboxes' not in mm_inputs, 'already have boxes'
                    mm_inputs['gt_bboxes'] = label['tlbr']

                if 'class_masks' in label:
                    mm_inputs['gt_masks'] = label['class_masks']

                if 'weight' in label:
                    gt_bboxes_ignore = [[w < ignore_thresh for w in ws]
                                        for ws in label['weight']]
                    mm_inputs['gt_bboxes_ignore'] = gt_bboxes_ignore

            else:

                # Old padded way
                gt_bboxes = []
                gt_labels = []
                gt_masks = []
                batch_tlbr = None
                batch_cxywh = None
                batch_has_mask = None
                batch_mask = None

                batch_cidxs = label['class_idxs'].view(B, -1)
                batch_flags = batch_cidxs > -1
                if 'tlbr' in label:
                    batch_tlbr = label['tlbr'].view(B, -1, 4)
                if 'cxywh' in label:
                    batch_cxywh = label['cxywh'].view(B, -1, 4)

                if 'class_masks' in label:
                    batch_has_mask = label['has_mask'].view(B, -1)
                    batch_mask = label['class_masks'].view(B, -1, H, W)

                for bx in range(B):
                    flags = batch_flags[bx]
                    flags = flags.view(-1)
                    if batch_tlbr is not None:
                        gt_bboxes.append(batch_tlbr[bx][flags])
                    if batch_cxywh is not None:
                        assert len(gt_bboxes) == 0, 'already have boxes'
                        _boxes = kwimage.Boxes(batch_cxywh[bx][flags], 'cxywh')
                        gt_bboxes.append(_boxes.to_tlbr().data)
                    if batch_has_mask is not None:
                        mask_flags = (batch_has_mask[bx] > 0)
                        _masks = batch_mask[bx][mask_flags]
                        gt_masks.append(_masks)

                    gt_labels.append(batch_cidxs[bx][flags].view(-1).long())
                    # gt_bboxes_ignore = weight < 0.5
                    # weight = label.get('weight', None)

                mm_inputs.update({
                    'gt_bboxes': gt_bboxes,
                    'gt_labels': gt_labels,
                    'gt_bboxes_ignore': None,
                })
                if gt_masks:
                    mm_inputs['gt_masks'] = gt_masks

    return mm_inputs


def _demo_batch(bsize=1, in_channels=3, h=256, w=256, classes=3,
                with_mask=True):
    """
    Input data for testing this detector

    Example:
        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> globals().update(**xdev.get_func_kwargs(_demo_batch))
        >>> batch = _demo_batch(with_mask=False)
        >>> mm_inputs = _batch_to_mm_inputs(batch)
    """
    USE_PHOTOS = False
    import kwarray
    rng = kwarray.ensure_rng(0)
    if USE_PHOTOS:
        import ndsampler
        coco_dset = ndsampler.CocoDataset.demo('photos')
        sampler = ndsampler.CocoSampler(coco_dset)

        batch_items = []

        for gid in [1]:
            img, annots = sampler.load_image_with_annots(gid)
            annots = [ann for ann in annots if 'bbox' in ann]
            hwc = img['imdata'][:]
            xywh = kwimage.Boxes([ann['bbox'] for ann in annots], 'xywh')
            cids = [ann['category_id'] for ann in annots]

            label = {
                'tlbr': torch.FloatTensor(xywh.to_tlbr().data),
                'class_idxs': torch.LongTensor(cids),
            }
            item = {
                'im': torch.FloatTensor(hwc.transpose(2, 0, 1) / 255),
                'label': label,
            }
            batch_items.append(item)

    else:
        if isinstance(bsize, list):
            item_sizes = bsize
            bsize = len(item_sizes)
        else:
            item_sizes = [rng.randint(0, 10) for bx in range(bsize)]

        input_shape = B, C, H, W = (bsize, in_channels, h, w)
        imgs = torch.rand(*input_shape)

        batch_items = []
        for bx in range(B):

            item_sizes[bx]

            dets = kwimage.Detections.random(num=item_sizes[bx],
                                             classes=classes,
                                             segmentations=True)
            dets = dets.scale((W, H))

            # Extract segmentations if they exist
            if with_mask:
                has_mask_list = []
                class_mask_list = []
                for sseg in dets.data['segmentations']:
                    if sseg is not None:
                        mask = sseg.to_mask(dims=(H, W))
                        c_mask = mask.to_c_mask().data
                        mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)
                        class_mask_list.append(mask_tensor[None, :])
                        has_mask_list.append(1)
                    else:
                        class_mask_list.append(None)
                        has_mask_list.append(-1)

                has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
                if len(class_mask_list) == 0:
                    class_masks = torch.empty((0, H, W), dtype=torch.uint8)
                else:
                    class_masks = torch.cat(class_mask_list, dim=0)
            else:
                class_masks = None

            dets = dets.tensor()
            label = {
                'tlbr': dets.boxes.to_tlbr().data.float(),
                'class_idxs': dets.class_idxs,
            }

            if with_mask:
                label['class_masks'] = class_masks
                label['has_mask'] = has_mask

            item = {
                'im': imgs[bx],
                'label': label,
            }
            batch_items.append(item)

    import netharn as nh
    batch = nh.data.collate.padded_collate(batch_items)
    return batch


class MM_Coder(object):
    """
    Standardize network inputs and outputs
    """

    def __init__(self, classes):
        self.classes = classes

    def decode_batch(self, outputs):
        """
        Transform mmdet outputs into a list of detections objects

        Args:
            outputs (Dict): dict containing loss_parts and batch_results

                b = 0  # indexes over batches
                k = 0  # indexes over the different classes

                # batch_results are an mmdet based list format
                batch_results = outputs['batch_results']
                result = batch_results[b]

                # result - can be a list with
                result[k] is an (N, 5) tensor encoding bboxes for class k

                # result - can be a 2-tuple with
                result[0][k] is a (N, 5) tensor encoding bboxes for class k
                result[1][k] is a N-len list of coco sseg dicts for class k

        Example:
            >>> from bioharn.models.mm_models import *  # NOQA
            >>> import torch
            >>> classes = ['a', 'b', 'c']
            >>> xpu = nh.XPU('cpu')
            >>> model = MM_CascadeRCNN(classes).to(xpu.main_device)
            >>> batch = model.demo_batch(bsize=1, h=256, w=256)
            >>> batch = xpu.move(batch)
            >>> outputs = model.forward(batch)
            >>> self = model.coder
            >>> batch_dets = model.coder.decode_batch(outputs)

        Example:
            >>> from bioharn.models.mm_models import *  # NOQA
            >>> classes = ['a', 'b', 'c']
            >>> xpu = nh.XPU(0)
            >>> model = MM_MaskRCNN(classes, in_channels=4).to(xpu.main_device)
            >>> batch = model.demo_batch(bsize=1, h=256, w=256)
            >>> batch = xpu.move(batch)
            >>> outputs = model.forward(batch)
            >>> self = model.coder
            >>> batch_dets = model.coder.decode_batch(outputs)
        """
        batch_results = outputs['batch_results']
        batch_dets = []

        from bioharn._hacked_distributed import BatchContainer
        if isinstance(batch_results, BatchContainer):
            batch_results = batch_results.data

        # HACK for the way mmdet handles background
        if 'backround' in self.classes:
            start = 1
        else:
            start = 0

        for result in batch_results:
            if isinstance(result, tuple) and len(result) == 2:
                # bbox and segmentation result
                mm_bbox_results = result[0]
                mm_sseg_results = result[1]
            elif isinstance(result, list):
                # bbox only result
                mm_bbox_results = result
                mm_sseg_results = None
            else:
                raise NotImplementedError(
                    'unknown mmdet result format. '
                    'type(result) = {}'.format(type(result))
                )

            if mm_bbox_results is not None:
                # Stack the results into a detections object
                pred_cidxs = []
                for cidx, cls_results in enumerate(mm_bbox_results, start=start):
                    pred_cidxs.extend([cidx + 1] * len(cls_results))
                pred_tlbr = np.vstack([r[:, 0:4] for r in mm_bbox_results])
                pred_score = np.hstack([r[:, 4] for r in mm_bbox_results])
            else:
                raise AssertionError('should always have bboxes')

            if mm_sseg_results is not None:
                pred_ssegs = []
                for cidx, cls_ssegs in enumerate(mm_sseg_results, start=start):
                    pred_ssegs.extend([
                        kwimage.Mask(sseg, 'bytes_rle') for sseg in cls_ssegs])
                pred_ssegs = kwimage.MaskList(pred_ssegs)
            else:
                pred_ssegs = kwimage.MaskList([None] * len(pred_cidxs))

            det = kwimage.Detections(
                boxes=kwimage.Boxes(pred_tlbr, 'tlbr'),
                scores=pred_score,
                class_idxs=np.array(pred_cidxs, dtype=np.int),
                segmentations=pred_ssegs,
                classes=self.classes
            )
            batch_dets.append(det)
        return batch_dets


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

        self.coder = MM_Coder(self.classes)

    def demo_batch(self, bsize=3, in_channels=None, h=256, w=256,
                   with_mask=None):
        """
        Input data for testing this detector

        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> self = MM_RetinaNet(classes).mm_detector
        >>> globals().update(**xdev.get_func_kwargs(MM_Detector.demo_batch))
        >>> self.demo_batch()
        """
        if in_channels is None:
            # use native in_channels if possible
            in_channels = getattr(self, 'in_channels', 3)
        if with_mask is None:
            with_mask = getattr(self.detector, 'with_mask', False)

        batch = _demo_batch(bsize, in_channels, h, w, with_mask=with_mask)
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

                OR an mmdet style batch containing:
                    imgs
                    img_metas
                    gt_bboxes
                    gt_labels
                    etc...

            return_loss (bool): compute the loss
            return_result (bool): compute the result
                TODO: make this more efficient if loss was computed as well

        Returns:
            Dict: containing results and losses depending on if return_loss and
                return_result were specified.
        """
        from bioharn._hacked_distributed import BatchContainer
        if 'img_metas' in batch and 'imgs' in batch:
            # already in mm_inputs format
            orig_mm_inputs = batch
        else:
            orig_mm_inputs = _batch_to_mm_inputs(batch)

        mm_inputs = orig_mm_inputs.copy()

        # from bioharn._hacked_distributed import _report_data_shape
        # print('--------------')
        # print('--------------')
        # _report_data_shape(mm_inputs)
        # print('--------------')
        # print('--------------')

        # Hack: remove data containers if it hasn't been done already
        import netharn as nh
        xpu = nh.XPU.from_data(self)
        for key in mm_inputs.keys():
            value = mm_inputs[key]
            if isinstance(value, BatchContainer):
                if len(value.data) != 1:
                    raise ValueError('data not scattered correctly')
                if value.cpu_only:
                    value = value.data[0]
                else:
                    value = xpu.move(value.data[0])
                mm_inputs[key] = value

        imgs = mm_inputs.pop('imgs')
        img_metas = mm_inputs.pop('img_metas')

        outputs = {}
        if return_loss:
            gt_bboxes = mm_inputs['gt_bboxes']
            gt_labels = mm_inputs['gt_labels']

            # _report_data_shape(mm_inputs)
            gt_bboxes_ignore = mm_inputs.get('gt_bboxes_ignore', None)

            trainkw = {}
            if self.detector.with_mask:
                if 'gt_masks' in mm_inputs:
                    # mmdet only allows numpy inputs
                    import kwarray
                    numpy_masks = [kwarray.ArrayAPI.numpy(mask)
                                   for mask in mm_inputs['gt_masks']]
                    trainkw['gt_masks'] = numpy_masks

            # Compute input normalization
            imgs_norm = self.input_norm(imgs)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'indexing with dtype')

                losses = self.detector.forward(imgs_norm, img_metas,
                                               gt_bboxes=gt_bboxes,
                                               gt_labels=gt_labels,
                                               gt_bboxes_ignore=gt_bboxes_ignore,
                                               return_loss=True, **trainkw)
                loss_parts = OrderedDict()
                for loss_name, loss_value in losses.items():
                    if 'loss' in loss_name:
                        if isinstance(loss_value, torch.Tensor):
                            loss_parts[loss_name] = loss_value.mean()
                        elif isinstance(loss_value, list):
                            loss_parts[loss_name] = sum(_loss.mean() for _loss in loss_value)
                        else:
                            raise TypeError(
                                '{} is not a tensor or list of tensors'.format(loss_name))

                outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                imgs_norm = self.input_norm(imgs)
                hack_imgs = [g[None, :] for g in imgs_norm]
                # For whaver reason we cant run more than one test image at the
                # same time.
                batch_results = []
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'indexing with dtype')
                    for one_img, one_meta in zip(hack_imgs, img_metas):
                        result = self.detector.forward([one_img], [[one_meta]],
                                                       return_loss=False)
                        batch_results.append(result)
                outputs['batch_results'] = BatchContainer(
                    batch_results, stack=False, cpu_only=True)
        return outputs


class MM_RetinaNet(MM_Detector):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_RetinaNet


    SeeAlso:
        ~/code/mmdetection/mmdet/models/detectors/cascade_rcnn.py
        ~/code/mmdetection/mmdet/models/detectors/retinanet.py

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> import mmcv
        >>> classes = ['class_{:0d}'.format(i) for i in range(80)]
        >>> self = MM_RetinaNet(classes)
        >>> head = self.detector.bbox_head

        >>> xpu = nh.XPU(0)
        >>> self = self.to(xpu.main_device)
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
            assert classes.node_to_idx['background'] == 0
            num_classes = len(classes)
        else:
            num_classes = len(classes) + 1

        self.in_channels = in_channels

        # model settings
        mm_config = dict(
            type='RetinaNet',
            pretrained='torchvision://resnet50',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=0,
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

                # anchor_ratios=[0.5, 1.0, 2.0],
                anchor_ratios=[0.75, 1.0, 1.5],
                # anchor_ratios=[0.8, 0.9, 1.0, 1.1, 1.2],

                anchor_strides=[8, 16, 32, 64, 128],
                # anchor_strides=[4, 8, 16, 32, 64],
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


class MM_CascadeRCNN(MM_Detector):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_CascadeRCNN

    Example:
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU(0)
        >>> self = MM_CascadeRCNN(classes).to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)

        import kwplot
        kwplot.autompl()
        kwplot.imshow(batch['im'].cpu()[0])
        batch_dets[0].draw()


        >>> #batch = xpu.move(batch)

        >>> #filename = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        >>> #_ = mmcv.runner.checkpoint.load_checkpoint(self.detector, filename)


        >>> mm_inputs = _batch_to_mm_inputs(batch)
        >>> gt_bboxes = mm_inputs['gt_bboxes']
        >>> gt_labels = mm_inputs['gt_labels']
        >>> img = mm_inputs.pop('imgs')
        >>> img_meta = mm_inputs.pop('img_metas')
        >>> imgs_hack = [g[None, :] for g in img]

        >>> losses = self.detector.forward(img, img_meta, **mm_inputs, return_loss=True)
        >>> outputs = self.detector.forward(imgs_hack, img_meta, return_loss=False)


        >>> results = self.detector.simple_test(imgs[0], img_meta)
        >>> proposals=None
        >>> rescale=False
        >>> self = self.detector

        outputs = self.detector.extract_feat(inputs)

        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU(None)
        >>> self = MM_CascadeRCNN(classes).to(xpu.main_device)
        >>> batch = _demo_batch([0, 0, 0], h=256, w=256)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)

        >>> self = self.detector
        >>> img = mm_inputs['imgs']
        >>> img_meta = mm_inputs['img_metas']


    Example:
        >>> # Test multiple channels
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU('cpu')
        >>> self = MM_CascadeRCNN(classes, in_channels=4).to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
    """
    def __init__(self, classes, in_channels=3, input_stats=None):
        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        if 'background' in classes:
            assert classes.node_to_idx['background'] == 0
            num_classes = len(classes)
        else:
            num_classes = len(classes) + 1

        pretrained = 'open-mmlab://resnext101_32x4d'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth'

        self.in_channels = in_channels

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
        super(MM_CascadeRCNN, self).__init__(mm_config, train_cfg=train_cfg,
                                             test_cfg=test_cfg,
                                             classes=classes,
                                             input_stats=input_stats)


class MM_MaskRCNN(MM_Detector):
    """
    Example:
        >>> # Test multiple channels
        >>> from bioharn.models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU(0)
        >>> self = MM_MaskRCNN(classes, in_channels=4).to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
        >>> batch_dets[0].data['segmentations'][0]
    """
    def __init__(self, classes, in_channels=3, input_stats=None):
        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)
        if 'background' in classes:
            assert classes.node_to_idx['background'] == 0
            num_classes = len(classes)
        else:
            num_classes = len(classes) + 1

        pretrained = 'torchvision://resnet50'
        self.in_channels = in_channels

        # model settings
        mm_config = dict(
            type='MaskRCNN',
            pretrained=pretrained,
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                in_channels=in_channels,
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
            bbox_head=dict(
                type='SharedFCBBoxHead',
                num_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
        # model training and testing settings
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
            rcnn=dict(
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
                debug=False))
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
                mask_thr_binary=0.5))

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)
        super(MM_MaskRCNN, self).__init__(mm_config, train_cfg=train_cfg,
                                          test_cfg=test_cfg, classes=classes,
                                          input_stats=input_stats)
