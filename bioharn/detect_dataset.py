from os.path import exists
from os.path import join
import netharn as nh
import numpy as np
import torch
import ubelt as ub
import kwarray
import kwimage
import torch.utils.data.sampler as torch_sampler


class DetectFitDataset(torch.utils.data.Dataset):
    """
    Loads data with ndsampler.CocoSampler and formats it in a way suitable for
    object detection.

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/bioharn'))
        >>> from bioharn.detect_dataset import *  # NOQA
        >>> self = DetectFitDataset.demo(augment='heavy', window_dims=(256, 256))
        >>> index = 15
        >>> item = self[index]
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> hwc01 = item['im'].numpy().transpose(1, 2, 0)
        >>> boxes = kwimage.Boxes(item['label']['cxywh'].numpy(), 'cxywh')
        >>> canvas = kwimage.ensure_uint255(hwc01)
        >>> canvas = boxes.draw_on(canvas)
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, sampler, augment='simple', window_dims=[512, 512],
                 input_dims='window', window_overlap=0.5, scales=[-3, 6],
                 factor=32):
        super(DetectFitDataset, self).__init__()

        self.sampler = sampler

        if input_dims == 'window':
            input_dims = window_dims

        self.factor = factor  # downsample factor of yolo grid
        self.input_dims = np.array(input_dims, dtype=np.int)
        self.window_dims = window_dims
        self.window_overlap = window_overlap

        # Can we do this lazilly?
        self._prebuild_pool()

        # assert np.all(self.input_dims % self.factor == 0)
        # FIXME: multiscale training is currently not enabled
        if not scales:
            scales = [1]

        rng = None
        self.rng = kwarray.ensure_rng(rng)

        if not augment:
            self.augmenter = None
        else:
            self.augmenter = DetectionAugmentor(mode=augment, rng=self.rng)

        # Used to resize images to the appropriate inp_size without changing
        # the aspect ratio.
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

    @ub.memoize_property
    def input_id(self):
        # Use the sampler to compute an input id
        depends = [
            self.augmenter and self.augmenter.json_id(),
            self.sampler._depends(),
            self.window_dims,
            self.input_dims,
        ]
        input_id = ub.hash_data(depends, hasher='sha512', base='abc')[0:32]
        return input_id

    def _prebuild_pool(self):
        print('Prebuild pool')
        positives, negatives = preselect_regions(
            self.sampler, self.window_overlap, self.window_dims)

        positives = kwarray.shuffle(positives, rng=971493943902)
        negatives = kwarray.shuffle(negatives, rng=119714940901)

        ratio = 2.0

        num_neg = int(len(positives) * ratio)
        chosen_neg = negatives[0:num_neg]

        chosen_regions = positives + chosen_neg
        self.chosen_regions = chosen_regions

    @classmethod
    def demo(cls, key='habcam', augment='simple', window_dims=(512, 512), **kw):
        """
        self = DetectFitDataset.demo()
        """
        import ndsampler
        if key == 'habcam':
            fpath = ub.expandpath('$HOME/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')
            dset = ndsampler.CocoDataset(fpath)
            from bioharn.detect_fit import DetectFitConfig
            config = DetectFitConfig()
            sampler = ndsampler.CocoSampler(dset, workdir=config['workdir'])
        else:
            sampler = ndsampler.CocoSampler.demo(key, **kw)
        self = cls(sampler, augment=augment, window_dims=window_dims)
        return self

    def __len__(self):
        # TODO: Use sliding windows so detection can be run and trained on
        # larger images
        return len(self.chosen_regions)
        # return len(self.sampler.image_ids)

    def __getitem__(self, spec):
        """
        Example:
            >>> # DISABLE_DOCTSET
            >>> from bioharn.detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo(key='shapes8', augment='complex', window_dims=(512, 512), gsize=(1920, 1080))
            >>> index = 1
            >>> item = self[{'index': index, 'input_dims': (300, 300)}]
            >>> hwc01 = item['im'].numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].numpy(), 'cxywh')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(hwc01)
            >>> boxes.draw()
            >>> for mask in item['label']['class_masks']:
            ...     kwimage.Mask(mask.data.cpu().numpy(), 'c_mask').draw()
            >>> kwplot.show_if_requested()

        Ignore:
            >>> from bioharn.detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo(key='habcam')
            >>> spec = {'index': 1, 'input_dims': (300, 300)}
        """
        import ndsampler
        if isinstance(spec, dict):
            index = spec['index']
            input_dims = spec['input_dims']
        elif isinstance(spec, (np.int64, int)):
            index = int(spec)
            input_dims = self.input_dims
        else:
            raise TypeError(type(spec))

        inp_size = np.array(input_dims[::-1])

        gid, slices, aids = self.chosen_regions[index]
        tr = {'gid': gid, 'slices': slices}

        # TODO: instead of forcing ourselfs to compute an iffy pad, we could
        # instead separate out all the non-square geometric augmentations and
        # then augment a bounding polygon representing the original region.
        # Based on that result we sample the appropriate data at the
        # appropriate scale. Then we apply the intensity based augmentors
        # after.
        pad = int((slices[0].stop - slices[0].start) * 0.3)

        img = self.sampler.dset.imgs[gid]
        if img.get('source', '') == 'habcam_2015_stereo':
            # Hack: dont pad next to the habcam border
            maxpad = ((img['width'] // 2) - slices[1].stop)
            pad = min(maxpad, pad)

            DO_HABCAM_DISPARITY = True
            if DO_HABCAM_DISPARITY:
                img_hashid = self.sampler.frames._lookup_hashid(gid)
                disp_cache_dpath = ub.ensuredir((self.sampler.frames.workdir, '_cache', '_disp_v1'))
                disp_cache_fpath = join(disp_cache_dpath, img_hashid + '_disp_v1.cog.tif')
                if not exists(disp_cache_fpath):
                    # Note: probably should be atomic
                    img3 = self.sampler.dset.load_image(gid)
                    imgL = img3[:, 0:img3.shape[1] // 2]
                    imgR = img3[:, img3.shape[1] // 2:]
                    disparity = compute_disparity(imgL, imgR, scale=0.5)
                    disparity = disparity.astype(np.float32)

                    ndsampler.utils.util_gdal._imwrite_cloud_optimized_geotiff(
                        disp_cache_fpath, disparity, compress='DEFLATE')
                disp_frame = ndsampler.utils.util_gdal.LazyGDalFrameFile(disp_cache_fpath)

                data_dims = ((img['width'] // 2), img['height'])
                data_slice, extra_padding, st_dims = self.sampler._rectify_tr(
                    tr, data_dims, window_dims=None, pad=pad)
                # Load the image data
                disp_im = disp_frame[data_slice]
                if extra_padding:
                    if disp_im.ndim != len(extra_padding):
                        extra_padding = extra_padding + [(0, 0)]  # Handle channels
                    disp_im = np.pad(disp_im, extra_padding, **{'mode': 'constant'})

        sample = self.sampler.load_sample(
            tr, visible_thresh=0.05,
            with_annots=['boxes', 'segmentation'], pad=pad)

        imdata = kwimage.atleast_3channels(sample['im'])[..., 0:3]

        boxes = sample['annots']['rel_boxes']
        cids = sample['annots']['cids']
        aids = sample['annots']['aids']
        ssegs = sample['annots']['rel_ssegs']
        anns = list(ub.take(self.sampler.dset.anns, aids))
        weights = [ann.get('weight', 1.0) for ann in anns]

        classes = self.sampler.classes
        dets = kwimage.Detections(
            boxes=boxes,
            segmentations=ssegs,
            class_idxs=np.array([classes.id_to_idx[cid] for cid in cids]),
            weights=np.array(weights),
            classes=classes,
        )
        orig_size = np.array(imdata.shape[0:2][::-1])

        if self.augmenter:
            inte_aug_det = self._intensity.to_deterministic()
            geom_aug_det = self._geometric.to_deterministic()

            input_dims = imdata.shape[0:2]
            imdata = seq_det.augment_image(imdata)
            output_dims = imdata.shape[0:2]

            if len(dets):
                dets = dets.warp(seq_det, input_dims=input_dims,
                                 output_dims=output_dims)

        pad = sample['params']['pad']
        if np.any(pad):
            # if we gave extra padding, crop back to the original shape
            y_sl, x_sl = [slice(d_pad, d - d_pad) for d, d_pad in
                          zip(imdata.shape[0:2], pad)]
            imdata = imdata[y_sl, x_sl]
            dets = dets.translate([-x_sl.start, -y_sl.start])

        # Clip any bounding boxes that went out of bounds
        h, w = imdata.shape[0:2]
        tlbr = dets.boxes.to_tlbr()
        old_area = tlbr.area
        tlbr = tlbr.clip(0, 0, w - 1, h - 1, inplace=True)
        new_area = tlbr.area
        dets.data['boxes'] = tlbr

        # Remove any boxes that have gone significantly out of bounds.
        remove_thresh = 0.1
        flags = (new_area / old_area).ravel() > remove_thresh
        dets = dets.compress(flags)

        # Apply letterbox resize transform to train and test
        self.letterbox.target_size = inp_size
        input_dims = imdata.shape[0:2]
        imdata = self.letterbox.augment_image(imdata)
        output_dims = imdata.shape[0:2]
        if len(dets):
            dets = dets.warp(self.letterbox, input_dims=input_dims,
                             output_dims=output_dims)

        # Remove any boxes that are no longer visible or out of bounds
        flags = (dets.boxes.area > 0).ravel()
        dets = dets.compress(flags)

        chw01 = torch.FloatTensor(imdata.transpose(2, 0, 1) / 255.0)

        # FIXME: yolo expects normalized boxes, but boxes relative to the input
        # chip dimensions are really what you want here. MMdet models work.
        cxwh = dets.boxes.toformat('cxywh')

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        bg_weight = torch.FloatTensor([1.0])

        label = {
            'cxywh': torch.FloatTensor(cxwh.data),
            'class_idxs': torch.LongTensor(dets.class_idxs[:, None]),
            'weight': torch.FloatTensor(dets.weights),

            'indices': index,
            'orig_sizes': orig_size,
            'bg_weights': bg_weight
        }

        if 'segmentations' in dets.data:
            has_mask_list = []
            class_mask_list = []
            for sseg in dets.data['segmentations']:
                if sseg is not None:
                    mask = sseg.to_mask(dims=chw01.shape[1:])
                    c_mask = mask.to_c_mask().data
                    mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)
                    class_mask_list.append(mask_tensor[None, :])
                    has_mask_list.append(1)
                else:
                    class_mask_list.append(None)
                    has_mask_list.append(-1)

            has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
            if len(class_mask_list) == 0:
                h, w = chw01.shape[1:]
                class_masks = torch.empty((0, h, w), dtype=torch.uint8)
            else:
                class_masks = torch.cat(class_mask_list, dim=0)
            label['class_masks'] = class_masks
            label['has_mask'] = has_mask

        item = {
            'im': chw01,
            'label': label,
            'tr': sample['tr'],
        }
        return item

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False, drop_last=False, multiscale=False):
        """
        Example:
            >>> # DISABLE_DOCTSET
            >>> self = DetectFitDataset.demo()
            >>> self.augmenter = None
            >>> loader = self.make_loader(batch_size=1, shuffle=True)
            >>> # training batches should have multiple shapes
            >>> shapes = set()
            >>> for raw_batch in ub.ProgIter(iter(loader), total=len(loader)):
            >>>     inputs = raw_batch['im']
            >>>     # test to see multiscale works
            >>>     shapes.add(inputs.shape[-1])
            >>>     if len(shapes) > 1:
            >>>         break
        """
        import torch.utils.data.sampler as torch_sampler
        assert len(self) > 0, 'must have some data'
        if shuffle:
            sampler = torch_sampler.RandomSampler(self)
        else:
            sampler = torch_sampler.SequentialSampler(self)

        if multiscale:
            batch_sampler = MultiScaleBatchSampler2(
                sampler, batch_size=batch_size, drop_last=drop_last,
                factor=32, scales=[-9, 1])
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last)

        def worker_init_fn(worker_id):
            # Make loaders more random
            kwarray.seed_global(np.random.get_state()[1][0] + worker_id)

        # torch.utils.data.sampler.WeightedRandomSampler
        loader = torch.utils.data.DataLoader(
            self, batch_sampler=batch_sampler,
            collate_fn=nh.data.collate.padded_collate, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return loader


class MultiScaleBatchSampler2(torch_sampler.BatchSampler):
    """
    Indicies returned in the batch are tuples indicating data index and scale
    index. Requires that dataset has a `multi_scale_inp_size` attribute.

    Args:
        sampler (Sampler): Base sampler. Must have a data_source attribute.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        resample_freq (int): how often to change scales. if None, then
            only one scale is used.

    Example:
        >>> import torch.utils.data as torch_data
        >>> class DummyDatset(torch_data.Dataset):
        >>>     def __init__(self):
        >>>         super(DummyDatset, self).__init__()
        >>>         self.input_dims = (512, 512)
        >>>     def __len__(self):
        >>>         return 1000
        >>> batch_size = 16
        >>> data_source = DummyDatset()
        >>> sampler = sampler1 = torch_sampler.RandomSampler(data_source)
        >>> self = rand = MultiScaleBatchSampler2(sampler1, resample_freq=10)
        >>> sampler2 = torch_sampler.SequentialSampler(data_source)
        >>> seq = MultiScaleBatchSampler2(sampler2, resample_freq=None)
        >>> rand_idxs = list(iter(rand))
        >>> seq_idxs = list(iter(seq))
    """

    def __init__(self, sampler, batch_size=16, resample_freq=10, factor=32,
                 scales=[-9, 1], drop_last=False):

        self.sampler = sampler
        self.drop_last = drop_last

        self.resample_interval = resample_freq

        input_dims = np.array(sampler.data_source.input_dims)

        self.base_input_dims = input_dims
        self.base_batch_size = batch_size

        factor_coeff = sorted(range(*scales), key=abs)
        factor = 32
        self.multi_scale_inp_size = [
            input_dims + (factor * i) for i in factor_coeff]

        self._batch_dynamics = [
            {'batch_size': batch_size, 'input_dims': dims}
            for dims in self.multi_scale_inp_size
        ]
        self.batch_size = None
        if ub.allsame(d['batch_size'] for d in self._batch_dynamics):
            self.batch_size = self._batch_dynamics[0]['batch_size']

        self.num_batches = None
        total = len(sampler)
        if self.drop_last:
            self.num_batches = int(np.floor(total / self.base_batch_size))
        else:
            self.num_batches = int(np.ceil(total / self.base_batch_size))

        self._dynamic_schedule = None
        self.rng = kwarray.ensure_rng(None)

    def __nice__(self):
        return str(len(self))

    def __len__(self):
        return self.num_batches

    def _init_dynamic_schedule(self):
        # print("INIT NEW DYNAMIC SCHEDULE")
        self._dynamic_schedule = ub.odict()
        total = len(self.sampler)
        remain = total

        # Always end on the native dynamic
        native_dynamic = {
            'batch_size': self.base_batch_size,
            'input_dims': self.base_input_dims,
        }

        if self.resample_interval is not None:
            final_native = self.resample_interval

            num_final = final_native * native_dynamic['batch_size']

            bx = 0
            while remain > 0:
                if remain <= num_final or bx == 0:
                    # The first and last batches will use the native
                    # input_dims.
                    current = native_dynamic.copy()
                    current['remain'] = remain
                    self._dynamic_schedule[bx] = current
                elif bx % self.resample_interval == 0:
                    dyn_idx = self.rng.randint(len(self._batch_dynamics))
                    current = self._batch_dynamics[dyn_idx]
                    current = current.copy()
                    if remain < 0:
                        current['batch_size'] += remain
                    current['remain'] = remain
                    self._dynamic_schedule[bx] = current

                    if remain < num_final:
                        # Ensure there are enough items for final batches
                        current['remain'] = remain
                        current['batch_size'] -= (num_final - remain)
                        self._dynamic_schedule[bx] = current

                if remain <= current['batch_size']:
                    current['batch_size'] = remain
                    current['remain'] = remain
                    current = current.copy()
                    self._dynamic_schedule[bx] = current

                bx += 1
                remain = remain - current['batch_size']
        else:
            self._dynamic_schedule[0] = {
                'batch_size': self.batch_size,
                'remain': total,
                'input_dims': self.base_input_dims,
            }

        final_bx, final_dynamic = list(self._dynamic_schedule.items())[-1]

        if self.drop_last:
            last = int(np.floor(final_dynamic['remain'] / final_dynamic['batch_size']))
        else:
            last = int(np.ceil(final_dynamic['remain'] / final_dynamic['batch_size']))

        num_batches = final_bx + last
        self.num_batches = num_batches

        # print(ub.repr2(self._dynamic_schedule, nl=1))
        # print('NEW SCHEDULE')

    def __iter__(self):
        # Start first batch
        self._init_dynamic_schedule()

        bx = 0
        batch = []
        if bx in self._dynamic_schedule:
            current_dynamic = self._dynamic_schedule[bx]
        # print('RESAMPLE current_dynamic = {!r}'.format(current_dynamic))

        for idx in self.sampler:
            # Specify dynamic information to the dataset
            index = {
                'index': idx,
                'input_dims': current_dynamic['input_dims'],
            }
            batch.append(index)
            if len(batch) == current_dynamic['batch_size']:
                yield batch

                # Start next batch
                bx += 1
                batch = []
                if bx in self._dynamic_schedule:
                    current_dynamic = self._dynamic_schedule[bx]
                    # print('RESAMPLE current_dynamic = {!r}'.format(current_dynamic))

        if len(batch) > 0 and not self.drop_last:
            yield batch


def preselect_regions(sampler, window_overlap, window_dims):
    """
    TODO: this might be generalized and added to ndsampler

    window_overlap = 0.5
    window_dims = (512, 512)
    """
    import netharn as nh

    keepbound = True

    gid_to_slider = {}
    for img in sampler.dset.imgs.values():
        if img.get('source', '') == 'habcam_2015_stereo':
            # Hack: todo, cannoncial way to get this effect
            full_dims = [img['height'], img['width'] // 2]
        else:
            full_dims = [img['height'], img['width']]

        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = nh.util.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap, keepbound=keepbound,
                                       allow_overshoot=True)
        gid_to_slider[img['id']] = slider

    from ndsampler import isect_indexer
    _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(sampler.dset)

    positives = []
    negatives = []
    for gid, slider in gid_to_slider.items():
        boxes = []
        regions = list(slider)
        for region in regions:
            y_sl, x_sl = region
            boxes.append([x_sl.start,  y_sl.start, x_sl.stop, y_sl.stop])
        boxes = kwimage.Boxes(np.array(boxes), 'tlbr')

        for region, box in zip(regions, boxes):
            aids = _isect_index.overlapping_aids(gid, box)
            # aids = sampler.regions.overlapping_aids(gid, box, visible_thresh=0.001)
            if len(aids):
                positives.append((gid, region, aids))
            else:
                negatives.append((gid, region, aids))

    print('Found {} positives'.format(len(positives)))
    print('Found {} negatives'.format(len(negatives)))
    return positives, negatives
    # len([gid for gid, a in sampler.dset.gid_to_aids.items() if len(a) > 0])


class DetectionAugmentor(object):
    """
    Ignore:
        self = DetectionAugmentor(mode='heavy')
    """
    def __init__(self, mode='simple', rng=None):
        import imgaug as ia
        from imgaug import augmenters as iaa
        self.rng = kwarray.ensure_rng(rng)

        self._intensity = None
        self._geometric = None

        if mode == 'simple':
            self._geometric = iaa.Sequential([
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.CropAndPad(px=(0, 4)),
            ])
            self._intensity = None
        elif mode == 'low':
            scale = .25
            rot = 30
            scale_base = ia.parameters.TruncatedNormal(
                1.0, (scale * 2) / 6, low=1 - scale, high=1 + scale)
            rot_base = ia.parameters.TruncatedNormal(
                0.0, (rot * 2) / 6, low=-rot, high=rot)
            scale_rv = ia.parameters.Choice([scale_base, 1], p=[.6, .4])
            rot_rv = ia.parameters.Choice([rot_base, 0], p=[.6, .4])

            self._geometric = iaa.Sequential([
                iaa.Affine(
                    scale=scale_rv,
                    rotate=rot_rv,
                    order=1,
                    cval=(0, 255),
                    backend='cv2',
                ),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.CropAndPad(px=(-3, 3)),
            ])
            self._intensity = None
        elif mode == 'medium':
            # The weather augmenters are very expensive, so we ditch them
            self._geometric = iaa.Sequential([
                iaa.Sometimes(0.55, iaa.Affine(
                    scale={"x": (1.0, 1.2), "y": (1.0, 1.2)},
                    rotate=(-40, 40),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    # order=[0, 1],
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.Sometimes(.9, iaa.CropAndPad(px=(-4, 4))),
            ], random_order=False)

            self._intensity = iaa.Sequential([
                iaa.Sequential([
                    iaa.Sometimes(.1, iaa.GammaContrast((0.5, 2.0))),
                    iaa.Sometimes(.1, iaa.LinearContrast((0.5, 1.5))),
                ], random_order=True),
                iaa.Sometimes(.5, iaa.Grayscale(alpha=(0, 1))),
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ]
        elif mode == 'heavy':
            scale = .25
            rot = 45
            self._geometric = iaa.Sequential([
                # Geometric
                iaa.Sometimes(0.55, iaa.Affine(
                    scale=ia.parameters.TruncatedNormal(1.0, (scale * 2) / 6, low=1 - scale, high=1 + scale),
                    rotate=ia.parameters.TruncatedNormal(0.0, (rot * 2) / 6, low=-rot, high=rot),
                    shear=ia.parameters.TruncatedNormal(0.0, 2.5, low=-16, high=16),
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),
            ])

            self._intensity = iaa.Sequential([
                # Color, brightness, saturation, and contrast
                iaa.Sometimes(.10, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Sometimes(.10, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(.10, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(.10, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Add((-10, 10), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Grayscale(alpha=(0, 1))),

                # Speckle noise
                iaa.Sometimes(.05, iaa.AddElementwise((-40, 40))),
                iaa.Sometimes(.05, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (.09, .31), size_percent=(0.19, 0.055),
                        per_channel=0.15
                    ),
                ])),
                # Blurring
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.GaussianBlur((0, 2.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ])),
            ], random_order=True)

        elif mode == 'complex':
            """
            notes:
                We have N independent random variables V[n] with
                `P(V[n] == 1) = p[n]`.

                The chance we draw none of them is
                >>> prod = np.prod
                >>> N = 12
                >>> p = [0.1 for n in range(N)]
                >>> prod([1 - p[n] for n in range(N)])

                More generally this is a binomial distribution when all p[n]
                are equal. (Unequal probabilities require poisson binomial,
                which is fairly expensive to compute). See
                https://github.com/scipy/scipy/issues/6000

                >>> from scipy.special import comb
                >>> n = 12
                >>> p = 0.1
                >>> dist = scipy.stats.binom(p=p, n=N)
                >>> # The probability sum(V) <= x is
                >>> # (ie we use at least x augmentors)
                >>> print('P(we use 0 augmentors) = {:.4f}'.format(dist.cdf(x=0)))
                >>> print('P(we use 1 augmentor)  = {:.4f}'.format(dist.cdf(x=1) - dist.cdf(x=0)))
                >>> print('P(we use 2 augmentors) = {:.4f}'.format(dist.cdf(x=2) - dist.cdf(x=1)))
                >>> print('P(we use 3 augmentors) = {:.4f}'.format(dist.cdf(x=3) - dist.cdf(x=2)))
                >>> print('P(we use 4 augmentors) = {:.4f}'.format(dist.cdf(x=4) - dist.cdf(x=3)))
                >>> print('P(we use 5 augmentors) = {:.4f}'.format(dist.cdf(x=5) - dist.cdf(x=4)))
                >>> print('P(we use 6 augmentors) = {:.4f}'.format(dist.cdf(x=6) - dist.cdf(x=5)))
                >>> print('P(we use 7 augmentors) = {:.4f}'.format(dist.cdf(x=7) - dist.cdf(x=6)))
                >>> print('P(we use 8 augmentors) = {:.4f}'.format(dist.cdf(x=8) - dist.cdf(x=7)))
            """
            self._geometric = iaa.Sequential([
                iaa.Sometimes(0.55, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=ia.parameters.TruncatedNormal(0.0, 2.5, low=-16, high=16),
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),

            ])
            self._intensity = iaa.Sequential([
                # Color, brightness, saturation, and contrast
                iaa.Sometimes(0.1, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Sometimes(.10, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(.10, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(.10, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Add((-10, 10), per_channel=0.5)),
                iaa.Sometimes(.1, iaa.Grayscale(alpha=(0, 1))),

                # Speckle noise
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
                iaa.Sometimes(.1, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (.09, .31), size_percent=(0.19, 0.055),
                        per_channel=0.15
                    ),
                ])),

                # Blurring
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.GaussianBlur((0, 2.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ])),

                # Sharpening
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.Sometimes(.1, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                    iaa.Sometimes(.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                ])),

                # Misc
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
            ], random_order=True)
        else:
            raise KeyError(mode)
        self.mode = mode
        self._augmenter.reseed(self.rng)

    def json_id(self):
        def imgaug_json_id(aug):
            # Ripped from netharn
            # TODO: submit a PR to imgaug that registers parameters
            # with classes
            import imgaug
            if isinstance(aug, tuple):
                return [imgaug_json_id(item) for item in aug]
            elif isinstance(aug, imgaug.parameters.StochasticParameter):
                return str(aug)
            else:
                try:
                    info = ub.odict()
                    info['__class__'] = aug.__class__.__name__
                    params = aug.get_parameters()
                    if params:
                        info['params'] = [imgaug_json_id(p) for p in params]
                    if isinstance(aug, list):
                        children = aug[:]
                        children = [imgaug_json_id(c) for c in children]
                        info['children'] = children
                    return info
                except Exception:
                    # imgaug is weird and buggy
                    return str(aug)

        params = {
            'intensity': imgaug_json_id(self._intensity),
            'geometric': imgaug_json_id(self._geometric),
        }
        return params

    def reseed(self, rng):
        return self._intensity.reseed(rng)

    def augment_detections(self, imdata, dets):

        return imdata, dets


def compute_disparity(imgL, imgR, scale=0.5):
    import cv2
    imgL1 = kwimage.imresize(imgL, scale=scale)
    imgR1 = kwimage.imresize(imgR, scale=scale)
    disp_alg = cv2.StereoSGBM_create(numDisparities=16, minDisparity=0,
                                     uniquenessRatio=5, blockSize=15,
                                     speckleWindowSize=50, speckleRange=2,
                                     P1=500, P2=2000, disp12MaxDiff=1000,
                                     mode=cv2.STEREO_SGBM_MODE_HH)
    disparity = disp_alg.compute(
        kwimage.convert_colorspace(imgL1, 'rgb', 'gray'),
        kwimage.convert_colorspace(imgR1, 'rgb', 'gray')
    )
    disparity = disparity - disparity.min()
    disparity = disparity / disparity.max()

    full_dsize = tuple(map(int, imgL1.shape[0:2][::-1]))
    disparity = kwimage.imresize(disparity, dsize=full_dsize)
    return disparity
