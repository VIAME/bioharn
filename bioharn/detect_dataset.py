import netharn as nh
import numpy as np
import torch
import ubelt as ub
import kwarray
import kwimage
import torch.utils.data.sampler as torch_sampler
from bioharn.channel_spec import ChannelSpec
from bioharn.data_containers import ItemContainer
from bioharn.data_containers import container_collate
from functools import partial
import numbers

# _debug = print
_debug = ub.identity


class DetectFitDataset(torch.utils.data.Dataset):
    """
    Loads data with ndsampler.CocoSampler and formats it in a way suitable for
    object detection.

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/bioharn'))
        >>> from bioharn.detect_dataset import *  # NOQA
        >>> self = DetectFitDataset.demo(key='shapes', channels='rgb|disparity', augment='heavy', window_dims=(390, 390))
        >>> index = 15
        >>> item = self[index]
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> components = self.channels.decode(item['inputs'], axis=0)
        >>> rgb01 = components['rgb'].data.numpy().transpose(1, 2, 0)
        >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
        >>> canvas_rgb = np.ascontiguousarray(kwimage.ensure_uint255(rgb01))
        >>> canvas_rgb = boxes.draw_on(canvas_rgb)
        >>> kwplot.imshow(canvas_rgb, pnum=(1, 2, 1), fnum=1)
        >>> if 'disparity' in components:
        >>>     disp = components['disparity'].data.numpy().transpose(1, 2, 0)
        >>>     disp_canvs = np.ascontiguousarray(disp.copy())
        >>>     disp_canvs = disp_canvs - disp_canvs.min()
        >>>     disp_canvs = disp_canvs / disp_canvs.max()
        >>>     disp_canvs = boxes.draw_on(disp_canvs)
        >>>     kwplot.imshow(disp_canvs, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, sampler, augment='simple', window_dims=[512, 512],
                 input_dims='window', window_overlap=0.5, scales=[-3, 6],
                 factor=32, use_segmentation=True, gravity=0.0,
                 classes_of_interest=None, channels='rgb'):
        super(DetectFitDataset, self).__init__()

        self.sampler = sampler

        if input_dims == 'window':
            input_dims = window_dims

        self.use_segmentation = use_segmentation
        self.channels = ChannelSpec.coerce(channels)

        self.factor = factor  # downsample factor of yolo grid
        self.input_dims = np.array(input_dims, dtype=np.int)
        self.window_dims = window_dims
        self.window_overlap = window_overlap

        if classes_of_interest is None:
            classes_of_interest = []
        self.classes_of_interest = {c.lower() for c in classes_of_interest}

        # Can we do this lazilly?
        self._prebuild_pool()

        window_jitter = 0.5 if augment == 'complex' else 0
        window_jitter = 0.1 if augment == 'medium' else 0
        self.window_jitter = window_jitter

        # assert np.all(self.input_dims % self.factor == 0)
        # FIXME: multiscale training is currently not enabled
        if not scales:
            scales = [1]

        rng = None
        self.rng = kwarray.ensure_rng(rng)

        if not augment:
            self.augmenter = None
        else:
            self.augmenter = DetectionAugmentor(mode=augment, gravity=gravity,
                                                rng=self.rng)

        self.disable_augmenter = False  # flag for forcing augmentor off

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

        ratio = 0.1

        num_neg = int(len(positives) * ratio)
        chosen_neg = negatives[0:num_neg]

        chosen_regions = positives + chosen_neg
        self.chosen_regions = chosen_regions

    @classmethod
    def demo(cls, key='habcam', augment='simple', channels='rgb', window_dims=(512, 512), **kw):
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
            sampler = ndsampler.CocoSampler.demo(key, aux='disparity', **kw)
        self = cls(sampler, augment=augment, window_dims=window_dims, channels=channels)
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
            >>> torch_dset = self = DetectFitDataset.demo(key='shapes8', augment='complex', window_dims=(512, 512), gsize=(1920, 1080))
            >>> index = 1
            >>> spec = {'index': index, 'input_dims': (120, 120)}
            >>> item = self[spec]
            >>> hwc01 = item['inputs']['rgb'].data.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()  # xdoc: +SKIP
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.imshow(hwc01)
            >>> labels = ['w={}'.format(w) for w in item['label']['weight'].data]
            >>> boxes.draw(labels=labels)
            >>> for mask in item['label']['class_masks'].data:
            ...     kwimage.Mask(mask.data.cpu().numpy(), 'c_mask').draw()
            >>> fig = plt.gcf()
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()

        Ignore:
            >>> from bioharn.detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo(key='habcam', augment='complex')
            >>> spec = {'index': 954, 'input_dims': (300, 300)}
            >>> item = self[spec]
            >>> hwc01 = item['inputs']['rgb'].data.numpy().transpose(1, 2, 0)
            >>> disparity = item['disparity'].data
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=True, fnum=1, pnum=(1, 2, 1))
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(hwc01, fnum=1, pnum=(1, 2, 1))
            >>> boxes.draw()
            >>> for mask, flag in zip(item['label']['class_masks'].data, item['label']['has_mask'].data):
            >>>      if flag > 0:
            >>>          kwimage.Mask(mask.data.cpu().numpy(), 'c_mask').draw()
            >>> kwplot.imshow(disparity, fnum=1, pnum=(1, 2, 2))
            >>> boxes.draw()
            >>> kwplot.show_if_requested()
        """

        if isinstance(spec, dict):
            index = spec['index']
            input_dims = spec['input_dims']
        elif isinstance(spec, numbers.Integral):
            index = int(spec)
            input_dims = self.input_dims
        else:
            raise TypeError(type(spec))

        inp_size = np.array(input_dims[::-1])

        gid, slices, _ = self.chosen_regions[index]

        if self.augmenter is not None and self.window_jitter and not self.disable_augmenter:
            # jitter the sliding window location a little bit.
            jitter = self.window_jitter
            y1 = slices[0].start
            y2 = slices[0].stop
            x1 = slices[1].start
            x2 = slices[1].stop
            box = kwimage.Boxes([[x1, y1, x2, y2]], 'tlbr')
            rng = self.rng
            offset = (int(box.width[0, 0] * jitter * (0.5 - rng.rand())),
                      int(box.height[0, 0] * jitter * (.5 - rng.rand())))
            box = box.translate(offset)
            x1, y1, x2, y2 = map(int, box.data[0])
            slices = tuple([slice(y1, y2), slice(x1, x2)])

        tr = {'gid': gid, 'slices': slices}
        _debug('tr = {!r}'.format(tr))

        # TODO: instead of forcing ourselfs to compute an iffy pad, we could
        # instead separate out all the non-square geometric augmentations and
        # then augment a bounding polygon representing the original region.
        # Based on that result we sample the appropriate data at the
        # appropriate scale. Then we apply the intensity based augmentors
        # after.
        pad = int((slices[0].stop - slices[0].start) * 0.3)

        img = self.sampler.dset.imgs[gid]

        disp_im = None
        _debug('self.channels = {!r}'.format(self.channels))
        if 'disparity' in self.channels:

            sampler = self.sampler
            # First check if the dataset defines a proper disparity channel
            if 'auxillary' in img:
                from ndsampler.utils import util_gdal
                disp_fpath = sampler.dset.get_auxillary_fpath(gid, 'disparity')
                disp_frame = util_gdal.LazyGDalFrameFile(disp_fpath)
                data_dims = disp_frame.shape[0:2]
                data_slice, extra_padding, st_dims = self.sampler._rectify_tr(
                    tr, data_dims, window_dims=None, pad=pad)
                # Load the image data
                disp_im = disp_frame[data_slice]
                if extra_padding:
                    if disp_im.ndim != len(extra_padding):
                        extra_padding = extra_padding + [(0, 0)]  # Handle channels
                    disp_im = np.pad(disp_im, extra_padding, **{'mode': 'constant'})

                # if disp_im.max() > 1.0:
                #     raise AssertionError('gid={} {}'.format(gid, ub.repr2(kwarray.stats_dict(disp_im))))

            if disp_im is None:
                raise Exception('no auxillary disparity')
                disp_im = np.zeros()

        with_annots = ['boxes']
        _debug('self.use_segmentation = {!r}'.format(self.use_segmentation))
        if self.use_segmentation:
            with_annots += ['segmentation']

        # NOTE: using the gdal backend samples HABCAM images in 16ms, and no
        # backend samples clocks in at 72ms. The disparity speedup is about 2x
        sample = self.sampler.load_sample(tr, visible_thresh=0.05,
                                          with_annots=with_annots, pad=pad)

        _debug('sample = {!r}'.format(sample))
        imdata = kwimage.atleast_3channels(sample['im'])[..., 0:3]

        boxes = sample['annots']['rel_boxes'].view(-1, 4)
        cids = sample['annots']['cids']
        aids = sample['annots']['aids']
        ssegs = sample['annots']['rel_ssegs']
        anns = list(ub.take(self.sampler.dset.anns, aids))
        weights = [ann.get('weight', 1.0) for ann in anns]

        for idx, cid in enumerate(cids):
            # set weights of uncertain varaibles to zero
            catname = self.sampler.dset._resolve_to_cat(cid)['name']
            if catname.lower() in {'unknown', 'ignore'}:
                weights[idx] = 0

            if self.classes_of_interest:
                if catname.lower() not in self.classes_of_interest:
                    weights[idx] = 0

        # TODO: remove anything marked as "negative"
        HACK_SSEG = False
        if HACK_SSEG:
            if img.get('source', '') in ['habcam_2015_stereo', 'habcam_stereo']:
                ssegs = []
                for xy, w in zip(boxes.xy_center, boxes.width):
                    r = w / 2.0
                    circle = kwimage.Polygon.circle(xy, r)
                    ssegs.append(circle)
                ssegs = kwimage.PolygonList(ssegs)

        classes = self.sampler.classes
        dets = kwimage.Detections(
            boxes=boxes,
            segmentations=ssegs,
            class_idxs=np.array([classes.id_to_idx[cid] for cid in cids]),
            weights=np.array(weights),
            classes=classes,
        )
        _debug('dets = {!r}'.format(dets))
        orig_size = np.array(imdata.shape[0:2][::-1])

        if self.augmenter and not self.disable_augmenter:
            _debug('augment')
            imdata, dets, disp_im = self.augmenter.augment_data(
                imdata, dets, disp_im)
            # disp_im.dtype

        _debug('un-pad')
        pad = sample['params']['pad']
        if np.any(pad):
            # if we gave extra padding, crop back to the original shape
            y_sl, x_sl = [slice(d_pad, d - d_pad) for d, d_pad in
                          zip(imdata.shape[0:2], pad)]
            imdata = imdata[y_sl, x_sl]
            if disp_im is not None:
                disp_im = disp_im[y_sl, x_sl]
            dets = dets.translate([-x_sl.start, -y_sl.start])

        # Ignore any box that is cutoff.
        ignore_thresh = 0.4
        h, w = imdata.shape[0:2]
        frame_box = kwimage.Boxes([[0, 0, w, h]], 'xywh')
        isect = dets.boxes.isect_area(frame_box)
        visibility = (isect / dets.boxes.area)[:, 0]
        ignore_flags = (visibility < ignore_thresh).astype(np.float)
        dets.data['weights'] *= (1 - ignore_flags)

        dets = dets.compress(visibility > 0)

        # Apply letterbox resize transform to train and test
        _debug('imresize')
        self.letterbox.target_size = inp_size
        prelb_dims = imdata.shape[0:2]
        imdata = self.letterbox.augment_image(imdata)
        postlb_dims = imdata.shape[0:2]
        if disp_im is not None:
            # note: the letterbox augment doesn't handle floats wello
            # use the kwimage.imresize instead
            # disp_im = self.letterbox.augment_image(disp_im)
            disp_im = kwimage.imresize(
                disp_im, dsize=self.letterbox.target_size,
                letterbox=True).clip(0, 1)
        if len(dets):
            _debug('warp')
            dets = dets.warp(self.letterbox,
                             input_dims=prelb_dims,
                             output_dims=postlb_dims)

        # Remove any boxes that are no longer visible or out of bounds
        flags = (dets.boxes.area > 0).ravel()
        dets = dets.compress(flags)

        chw01 = torch.FloatTensor(imdata.transpose(2, 0, 1) / 255.0)
        cxwh = dets.boxes.toformat('cxywh')

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        bg_weight = torch.FloatTensor([1.0])

        label = {
            'cxywh': ItemContainer(torch.FloatTensor(cxwh.data), stack=False),
            'class_idxs': ItemContainer(torch.LongTensor(dets.class_idxs), stack=False),
            'weight': ItemContainer(torch.FloatTensor(dets.weights), stack=False),

            'indices': ItemContainer(index, stack=False),
            'orig_sizes': ItemContainer(orig_size, stack=False),
            'bg_weights': ItemContainer(bg_weight, stack=False),
        }
        _debug('label = {!r}'.format(label))

        if 'segmentations' in dets.data and self.use_segmentation:
            # Convert segmentations to masks
            has_mask_list = []
            class_mask_list = []
            h, w = chw01.shape[1:]
            for sseg in dets.data['segmentations']:
                if sseg is not None:
                    mask = sseg.to_mask(dims=chw01.shape[1:])
                    c_mask = mask.to_c_mask().data
                    mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)
                    class_mask_list.append(mask_tensor[None, :])
                    has_mask_list.append(1)
                else:
                    bad_mask = torch.empty((h, w), dtype=torch.uint8)
                    class_mask_list.append(bad_mask)
                    has_mask_list.append(-1)

            has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
            if len(class_mask_list) == 0:
                class_masks = torch.empty((0, h, w), dtype=torch.uint8)
            else:
                class_masks = torch.cat(class_mask_list, dim=0)
            label['class_masks'] = ItemContainer(class_masks, stack=False)
            label['has_mask'] = ItemContainer(has_mask, stack=False)

        compoments = {
            'rgb': chw01,
        }
        _debug('compoments = {!r}'.format(compoments))
        _debug('disp_im = {!r}'.format(disp_im))
        if disp_im is not None:
            disp_im = kwarray.atleast_nd(disp_im, 3)
            compoments['disparity'] = torch.FloatTensor(
                disp_im.transpose(2, 0, 1))

        inputs = {
            k: ItemContainer(v, stack=True)
            for k, v in self.channels.encode(compoments).items()
        }
        _debug('inputs = {!r}'.format(inputs))

        item = {
            'inputs': inputs,
            'label': label,
            'tr': ItemContainer(sample['tr'], stack=False),
        }
        _debug('item = {!r}'.format(item))
        return item

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False, drop_last=False, multiscale=False,
                    balance=False, xpu=None):
        """
        CommandLine:
            xdoctest -m /home/joncrall/code/bioharn/bioharn/detect_dataset.py DetectFitDataset.make_loader

        Example:
            >>> from bioharn.detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo('shapes32')
            >>> self.augmenter = None
            >>> loader = self.make_loader(batch_size=4, shuffle=True, balance='tfidf')
            >>> loader.batch_sampler.index_to_prob
            >>> loader = self.make_loader(batch_size=1, shuffle=True)
            >>> # training batches should have multiple shapes
            >>> shapes = set()
            >>> for raw_batch in ub.ProgIter(iter(loader), total=len(loader)):
            >>>     inputs = raw_batch['inputs']['rgb']
            >>>     # test to see multiscale works
            >>>     shapes.add(inputs.data[0].shape[-1])
            >>>     if len(shapes) > 1:
            >>>         break
        """
        # dataset = self
        import torch.utils.data.sampler as torch_sampler
        assert len(self) > 0, 'must have some data'
        if shuffle:
            sampler = torch_sampler.RandomSampler(self)
        else:
            sampler = torch_sampler.SequentialSampler(self)

        if balance == 'tfidf':
            if not shuffle:
                raise AssertionError('for now you must shuffle when you balance')
            if balance != 'tfidf':
                raise AssertionError('for now balance must be tfidf')
            import xdev
            xdev.embed()

            # label_freq = ub.map_vals(len, self.sampler.dset.index.cid_to_aids)
            anns = self.sampler.dset.anns
            cats = self.sampler.dset.cats

            label_to_weight = None
            if self.classes_of_interest:
                # Only give sampling weight to categories we care about
                label_to_weight = {cat['name']: 0 for cat in cats.values()}
                for cname in self.classes_of_interest:
                    label_to_weight[cname] = 1

            index_to_labels = [
                np.array([cats[anns[aid]['category_id']]['name'] for aid in aids], dtype=np.str)
                for gid, slices, aids in self.chosen_regions
            ]

            batch_sampler = nh.data.batch_samplers.GroupedBalancedBatchSampler(
                index_to_labels, batch_size=batch_size, num_batches='auto',
                shuffle=shuffle, label_to_weight=label_to_weight, rng=None
            )
            print('balanced batch_sampler = {!r}'.format(batch_sampler))
        elif multiscale:
            batch_sampler = MultiScaleBatchSampler2(
                sampler, batch_size=batch_size, drop_last=drop_last,
                factor=32, scales=[-9, 1])
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last)

        if ub.WIN32:
            # Hack for win32 because of pickle loading issues with local vars
            worker_init_fn = None
        else:
            def worker_init_fn(worker_id):
                # Make loaders more random
                kwarray.seed_global(np.random.get_state()[1][0] + worker_id)
                if self.augmenter:
                    rng = kwarray.ensure_rng(None)
                    reseed_(self.augmenter, rng)

        # torch.utils.data.sampler.WeightedRandomSampler

        if xpu is None:
            num_devices = 1
        else:
            num_devices = len(xpu.devices)

        collate_fn = partial(container_collate, num_devices=num_devices)
        # collate_fn = nh.data.collate.padded_collate

        loader = torch.utils.data.DataLoader(
            self, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return loader


def reseed_(auger, rng):
    if hasattr(auger, 'seed_'):
        return auger.seed_(rng)
    else:
        return auger.reseed(rng)


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
    def __init__(self, mode='simple', gravity=0, rng=None):
        import imgaug as ia
        from imgaug import augmenters as iaa
        self.rng = kwarray.ensure_rng(rng)

        self.mode = mode

        print('gravity = {!r}'.format(gravity))
        self._intensity = iaa.Sequential([])
        self._geometric = iaa.Sequential([])
        self._disp_intensity = iaa.Sequential([])

        if mode == 'simple':
            self._geometric = iaa.Sequential([
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.CropAndPad(px=(0, 4)),
            ])
            self._intensity = iaa.Sequential([])
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
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.CropAndPad(px=(-3, 3)),
            ])
            self._intensity = iaa.Sequential([])
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
                iaa.Flipud(p=.5 * (1 - gravity)),
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
            ])
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
                iaa.Flipud(p=.5 * (1 - gravity)),
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
            self._disp_intensity = iaa.Sequential([
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
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
                >>> print('P(we use 6 augmentors) = {:.4f}'.format(dist.cdf(x=6) - dist.cdf(x=5)))
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
                iaa.Flipud(p=.5 * (1 - gravity)),
            ] + ([iaa.Rot90(k=[0, 1, 2, 3])] * int(1 - gravity))  +
                [
                    iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),
                 ],
            )
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
            self._disp_intensity = iaa.Sequential([
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ], random_order=True)
        else:
            raise KeyError(mode)

        self._augers = ub.odict([
            ('intensity', self._intensity),
            ('geometric', self._geometric),
            ('disp_intensity', self._disp_intensity),
        ])
        self.mode = mode
        self.seed_(self.rng)

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

        params = ub.map_vals(imgaug_json_id, self._augers)
        return params

    def seed_(self, rng):
        for auger in self._augers.values():
            if auger is not None:
                reseed_(auger, rng)

    def augment_data(self, imdata, dets, disp_im=None):
        """
        Ignore:
            self = DetectionAugmentor(mode='heavy')
            s = 128
            rng = kwarray.ensure_rng(0)
            dets = kwimage.Detections.random(segmentations=True, rng=rng).scale(s)

            imdata = (rng.rand(s, s, 3) * 255).astype(np.uint8)
            disp_im = rng.rand(s, s).astype(np.float32).clip(0, 1)

            import kwplot
            kwplot.imshow(imdata, fnum=1, pnum=(2, 2, 1), doclf=True)
            dets.draw()
            kwplot.imshow(disp_im, fnum=1, pnum=(2, 2, 2))
            dets.draw()

            imdata1, dets1, disp_im1 = self.augment_data(imdata, dets, disp_im)

            kwplot.imshow(imdata1, fnum=1, pnum=(2, 2, 3))
            dets1.draw()
            kwplot.imshow(disp_im1, fnum=1, pnum=(2, 2, 4))
            dets1.draw()
        """

        _debug('to det')
        rgb_im_aug_det = self._intensity.to_deterministic()
        geom_aug_det = self._geometric.to_deterministic()
        disp_im_aug_det = self._augers['disp_intensity'].to_deterministic()

        input_dims = imdata.shape[0:2]
        _debug('aug gdo')
        _debug('imdata.dtype = {!r}'.format(imdata.dtype))
        _debug('imdata.shape = {!r}'.format(imdata.shape))
        imdata = geom_aug_det.augment_image(imdata)
        _debug('aug rgb')
        imdata = rgb_im_aug_det.augment_image(imdata)

        _debug('disp_im = {!r}'.format(disp_im))
        if disp_im is not None:
            # _debug(kwarray.stats_dict(disp_im))
            disp_im = kwimage.ensure_uint255(disp_im)
            disp_im = disp_im_aug_det.augment_image(disp_im)
            disp_im = geom_aug_det.augment_image(disp_im)
            disp_im = disp_im_aug_det.augment_image(disp_im)
            disp_im = kwimage.ensure_float01(disp_im)
            # _debug(kwarray.stats_dict(disp_im))

        output_dims = imdata.shape[0:2]

        if len(dets):
            _debug('aug dets')
            dets = dets.warp(geom_aug_det, input_dims=input_dims,
                             output_dims=output_dims)

        return imdata, dets, disp_im
