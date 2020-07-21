"""

Notes:

    https://data.kitware.com/#collection/58b747ec8d777f0aef5d0f6a

    source $HOME/internal/secrets

    girder-client --api-url https://data.kitware.com/api/v1 list 58b747ec8d777f0aef5d0f6a

    girder-client --api-url https://data.kitware.com/api/v1 list 58c49f668d777f0aef5d7960

    girder-client --api-url https://data.kitware.com/api/v1 list 5c423a5f8d777f072b0ba58f

    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3181eaf2e2eed3505827c

    girder-client --api-url https://data.kitware.com/api/v1 list 5aac22638d777f068578d53c --columns=id,type,name

    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3eb8eaf2e2eed3508d604

    girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604


"""
from os.path import exists
from os.path import isfile
from os.path import basename
from os.path import join
import ubelt as ub
import torch.utils.data as torch_data
import netharn as nh
import numpy as np
import torch
import six
import scriptconfig as scfg
import kwimage
import warnings
from netharn.data.channel_spec import ChannelSpec
from netharn.data.data_containers import ContainerXPU

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class DetectPredictConfig(scfg.Config):
    default = {

        'deployed': None,
        'batch_size': 4,
        'xpu': 'auto',

        'window_dims': scfg.Value('native', help='size of a sliding window'),  # (512, 512),
        'input_dims': scfg.Value('window', help='The size of the inputs to the network'),

        'workers': 0,

        'window_overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        'channels': scfg.Value(
            'native',
            help='list of channels needed by the model. '
            'Typically this can be inferred from the model'),

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.1,

        'verbose': 1,
    }


class DetectPredictor(object):
    """
    A detector API for bioharn trained models

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> import ubelt as ub
        >>> from bioharn.detect_predict import *  # NOQA
        >>> deployed_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5dd3eb8eaf2e2eed3508d604/download',
        >>>     fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3_mm2x.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='63b7c3981b3446b079c1d83541a5666c496')
        >>> image_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5dcf0d1faf2e2eed35fad5d1/download',
        >>>     fname='scallop.jpg', appname='viame', hasher='sha512',
        >>>     hash_prefix='3bd290526c76453bec7')
        >>> path_or_image = full_rgb = kwimage.imread(image_fpath)
        >>> config = dict(
        >>>     deployed=deployed_fpath,
        >>> )
        >>> predictor = DetectPredictor(config)
        >>> final = predictor.predict(full_rgb)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(full_rgb, doclf=True)
        >>> final2 = final.compress(final.scores > .0)
        >>> final2.draw()
    """
    def __init__(predictor, config):
        predictor.config = DetectPredictConfig(config)
        predictor.model = None
        predictor.xpu = None
        predictor.coder = None

        # This is populated if we need to modify behavior for backwards
        # compatibility.
        predictor._compat_hack = None

    def info(predictor, text):
        if predictor.config['verbose']:
            print(text)

    @classmethod
    def _infer_native(cls, config):
        """
        Preforms whatever hacks are necessary to introspect the correct
        values of special "native" config options depending on the model.
        """
        # Set default fallback values
        native_defaults = {
            'input_dims': (512, 512),
            'window_dims': 'full',
            'channels': 'hack_old_method'
        }
        @ub.memoize
        def _native_config():
            deployed = nh.export.DeployedModel.coerce(config['deployed'])
            # New models should have relevant params here, which is slightly
            # less hacky than using the eval.
            native_config = deployed.train_info()['other']
            common = set(native_defaults) & set(native_config)
            if len(common) != len(native_defaults):
                # Fallback on the hacky string encoding of the configs
                cfgstr = deployed.train_info()['extra']['config']
                # import ast
                # parsed = ast.literal_eval(cfgstr)
                parsed = eval(cfgstr, {'inf': float('inf')})
                native_config.update(parsed)
            return native_config

        native = {}
        native_config = _native_config()
        for key in list(native_defaults.keys()):
            if config[key] == 'native':
                try:
                    native[key] = native_config[key]
                except Exception:
                    warnings.warn((
                        'WARNING: Unable to determine native {} from model. '
                        'Defaulting to {}! Please ensure this is OK.').format(
                            key, native_defaults[key]
                    ))
                    native[key] = native_defaults[key]
            else:
                native[key] = config[key]

        if native['channels'] == 'hack_old_method':
            # Hueristic to determine what channels an older model wants.  This
            # should not be necessary for newer models which directly encode
            # this.
            native['channels'] = 'rgb'
            if native_config.get('use_disparity', False):
                native['channels'] += '|disparity'

        return native

    def _ensure_model(predictor):
        # Just make sure the model is in memory (it might not be on the XPU yet)
        if predictor.model is None:
            # TODO: we want to use ContainerXPU when dealing with an mmdet
            # model but we probably want regular XPU otherwise. Not sure what
            # the best way to do this is yet.
            # NOTE: ContainerXPU might actually work with non-container returns
            # need to test this.
            xpu = ContainerXPU.coerce(predictor.config['xpu'])
            deployed = nh.export.DeployedModel.coerce(predictor.config['deployed'])
            model = deployed.load_model()
            model.train(False)
            predictor.xpu = xpu
            predictor.model = model
            # The model must have a coder
            predictor.raw_model = predictor.xpu.raw(predictor.model)
            predictor.coder = predictor.raw_model.coder

    def _ensure_mounted_model(predictor):
        predictor._ensure_model()
        model = predictor.model
        _ensured_mount = getattr(model, '_ensured_mount', False)
        if not _ensured_mount:
            xpu = predictor.xpu
            if xpu != ContainerXPU.from_data(model):
                predictor.info('Mount model on {}'.format(xpu))
                model = xpu.mount(model)
                predictor.model = model
                # The model must have a coder
                predictor.raw_model = predictor.xpu.raw(predictor.model)
                predictor.coder = predictor.raw_model.coder
            # hack to prevent multiple XPU data checks
            predictor.model._ensured_mount = True

    def _rectify_image(predictor, path_or_image):
        if isinstance(path_or_image, six.string_types):
            predictor.info('Reading {!r}'.format(path_or_image))
            full_rgb = kwimage.imread(path_or_image, space='rgb')
        else:
            full_rgb = path_or_image
        return full_rgb

    def predict(predictor, path_or_image):
        """
        Predict on a single large image using a sliding window_dims

        Args:
            path_or_image (PathLike | ndarray): An 8-bit RGB numpy image or a
                path to the image.

        Returns:
            kwimage.Detections: a wrapper around predicted boxes, scores,
                and class indices. See the `.data` attribute for more info.

        SeeAlso:
            :method:`predict_sampler` - this can be a faster alternative to
            predict, but it requires that your dataset is formatted as a
            sampler.

        TODO:
            - [X] Handle auxillary inputs
        """
        predictor.info('Begin detection prediction')

        # Ensure model is in prediction mode and disable gradients for speed
        predictor._ensure_mounted_model()

        full_rgb = predictor._rectify_image(path_or_image)
        predictor.info('Detect objects in image (shape={})'.format(full_rgb.shape))

        full_rgb, pad_offset_rc, window_dims = predictor._prepare_image(full_rgb)
        pad_offset_xy = torch.FloatTensor(np.ascontiguousarray(pad_offset_rc[::-1], dtype=np.float32))

        slider_dataset = predictor._make_dataset(full_rgb, window_dims)

        # Its typically faster to use num_workers=0 here because the full image
        # is already in memory. We only need to slice and cast to float32.
        slider_loader = torch.utils.data.DataLoader(
            slider_dataset, shuffle=False, num_workers=predictor.config['workers'],
            batch_size=predictor.config['batch_size'])

        # TODO:
        # mmdetection models need to modify predictor._raw_model.detector.test_cfg
        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           desc='predict', enabled=predictor.config['verbose'] > 1)
        accum_dets = []
        with torch.set_grad_enabled(False):
            for raw_batch in prog:
                batch = {
                    'inputs': predictor.xpu.move(raw_batch['inputs']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                    'pad_offset_xy': pad_offset_xy,
                }
                results = predictor._predict_batch(batch)
                for dets in results:
                    accum_dets.append(dets)

        # Stitch predicted detections together
        predictor.info('Accumulate detections')
        all_dets = kwimage.Detections.concatenate(accum_dets)

        # Perform final round of NMS on the stiched boxes
        predictor.info('Finalize detections')

        if len(all_dets) > 0:
            keep = all_dets.non_max_supression(
                thresh=predictor.config['nms_thresh'],
                daq={'diameter': all_dets.boxes.width.max()},
            )
            final_dets = all_dets.take(keep)
        else:
            final_dets = all_dets

        predictor.info('Finished prediction')
        return final_dets

    @profile
    def predict_sampler(predictor, sampler, gids=None):
        """
        Predict on all images in a dataset wrapped in a ndsampler.CocoSampler

        Args:
            sampler (ndsampler.CocoDataset): dset wrapped in a sampler
            gids (List[int], default=None): if specified, then only predict
                on these image ids.

        Yields:
            Tuple[int, Detections] : image_id, detection pairs

        SeeAlso:
            :method:`predict` - this is a simpler alternative to
            predict_sampler. It only requires that you pass your data in as an
            image.
        """
        predictor._ensure_mounted_model()

        native = predictor._infer_native(predictor.config)
        input_dims = native['input_dims']
        window_dims = native['window_dims']
        channels = native['channels']

        torch_dset = WindowedSamplerDataset(sampler, window_dims=window_dims,
                                            input_dims=input_dims,
                                            channels=channels, gids=gids)
        if len(torch_dset) == 0:
            return
        slider_loader = torch.utils.data.DataLoader(
            torch_dset, shuffle=False, num_workers=predictor.config['workers'],
            batch_size=predictor.config['batch_size'])

        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           chunksize=predictor.config['batch_size'],
                           desc='predict', enabled=predictor.config['verbose'] > 1)

        xpu = predictor.xpu

        # raw_batch = ub.peek(prog)
        with torch.set_grad_enabled(False):

            # ----
            buffer_gids = []
            buffer_dets = []

            for raw_batch in prog:
                batch = {
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                }
                if 'inputs' in raw_batch:
                    batch['inputs'] = xpu.move(raw_batch['inputs'])
                else:
                    raise NotImplementedError

                batch_gids = raw_batch['gid'].view(-1).numpy()
                batch_dets = list(predictor._predict_batch(batch))

                # Determine if we have finished an image (assuming images are
                # passed in sequentially in order)
                can_yield = (
                    np.any(np.diff(batch_gids)) or
                    (len(buffer_gids) and buffer_gids[-1] != batch_gids[0])
                )

                buffer_gids.extend(batch_gids)
                buffer_dets.extend(batch_dets)

                if can_yield:
                    ready_idx = max(np.where(np.diff(buffer_gids))[0]) + 1
                    ready_gids = buffer_gids[:ready_idx]
                    ready_dets = buffer_dets[:ready_idx]

                    #
                    buffer_gids = buffer_gids[ready_idx:]
                    buffer_dets = buffer_dets[ready_idx:]
                    for gid, dets in predictor._finalize_dets(ready_dets, ready_gids):
                        yield gid, dets
            # ----

            # Finalize anything that remains
            ready_gids = buffer_gids
            ready_dets = buffer_dets
            for gid, dets in predictor._finalize_dets(ready_dets, ready_gids):
                yield gid, dets

    @profile
    def _finalize_dets(predictor, ready_dets, ready_gids):
        """ Helper for predict_sampler """
        gid_to_ready_dets = ub.group_items(ready_dets, ready_gids)
        for gid, dets_list in gid_to_ready_dets.items():
            if len(dets_list) == 0:
                dets = kwimage.Detections.concatenate([])
            elif len(dets_list) == 1:
                dets = dets_list[0]
            elif len(dets_list) > 1:
                dets = kwimage.Detections.concatenate(dets_list)
                keep = dets.non_max_supression(
                    thresh=predictor.config['nms_thresh'],
                )
                dets = dets.take(keep)
            yield (gid, dets)

    @profile
    def _prepare_image(predictor, full_rgb):
        full_dims = tuple(full_rgb.shape[0:2])

        if predictor.config['window_dims'] == 'native':
            native = predictor._infer_native(predictor.config)
            window_dims = native['window_dims']
        else:
            window_dims = predictor.config['window_dims']

        if window_dims == 'full':
            window_dims = full_dims

        # Pad small images to be at least the minimum window_dims size
        dims_delta = np.array(full_dims) - np.array(window_dims)
        if np.any(dims_delta < 0):
            padding = np.maximum(-dims_delta, 0)
            lower_pad = padding // 2
            upper_pad = padding - lower_pad
            pad_width = list(zip(lower_pad, upper_pad))
            ndims_all = len(full_rgb.shape)
            ndims_spti = len(padding)
            if ndims_all > ndims_spti:
                # Handle channels
                extra = [(0, 0)] * (ndims_all - ndims_spti)
                pad_width = pad_width + extra
            full_rgb = np.pad(full_rgb, pad_width, mode='constant',
                              constant_values=127)
            full_dims = tuple(full_rgb.shape[0:2])
            pad_offset_rc = lower_pad[0:2]
        else:
            pad_offset_rc = np.array([0, 0])

        return full_rgb, pad_offset_rc, window_dims

    def _make_dataset(predictor, full_rgb, window_dims):
        """ helper for predict """
        full_dims = tuple(full_rgb.shape[0:2])

        native = predictor._infer_native(predictor.config)

        assert native['channels'] == 'rgb', 'cant handle non-rgb yet'

        # Break large images into chunks to fit on the GPU
        slider = nh.util.SlidingWindow(full_dims, window=window_dims,
                                       overlap=predictor.config['window_overlap'],
                                       keepbound=True, allow_overshoot=True)

        input_dims = predictor.config['input_dims']
        if input_dims == 'full' or input_dims == window_dims:
            input_dims = None

        slider_dataset = SingleImageDataset(full_rgb, slider, input_dims)
        return slider_dataset

    @profile
    def _predict_batch(predictor, batch):
        """
        Runs the torch network on a single batch and postprocesses the outputs

        Yields:
            kwimage.Detections
        """
        tf_chip_to_full = batch['tf_chip_to_full']

        scale_xy = tf_chip_to_full['scale_xy']
        shift_xy = tf_chip_to_full['shift_xy']

        if 'pad_offset_xy' in batch:
            pad_offset_xy = batch['pad_offset_xy']
            shift_xy_ = shift_xy - pad_offset_xy[None, :]
        else:
            shift_xy_ = shift_xy

        import xdev
        with xdev.embed_on_exception_context:
            outputs = None

            if predictor._compat_hack is None:
                # All GPU work happens in this line
                if hasattr(predictor.model.module, 'detector'):
                    # HACK FOR MMDET MODELS
                    # TODO: hack for old detectors that require "im" inputs
                    try:
                        outputs = predictor.model.forward(
                            batch, return_loss=False, return_result=True)
                    except KeyError:
                        predictor._compat_hack = 'old_mmdet_im_model'
                    except NotImplementedError:
                        predictor._compat_hack = 'fixup_mm_inputs'
                    if predictor._compat_hack:
                        warnings.warn(
                            'Normal mm-detection input failed. '
                            'Attempting to find backwards compatible solution')
                else:
                    assert len(batch['inputs']) == 1
                    try:
                        im = ub.peek(batch['inputs'].values())
                        outputs = predictor.model.forward(batch['inputs'])
                    except Exception:
                        try:
                            # Hack for old efficientdet models with bad input checking
                            from netharn.data.data_containers import BatchContainer
                            if isinstance(batch['inputs']['rgb'], torch.Tensor):
                                batch['inputs']['rgb'] = BatchContainer([batch['inputs']['rgb']])
                            outputs = predictor.model.forward(batch)
                            predictor._compat_hack = 'efficientdet_hack'
                        except Exception:
                            raise Exception('Unsure about expected model inputs')
                    # raise NotImplementedError('only works on mmdet models')

            if outputs is None:
                # HACKS FOR BACKWARDS COMPATIBILITY
                if predictor._compat_hack == 'old_mmdet_im_model':
                    batch['im'] = batch.pop('inputs')['rgb']
                    outputs = predictor.model.forward(batch, return_loss=False)
                if predictor._compat_hack == 'fixup_mm_inputs':
                    from bioharn.models.mm_models import _batch_to_mm_inputs
                    mm_inputs = _batch_to_mm_inputs(batch)
                    outputs = predictor.model.forward(mm_inputs, return_loss=False)
                if predictor._compat_hack == 'efficientdet_hack':
                    from netharn.data.data_containers import BatchContainer
                    batch['inputs']['rgb'] = BatchContainer([batch['inputs']['rgb']])
                    outputs = predictor.model.forward(batch)

            # Postprocess GPU outputs
            if 'Container' in str(type(outputs)):
                # HACK
                outputs = outputs.data

            batch_dets = predictor.coder.decode_batch(outputs)

        for idx, det in enumerate(batch_dets):
            item_scale_xy = scale_xy[idx].numpy()
            item_shift_xy = shift_xy_[idx].numpy()
            det = det.numpy()
            det = det.compress(det.scores > predictor.config['conf_thresh'])

            if True and len(det) and np.all(det.boxes.width <= 1):
                # HACK FOR YOLO
                # TODO: decode should return detections in batch input space
                assert len(batch['inputs']) == 1
                im = ub.peek(batch['inputs'].values())
                inp_size = np.array(im.shape[-2:][::-1])
                det = det.scale(inp_size)

            det = det.scale(item_scale_xy)
            det = det.translate(item_shift_xy)
            # Fix type issue
            if 'class_idxs' in det.data:
                det.data['class_idxs'] = det.data['class_idxs'].astype(np.int)
            yield det


class SingleImageDataset(torch_data.Dataset):
    """
    Wraps a SlidingWindow in a torch dataset for fast data loading

    This maps image slices into an indexable set for the torch dataloader.

    Calling __getitem__ will result in a dictionary containing a chip for a
    particular window and that chip's offset in the original image.
    """

    def __init__(self, full_image, slider, input_dims, channels='rgb'):
        self.full_image = full_image
        self.slider = slider
        self.input_dims = input_dims
        self.window_dims = self.slider.window
        self.channels = ChannelSpec.coerce(channels)

    def __len__(self):
        return self.slider.n_total

    def __getitem__(self, index):
        # Lookup the window location
        slider = self.slider
        basis_idx = np.unravel_index(index, slider.basis_shape)
        slice_ = tuple([bdim[i] for bdim, i in zip(slider.basis_slices, basis_idx)])

        # Sample the image patch
        chip_hwc = self.full_image[slice_]

        # Resize the image patch if necessary
        if self.input_dims is not None and self.input_dims != 'window':
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            window_size = self.window_dims[::-1]
            input_size = self.input_dims[::-1]
            shift, scale, embed_size = letterbox._letterbox_transform(window_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)
        else:
            shift = [0, 0]
            scale = [1, 1]
        scale_xy = torch.FloatTensor(scale)

        # Assume 8-bit image inputs
        chip_chw = np.transpose(chip_hwc, (2, 0, 1))
        tensor_rgb = torch.FloatTensor(np.ascontiguousarray(chip_chw)) / 255.0
        offset_xy = torch.FloatTensor([slice_[1].start, slice_[0].start])

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        if False:
            tf_mat = np.array([
                [tf_full_to_chip['scale_xy'][0], 0, tf_full_to_chip['shift_xy'][0]],
                [0, tf_full_to_chip['scale_xy'][1], tf_full_to_chip['shift_xy'][1]],
                [0, 0, 1],
            ])
            np.linalg.inv(tf_mat)

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }

        assert self.channels.spec == 'rgb'

        return {
            'inputs': {'rgb': tensor_rgb},
            'tf_chip_to_full': tf_chip_to_full,
        }


class WindowedSamplerDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Dataset that breaks up images into windows and optionally resizes those
    windows.

    TODO: Use as a base class for training detectors. This should ideally be
    used as an input to another dataset which handles augmentation.

    Args:
        window_dims: size of a sliding window
        input_dims: size to resize sampled windows to
        window_overlap: amount of overlap between windows
        gids : images to sample from, if None use all of them
    """

    def __init__(self, sampler, window_dims='full', input_dims='native',
                 window_overlap=0.0, gids=None, channels='rgb'):
        self.sampler = sampler
        self.input_dims = input_dims
        self.window_dims = window_dims
        self.window_overlap = window_overlap
        self.channels = ChannelSpec.coerce(channels)
        self.subindex = None
        self.gids = gids
        self._build_sliders()

    @classmethod
    def demo(WindowedSamplerDataset, key='habcam', **kwargs):
        import ndsampler
        if key == 'habcam':
            dset_fpath = ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')
            workdir = ub.expandpath('~/work/bioharn')
            dset = ndsampler.CocoDataset(dset_fpath)
            sampler = ndsampler.CocoSampler(dset, workdir=workdir, backend=None)
        else:
            sampler = ndsampler.CocoSampler.demo(key)
        self = WindowedSamplerDataset(sampler, **kwargs)
        return self

    def _build_sliders(self):
        """
        Use the ndsampler.Sampler and sliders to build a flat index that can
        reach every subregion of every image in the training set.

        Ignore:
            window_dims = (512, 512)
            input_dims = 'native'
            window_overlap = 0
        """
        import netharn as nh
        window_overlap = self.window_overlap
        window_dims = self.window_dims
        sampler = self.sampler

        gids = self.gids
        if gids is None:
            gids = list(sampler.dset.imgs.keys())

        gid_to_slider = {}
        for gid in gids:
            img = sampler.dset.imgs[gid]
            # if img.get('source', '') == 'habcam_2015_stereo':
            # Hack: todo, cannoncial way to get this effect
            full_dims = [img['height'], img['width']]
            # else:
            #     full_dims = [img['height'], img['width']]

            window_dims_ = full_dims if window_dims == 'full' else window_dims
            slider = nh.util.SlidingWindow(full_dims, window_dims_,
                                           overlap=window_overlap,
                                           keepbound=True,
                                           allow_overshoot=True)
            gid_to_slider[img['id']] = slider

        self.gid_to_slider = gid_to_slider
        self._gids = list(gid_to_slider.keys())
        self._sliders = list(gid_to_slider.values())
        self.subindex = nh.util.FlatIndexer.fromlist(self._sliders)

    def __len__(self):
        return len(self.subindex)

    @profile
    def __getitem__(self, index):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--data)
            >>> self = WindowedSamplerDataset.demo(window_dims=(512, 512))
            >>> index = 0
            >>> item = self[1]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(item['inputs']['rgb'])
        """
        outer, inner = self.subindex.unravel(index)
        gid = self._gids[outer]
        slider = self._sliders[outer]
        slices = slider[inner]

        tr = {'gid': gid, 'slices': slices}

        # for now load sample only returns rgb
        unique_channels = self.channels.unique()

        assert 'rgb' in unique_channels
        sample = self.sampler.load_sample(tr, with_annots=False)
        chip_hwc = kwimage.atleast_3channels(sample['im'])

        chip_dims = tuple(chip_hwc.shape[0:2])

        # Resize the image patch if necessary
        if self.input_dims != 'native' and self.input_dims != 'window':
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            chip_size = np.array(chip_dims[::-1])
            input_size = np.array(self.input_dims[::-1])
            shift, scale, embed_size = letterbox._letterbox_transform(chip_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)
        else:
            letterbox = None
            shift = [0, 0]
            scale = [1, 1]
        scale_xy = torch.FloatTensor(scale)

        # Assume 8-bit image inputs
        chip_chw = np.transpose(chip_hwc, (2, 0, 1))
        tensor_rgb = torch.FloatTensor(np.ascontiguousarray(chip_chw)) / 255.0
        offset_xy = torch.FloatTensor([slices[1].start, slices[0].start])

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        if False:
            tf_mat = np.array([
                [tf_full_to_chip['scale_xy'][0], 0, tf_full_to_chip['shift_xy'][0]],
                [0, tf_full_to_chip['scale_xy'][1], tf_full_to_chip['shift_xy'][1]],
                [0, 0, 1],
            ])
            np.linalg.inv(tf_mat)

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }
        components = {
            'rgb': tensor_rgb,
        }
        item = {
            'gid': torch.LongTensor([gid]),
            'tf_chip_to_full': tf_chip_to_full,
        }

        sampler = self.sampler

        # if img.get('source', '') in ['habcam_2015_stereo', 'habcam_stereo']:
        if 'disparity' in unique_channels:
            from ndsampler.utils import util_gdal
            disp_fpath = sampler.dset.get_auxillary_fpath(gid, 'disparity')
            disp_frame = util_gdal.LazyGDalFrameFile(disp_fpath)
            data_dims = disp_frame.shape[0:2]
            pad = 0
            data_slice, extra_padding, st_dims = sampler._rectify_tr(
                tr, data_dims, window_dims=None, pad=pad)
            # Load the image data
            disp_im = disp_frame[data_slice]
            if extra_padding:
                if disp_im.ndim != len(extra_padding):
                    extra_padding = extra_padding + [(0, 0)]  # Handle channels
                disp_im = np.pad(disp_im, extra_padding, **{'mode': 'constant'})
            if letterbox is not None:
                disp_im = letterbox.augment_image(disp_im)
            if len(disp_im.shape) == 2:
                disp_im = disp_im[None, :, :]
            else:
                disp_im = disp_im.transpose(2, 0, 1)
            components['disparity'] = torch.FloatTensor(disp_im)

        item['inputs'] = self.channels.encode(components)
        return item


################################################################################
# CLI


def _coerce_sampler(config):
    import six
    import ndsampler
    from bioharn import util
    from os.path import isdir

    # Running prediction is much faster if you can build a sampler.
    sampler_backend = config['sampler_backend']

    if isinstance(config['dataset'], six.string_types):
        if config['dataset'].endswith('.json'):
            dataset_fpath = ub.expandpath(config['dataset'])
            coco_dset = ndsampler.CocoDataset(dataset_fpath)
            print('coco hashid = {}'.format(coco_dset._build_hashid()))
        else:
            image_path = ub.expandpath(config['dataset'])
            path_exists = exists(image_path)
            if path_exists and isfile(image_path):
                # Single image case
                coco_dset = ndsampler.CocoDataset()
                coco_dset.add_image(image_path)
            elif path_exists and isdir(image_path):
                # Directory of images case
                IMG_EXTS = [
                    '.bmp', '.pgm', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
                    '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif', '.r0',
                    '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
                ]
                img_globs = ['*' + ext for ext in IMG_EXTS]
                fpaths = list(util.find_files(image_path, img_globs))
                if len(fpaths):
                    coco_dset = ndsampler.CocoDataset.from_image_paths(fpaths)
                else:
                    raise Exception('no images found')
            else:
                # Glob pattern case
                import glob
                fpaths = list(glob.glob(image_path))
                if len(fpaths):
                    coco_dset = ndsampler.CocoDataset.from_image_paths(fpaths)
                else:
                    raise Exception('not an image path')

    elif isinstance(config['dataset'], list):
        # Multiple image case
        gpaths = config['dataset']
        gpaths = [ub.expandpath(g) for g in gpaths]
        coco_dset = ndsampler.CocoDataset.from_image_paths(gpaths)
    else:
        raise TypeError(config['dataset'])

    print('Create sampler')
    workdir = ub.expandpath(config.get('workdir'))
    sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                    backend=sampler_backend)
    return sampler


@profile
def _cached_predict(predictor, sampler, out_dpath='./cached_out', gids=None,
                    draw=False, enable_cache=True, async_buffer=False,
                    verbose=1, draw_truth=True):
    """
    Helper to only do predictions that havent been done yet.

    Note that this currently requires you to ensure that the dest folder is
    unique to this particular dataset.

    Ignore:
        >>> import ndsampler
        >>> config = {}
        >>> config['deployed'] = ub.expandpath('~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000042.pt')
        >>> predictor = DetectPredictor(config)
        >>> predictor._ensure_model()
        >>> out_dpath = './cached_out'
        >>> gids = None
        >>> coco_dset = ndsampler.CocoDataset(ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json'))
        >>> sampler = ndsampler.CocoSampler(coco_dset, workdir=None,
        >>>                                 backend=None)
    """
    import kwarray
    import ndsampler
    from bioharn import util
    import tempfile
    coco_dset = sampler.dset
    # predictor.config['verbose'] = 1

    det_outdir = ub.ensuredir((out_dpath, 'pred'))
    tmp_fpath = tempfile.mktemp()

    if gids is None:
        gids = list(coco_dset.imgs.keys())

    gid_to_pred_fpath = {
        gid: join(det_outdir, 'dets_gid_{:08d}_v2.mscoco.json'.format(gid))
        for gid in gids
    }

    if enable_cache:
        # Figure out what gids have already been computed
        have_gids = [gid for gid, fpath in gid_to_pred_fpath.items() if exists(fpath)]
    else:
        have_gids = []

    print('enable_cache = {!r}'.format(enable_cache))
    print('Found {} / {} existing predictions'.format(len(have_gids), len(gids)))

    gids = ub.oset(gids) - have_gids
    pred_gen = predictor.predict_sampler(sampler, gids=gids)

    if async_buffer:
        desc = 'buffered detect'
        buffered_gen = util.AsyncBufferedGenerator(pred_gen,
                                                   size=coco_dset.n_images)
        gen = buffered_gen
    else:
        desc = 'unbuffered detect'
        gen = pred_gen

    gid_to_pred = {}
    prog = ub.ProgIter(gen, total=len(gids), desc=desc, verbose=verbose)
    for img_idx, (gid, dets) in enumerate(prog):
        gid_to_pred[gid] = dets

        img = coco_dset.imgs[gid]

        # TODO: need to either add the expected img_root to the coco dataset or
        # reroot the file name to be a full path so the predicted dataset can
        # reference the source images if needed.
        single_img_coco = ndsampler.CocoDataset()
        gid = single_img_coco.add_image(**img)

        for cat in dets.classes.to_coco():
            single_img_coco.add_category(**cat)

        # for cat in coco_dset.cats.values():
        #     single_img_coco.add_category(**cat)

        for ann in dets.to_coco():
            ann['image_id'] = gid
            if 'category_name' in ann:
                catname = ann['category_name']
                cid = single_img_coco.ensure_category(catname)
                ann['category_id'] = cid
            single_img_coco.add_annotation(**ann)

        single_pred_fpath = gid_to_pred_fpath[gid]

        # prog.ensure_newline()
        # print('write single_pred_fpath = {!r}'.format(single_pred_fpath))
        # TODO: use safer?
        single_img_coco.dump(tmp_fpath, newlines=True)
        util.atomic_move(tmp_fpath, single_pred_fpath)

        if draw is True or (draw and img_idx < draw):
            draw_outdir = ub.ensuredir((out_dpath, 'draw'))
            img_fpath = coco_dset.get_image_fpath(gid)
            gname = basename(img_fpath)
            viz_fname = ub.augpath(gname, prefix='detect_', ext='.jpg')
            viz_fpath = join(draw_outdir, viz_fname)

            image = kwimage.imread(img_fpath)

            if draw_truth:
                # draw truth if available
                anns = list(ub.take(coco_dset.anns, coco_dset.gid_to_aids[gid]))
                true_dets = kwimage.Detections.from_coco_annots(anns,
                                                                dset=coco_dset)
                true_dets.draw_on(image, alpha=None, color='green')

            flags = dets.scores > .2
            flags[kwarray.argmaxima(dets.scores, num=10)] = True
            top_dets = dets.compress(flags)
            toshow = top_dets.draw_on(image, alpha=None)
            # kwplot.imshow(toshow)
            kwimage.imwrite(viz_fpath, toshow, space='rgb')

    if enable_cache:
        pred_fpaths = [gid_to_pred_fpath[gid] for gid in have_gids]
        cached_dets = _load_dets(pred_fpaths, workers=6)
        assert have_gids == [d.meta['gid'] for d in cached_dets]
        gid_to_cached = ub.dzip(have_gids, cached_dets)
        gid_to_pred.update(gid_to_cached)

    return gid_to_pred, gid_to_pred_fpath


def _load_dets(pred_fpaths, workers=6):
    # Process mode is much faster than thread.
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool(mode='process', max_workers=workers)
    for single_pred_fpath in ub.ProgIter(pred_fpaths, desc='submit load dets jobs'):
        job = jobs.submit(_load_dets_worker, single_pred_fpath)
    dets = []
    for job in ub.ProgIter(jobs.jobs, total=len(jobs), desc='loading cached dets'):
        dets.append(job.result())
    return dets


def _load_dets_worker(single_pred_fpath):
    """
    single_pred_fpath = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/eval/habcam_cfarm_v6_test.mscoc/bioharn-det-mc-cascade-rgbd-v36__epoch_00000018/c=0.2,i=window,n=0.5,window_d=512,512,window_o=0.0/pred/dets_gid_00004070_v2.mscoco.json')
    """
    import kwcoco
    single_img_coco = kwcoco.CocoDataset(single_pred_fpath, autobuild=False)
    if len(single_img_coco.dataset['images']) != 1:
        raise Exception('Expected predictions for a single image only')
    gid = single_img_coco.dataset['images'][0]['id']
    dets = kwimage.Detections.from_coco_annots(single_img_coco.dataset['annotations'],
                                               dset=single_img_coco)
    dets.meta['gid'] = gid
    return dets


class DetectPredictCLIConfig(scfg.Config):
    default = ub.dict_union(
        {
            'dataset': scfg.Value(None, help='coco dataset, path to images or folder of images'),
            'out_dpath': scfg.Path('./out', help='output directory'),
            'draw': scfg.Value(False),
            'sampler_backend': scfg.Value(None),
            'workdir': scfg.Path('~/work/bioharn', help='work directory for sampler if needed'),

            'async_buffer': scfg.Value(False, help="I've seen this increase prediction rate from 2.0Hz to 2.3Hz, but it increases instability, unsure of the reason"),
        },
        DetectPredictConfig.default
    )


def detect_cli(config={}):
    """
    CommandLine:
        python -m bioharn.detect_predict --help

    CommandLine:
        python -m bioharn.detect_predict \
            --dataset=~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_test.mscoco.json \
            --deployed=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip \
            --out_dpath=~/work/bioharn/habcam_test_out \
            --draw=100 \
            --input_dims=512,512 \
            --xpu=0 --batch_size=1


        python -m bioharn.detect_predict \
            --dataset=/data/projects/GOOD/pyrosome-test/US_NW_2017_NWFSC_PYROSOME_TEST \
            --deployed=$HOME/work/bioharn/fit/nice/test-pyrosome/deploy_MM_CascadeRCNN_lqufwadq_031_HNSZYA.zip \
            --out_dpath=$HOME/work/bioharn/predict_pyrosome_test \
            --draw=100 \
            --xpu=auto --batch_size=2

    Ignore:
        >>> config = {}
        >>> config['dataset'] = '~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json'
        >>> config['deployed'] = '/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip'
        >>> config['out_dpath'] = 'out'
    """
    config = DetectPredictCLIConfig(config, cmdline=True)
    print('config = {}'.format(ub.repr2(config.asdict())))

    out_dpath = ub.expandpath(config.get('out_dpath'))
    det_outdir = ub.ensuredir((out_dpath, 'pred'))

    sampler = _coerce_sampler(config)
    print('prepare frames')
    sampler.frames.prepare(workers=config['workers'])

    print('Create predictor')
    pred_config = ub.dict_subset(config, DetectPredictConfig.default)
    if config['verbose'] < 2:
        pred_config['verbose'] = 0
    predictor = DetectPredictor(pred_config)
    print('Ensure model')
    predictor._ensure_model()

    # async_buffer = not ub.argval('--serial') and config['workers'] > 0
    async_buffer = ub.argval('--async-buffer')

    gid_to_pred, gid_to_pred_fpath = _cached_predict(
        predictor, sampler, out_dpath=out_dpath, gids=None,
        draw=config['draw'], enable_cache=True, async_buffer=async_buffer)

    import ndsampler
    coco_dsets = []
    for gid, pred_fpath in gid_to_pred_fpath.items():
        single_img_coco = ndsampler.CocoDataset(pred_fpath)
        coco_dsets.append(single_img_coco)

    pred_dset = ndsampler.CocoDataset.union(*coco_dsets)
    pred_fpath = join(det_outdir, 'detections.mscoco.json')
    print('Dump detections to pred_fpath = {!r}'.format(pred_fpath))
    pred_dset.dump(pred_fpath, newlines=True)


if __name__ == '__main__':
    detect_cli()
