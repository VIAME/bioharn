"""
"""
import ubelt as ub
import torch.utils.data as torch_data
import netharn as nh
import numpy as np
import torch
import six
import scriptconfig as scfg
import kwimage


class DetectPredictConfig(scfg.Config):
    default = {

        'deployed': None,
        'batch_size': 4,
        'xpu': 'auto',

        'window_dims': scfg.Value('full', help='size of a sliding window'),  # (512, 512),
        'input_dims': scfg.Value((512, 512), help='The size of the inputs to the network'),

        'workers': 0,

        'overlap': 0.0,

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.001,

        'verbose': 3,
    }


class DetectPredictor(object):
    """
    A detector API for bioharn trained models

    Example:
        >>> path_or_image = kwimage.imread('/home/joncrall/data/noaa/2015_Habcam_photos/201503.20150522.131445618.413800.png')[:, :1360]
        >>> full_rgb = path_or_image
        >>> config = dict(
        >>>     deployed='/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip',
        >>>     window_dims=(512, 512),
        >>>     input_dims=(256, 256),
        >>> )
        >>> self = DetectPredictor(config)
        >>> final = self.predict(full_rgb)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(full_rgb, doclf=True)
        >>> final2 = final.compress(final.scores > .0)
        >>> final2.draw()
    """
    def __init__(self, config):
        self.config = DetectPredictConfig(config)
        self.model = None
        self.xpu = None

    def info(self, text):
        if self.config['verbose']:
            print(text)

    def _ensure_model(self):
        if self.model is None:
            xpu = nh.XPU.coerce(self.config['xpu'])
            deployed = nh.export.DeployedModel.coerce(self.config['deployed'])
            model = deployed.load_model()
            if xpu != nh.XPU.from_data(model):
                self.info('Mount {} on {}'.format(deployed, xpu))
                model = xpu.mount(model)
            model.train(False)
            self.model = model
            self.xpu = xpu

    def _rectify_image(self, path_or_image):
        if isinstance(path_or_image, six.string_types):
            self.info('Reading {!r}'.format(path_or_image))
            full_rgb = kwimage.imread(path_or_image, space='rgb')
        else:
            full_rgb = path_or_image
        return full_rgb

    def predict(self, path_or_image):
        """
        Predict on a single large image using a sliding window_dims

        Args:
            path_or_image (PathLike | ndarray): An 8-bit RGB numpy image or a
                path to the image.

        Returns:
            kwimage.Detections: a wrapper around predicted boxes, scores,
                and class indices. See the `.data` attribute for more info.
        """
        self.info('Begin detection prediction')

        # Ensure model is in prediction mode and disable gradients for speed
        self._ensure_model()
        self.model.eval()

        # The model must have a coder
        self._raw_model = self.xpu.raw(self.model)
        self._coder = self._raw_model.coder

        full_rgb = self._rectify_image(path_or_image)
        self.info('Detect objects in image (shape={})'.format(full_rgb.shape))

        full_rgb, pad_offset_rc, window_dims = self._prepare_image(full_rgb)
        pad_offset_xy = torch.FloatTensor(np.ascontiguousarray(pad_offset_rc[::-1]))

        slider_dataset = self._make_dataset(full_rgb, window_dims)

        # Its typically faster to use num_workers=0 here because the full image
        # is already in memory. We only need to slice and cast to float32.
        slider_loader = torch.utils.data.DataLoader(
            slider_dataset, shuffle=False, num_workers=self.config['workers'],
            batch_size=self.config['batch_size'])

        # TODO:
        # mmdetection models need to modify self._raw_model.detector.test_cfg
        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           desc='predict', enabled=self.config['verbose'] > 1)
        accum_dets = []
        with torch.set_grad_enabled(False):
            for raw_batch in prog:
                batch = {
                    'im': self.xpu.move(raw_batch['im']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                    'pad_offset_xy': pad_offset_xy,
                }
                for dets in self._predict_batch(batch):
                    accum_dets.append(dets)

        # Stitch predicted detections together
        self.info('Accumulate detections')
        all_dets = kwimage.Detections.concatenate(accum_dets)

        # Perform final round of NMS on the stiched boxes
        self.info('Finalize detections')
        keep = all_dets.non_max_supression(
            thresh=self.config['nms_thresh'],
            daq={'diameter': all_dets.boxes.width.max()},
        )

        final_dets = all_dets.take(keep)
        self.info('Finished prediction')
        return final_dets

    def _prepare_image(self, full_rgb):
        full_dims = tuple(full_rgb.shape[0:2])
        if self.config['window_dims'] == 'full':
            window_dims = full_dims
        else:
            window_dims = self.config['window_dims']

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

    def _make_dataset(self, full_rgb, window_dims):
        full_dims = tuple(full_rgb.shape[0:2])

        # Break large images into chunks to fit on the GPU
        slider = nh.util.SlidingWindow(full_dims, window=window_dims,
                                       overlap=self.config['overlap'],
                                       keepbound=True, allow_overshoot=True)

        input_dims = self.config['input_dims']
        if input_dims == 'full' or input_dims == window_dims:
            input_dims = None

        slider_dataset = SliderDataset(full_rgb, slider, input_dims)
        return slider_dataset

    def _predict_batch(self, batch):
        """
        Runs the torch network on a single batch and postprocesses the outputs

        Yields:
            kwimage.Detections
        """
        chips = batch['im']
        tf_chip_to_full = batch['tf_chip_to_full']
        pad_offset_xy = batch['pad_offset_xy']

        scale_xy = tf_chip_to_full['scale_xy']
        shift_xy = tf_chip_to_full['shift_xy']

        shift_xy_ = shift_xy - pad_offset_xy[None, :]

        # All GPU work happens in this line
        if True:
            # HACK FOR MMDET MODELS
            from bioharn.models.mm_models import _batch_to_mm_inputs
            mm_inputs = _batch_to_mm_inputs(batch)
            imgs = mm_inputs.pop('imgs')
            img_metas = mm_inputs.pop('img_metas')
            hack_imgs = [g[None, :] for g in imgs]
            # For whaver reason we cant run more than one test image at the
            # same time.
            batch_results = []
            outputs = {}
            for one_img, one_meta in zip(hack_imgs, img_metas):
                result = self.model.module.detector.forward(
                    [one_img], [one_meta], return_loss=False)
                batch_results.append(result)
            outputs['batch_results'] = batch_results
        else:
            outputs = self.model.forward(chips, return_loss=False)

        # Postprocess GPU outputs
        batch_dets = self._coder.decode_batch(outputs)
        for idx, det in enumerate(batch_dets):
            item_scale_xy = scale_xy[idx].numpy()
            item_shift_xy = shift_xy_[idx].numpy()
            det = det.numpy()
            det = det.scale(item_scale_xy)
            det = det.translate(item_shift_xy)
            # Fix type issue
            det.data['class_idxs'] = det.data['class_idxs'].astype(np.int)
            yield det


class SliderDataset(torch_data.Dataset):
    """
    Wraps a SlidingWindow in a torch dataset for fast data loading

    This maps image slices into an indexable set for the torch dataloader.

    Calling __getitem__ will result in a dictionary containing a chip for a
    particular window and that chip's offset in the original image.
    """

    def __init__(self, full_image, slider, input_dims):
        self.full_image = full_image
        self.slider = slider
        self.input_dims = input_dims
        self.window_dims = self.slider.window

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
        if self.input_dims is not None:
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
        tensor_chip = torch.FloatTensor(np.ascontiguousarray(chip_chw)) / 255.0
        offset_xy = torch.LongTensor([slice_[1].start, slice_[0].start])

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
        return {
            'im': tensor_chip,
            'tf_chip_to_full': tf_chip_to_full,
        }
