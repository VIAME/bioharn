"""
This example code trains a baseline object detection algorithm given mscoco
inputs.
"""
import netharn as nh
import numpy as np
import os
import torch
import ubelt as ub
import kwarray
import kwimage
# import numpy as np
# import torch
# import netharn as nh
# import ubelt as ub
import scriptconfig as scfg
# from os.path import join
from netharn.models.yolo2 import yolo2


class DetectFitConfig(scfg.Config):
    default = {
        # Personal Preference
        'nice': scfg.Value(
            'untitled',
            help=('a human readable tag for your experiment (we also keep a '
                  'failsafe computer readable tag in case you update hyperparams, '
                  'but forget to update this flag)')),

        # System Options
        'workdir': scfg.Path('~/work/bioharn', help='path where this script can dump stuff'),
        'workers': scfg.Value(0, help='number of DataLoader processes'),
        'xpu': scfg.Value('argv', help='a CUDA device or a CPU'),

        # Data (the hardest part of machine learning)
        'datasets': scfg.Value('special:shapes1024', help='special dataset key'),
        'train_dataset': scfg.Value(None, help='override train with a custom coco dataset'),
        'vali_dataset': scfg.Value(None, help='override vali with a custom coco dataset'),
        'test_dataset': scfg.Value(None, help='override test with a custom coco dataset'),

        'sampler_backend': scfg.Value('auto', help='backend for ndsampler'),

        # Dataset options
        'multiscale': False,
        # 'visible_thresh': scfg.Value(0.5, help='percentage of a box that must be visible to be included in truth'),
        'input_dims': scfg.Value('window', help='size of input to the system '),
        'window_dims': scfg.Value((512, 512), help='size of window to place over the dataset'),
        'window_overlap': scfg.Value(0.0, help='amount of overlap in the sliding windows'),
        'normalize_inputs': scfg.Value(False, help='if True, precompute training mean and std for data whitening'),

        # 'augment': scfg.Value('simple', help='key indicating augmentation strategy', choices=['medium', 'simple']),
        'augment': scfg.Value('medium', help='key indicating augmentation strategy', choices=['medium', 'low', 'simple', 'complex', None]),

        'use_disparity': False,

        'ovthresh': 0.5,

        # High level options
        'arch': scfg.Value(
            'yolo2', help='network toplogy',
            # choices=['yolo2']
        ),

        'optim': scfg.Value('sgd', help='torch optimizer',
                            choices=['sgd', 'adam', 'adamw']),
        'batch_size': scfg.Value(4, help='number of images that run through the network at a time'),
        'bstep': scfg.Value(8, help='num batches before stepping'),
        'lr': scfg.Value(1e-3, help='learning rate'),  # 1e-4,
        'decay': scfg.Value(1e-4, help='weight decay'),

        'schedule': scfg.Value('ReduceLROnPlateau', help='learning rate / momentum scheduler'),
        'max_epoch': scfg.Value(200, help='Maximum number of epochs'),
        'patience': scfg.Value(10, help='Maximum number of bad epochs on validation before stopping'),
        'min_lr': scfg.Value(1e-9, help='minimum learning rate before termination'),

        # Initialization
        'init': scfg.Value('imagenet', help='initialization strategy'),

        'pretrained': scfg.Path(help='path to a netharn deploy file'),

        # Loss Terms
        'focus': scfg.Value(0.0, help='focus for Focal Loss'),

        # Hacked Dynamics
        'warmup_iters': 800,
        'warmup_ratio': 0.1,
        'grad_norm_max': 35,
        'grad_norm_type': 2,

        # preference
        'num_draw': scfg.Value(4, help='Number of initial batchs to draw per epoch'),
        'draw_interval': scfg.Value(1, help='Minutes to wait between drawing'),
        'draw_per_batch': scfg.Value(8, help='Number of items to draw within each batch'),

        'collapse_classes': scfg.Value(False, help='force one-class detector'),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['datasets'] == 'special:voc':
            self['train_dataset'] = ub.expandpath('~/data/VOC/voc-trainval.mscoco.json')
            self['vali_dataset'] = ub.expandpath('~/data/VOC/voc-test-2007.mscoco.json')
        elif self['datasets'] == 'special:habcam':
            self['train_dataset'] = ub.expandpath('~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json')
            self['vali_dataset'] = ub.expandpath('~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')

        key = self.get('pretrained', None) or self.get('init', None)
        if key == 'imagenet':
            self['pretrained'] = yolo2.initial_imagenet_weights()
        elif key == 'lightnet':
            self['pretrained'] = yolo2.demo_voc_weights()

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class DetectHarn(nh.FitHarn):
    def __init__(harn, **kw):
        super(DetectHarn, harn).__init__(**kw)
        # Dictionary of detection metrics
        harn.dmets = {}  # Dict[str, nh.metrics.DetectionMetrics]
        harn.chosen_indices = {}

    def after_initialize(harn):
        # hack the coder into the criterion
        if harn.criterion is not None:
            harn.criterion.coder = harn.raw_model.coder

        # Prepare structures we will use to measure and quantify quality
        for tag, voc_dset in harn.datasets.items():
            dmet = nh.metrics.DetectionMetrics()
            dmet._pred_aidbase = getattr(dmet, '_pred_aidbase', 1)
            dmet._true_aidbase = getattr(dmet, '_true_aidbase', 1)
            harn.dmets[tag] = dmet

    def before_epochs(harn):
        # Seed the global random state before each epoch
        kwarray.seed_global(473282 + np.random.get_state()[1][0] + harn.epoch,
                            offset=57904)

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        if 0:
            batch = harn.xpu.move(raw_batch)
            # Fix batch shape
            bsize = batch['im'].shape[0]
            batch['label']['cxywh'] = batch['label']['cxywh'].view(bsize, -1, 4)
            batch['label']['class_idxs'] = batch['label']['class_idxs'].view(bsize, -1)
            batch['label']['weight'] = batch['label']['weight'].view(bsize, -1)
        else:
            batch = raw_batch
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        Example:
            >>> # DISABLE_DOCTSET
            >>> #harn = setup_harn(bsize=2, datasets='special:habcam', arch='cascade', init='noop', xpu=None, use_disparity=True, workers=0, normalize_inputs=False)
            >>> harn = setup_harn(bsize=2, datasets='special:shapes5',
            >>>                   arch='cascade', init='noop', xpu=(0,1),
            >>>                   workers=0, batch_size=3, normalize_inputs=False)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(2, 'vali')
            >>> outputs, loss = harn.run_batch(batch)

        """
        # Compute how many images have been seen before
        if getattr(harn.raw_model, '__BUILTIN_CRITERION__', False):

            bx = harn.bxs[harn.current_tag]
            harn.raw_model.detector.test_cfg['score_thr'] = 0.0

            if not getattr(harn, '_draw_timer', None):
                harn._draw_timer = ub.Timer().tic()
            # need to hack do draw here, because we need to call
            # mmdet forward in a special way

            harn._hack_do_draw = (harn.batch_index <= harn.script_config['num_draw'])
            harn._hack_do_draw |= (harn._draw_timer.toc() > 60 * harn.script_config['draw_interval'])

            return_result = harn._hack_do_draw

            if harn.script_config['use_disparity']:
                batch = batch.copy()
                # hack in 4th channel
                orig_im = batch['im']
                if len(batch['disparity'].shape) == 3:
                    disparity = batch['disparity'].unsqueeze(1)
                else:
                    disparity = batch['disparity']
                batch['im'] = torch.cat([orig_im, disparity], dim=1)

            batch = batch.copy()
            batch.pop('tr')

            if False:

                from bioharn._hacked_distributed import _report_data_shape
                _report_data_shape(batch)

                from bioharn.models import mm_models
                mm_inputs = mm_models._batch_to_mm_inputs(batch)
                _report_data_shape(mm_inputs)

                _report_data_shape(batch)

                self = harn.model
                _inputs, _kwargs = self.scatter([mm_inputs], dict(return_loss=True, return_result=True), self.device_ids)
                _report_data_shape(_inputs)
                replicas = self.replicate(self.module, self.device_ids[:len(_inputs)])

                if 0:
                    _single_out = self.module(_inputs[0][0], **_kwargs[0])
                    _report_data_shape(_single_out)

                _outputs = self.parallel_apply(replicas, _inputs, _kwargs)
                _gathered = self.gather(_outputs, self.output_device)
                _report_data_shape(_gathered)

                _report_data_shape(_outputs)

                self.module(*_inputs[0], **_kwargs[0])
                batch = _inputs[0][0]

                _report_data_shape(_inputs)

                _inputs, _kwargs = self.scatter(batch, dict(return_loss=True, return_result=True), self.device_ids)
                _report_data_shape(_inputs)

                _inputs, _kwargs = self.scatter(mm_inputs, dict(return_loss=True, return_result=True), self.device_ids)

                outputs = harn.model.forward(mm_inputs, return_loss=True,
                                             return_result=return_result)
                _report_data_shape(outputs)

            outputs = harn.model.forward(batch, return_loss=True,
                                         return_result=return_result)

            # Hack away the DataContainer in the DataSerial case
            from bioharn._hacked_distributed import DataContainer
            if 'batch_results' in outputs:
                if isinstance(outputs['batch_results'], DataContainer):
                    outputs['batch_results'] = outputs['batch_results'].data
            # Criterion was computed in the forward pass
            loss_parts = {k: v.sum() for k, v in outputs['loss_parts'].items()}

        else:

            assert False, 'out of date'
            bsize = harn.loaders['train'].batch_sampler.batch_size
            nitems = (len(harn.datasets['train']) // bsize) * bsize
            bx = harn.bxs['train']
            n_seen = (bx * bsize) + (nitems * harn.epoch)

            inputs = batch['im']
            target = batch['label']
            outputs = harn.model(inputs)
            loss_parts = harn.criterion(outputs, target, seen=n_seen)
        return outputs, loss_parts

    def on_batch(harn, batch, outputs, losses):
        """
        custom callback

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harn(bsize=2, datasets='special:habcam', arch='retinanet', init='noop')
            >>> harn.initialize()

            >>> weights_fpath = '/home/joncrall/work/bioharn/fit/nice/bioharn-det-v8-test-retinanet/torch_snapshots/_epoch_00000021.pt'

            >>> initializer = nh.initializers.Pretrained(weights_fpath)
            >>> init_info = initializer(harn.raw_model)

            >>> batch = harn._demo_batch(1, 'train')
            >>> outputs, losses = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, losses)
            >>> # xdoc: +REQUIRES(--show)
            >>> batch_dets = harn.raw_model.coder.decode_batch(outputs)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.01)
            >>> kwplot.imshow(stacked)
            >>> kwplot.show_if_requested()
        """
        bx = harn.bxs[harn.current_tag]
        try:
            if harn._hack_do_draw:
                detections = harn.raw_model.coder.decode_batch(outputs)
                harn._draw_timer.tic()
                stacked = harn.draw_batch(batch, outputs, detections, thresh=0.0)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag, 'batch'))
                dump_fname = 'pred_bx{:04d}_epoch{:08d}.png'.format(bx, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                harn.debug('dump viz fpath = {}'.format(fpath))
                kwimage.imwrite(fpath, stacked)
        except Exception as ex:
            harn.error('\n\n\n')
            harn.error('ERROR: FAILED TO POSTPROCESS OUTPUTS')
            harn.error('DETAILS: {!r}'.format(ex))
            raise

        if bx % 10 == 0:
            # If multiscale shuffle the input dims
            pass

        metrics_dict = ub.odict()
        return metrics_dict

    def draw_batch(harn, batch, outputs, batch_dets, idx=None, thresh=None,
                   num_extra=3):
        """
        Returns:
            np.ndarray: numpy image

        Example:
            >>> # DISABLE_DOCTSET
            >>> from bioharn.detect_fit import *  # NOQA
            >>> #harn = setup_harn(bsize=1, datasets='special:voc', pretrained='lightnet')
            >>> #harn = setup_harn(bsize=1, datasets='special:habcam', arch='cascade', pretrained='/home/joncrall/work/bioharn/fit/nice/bioharn-det-v14-cascade/deploy_MM_CascadeRCNN_iawztlag_032_ETMZBH.zip', normalize_inputs=False, use_disparity=True)
            >>> #harn = setup_harn(bsize=1, datasets='special:shapes8', arch='cascade', xpu=[1])
            >>> harn = setup_harn(bsize=1, datasets='special:shapes8', arch='cascade', xpu=[1, 0])
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')

            >>> outputs, loss = harn.run_batch(batch)
            >>> batch_dets = harn.raw_model.coder.decode_batch(outputs)

            >>> idx = None
            >>> thresh = None
            >>> num_extra = 3

            >>> stacked = harn.draw_batch(batch, outputs, batch_dets)

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(stacked)
            >>> kwplot.show_if_requested()
        """

        # hack for data container
        im_batch = batch['im'].data
        im_batch = torch.cat(im_batch, dim=0)

        labels = {
            k: v.data for k, v in batch['label'].items()
        }
        labels = nh.XPU('cpu').move(labels)

        # TODO: FIX YOLO SO SCALE IS NOT NEEDED

        if harn.script_config['use_disparity'] and 'disparity' in batch:
            batch_disparity = kwarray.ArrayAPI.numpy(batch['disparity'])
        else:
            batch_disparity = None

        classes = harn.datasets['train'].sampler.classes

        if idx is None:
            bsize = len(im_batch)
            # Get a varied sample of the batch
            # (the idea is ensure that we show things on the non-dominat gpu)
            num_want = harn.script_config['draw_per_batch']
            num_want = min(num_want, bsize)
            # This will never produce duplicates (difference between
            # consecutive numbers will always be > 1 there fore they will
            # always round to a different number)
            idxs = np.linspace(bsize - 1, 0, num_want).round().astype(np.int).tolist()
            idxs = sorted(idxs)
            # assert len(set(idxs)) == len(idxs)
            # idxs = idxs[0:4]
        else:
            idxs = [idx]

        imgs = []
        for idx in idxs:
            chw01 = im_batch[idx]

            if 1:
                class_idxs = list(ub.flatten(labels['class_idxs']))
                cxywh = list(ub.flatten(labels['cxywh']))
                weights = list(ub.flatten(labels['weight']))
                # Convert true batch item to detections object
                true_dets = kwimage.Detections(
                    boxes=kwimage.Boxes(cxywh[idx], 'cxywh'),
                    class_idxs=class_idxs[idx],
                    weights=weights[idx],
                    classes=classes,
                )
                if 'class_masks' in labels:
                    # Add in truth segmentation masks
                    try:
                        masks = list(ub.flatten(labels['class_masks']))
                        item_masks = masks[idx]
                        ssegs = []
                        for mask in item_masks:
                            ssegs.append(kwimage.Mask(mask.numpy(), 'c_mask'))
                        true_dets.data['segmentations'] = kwimage.MaskList(ssegs)
                    except Exception as ex:
                        harn.warn('issue building sseg viz due to {!r}'.format(ex))
            else:
                # Convert true batch item to detections object
                _true_cidxs = labels['class_idxs'][idx].view(-1)
                flags = _true_cidxs > -1
                true_dets = kwimage.Detections(
                    boxes=kwimage.Boxes(labels['cxywh'][idx][flags], 'cxywh'),
                    class_idxs=_true_cidxs[flags],
                    weights=labels['weight'][idx][flags],
                    classes=classes,
                )
                if 'has_mask' in labels:
                    # Add in truth segmentation masks
                    try:
                        mask_flags = labels['has_mask'][idx][flags] > 0
                        item_masks = labels['class_masks'][idx][flags]
                        ssegs = []
                        for mask, flag in zip(item_masks, mask_flags):
                            if flag:
                                ssegs.append(kwimage.Mask(mask.numpy(), 'c_mask'))
                            else:
                                ssegs.append(None)
                        true_dets.data['segmentations'] = kwimage.MaskList(ssegs)
                    except Exception as ex:
                        harn.warn('issue building sseg viz due to {!r}'.format(ex))
            true_dets = true_dets.numpy()

            # Read out the predicted detections
            pred_dets = batch_dets[idx]
            pred_dets = pred_dets.numpy()
            if thresh is not None:
                pred_dets = pred_dets.compress(pred_dets.scores > thresh)

            # only show so many predictions
            num_max = len(true_dets) + num_extra
            sortx = pred_dets.argsort(reverse=True)
            pred_dets = pred_dets.take(sortx[0:num_max])

            hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
            if batch_disparity is None:
                canvas = hwc01.copy()
            else:
                # Show disparity in the canvas
                disparity = batch_disparity[idx].transpose(1, 2, 0)
                canvas = kwimage.stack_images([hwc01, disparity], axis=1)

            canvas = kwimage.ensure_uint255(canvas)
            canvas = true_dets.draw_on(canvas, color='green')
            canvas = pred_dets.draw_on(canvas, color='blue')

            # canvas = cv2.resize(canvas, (300, 300))
            imgs.append(canvas)

        stacked = imgs[0] if len(imgs) == 1 else kwimage.stack_images_grid(imgs)
        return stacked


def setup_harn(cmdline=True, **kw):
    """
    Ignore:
        >>> from bioharn.detect_fit import *  # NOQA
        >>> cmdline = False
        >>> kw = {
        >>>     'train_dataset': '~/data/VOC/voc-trainval.mscoco.json',
        >>>     'vali_dataset': '~/data/VOC/voc-test-2007.mscoco.json',
        >>> }
        >>> harn = setup_harn(**kw)
    """
    import ndsampler
    from ndsampler import coerce_data
    config = DetectFitConfig(default=kw, cmdline=cmdline)

    nh.configure_hacks(config)  # fix opencv bugs
    ub.ensuredir(config['workdir'])

    # Load ndsampler.CocoDataset objects from info in the config
    subsets = coerce_data.coerce_datasets(config)

    # HACK: ENSURE BACKGROUND IS CLASS IDX 0 for mmdet
    classes = subsets['train'].object_categories()
    for k, subset in subsets.items():
        # TODO: better handling
        special_catnames = ['negative', 'ignore', 'test']
        for k in special_catnames:
            try:
                subset.remove_categories([k], keep_annots=False, verbose=1)
            except KeyError:
                pass

    if config['collapse_classes']:
        print('Collapsing all category labels')
        import six
        if isinstance(config['collapse_classes'], six.string_types):
            hacklbl = config['collapse_classes']
        else:
            hacklbl = 'object'
        print('Hacking all labels to ' + hacklbl)
        for tag, subset in subsets.items():
            mapper = {c['name']: hacklbl for c in subset.cats.values()
                      if c['name'] != 'background'}
            subset.rename_categories(mapper)

    classes = subsets['train'].object_categories()
    print('classes = {!r}'.format(classes))
    if 'background' not in classes:
        for k, subset in subsets.items():
            subset.add_category('background', id=0)

    classes = subsets['train'].object_categories()
    print('classes = {!r}'.format(classes))

    samplers = {}
    for tag, subset in subsets.items():
        print('subset = {!r}'.format(subset))
        sampler = ndsampler.CocoSampler(subset, workdir=config['workdir'],
                                        backend=config['sampler_backend'])

        sampler.frames.prepare(workers=config['workers'])

        samplers[tag] = sampler

    from bioharn.detect_dataset import DetectFitDataset
    torch_datasets = {
        tag: DetectFitDataset(
            sampler,
            use_segmentation='mask' in config['arch'].lower(),
            input_dims=config['input_dims'],
            window_dims=config['window_dims'],
            window_overlap=config['window_overlap'] if (tag == 'train') else 0.0,
            augment=config['augment'] if (tag == 'train') else False,
        )
        for tag, sampler in samplers.items()
    }

    from bioharn._hacked_distributed import Hacked_XPU
    xpu = Hacked_XPU.coerce(config['xpu'])
    print('xpu = {!r}'.format(xpu))

    print('make loaders')
    loaders_ = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            multiscale=(tag == 'train') and config['multiscale'],
            pin_memory=True, xpu=xpu)
        for tag, dset in torch_datasets.items()
    }

    if config['normalize_inputs']:
        # Get stats on the dataset (todo: turn off augmentation for this)
        _dset = torch_datasets['train']
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(1000, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)
        cacher = ub.Cacher('dset_mean', cfgstr=_dset.input_id + 'v2')
        input_stats = cacher.tryload()
        if input_stats is None:
            # Use parallel workers to load data faster
            from bioharn._hacked_distributed import container_collate
            # collate_fn = container_collate
            from functools import partial
            collate_fn = partial(container_collate, samples_per_gpu=1)

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True, batch_size=config['batch_size'])
            # Track moving average
            running = nh.util.RunningStats()
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                try:
                    for im in batch['im'].data:
                        running.update(im.numpy())
                except ValueError:  # final batch broadcast error
                    pass
            input_stats = {
                'std': running.simple(axis=None)['mean'].round(3),
                'mean': running.simple(axis=None)['std'].round(3),
            }
            cacher.save(input_stats)
    else:
        input_stats = None
    print('input_stats = {!r}'.format(input_stats))

    initializer_ = nh.Initializer.coerce(config, leftover='kaiming_normal')
    print('initializer_ = {!r}'.format(initializer_))

    arch = config['arch']
    classes = samplers['train'].classes

    if config['use_disparity']:
        in_channels = 4
    else:
        in_channels = 3

    criterion_ = None
    if arch == 'retinanet':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
            in_channels=in_channels,
            input_stats=input_stats,
        )
        model = mm_models.MM_RetinaNet(**initkw)
        model._initkw = initkw
    elif arch == 'maskrcnn':
        from xviewharn.models import mm_models
        initkw = dict(
            classes=classes,
            in_channels=in_channels,
            input_stats=input_stats,
        )
        model = mm_models.MM_MaskRCNN(**initkw)
        model._initkw = initkw
    elif arch == 'cascade':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
            in_channels=in_channels,
            input_stats=input_stats,
        )
        model = mm_models.MM_CascadeRCNN(**initkw)
        model._initkw = initkw
    elif arch == 'yolo2':
        if False:
            dset = samplers['train'].dset
            print('dset = {!r}'.format(dset))
            # anchors = yolo2.find_anchors(dset)
            # anchors = yolo2.find_anchors2(dset.sampler)

        # HACKED IN:
        anchors = np.array([[1.0, 1.0],
                            [0.1, 0.1 ],
                            [0.01, 0.01],
                            [0.07781961, 0.10329947],
                            [0.03830135, 0.05086466]])

        # anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
        #                     (5.05587, 8.09892), (9.47112, 4.84053),
        #                     (11.2364, 10.0071)])

        model_ = (yolo2.Yolo2, {
            'classes': classes,
            'anchors': anchors,
            'input_stats': input_stats,
            'conf_thresh': 0.001,
            'nms_thresh': 0.5,
        })
        model = model_[0](**model_[1])
        model._initkw = model_[1]

        criterion_ = (yolo2.YoloLoss, {
            'coder': model.coder,
            'seen': 0,
            'coord_scale'    : 1.0,
            'noobject_scale' : 1.0,
            'object_scale'   : 5.0,
            'class_scale'    : 1.0,
            'thresh'         : 0.6,  # iou_thresh
            # 'seen_thresh': 12800,
        })
    else:
        raise KeyError(arch)

    scheduler_ = nh.Scheduler.coerce(config)
    print('scheduler_ = {!r}'.format(scheduler_))

    optimizer_ = nh.Optimizer.coerce(config)
    print('optimizer_ = {!r}'.format(optimizer_))

    dynamics_ = nh.Dynamics.coerce(config)
    print('dynamics_ = {!r}'.format(dynamics_))

    import sys

    hyper = nh.HyperParams(**{
        'nice': config['nice'],
        'workdir': config['workdir'],

        'datasets': torch_datasets,
        'loaders': loaders_,

        'xpu': xpu,

        'model': model,

        'criterion': criterion_,

        'initializer': initializer_,

        'optimizer': optimizer_,
        'dynamics': dynamics_,

        'scheduler': scheduler_,

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            # 'maximize': ['mAP'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'min_lr': config['min_lr'],
            'smoothing': .6,
        }),

        'other': {
            # Other params are not used internally, so you are free to set any
            # extra params specific to your algorithm, and still have them
            # logged in the hyperparam structure. For YOLO this is `ovthresh`.
            'batch_size': config['batch_size'],
            'nice': config['nice'],
            'ovthresh': config['ovthresh'],  # used in mAP computation
        },
        'extra': {
            'config': ub.repr2(config.asdict()),
            'argv': sys.argv,
        }
    })
    print('hyper = {!r}'.format(hyper))
    print('make harn')
    harn = DetectHarn(hyper=hyper)
    harn.config.update({
        'num_keep': 2,
        'keep_freq': 30,
        'export_modules': ['bioharn'],  # TODO
        'prog_backend': 'progiter',  # alternative: 'tqdm'
        'keyboard_debug': True,
    })
    harn.intervals.update({
        'log_iter_train': 50,
    })
    harn.script_config = config

    print('harn = {!r}'.format(harn))
    print('samplers = {!r}'.format(samplers))
    return harn


def fit():
    harn = setup_harn()
    harn.initialize()
    with harn.xpu:
        harn.run()


if __name__ == '__main__':
    """

    CommandLine:
        # Uses defaults with demo data
        python ~/code/netharn/examples/object_detection.py

        python ~/code/netharn/examples/grab_voc.py

        python ~/code/netharn/examples/object_detection.py --datasets=special:voc

        python -m bioharn.detect_fit \
            --nice=demo_shapes2 \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=3 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=demo_shapes \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=1 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-test-yolo \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --pretrained=imagenet \
            --schedule=step90 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=16 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-test-yolo-v5 \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --pretrained=/home/joncrall/work/bioharn/fit/nice/bioharn-test-yolo/torch_snapshots/_epoch_00000011.pt \
            --schedule=ReduceLROnPlateau \
            --optim=adamw --lr=3e-4 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=16 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v12-test-retinanet \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau \
            --arch=retinanet \
            --augment=medium \
            --init=noop \
            --optim=sgd --lr=1e-3 \
            --input_dims=512,512 \
            --workers=6 --xpu=1 --batch_size=12 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v9-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --arch=cascade \
            --init=noop \
            --optim=sgd --lr=1e-2 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v11-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v9-test-cascade/zjolejwz/deploy_MM_CascadeRCNN_zjolejwz_010_LUAKQJ.zip \
            --augment=medium \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v11-test-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p1-c2 \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip \
            --augment=medium \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=1024,1024 \
            --workers=4 --xpu=1 --batch_size=1 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v12-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p1-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-4 \
            --input_dims=512,512 \
            --window_dims=512,512 \
            --workers=4 --xpu=1 --batch_size=4 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v13-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-4 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v14-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v16-cascade \
            --train_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json \
            --vali_dataset=~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --use_disparity=True \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v16-cascade/ozeaiwmm/explit_checkpoints/_epoch_00000000_2019-12-12T185656+5.pt \
            --normalize_inputs=False \
            --workers=4 --xpu=1 --batch_size=8 --bstep=4

        python -m bioharn.detect_fit \
            --nice=bioharn-det-v17-cascade-mc-disp \
            --train_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --vali_dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --use_disparity=True \
            --optim=adamw --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=False \
            --workers=4 --xpu=0 --batch_size=16 --bstep=1


        $HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Habcam_2015_g027250_a00111034_c0016_v3_vali.mscoco.json

        # --pretrained='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth' \


        python -m bioharn.detect_fit \
            --nice=demo_shapes_gpu10 \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=1,0 --batch_size=4 --bstep=1

        python -m bioharn.detect_fit \
            --nice=demo_shapes_gpu1 \
            --datasets=special:shapes256 \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.2 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=0 --xpu=1 --batch_size=4 --bstep=1


        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v2 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v2.mscoco.json \
            --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v2.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --pretrained=/home/joncrall/work/sealions/fit/runs/detect-singleclass-cascade-v2/nkdvpjss/explit_checkpoints/_epoch_00000000_2020-01-16T182834+5.pt \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=1 --batch_size=8 --bstep=1


        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v3 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v3.mscoco.json \
            --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v3.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=True \
            --workers=4 --xpu=0 --batch_size=16 --bstep=1

        python -m bioharn.detect_fit \
            --nice=detect-singleclass-cascade-v4 \
            --workdir=$HOME/work/sealions \
            --train_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_train_v3.mscoco.json \
            --vali_dataset=/home/joncrall/data/US_ALASKA_MML_SEALION/sealions_vali_v3.mscoco.json \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --optim=sgd --lr=1e-2 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --multiscale=True \
            --normalize_inputs=True \
            --min_lr=1e-6 \
            --workers=4 --xpu=1,0 --batch_size=8 --bstep=1


    """
    import warnings
    import traceback
    _orig_formatwarning = warnings.formatwarning
    def _monkeypatch_formatwarning_tb(*args, **kwargs):
        s = _orig_formatwarning(*args, **kwargs)
        tb = traceback.format_stack()
        s += ''.join(tb[:-1])
        return s
    warnings.formatwarning = _monkeypatch_formatwarning_tb
    fit()
