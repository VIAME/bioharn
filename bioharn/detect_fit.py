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
        'normalize_inputs': scfg.Value(True, help='if True, precompute training mean and std for data whitening'),

        # 'augment': scfg.Value('simple', help='key indicating augmentation strategy', choices=['medium', 'simple']),
        'augment': scfg.Value('medium', help='key indicating augmentation strategy', choices=['medium', 'low', 'simple', 'complex', None]),

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

        # Initialization
        'init': scfg.Value('imagenet', help='initialization strategy'),

        'pretrained': scfg.Path(help='path to a netharn deploy file'),

        # Loss Terms
        'focus': scfg.Value(0.0, help='focus for Focal Loss'),

        # Hacked Dynamics
        'warmup_iters': 800,
        'warmup_ratio': 1.0 / 10.0,
        'grad_norm_max': 35,
        'grad_norm_type': 2,
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
        nh.util.seed_global(473282 + np.random.get_state()[1][0] + harn.epoch,
                            offset=57904)

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        batch = harn.xpu.move(raw_batch)
        # Fix batch shape
        bsize = batch['im'].shape[0]
        batch['label']['cxywh'] = batch['label']['cxywh'].view(bsize, -1, 4)
        batch['label']['class_idxs'] = batch['label']['class_idxs'].view(bsize, -1)
        batch['label']['weight'] = batch['label']['weight'].view(bsize, -1)
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harn(bsize=2, datasets='special:habcam', arch='retinanet', init='noop')
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'vali')
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
            harn._hack_do_draw = (harn.batch_index <= 4)
            harn._hack_do_draw |= (harn._draw_timer.toc() > 60 * 1)

            return_result = harn._hack_do_draw
            outputs = harn.model.forward(batch, return_loss=True,
                                         return_result=return_result)

            # Criterion was computed in the forward pass
            loss = outputs['loss_parts']

        else:
            bsize = harn.loaders['train'].batch_sampler.batch_size
            nitems = (len(harn.datasets['train']) // bsize) * bsize
            bx = harn.bxs['train']
            n_seen = (bx * bsize) + (nitems * harn.epoch)

            inputs = batch['im']
            target = batch['label']
            outputs = harn.model(inputs)
            loss = harn.criterion(outputs, target, seen=n_seen)
        return outputs, loss

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
            >>> nh.util.autompl()  # xdoc: +SKIP
            >>> stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.01)
            >>> nh.util.imshow(stacked)
            >>> nh.util.show_if_requested()
        """
        bx = harn.bxs[harn.current_tag]
        try:
            if harn._hack_do_draw:
                detections = harn.raw_model.coder.decode_batch(outputs)
                harn._draw_timer.tic()
                stacked = harn.draw_batch(batch, outputs, detections, thresh=0.0)
                # img = nh.util.render_figure_to_image(fig)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag, 'batch'))
                dump_fname = 'pred_bx{:04d}_epoch{:08d}.png'.format(bx, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                harn.debug('dump viz fpath = {}'.format(fpath))
                nh.util.imwrite(fpath, stacked)
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
                   orig_img=None, num_extra=3):
        """
        Returns:
            np.ndarray: numpy image

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harn(bsize=1, datasets='special:voc', pretrained='lightnet')
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')

            >>> outputs, loss = harn.run_batch(batch)
            >>> batch_dets = harn.raw_model.coder.decode_batch(outputs)

            >>> stacked = harn.draw_batch(batch, outputs, batch_dets)

            >>> # xdoc: +REQUIRES(--show)
            >>> nh.util.autompl()  # xdoc: +SKIP
            >>> nh.util.imshow(stacked)
            >>> nh.util.show_if_requested()
        """
        inputs = batch['im']
        labels = batch['label']
        orig_sizes = labels['orig_sizes']

        classes = harn.datasets['train'].sampler.classes

        if idx is None:
            idxs = range(len(inputs))
        else:
            idxs = [idx]

        idxs = idxs[0:4]

        imgs = []
        for idx in idxs:
            chw01 = inputs[idx]
            pred_dets = batch_dets[idx]
            # pred_dets.meta['classes'] = classes

            true_dets = kwimage.Detections(
                boxes=kwimage.Boxes(labels['cxywh'][idx], 'cxywh'),
                class_idxs=labels['class_idxs'][idx].view(-1),
                weights=labels['weight'][idx],
                classes=classes,
            )

            pred_dets = pred_dets.numpy()
            true_dets = true_dets.numpy()

            true_dets = true_dets.compress(true_dets.class_idxs != -1)

            if thresh is not None:
                pred_dets = pred_dets.compress(pred_dets.scores > thresh)

            # only show so many predictions
            num_max = len(true_dets) + num_extra
            sortx = pred_dets.argsort(reverse=True)
            pred_dets = pred_dets.take(sortx[0:num_max])

            hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
            inp_size = np.array(hwc01.shape[0:2][::-1])

            # TODO: FIX YOLO SO SCALE IS NOT NEEDED
            # true_dets.boxes.scale(inp_size, inplace=True)
            # pred_dets.boxes.scale(inp_size, inplace=True)

            letterbox = harn.datasets[harn.current_tag].letterbox
            orig_size = orig_sizes[idx].cpu().numpy()
            target_size = inp_size
            img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
            img = np.clip(img, 0, 1)
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img is None or orig_img.shape == img.shape

            true_dets.data['boxes'] = letterbox._boxes_letterbox_invert(
                true_dets.boxes, orig_size, target_size)
            pred_dets.data['boxes'] = letterbox._boxes_letterbox_invert(
                pred_dets.boxes, orig_size, target_size)

            # shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)
            # fig = nh.util.figure(doclf=True, fnum=1)
            # nh.util.imshow(img, colorspace='rgb')
            canvas = (img * 255).astype(np.uint8)
            canvas = true_dets.draw_on(canvas, color='green')
            canvas = pred_dets.draw_on(canvas, color='blue')

            # canvas = cv2.resize(canvas, (300, 300))
            imgs.append(canvas)

        stacked = imgs[0] if len(imgs) == 1 else nh.util.stack_images_grid(imgs)
        return stacked


def setup_harn(cmdline=True, **kw):
    """
    Ignore:
        >>> from object_detection import *  # NOQA
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
    if 'background' not in classes:
        for k, subset in subsets.items():
            subset.add_category('background', id=0)

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
            input_dims=config['input_dims'],
            window_dims=config['window_dims'],
            window_overlap=config['window_overlap'] if (tag == 'train') else 0.0,
            augment=config['augment'] if (tag == 'train') else False,
        )
        for tag, sampler in samplers.items()
    }

    print('make loaders')
    loaders_ = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            multiscale=(tag == 'train') and config['multiscale'],
            pin_memory=True)
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
            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=nh.data.collate.padded_collate,
                num_workers=config['workers'],
                shuffle=True, batch_size=config['batch_size'])
            # Track moving average
            running = nh.util.RunningStats()
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                try:
                    running.update(batch['im'].numpy())
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

    criterion_ = None
    if arch == 'retinanet':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
        )
        model = mm_models.MM_RetinaNet(**initkw)
        model._initkw = initkw
    elif arch == 'cascade':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
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
    if True:
        dynamics_['warmup_iters'] = config['warmup_iters']
        dynamics_['warmup_ratio'] = config['warmup_ratio']
        dynamics_['grad_norm_max'] = config['grad_norm_max']
        dynamics_['grad_norm_type'] = config['grad_norm_type']
    print('dynamics_ = {!r}'.format(dynamics_))

    xpu = nh.XPU.coerce(config['xpu'])
    print('xpu = {!r}'.format(xpu))

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
    harn.fit_config = config

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

        # --pretrained='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth' \


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
