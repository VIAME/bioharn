"""
This example code trains a baseline object detection algorithm given mscoco
inputs.
"""
from os.path import join
import netharn as nh
import numpy as np
import os
import torch
import ubelt as ub
import kwarray
import kwimage
import warnings
import scriptconfig as scfg


class DetectFitConfig(scfg.Config):
    default = {
        # Personal Preference
        'nice': scfg.Value(None, help=('deprecated use name instead')),

        'name': scfg.Value(
            'untitled',
            help=('a human readable tag for your experiment (we also keep a '
                  'failsafe computer readable tag in case you update hyperparams, '
                  'but forget to update this flag)')),

        # System Options
        'workdir': scfg.Path('~/work/bioharn', help='path where this script can dump stuff'),
        'workers': scfg.Value(0, help='number of DataLoader processes'),
        'xpu': scfg.Value('argv', help='a CUDA device or a CPU'),

        # Data (the hardest part of machine learning)
        'datasets': scfg.Value(None, help='special dataset key. Mutex with train_dataset, etc..'),
        'train_dataset': scfg.Value(None, help='override train with a custom coco dataset'),
        'vali_dataset': scfg.Value(None, help='override vali with a custom coco dataset'),
        'test_dataset': scfg.Value(None, help='override test with a custom coco dataset'),

        'sampler_backend': scfg.Value('auto', help='backend for ndsampler'),

        # Dataset options
        'multiscale': False,
        # 'visible_thresh': scfg.Value(0.5, help='percentage of a box that must be visible to be included in truth'),
        'input_dims': scfg.Value('window', help=ub.paragraph(
            '''
            After a window is sample, it is resized to this shape (using
            letterboxing to maintain aspect ratio). This is the size of input
            to the system, so if your network needs 224x224 pixel input, then
            this is the place to set it. By default the window size is
            unchanged.
            ''')),
        'window_dims': scfg.Value((512, 512), help=ub.paragraph(
            '''
            Size of window to place over the dataset.
            ''')),
        'window_overlap': scfg.Value(0.0, help=ub.paragraph(
            '''
            Amount of overlap in the sliding windows. This is given as a
            fraction between 0 and 1.
            ''')),
        'normalize_inputs': scfg.Value('imagenet', help=ub.paragraph(
            '''
            Specification for the mean and std for data whitening.

            If True, precompute using 1000 unaugmented training images.
            If an integer, use that many unaugmented training images.
            If 'imagenet' use standard mean/std values (default).
            ''')),
        # 'augment': scfg.Value('simple', help='key indicating augmentation strategy', choices=['medium', 'simple']),
        'augment': scfg.Value('medium', help='key indicating augmentation strategy', choices=['medium', 'low', 'simple', 'complex', None]),
        'gravity': scfg.Value(0.0, help='how often to assume gravity vector for augmentation'),
        'balance': scfg.Value(None),

        'channels': scfg.Value('rgb', type=str, help='special channel code. See ChannelSpec'),

        'ovthresh': 0.5,

        # High level options
        'arch': scfg.Value(
            'yolo2', help='network toplogy',
            # choices=['yolo2']
        ),

        'optim': scfg.Value('sgd', help='torch optimizer, sgd, adam, adamw, etc...'),
        'batch_size': scfg.Value(4, help='number of images that run through the network at a time'),

        'num_batches': scfg.Value('auto', help='Number of batches per epoch (mainly for balanced batch sampling)'),
        'num_vali_batches': scfg.Value('auto', help='number of val batches per epoch'),

        'bstep': scfg.Value(8, help='num batches before stepping'),
        'lr': scfg.Value(1e-3, help='learning rate'),  # 1e-4,
        'decay': scfg.Value(1e-4, help='weight decay'),

        'schedule': scfg.Value('Exponential-g0.98-s1', help='learning rate / momentum scheduler'),
        'max_epoch': scfg.Value(50, help='Maximum number of epochs'),
        'patience': scfg.Value(10, help='Maximum number of bad epochs on validation before stopping'),
        'min_lr': scfg.Value(1e-9, help='minimum learning rate before termination'),

        # Initialization
        'init': scfg.Value('noop', help='initialization strategy'),
        'pretrained': scfg.Path(None, help='path to a netharn deploy file'),

        'backbone_init': scfg.Value('url', help='path to backbone weights for mmdetection initialization'),
        'anchors': scfg.Value('auto', help='how to choose anchor boxes'),

        # Loss Terms
        'focus': scfg.Value(0.0, help='focus for Focal Loss'),

        'seen_thresh': scfg.Value(12800, help='for yolo criterion'),

        # Hacked Dynamics
        'warmup_iters': 800,
        'warmup_ratio': 0.1,
        'grad_norm_max': 35,
        'grad_norm_type': 2,

        # preference
        'num_draw': scfg.Value(4, help='Number of initial batchs to draw per epoch'),
        'draw_interval': scfg.Value(1, help='Minutes to wait between drawing'),
        'draw_per_batch': scfg.Value(8, help='Number of items to draw within each batch'),

        'classes_of_interest': scfg.Value([], help='if specified only these classes are given weight'),

        'collapse_classes': scfg.Value(False, help='force one-class detector'),

        'ensure_background_class': scfg.Value(False, help='ensure a background category exists'),
        'timeout': scfg.Value(float('inf'), help='maximum number of seconds to wait for training'),
        'test_on_finish': True,
        'vali_intervals': 1,
    }

    def normalize(self):
        if self['nice'] is not None:
            import warnings
            warnings.warn('Using "nice" is deprecated use "name" instead', DeprecationWarning)
            self['name'] = self['nice']

        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['datasets'] == 'special:voc':
            from netharn.data.grab_voc import ensure_voc_coco
            paths = ensure_voc_coco()
            self['train_dataset'] = paths['trainval']
            self['vali_dataset'] = paths['test']
        elif self['datasets'] == 'special:habcam':
            self['train_dataset'] = ub.expandpath('~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json')
            self['vali_dataset'] = ub.expandpath('~/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')

        key = self.get('pretrained', None) or self.get('init', None)
        if key == 'imagenet':
            from bioharn.models.yolo2 import yolo2
            self['pretrained'] = yolo2.initial_imagenet_weights()
        elif key == 'lightnet':
            from bioharn.models.yolo2 import yolo2
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
        # kwarray.seed_global(473282 + np.random.get_state()[1][0] + harn.epoch,
        #                     offset=57904)
        if harn.epoch == 0 or 1:
            harn._draw_conv_layers(suffix='_init')

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


        Ignore:
            >>> from bioharn.detect_fit import *  # NOQA
            >>> harn = setup_harn(bsize=2, datasets='special:habcam',
            >>>     arch='cascade', init='noop', xpu=0, channels='rgb|disparity',
            >>>     workers=0, normalize_inputs=False, sampler_backend=None)

            >>> from bioharn.detect_fit import *  # NOQA
            >>> harn = setup_harn(bsize=2, datasets='special:shapes256',
            >>>     arch='efficientdet', init='noop', xpu=0, channels='rgb',
            >>>     workers=0, normalize_inputs=False, sampler_backend=None)

            >>> from bioharn.detect_fit import *  # NOQA
            >>> harn = setup_harn(bsize=2, datasets='special:shapes256',
            >>>     arch='MM_HRNetV2_w18_MaskRCNN', xpu='auto',
            >>>     workers=0, normalize_inputs='imagenet', sampler_backend=None)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(1, 'vali')
            >>> del batch['label']['has_mask']
            >>> del batch['label']['class_masks']
            >>> outputs, loss = harn.run_batch(batch)

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harn(bsize=2, datasets='special:shapes5',
            >>>                   arch='cascade', init='noop', xpu=(0,1),
            >>>                   workers=0, batch_size=3, normalize_inputs=False)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(1, 'vali')
            >>> outputs, loss = harn.run_batch(batch)

        """
        if not getattr(harn, '_draw_timer', None):
            harn._draw_timer = ub.Timer().tic()
        # need to hack do draw here, because we need to call
        # mmdet forward in a special way
        harn._hack_do_draw = (harn.batch_index < harn.script_config['num_draw'])
        harn._hack_do_draw |= ((harn._draw_timer.toc() > 60 * harn.script_config['draw_interval']) and
                               (harn.script_config['draw_interval'] > 0))

        return_result = False
        return_result = harn._hack_do_draw

        if getattr(harn.raw_model, '__BUILTIN_CRITERION__', False):
            # try:
            #     # hack for mmdet
            #     harn.raw_model.detector.test_cfg['score_thr'] = 0.0
            # except AttributeError:
            #     pass

            batch = batch.copy()
            batch.pop('tr', None)
            from netharn.data.data_containers import BatchContainer

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'indexing with dtype')
                warnings.filterwarnings('ignore', 'Default upsampling behavior')
                # warnings.filterwarnings('ignore', 'asked to gather along dimension 0')
                outputs = harn.model.forward(batch, return_loss=True,
                                             return_result=return_result)

            # Hack away the BatchContainer in the DataSerial case
            if 'batch_results' in outputs:
                if isinstance(outputs['batch_results'], BatchContainer):
                    outputs['batch_results'] = outputs['batch_results'].data
            # Criterion was computed in the forward pass
            loss_parts = {k: v.sum() for k, v in outputs['loss_parts'].items()}

        else:
            inputs = batch['inputs']
            # unpack the BatchContainer
            im = {k: v.data[0] for k, v in inputs.items()}

            # Compute how many images have been seen before
            bsize = harn.loaders['train'].batch_sampler.batch_size
            nitems = (len(harn.datasets['train']) // bsize) * bsize
            bx = harn.bxs['train']
            n_seen = (bx * bsize) + (nitems * harn.epoch)

            batch = batch.copy()

            im = harn.xpu.move(im)
            outputs = harn.model(im)

            label = batch['label']
            unwrapped = {k: v.data[0] for k, v in label.items()}
            target = {
                'cxywh': nh.data.collate.padded_collate(unwrapped['cxywh']),
                'class_idxs': nh.data.collate.padded_collate(unwrapped['class_idxs']),
                'weight': nh.data.collate.padded_collate(unwrapped['weight']),
                # 'indices': nh.data.collate.padded_collate(unwrapped['indices']),
                # 'orig_sizes': nh.data.collate.padded_collate(unwrapped['orig_sizes']),
                # 'bg_weights': nh.data.collate.padded_collate(unwrapped['bg_weights']),
            }
            target = harn.xpu.move(target)
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
                batch_dets = harn.raw_model.coder.decode_batch(outputs)
                harn._draw_timer.tic()
                stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.0)
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

    def overfit(harn, batch, interactive=False):
        """
        Ensure that the model can overfit to a single batch.

        CommandLine:
            xdoctest -m /home/joncrall/code/bioharn/bioharn/detect_fit.py DetectHarn.overfit

        Example:
            >>> # DISABLE_DOCTSET
            >>> from bioharn.detect_fit import *  # NOQA
            >>> #harn = setup_harn(bsize=1, datasets='special:voc', pretrained='lightnet')
            >>> harn = setup_harn(
            >>>     name='overfit_test', batch_size=8,
            >>>     # datasets='special:voc',
            >>>     train_dataset='special:vidshapes8',
            >>>     # train_dataset=ub.expandpath('$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_dummy_sseg.mscoco.json'),
            >>>     gravity=1, augment=None,
            >>>     #arch='yolo2', pretrained='lightnet', lr=3e-5, normalize_inputs=False, anchors='lightnet', ensure_background_class=0, seen_thresh=110,
            >>>     arch='MM_HRNetV2_w18_MaskRCNN', init='noop', lr=1e-4, normalize_inputs='imagenet',
            >>>     #arch='retinanet', init='noop', normalize_inputs=True, lr=1e-3,
            >>>     #arch='cascade', init='noop', normalize_inputs=True, lr=1e-3,
            >>>     channels='rgb',
            >>>     sampler_backend=None)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(1, 'train')
            >>> import kwplot
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> harn.overfit(batch, interactive=True)
            >>> # xdoc: +REQUIRES(--show)
        """
        niters = 10000
        if interactive:
            import xdev
            import kwplot
            kwplot.autompl()

            """
            Ignore:

                model = harn.raw_model
                imgs, annotations = model._encode_batch(batch)
                imgs = model.input_norm(imgs)
                x = model.extract_feat(imgs)
                outs = model.bbox_head(x)
                cls_score, bbox_pred = outs
                classifications = torch.cat([out for out in cls_score], dim=1)
                regressions = torch.cat([out for out in bbox_pred], dim=1)
                anchors = model.anchors(imgs.shape[2:], imgs.device)
            """

            curves = ub.ddict(list)
            for bx in xdev.InteractiveIter(list(range(niters))):

                outputs, loss_parts = harn.run_batch(batch)
                print(ub.repr2(loss_parts, nl=1))

                for k, v in loss_parts.items():
                    curves[k].append(float(v.item()))

                batch_dets = harn.raw_model.coder.decode_batch(outputs)
                print('batch_dets = {!r}'.format(batch_dets))
                dets0 = batch_dets[0].numpy().sort()
                print('dets0.classes = {!r}'.format(dets0.classes))
                try:
                    print('dets0.probs =\n{}'.format(ub.repr2(dets0.probs, precision=2)))
                except Exception:
                    pass
                print('dets0.scores = {!r}'.format(dets0.scores[0:3]))
                print('dets0.boxes = {!r}'.format(dets0.boxes.to_cxywh()[0:3]))

                stacked = harn.draw_batch(batch, outputs, batch_dets)
                kwplot.imshow(stacked, fnum=1, pnum=(1, 2, 1))

                # ymax = [v.mean() for v in curves.values()]
                ymax = 4
                kwplot.multi_plot(ydata=curves, fnum=1, pnum=(1, 2, 2), ymax=ymax, ymin=0)
                xdev.InteractiveIter.draw()
                loss = sum(loss_parts.values())
                loss.backward()
                harn.optimizer.step()
                harn.optimizer.zero_grad()
        else:
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', 'overfit'))
            for bx in range(niters):

                outputs, loss_parts = harn.run_batch(batch)
                loss = sum(loss_parts.values())
                print('loss = {!r}'.format(loss))
                loss.backward()
                harn.optimizer.step()
                harn.optimizer.zero_grad()

                fpath = join(dpath, 'overfit_{:05d}.jpg'.format(bx))

                batch_dets = harn.raw_model.coder.decode_batch(outputs)

                stacked = harn.draw_batch(batch, outputs, batch_dets)
                print('fpath = {!r}'.format(fpath))
                kwimage.imwrite(fpath, stacked)

    def draw_batch(harn, batch, outputs, batch_dets, idx=None, thresh=None,
                   num_extra=3):
        """
        Returns:
            np.ndarray: numpy image

        Example:
            >>> # DISABLE_DOCTSET
            >>> from bioharn.detect_fit import *  # NOQA
            >>> #harn = setup_harn(bsize=1, datasets='special:voc', pretrained='lightnet')
            >>> harn = setup_harn(bsize=1, datasets='special:habcam', arch='cascade', pretrained='/home/joncrall/work/bioharn/fit/nice/bioharn-det-v14-cascade/deploy_MM_CascadeRCNN_iawztlag_032_ETMZBH.zip', normalize_inputs=False, channels='rgb|disparity', sampler_backend=None)
            >>> #harn = setup_harn(bsize=1, datasets='special:shapes8', arch='cascade', xpu=[1])
            >>> #harn = setup_harn(bsize=1, datasets='special:shapes8', arch='cascade', xpu=[1, 0])
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
        inputs = batch['inputs']

        channels = harn.raw_model.channels
        components = channels.decode(inputs)

        def _ensure_unpacked(item):
            if isinstance(item, torch.Tensor):
                item = item
            else:
                item = item.pack()
            return item

        rgb_batch = _ensure_unpacked(components['rgb'])

        labels = {
            k: v.data for k, v in batch['label'].items()
        }
        labels = nh.XPU('cpu').move(labels)

        # TODO: FIX YOLO SO SCALE IS NOT NEEDED

        if 'disparity' in components:
            batch_disparity = kwarray.ArrayAPI.numpy(
                _ensure_unpacked(components['disparity']))
        else:
            batch_disparity = None

        classes = harn.datasets['train'].sampler.classes

        if idx is None:
            bsize = len(rgb_batch)
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
            chw01 = rgb_batch[idx]

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
                    item_masks = list(ub.flatten(labels['class_masks']))[idx]
                    item_flags = list(ub.flatten(labels['has_mask']))[idx]
                    ssegs = []
                    for flag, mask in zip(item_flags, item_masks):
                        if flag > 0:
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
            try:
                canvas = true_dets.draw_on(canvas, color='green')
                canvas = pred_dets.draw_on(canvas, color='blue')
            except Exception as ex:
                import xdev
                xdev.embed()
                harn.warn('In draw_batch ex = {!r}'.format(ex))
                canvas = kwimage.draw_text_on_image(
                    canvas, 'drawing-error', org=(0, 0), valign='top')

            # canvas = cv2.resize(canvas, (300, 300))
            imgs.append(canvas)

        stacked = imgs[0] if len(imgs) == 1 else kwimage.stack_images_grid(imgs)
        return stacked

    def after_epochs(harn):
        """
        Callback after all train/vali/test epochs are complete.
        """
        harn._draw_conv_layers()

    def _draw_conv_layers(harn, suffix=''):
        """
        We use this to visualize the first convolutional layer
        """
        import kwplot
        # Visualize the first convolutional layer
        dpath = ub.ensuredir((harn.train_dpath, 'monitor', 'layers'))
        # fig = kwplot.figure(fnum=1)
        for key, layer in nh.util.trainable_layers(harn.model, names=True):
            # Typically the first convolutional layer returned here is the
            # first convolutional layer in the network
            if isinstance(layer, torch.nn.Conv2d):
                if max(layer.kernel_size) > 2:
                    fig = kwplot.plot_convolutional_features(
                        layer, fnum=1, normaxis=0)
                    kwplot.set_figtitle(key, subtitle=str(layer), fig=fig)
                    layer_dpath = ub.ensuredir((dpath, key))
                    fname = 'layer-{}-epoch_{}{}.jpg'.format(
                        key, harn.epoch, suffix)
                    fpath = join(layer_dpath, fname)
                    fig.savefig(fpath)
                    break

            if isinstance(layer, torch.nn.Linear):
                # TODO: visualize the FC layer
                pass

    def on_complete(harn):
        """
        Evaluate the trained model after training is complete

        Ignore:
            # test to make sure this works
            python -m netharn.data.grab_voc

            python -m bioharn.detect_fit \
                --name=bioharn_shape_example \
                --datasets=special:shapes1024 \
                --schedule=step-60-80 \
                --augment=simple \
                --init=lightnet \
                --arch=yolo2 \
                --optim=sgd --lr=1e-3 \
                --input_dims=window \
                --window_dims=512,512 \
                --window_overlap=0.0 \
                --normalize_inputs=False \
                --workers=4 --xpu=0 --batch_size=16 --bstep=1 \
                --sampler_backend=cog \
                --test_on_finish=True \
                --timeout=1

        """
        from bioharn import detect_eval

        if harn.script_config['test_on_finish']:
            eval_dataset = harn.datasets.get('test', None)
            if eval_dataset is None:
                harn.warn('No test dataset to evaluate, trying vali')
                eval_dataset = harn.datasets.get('vali', None)

            if eval_dataset is None:
                harn.warn('No evaluation dataset')
            else:
                # hack together special attributes into "deployed" to work with
                # what detect_eval expects. Eventually we should clean this up.
                if getattr(harn, 'deploy_fpath', None) is None:
                    harn.deploy_fpath = harn._deploy()

                import torch_liberator
                deployed = torch_liberator.DeployedModel.coerce(harn.deploy_fpath)
                deployed._model = harn.model
                deployed._train_info = harn.train_info
                deployed.train_dpath = harn.train_dpath

                print('eval_dataset = {!r}'.format(eval_dataset))
                eval_config = {
                    'xpu': harn.xpu,

                    'deployed': deployed,

                    # fixme: should be able to pass the dataset as an object
                    'dataset': eval_dataset.sampler.dset.fpath,

                    'input_dims': harn.script_config['input_dims'],
                    'window_dims': harn.script_config['window_dims'],
                    'window_overlap': harn.script_config['window_overlap'],
                    'workers': harn.script_config['workers'],
                    'channels': harn.script_config['channels'],
                    'out_dpath': ub.ensuredir(harn.train_dpath, 'out_eval'),  # fixme
                    'eval_in_train_dpath': True,
                    'draw': 10,
                    'batch_size': harn.script_config['batch_size'],
                }
                eval_config = detect_eval.DetectEvaluateConfig(eval_config)
                evaluator = detect_eval.DetectEvaluator(config=eval_config)

                # Fixme: the evaluator should be able to handle being passed the
                # sampler / dataset in the config.
                evaluator.sampler = eval_dataset.sampler
                evaluator.evaluate()


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
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    nh.configure_hacks(config)  # fix opencv bugs
    ub.ensuredir(config['workdir'])

    if False:
        # Hack to fix: https://github.com/pytorch/pytorch/issues/973
        # torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)
        try:
            import resource
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        except Exception:
            pass

    # Load ndsampler.CocoDataset objects from info in the config
    subsets = coerce_data.coerce_datasets(config)

    classes = subsets['train'].object_categories()
    for k, subset in subsets.items():
        # TODO: better handling
        special_catnames = ['negative',
                            # 'ignore',
                            'test']
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
    if config['ensure_background_class']:
        # HACK: ENSURE BACKGROUND IS CLASS IDX 0 for mmdet 1.x
        print('classes = {!r}'.format(classes))
        if 'background' not in classes:
            for k, subset in subsets.items():
                # mmdet 1.x wants id=0, but 2.x wants id=len(classes)
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
            classes_of_interest=config['classes_of_interest'],
            use_segmentation='mask' in config['arch'].lower(),
            input_dims=config['input_dims'],
            window_dims=config['window_dims'],
            window_overlap=config['window_overlap'] if (tag == 'train') else 0.0,
            augment=config['augment'] if (tag == 'train') else False,
            gravity=config['gravity'],
            channels=config['channels'],
        )
        for tag, sampler in samplers.items()
    }

    from netharn.data.data_containers import ContainerXPU
    xpu = ContainerXPU.coerce(config['xpu'])
    print('xpu = {!r}'.format(xpu))

    print('make loaders')
    loaders_ = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            # num_batches=config['num_batches'] if tag == 'train' else 'auto',
            num_batches=config['num_batches'] if tag == 'train' else config['num_vali_batches'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            balance=(tag == 'train' and config['balance']),
            multiscale=(tag == 'train') and config['multiscale'],
            # pin_memory=True,
            pin_memory=False,
            xpu=xpu)
        for tag, dset in torch_datasets.items()
    }

    from netharn.data.channel_spec import ChannelSpec
    channels = ChannelSpec.coerce(config['channels'])
    print('channels = {!r}'.format(channels))

    if config['normalize_inputs'] == 'imagenet':
        input_stats = {
            'rgb': {
                'mean':  torch.Tensor([[[[0.4850]], [[0.4560]], [[0.4060]]]]),
                'std':  torch.Tensor([[[[0.2290]], [[0.2240]], [[0.2250]]]]),
            }
        }
    elif config['normalize_inputs']:
        # TODO: this needs to be refactored and abstracted
        # Get stats on the dataset (todo: nice way to disable augmentation temporarilly for this)
        _dset = torch_datasets['train']
        num = config['normalize_inputs']
        num = num if isinstance(num, int) and num is not True else 1000
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(num, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)

        depends = ub.odict([
            ('input_id', _dset.input_id),
            ('num', num),
        ])
        cfgstr = ub.hash_data(depends)
        cacher = ub.Cacher('dset_mean', cfgstr=cfgstr + 'v8')
        input_stats = cacher.tryload()
        if input_stats is None:
            # Use parallel workers to load data faster
            from netharn.data.data_containers import container_collate
            from functools import partial
            collate_fn = partial(container_collate, num_devices=1)

            _dset.disable_augmenter = True

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True, batch_size=config['batch_size'])

            # Track moving average of each fused channel stream
            channel_stats = {key: kwarray.RunningStats()
                             for key in channels.keys()}

            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                for key, val in batch['inputs'].items():
                    try:
                        for batch_part in val.data:
                            for part in batch_part.numpy():
                                channel_stats[key].update(part)
                    except ValueError:  # final batch broadcast error
                        pass

            input_stats = {}
            for key, running in channel_stats.items():
                perchan_stats = running.summarize(axis=(1, 2))
                input_stats[key] = {
                    'std': perchan_stats['mean'].round(3),
                    'mean': perchan_stats['std'].round(3),
                }

            cacher.save(input_stats)
            _dset.disable_augmenter = False  # hack
    else:
        input_stats = None

    print('input_stats = {!r}'.format(input_stats))

    initializer_ = nh.Initializer.coerce(
        config, leftover='kaiming_normal', association='embedding')
    print('initializer_ = {!r}'.format(initializer_))

    arch = config['arch']
    classes = samplers['train'].classes

    criterion_ = None
    if arch == 'efficientdet':
        from bioharn.models import efficientdet
        initkw = dict(
            classes=classes,
            channels=config['channels'],
            input_stats=input_stats,
        )
        model = efficientdet.EfficientDet(**initkw)
        model._initkw = initkw
    elif arch == 'retinanet':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
            channels=config['channels'],
            input_stats=input_stats,
        )
        model = mm_models.MM_RetinaNet(**initkw)
        model._initkw = initkw
        if config['backbone_init'] == 'url':
            model._init_backbone_from_pretrained('torchvision://resnet50')
        elif config['backbone_init'] is not None:
            model._init_backbone_from_pretrained(config['backbone_init'])
    elif arch == 'MM_HRNetV2_w18_MaskRCNN':
        from bioharn.models import new_models_v1
        initkw = dict(
            classes=classes,
            channels=config['channels'],
            input_stats=input_stats,
        )
        model = new_models_v1.MM_HRNetV2_w18_MaskRCNN(**initkw)
        model._initkw = initkw
        if config['backbone_init'] == 'url':
            model._init_backbone_from_pretrained(model.pretrained_url)
        elif config['backbone_init'] is not None:
            model._init_backbone_from_pretrained(config['backbone_init'])
    elif arch == 'maskrcnn':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
            channels=config['channels'],
            input_stats=input_stats,
        )
        model = mm_models.MM_MaskRCNN(**initkw)
        model._initkw = initkw
        if config['backbone_init'] == 'url':
            model._init_backbone_from_pretrained('torchvision://resnet50')
        elif config['backbone_init'] is not None:
            model._init_backbone_from_pretrained(config['backbone_init'])
    elif arch == 'cascade':
        from bioharn.models import mm_models
        initkw = dict(
            classes=classes,
            channels=config['channels'],
            input_stats=input_stats,
        )
        model = mm_models.MM_CascadeRCNN(**initkw)
        model._initkw = initkw

        if config['backbone_init'] == 'url':
            model._init_backbone_from_pretrained('open-mmlab://resnext101_32x4d')
        elif config['backbone_init'] is not None:
            model._init_backbone_from_pretrained(config['backbone_init'])
    elif arch == 'yolo2':
        from bioharn.models.yolo2 import yolo2

        if config['anchors'] == 'auto':
            _dset = torch_datasets['train']
            cacher = ub.Cacher('dset_anchors', cfgstr=_dset.input_id + 'v4')
            anchors = cacher.tryload()
            if anchors is None:
                anchors = _dset.sampler.dset.boxsize_stats(anchors=5, perclass=False)['all']['anchors']
                anchors = anchors.round(1)
                cacher.save(anchors)
        elif config['anchors'] == 'lightnet':
            # HACKED IN:
            # anchors = np.array([[1.0, 1.0],
            #                     [0.1, 0.1 ],
            #                     [0.01, 0.01],
            #                     [0.07781961, 0.10329947],
            #                     [0.03830135, 0.05086466]])
            anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                                (5.05587, 8.09892), (9.47112, 4.84053),
                                (11.2364, 10.0071)]) * 32
        else:
            raise KeyError(config['anchors'])

        print('anchors = {!r}'.format(anchors))

        model_ = (yolo2.Yolo2, {
            'classes': classes,
            'anchors': anchors,
            'input_stats': input_stats,
            'channels': config['channels'],
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
            'seen_thresh': config['seen_thresh'],
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
        'name': config['name'],
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
            'smoothing': 0,
        }),

        'other': {
            # Other params are not used internally, so you are free to set any
            # extra params specific to your algorithm, and still have them
            # logged in the hyperparam structure. For YOLO this is `ovthresh`.
            'batch_size': config['batch_size'],
            'name': config['name'],
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
    harn.preferences.update({
        'num_keep': 10,
        'keep_freq': 5,
        'export_modules': ['bioharn'],  # TODO
        'prog_backend': 'progiter',  # alternative: 'tqdm'
        'keyboard_debug': False,
        'eager_dump_tensorboard': True,
        'deploy_after_error': True,
        'timeout': config['timeout'],
        'allow_unicode': True,
    })
    harn.intervals.update({
        'log_iter_train': 1000,
        'test': 0,
        'vali': config['vali_intervals'],
    })
    harn.script_config = config

    print('harn = {!r}'.format(harn))
    print('samplers = {!r}'.format(samplers))
    return harn


def fit(**kw):
    harn = setup_harn(**kw)
    harn.initialize()

    if ub.argflag('--lrtest'):
        """
        """
        # Undocumented hidden feature,
        # Perform an LR-test, then resetup the harness. Optionally draw the
        # results using matplotlib.
        from netharn.prefit.lr_tests import lr_range_test

        result = lr_range_test(
            harn, init_value=1e-4, final_value=0.5, beta=0.3,
            explode_factor=10, num_iters=200)

        if ub.argflag('--show'):
            import kwplot
            plt = kwplot.autoplt()
            result.draw()
            plt.show()

        # Recreate a new version of the harness with the recommended LR.
        config = harn.script_config.asdict()
        config['lr'] = (result.recommended_lr * 10)
        harn = setup_harn(**config)
        harn.initialize()

    # This starts the main loop which will run until the monitor's terminator
    # criterion is satisfied. If the initialize step loaded a checkpointed that
    # already met the termination criterion, then this will simply return.
    return harn.run()


if __name__ == '__main__':
    """

    CommandLine:
        python -m bioharn.detect_fit \
            --name=bioharn_shapes_example \
            --datasets=special:shapes256 \
            --schedule=step-10-30 \
            --augment=complex \
            --init=noop \
            --arch=retinanet \
            --optim=sgd --lr=1e-3 \
            --input_dims=window \
            --window_dims=128,128 \
            --window_overlap=0.0 \
            --normalize_inputs=imagenet \
            --workers=4 --xpu=0 --batch_size=8 --bstep=1 \
            --sampler_backend=None

        kwcoco toydata --key vidshapes32-aux --dst auxtrain.json
        kwcoco toydata --key vidshapes8-aux --dst auxvali.json

        python -m bioharn.detect_fit \
            --name=bioharn_shapes_example3 \
            --train_dataset=vidshapes32-aux \
            --vali_dataset=vidshapes8-aux \
            --augment=simple \
            "--channels=rgb|disparity,flowx|flowy" \
            --init=noop \
            --arch=MM_HRNetV2_w18_MaskRCNN \
            --optim=sgd --lr=1e-8 \
            --schedule=ReduceLROnPlateau-p10-c10 \
            --patience=100 \
            --input_dims=256,256 \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --normalize_inputs=True \
            --workers=0 --xpu=0 --batch_size=2 --bstep=4 \
            --sampler_backend=cog \
            --test_on_finish=True \
            --num_batches=10 \
            --max_epoch=10

        python -m bioharn.detect_fit \
            --name=bioharn_shapes_example3 \
            --train_dataset=vidshapes32-aux \
            --vali_dataset=vidshapes8-aux \
            --augment=simple \
            "--channels=rgb|disparity" \
            --init=noop \
            --arch=efficientdet \
            --optim=sgd --lr=1e-8 \
            --schedule=ReduceLROnPlateau-p10-c10 \
            --patience=100 \
            --input_dims=256,256 \
            --window_dims=512,512 \
            --window_overlap=0.0 \
            --normalize_inputs=True \
            --workers=0 --xpu=0 --batch_size=2 --bstep=4 \
            --sampler_backend=cog \
            --test_on_finish=True \
            --num_batches=10 \
            --max_epoch=10
    """
    if 0:
        def make_warnings_print_tracebacks():
            import warnings
            import traceback
            _orig_formatwarning = warnings.formatwarning
            warnings._orig_formatwarning = _orig_formatwarning
            def _monkeypatch_formatwarning_tb(*args, **kwargs):
                s = _orig_formatwarning(*args, **kwargs)
                if len(s.strip()):
                    tb = traceback.format_stack()
                    s += ''.join(tb[:-1])
                return s
            warnings.formatwarning = _monkeypatch_formatwarning_tb

        make_warnings_print_tracebacks()
    fit()
