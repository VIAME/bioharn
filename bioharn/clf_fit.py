# -*- coding: utf-8 -*-
"""
This is a simple generalized harness for training a classifier on a coco dataset.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import numpy as np
import sys
import torch
import ubelt as ub

import netharn as nh
import kwarray
import scriptconfig as scfg
from netharn.data.channel_spec import ChannelSpec
from bioharn import clf_dataset


class ClfConfig(scfg.Config):
    """
    This is the default configuration for running the classification example.

    Instances of this class behave like a dictionary. However, they can also be
    specified on the command line, via kwargs, or by pointing to a YAML/json
    file. See :module:``scriptconfig`` for details of how to use
    :class:`scriptconfig.Config` objects.
    """
    default = {
        'name': scfg.Value('clf_example', help='A human readable tag that is "name" for humans'),
        'workdir': scfg.Path('~/work/netharn', help='Dump all results in your workdir'),

        'workers': scfg.Value(2, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('auto', help='See netharn.XPU for details. can be auto/cpu/xpu/cuda0/0,1,2,3)'),

        'datasets': scfg.Value('special:shapes256', help='Either a special key or a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),
        'test_dataset': scfg.Value(None),

        'sampler_backend': scfg.Value(None, help='ndsampler backend'),

        'channels': scfg.Value('rgb', help='special channel code. See ChannelSpec'),

        'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('adam', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'min_dim': scfg.Value(64, help='absolute minimum window size'),
        'input_dims': scfg.Value((224, 224), help='Window size to input to the network'),
        'normalize_inputs': scfg.Value(True, help=(
            'if True, precompute training mean and std for data whitening')),

        'balance': scfg.Value(None, help='balance strategy. Can be classes or None'),

        'augmenter': scfg.Value('simple', help='type of training dataset augmentation'),  # TODO: rename to augment
        'gravity': scfg.Value(0.0, help='how often to assume gravity vector for augmentation'),

        'batch_size': scfg.Value(3, help='number of items per batch'),
        'num_batches': scfg.Value('auto', help='Number of batches per epoch (mainly for balanced batch sampling)'),

        'max_epoch': scfg.Value(140, help='Maximum number of epochs'),
        'patience': scfg.Value(140, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-4, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),
        'schedule': scfg.Value(
            'step90-120', help=(
                'Special coercible netharn code. Eg: onecycle50, step50, gamma, ReduceLROnPlateau-p10-c10')),

        'init': scfg.Value('noop', help='How to initialized weights: e.g. noop, kaiming_normal, path-to-a-pretrained-model)'),
        'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),

        # preference
        'num_draw': scfg.Value(4, help='Number of initial batchs to draw per epoch'),
        'draw_interval': scfg.Value(1, help='Minutes to wait between drawing'),
        'draw_per_batch': scfg.Value(32, help='Number of items to draw within each batch'),

        'timeout': scfg.Value(float('inf'), help='maximum number of seconds to wait for training'),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class ClfModel(nh.layers.Module):
    """
    A simple pytorch classification model.

    Example:
        >>> classes = ['a', 'b', 'c']
        >>> input_stats = {
        >>>     'mean': torch.Tensor([[[0.1]], [[0.2]], [[0.2]]]),
        >>>     'std': torch.Tensor([[[0.3]], [[0.3]], [[0.3]]]),
        >>> }
        >>> channels = 'rgb'
        >>> self = ClfModel(
        >>>     arch='resnet50', channels=channels,
        >>>     input_stats=input_stats, classes=classes)
        >>> inputs = torch.rand(4, 1, 256, 256)
        >>> outputs = self(inputs)
        >>> self.coder.decode_batch(outputs)
    """

    def __init__(self, arch='resnet50', classes=1000, channels='rgb',
                 input_stats=None):
        super(ClfModel, self).__init__()

        import ndsampler
        if input_stats is None:
            input_stats = {}
        input_norm = nh.layers.InputNorm(**input_stats)

        self.classes = ndsampler.CategoryTree.coerce(classes)

        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))
        num_classes = len(self.classes)

        if arch == 'resnet50':
            from torchvision import models
            model = models.resnet50()
            new_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                        stride=3, padding=3, bias=False)
            new_fc = torch.nn.Linear(2048, num_classes, bias=True)
            new_conv1.weight.data[:, 0:in_channels, :, :] = model.conv1.weight.data[0:, 0:in_channels, :, :]
            new_fc.weight.data[0:num_classes, :] = model.fc.weight.data[0:num_classes, :]
            new_fc.bias.data[0:num_classes] = model.fc.bias.data[0:num_classes]
            model.fc = new_fc
            model.conv1 = new_conv1
        elif arch == 'resnext101':
            from torchvision.models import resnet
            arch = 'resnext101_32x8d'
            model = resnet.resnext101_32x8d()
            state_dict = resnet.load_state_dict_from_url(
                    resnet.model_urls[arch])
            model.load_state_dict(state_dict)
            new_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                        stride=3, padding=3, bias=False)
            new_fc = torch.nn.Linear(2048, num_classes, bias=True)
            new_conv1.weight.data[:, 0:in_channels, :, :] = model.conv1.weight.data[0:, 0:in_channels, :, :]
            new_fc.weight.data[0:num_classes, :] = model.fc.weight.data[0:num_classes, :]
            new_fc.bias.data[0:num_classes] = model.fc.bias.data[0:num_classes]
            model.fc = new_fc
            model.conv1 = new_conv1
        else:
            raise KeyError(arch)

        self.arch = arch

        self.input_norm = input_norm
        self.model = model

        self.coder = ClfCoder(self.classes)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor | dict): Either the input images  (as a regulary
                pytorch BxCxHxW Tensor) or a dictionary mapping input
                modalities to the input imges.

        Returns:
             Dict[str, Tensor]: model output wrapped in a dictionary so its
                 clear what the return type is. In this case "energy" is class
                 probabilities **before** softmax / normalization is applied.
        """
        if isinstance(inputs, dict):
            # TODO: handle channel modalities later
            assert len(inputs) == 1, (
                'only support one fused stream: e.g. rgb for now ')
            im = ub.peek(inputs.values())
        else:
            im = inputs

        im = self.input_norm(im)
        class_energy = self.model(im)
        outputs = {
            'class_energy': class_energy,
        }
        return outputs


class ClfCoder(object):
    """
    The coder take the output of the classifier and transforms it into a
    standard format. Currently there is no standard "classification" format
    that I use other than a dictionary with special keys.
    """
    def __init__(self, classes):
        self.classes = classes

    def decode_batch(self, outputs):
        class_energy = outputs['class_energy']
        class_probs = self.classes.hierarchical_softmax(class_energy, dim=1)
        pred_cxs, pred_conf = self.classes.decision(
            class_probs, dim=1, thresh=0.1,
            criterion='entropy',
        )
        decoded = {
            'class_probs': class_probs,
            'pred_cxs': pred_cxs,
            'pred_conf': pred_conf,
        }
        return decoded


class ClfHarn(nh.FitHarn):
    """
    The Classification Harness
    ==========================

    The concept of a "Harness" at the core of netharn.  This our custom
    :class:`netharn.FitHarn` object for a classification problem.

    The Harness provides the important details to the training loop via the
    `run_batch` method. The rest of the loop boilerplate is taken care of by
    `nh.FitHarn` internals. In addition to `run_batch`, we also define several
    callbacks to perform customized monitoring of training progress.
    """

    def after_initialize(harn, **kw):
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }

    def prepare_batch(harn, raw_batch):
        return raw_batch

    def run_batch(harn, batch):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> harn = setup_harn(datasets='special:shapes256', batch_size=4).initialize()
            >>> batch = harn._demo_batch(0, tag='train')
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
        """
        classes = harn.raw_model.classes
        inputs = harn.xpu.move(batch['inputs'])
        labels = harn.xpu.move(batch['labels'])

        outputs = harn.model(inputs)

        class_energy = outputs['class_energy']
        class_logprobs = classes.hierarchical_log_softmax(
            class_energy, dim=1)

        class_idxs = labels['class_idxs']
        loss = nh.criterions.focal.nll_focal_loss(
            class_logprobs, class_idxs, focus=2.0, reduction='mean')

        loss_parts = {}
        loss_parts['clf'] = loss

        decoded = harn.raw_model.coder.decode_batch(outputs)

        outputs['class_probs'] = decoded['class_probs']
        outputs['pred_cxs'] = decoded['pred_cxs']
        outputs['true_cxs'] = class_idxs
        return outputs, loss_parts

    def on_batch(harn, batch, outputs, loss):
        """
        Custom code executed at the end of each batch.
        """
        bx = harn.bxs[harn.current_tag]

        if not getattr(harn, '_draw_timer', None):
            harn._draw_timer = ub.Timer().tic()
        # need to hack do draw here, because we need to call
        # mmdet forward in a special way
        harn._hack_do_draw = (harn.batch_index < harn.script_config['num_draw'])
        harn._hack_do_draw |= ((harn._draw_timer.toc() > 60 * harn.script_config['draw_interval']) and
                               (harn.script_config['draw_interval'] > 0))
        if harn._hack_do_draw:
            stacked = harn._draw_batch(batch, outputs)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            import kwimage
            kwimage.imwrite(fpath, stacked)

        y_pred = kwarray.ArrayAPI.numpy(outputs['pred_cxs'])
        y_true = outputs['true_cxs'].data.cpu().numpy()
        probs = outputs['class_probs'].data.cpu().numpy()
        harn._accum_confusion_vectors['y_true'].append(y_true)
        harn._accum_confusion_vectors['y_pred'].append(y_pred)
        harn._accum_confusion_vectors['probs'].append(probs)

    def _draw_batch(harn, batch, outputs, idxs=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> harn = setup_harn(batch_size=3).initialize()
            >>> batch = harn._demo_batch(0, tag='train')
            >>> outputs, loss = harn.run_batch(batch)
            >>> stacked = harn._draw_batch(batch, outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        inputs = batch['inputs']['rgb'].data.cpu().numpy()
        true_cxs = batch['labels']['class_idxs'].data.cpu().numpy()
        class_probs = outputs['class_probs'].data.cpu().numpy()
        pred_cxs = kwarray.ArrayAPI.numpy(outputs['pred_cxs'])

        dset = harn.datasets[harn.current_tag]
        classes = dset.classes

        if idxs is None:
            bsize = len(inputs)
            # Get a varied sample of the batch
            # (the idea is ensure that we show things on the non-dominat gpu)
            num_want = harn.script_config['draw_per_batch']
            if num_want is None:
                num_want = bsize
            num_want = min(num_want, bsize)
            # This will never produce duplicates (difference between
            # consecutive numbers will always be > 1 there fore they will
            # always round to a different number)
            idxs = np.linspace(bsize - 1, 0, num_want).round().astype(np.int).tolist()
            idxs = sorted(idxs)
        else:
            idxs = [idxs] if not ub.iterable(idxs) else idxs

        todraw = []
        for idx in idxs:
            im = inputs[idx]
            pcx = pred_cxs[idx]
            tcx = true_cxs[idx]
            probs = class_probs[idx]
            im_ = im.transpose(1, 2, 0)

            # Renormalize and resize image for drawing
            im_ = kwimage.normalize(im_)
            im_ = kwimage.ensure_uint255(im_)
            im_ = np.ascontiguousarray(im_)
            im_ = kwimage.imresize(im_, dsize=(200, 200),
                                   interpolation='nearest')

            # Draw classification information on the image
            im_ = kwimage.draw_clf_on_image(im_, classes=classes, tcx=tcx,
                                            pcx=pcx, probs=probs)
            todraw.append(im_)

        stacked = kwimage.stack_images_grid(todraw, overlap=-10,
                                            bg_value=(10, 40, 30),
                                            chunksize=8)
        return stacked

    def on_epoch(harn):
        """
        Custom code executed at the end of each epoch.

        This function can optionally return a dictionary containing any scalar
        quality metrics that you wish to log and monitor. (Note these will be
        plotted to tensorboard if that is installed).

        Returns:
            dict: dictionary of scalar metrics for netharn to log

        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> harn = setup_harn().initialize()
            >>> harn._demo_epoch('vali', max_iter=10)
            >>> harn.on_epoch()
        """
        from netharn.metrics import clf_report
        dset = harn.datasets[harn.current_tag]

        probs = np.vstack(harn._accum_confusion_vectors['probs'])
        y_true = np.hstack(harn._accum_confusion_vectors['y_true'])
        y_pred = np.hstack(harn._accum_confusion_vectors['y_pred'])

        # _pred = probs.argmax(axis=1)
        # assert np.all(_pred == y_pred)

        # from netharn.metrics import confusion_vectors
        # cfsn_vecs = confusion_vectors.ConfusionVectors.from_arrays(
        #     true=y_true, pred=y_pred, probs=probs, classes=dset.classes)
        # report = cfsn_vecs.classification_report()
        # combined_report = report['metrics'].loc['combined'].to_dict()

        # ovr_cfsn = cfsn_vecs.binarize_ovr()
        # Compute multiclass metrics (new way!)
        target_names = dset.classes
        ovr_report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])

        # percent error really isn't a great metric, but its easy and standard.
        errors = (y_true != y_pred)
        acc = 1.0 - errors.mean()
        percent_error = (1.0 - acc) * 100

        metrics_dict = ub.odict()
        metrics_dict['ave_brier'] = ovr_report['ave']['brier']
        metrics_dict['ave_mcc'] = ovr_report['ave']['mcc']
        metrics_dict['ave_auc'] = ovr_report['ave']['auc']
        metrics_dict['ave_ap'] = ovr_report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = acc

        harn.info(ub.color_text('ACC FOR {!r}: {!r}'.format(harn.current_tag, acc), 'yellow'))

        # Clear confusion vectors accumulator for the next epoch
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }
        return metrics_dict


def setup_harn(cmdline=True, **kw):
    """
    This creates the "The Classification Harness" (i.e. core ClfHarn object).
    This is where we programmatically connect our program arguments with the
    netharn HyperParameter standards. We are using :module:`scriptconfig` to
    capture these, but you could use click / argparse / etc.

    This function has the responsibility of creating our torch datasets,
    lazy computing input statistics, specifying our model architecture,
    schedule, initialization, optimizer, dynamics, XPU etc. These can usually
    be coerced using netharn API helpers and a "standardized" config dict. See
    the function code for details.

    Args:
        cmdline (bool, default=True):
            if True, behavior will be modified based on ``sys.argv``.
            Note this will activate the scriptconfig ``--help``, ``--dump`` and
            ``--config`` interactions.

    Kwargs:
        **kw: the overrides the default config for :class:`ClfConfig`.
            Note, command line flags have precedence if cmdline=True.

    Returns:
        ClfHarn: a fully-defined, but uninitialized custom :class:`FitHarn`
            object.

    Example:
        >>> # xdoctest: +SKIP
        >>> kw = {'datasets': 'special:shapes256'}
        >>> cmdline = False
        >>> harn = setup_harn(cmdline, **kw)
        >>> harn.initialize()
    """
    import ndsampler
    config = ClfConfig(default=kw)
    config.load(cmdline=cmdline)
    print('config = {}'.format(ub.repr2(config.asdict())))

    nh.configure_hacks(config)
    coco_datasets = nh.api.Datasets.coerce(config)

    print('coco_datasets = {}'.format(ub.repr2(coco_datasets, nl=1)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend=config['sampler_backend'])
        for tag, dset in coco_datasets.items()
    }

    for tag, sampler in ub.ProgIter(list(samplers.items()), desc='prepare frames'):
        sampler.frames.prepare(workers=config['workers'])

    torch_datasets = {
        'train': clf_dataset.ClfDataset(
            samplers['train'],
            input_dims=config['input_dims'],
            min_dim=config['min_dim'],
            augment=config['augmenter'],
            gravity=config['gravity'],
        ),
        'vali': clf_dataset.ClfDataset(
            samplers['vali'],
            input_dims=config['input_dims'],
            min_dim=config['min_dim'],
            augment=False),
    }

    channels = ChannelSpec.coerce(config['channels'])

    if config['normalize_inputs']:
        # Get stats on the dataset (todo: turn off augmentation for this)
        _dset = torch_datasets['train']
        prev = _dset.disable_augmenter
        _dset.disable_augmenter = False
        if config['normalize_inputs'] is True:
            est_size = 1000
        else:
            est_size = config['normalize_inputs']
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(est_size, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)

        depends = [
            config['normalize_inputs'],
            _dset.input_id
        ]

        cacher = ub.Cacher('dset_mean', cfgstr=ub.hash_data(depends) + 'v4')
        input_stats = cacher.tryload()

        if input_stats is None:
            # Use parallel workers to load data faster
            from netharn.data.data_containers import container_collate
            from functools import partial
            collate_fn = partial(container_collate, num_devices=1)

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True,
                batch_size=config['batch_size'])

            # Track moving average of each fused channel stream
            channel_stats = {key: nh.util.RunningStats()
                             for key in channels.keys()}
            assert len(channel_stats) == 1, (
                'only support one fused stream for now')
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                for key, val in batch['inputs'].items():
                    try:
                        for part in val.numpy():
                            channel_stats[key].update(part)
                    except ValueError:  # final batch broadcast error
                        pass

            perchan_input_stats = {}
            for key, running in channel_stats.items():
                running = ub.peek(channel_stats.values())
                perchan_stats = running.simple(axis=(1, 2))
                perchan_input_stats[key] = {
                    'std': perchan_stats['mean'].round(3),
                    'mean': perchan_stats['std'].round(3),
                }

            input_stats = ub.peek(perchan_input_stats.values())
            cacher.save(input_stats)
        _dset.disable_augmenter = prev
    else:
        input_stats = {}

    torch_loaders = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            balance=(config['balance'] if tag == 'train' else None),
            pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    classes = torch_datasets['train'].classes

    modelkw = {
        'arch': config['arch'],
        'input_stats': input_stats,
        'classes': classes.__json__(),
        'channels': channels,
    }
    model = ClfModel(**modelkw)
    model._initkw = modelkw

    initializer_ = nh.Initializer.coerce(config, association='prefix-hack')
    # initializer_ = nh.Initializer.coerce(config, association='embedding')

    hyper = nh.HyperParams(
        name=config['name'],

        workdir=config['workdir'],
        xpu=nh.XPU.coerce(config['xpu']),

        datasets=torch_datasets,
        loaders=torch_loaders,

        model=model,
        criterion=None,

        optimizer=nh.Optimizer.coerce(config),
        dynamics=nh.Dynamics.coerce(config),
        scheduler=nh.Scheduler.coerce(config),

        initializer=initializer_,

        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'smoothing': 0.0,
        }),
        other={
            'name': config['name'],
            'batch_size': config['batch_size'],
            'balance': config['balance'],
        },
        extra={
            'argv': sys.argv,
            'config': ub.repr2(config.asdict()),
        }
    )
    harn = ClfHarn(hyper=hyper)
    harn.preferences.update({
        'num_keep': 3,
        'keep_freq': 10,
        'tensorboard_groups': ['loss'],
        'eager_dump_tensorboard': True,
        'timeout': config['timeout'],
        'colored': not ub.WIN32,
        'allow_unicode': not ub.WIN32,
    })

    if ub.WIN32:
        ub.util_colors.NO_COLOR = True

    harn.intervals.update({})
    harn.script_config = config
    return harn


def main():
    """
    Main function for the generic classification example with an undocumented
    hack for the lrtest.
    """
    harn = setup_harn()
    harn.initialize()

    if ub.argflag('--lrtest'):
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
    deploy_fpath = harn.run()

    # The returned deploy_fpath is the path to an exported netharn model.
    # This model is the on with the best weights according to the monitor.
    print('deploy_fpath = {!r}'.format(deploy_fpath))
    return harn


if __name__ == '__main__':
    """

    TODO:
        - [ ] Construct training dataset based on truth boxes unioned with
              predicted boxes from a detector.

        - [ ] Evaluate trained models (create clf_predict / clf_eval)

        - [ ] Student Teacher

    Example:
        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-v001 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p5-c5 \
            --max_epoch=400 \
            --augment=complex \
            --init=noop \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x8d-8ba56ff5.pth \
            --workers=8 \
            --xpu=auto \
            --batch_size=32 \
            --num_batches=2000 \
            --balance=classes



        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-v002 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p5-c5 \
            --max_epoch=400 \
            --augment=complex \
            --init=noop \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --backbone_init=$HOME/.cache/torch/checkpoints/resnext101_32x8d-8ba56ff5.pth \
            --workers=8 \
            --xpu=auto \
            --batch_size=32 \
            --balance=None

        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-v003 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
            --schedule=ReduceLROnPlateau-p5-c5 \
            --max_epoch=400 \
            --augment=simple \
            --pretrained=/home/joncrall/work/bioharn/fit/runs/bioharn-clf-rgb-v001/nrorbmcb/deploy_ClfModel_nrorbmcb_051_UFCIUU.zip \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --workers=8 \
            --xpu=auto \
            --batch_size=32 \
            --balance=None


        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-hard-v004 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_hardbg1.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_hardbg1.mscoco.json \
            --schedule=ReduceLROnPlateau-p5-c5 \
            --max_epoch=400 \
            --augment=simple \
            --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --workers=8 \
            --xpu=auto \
            --batch_size=32 \
            --balance=classes

        python -m bioharn.clf_fit \
            --name=test-start-from-pretrained-pt \
            --train_dataset=special:shapes32 \
            --vali_dataset=special:shapes8 \
            --pretrained=/home/joncrall/.cache/torch/checkpoints/resnet50-19c8e357.pth \
            --workdir=$HOME/work/test \
            --arch=resnet50 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=False \
            --workers=0 \
            --xpu=auto \
            --batch_size=32
    """
    main()
