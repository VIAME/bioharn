from os.path import exists
from os.path import isfile
from os.path import basename
from os.path import join
import warnings
import ubelt as ub
import torch.utils.data as torch_data
import netharn as nh
import numpy as np
import torch
import scriptconfig as scfg
import kwimage
# import warnings
# from bioharn.channel_spec import ChannelSpec


class ClfPredictConfig(scfg.Config):
    default = {

        'deployed': None,
        'batch_size': 4,
        'xpu': 'auto',

        'input_dims': scfg.Value('native', help='The size of the inputs to the network'),
        'min_dim': scfg.Value(64, help='absolute minimum window size'),

        'workers': 0,

        'sharing_strategy': scfg.Value('default', help=(
            'torch backend data loader strategory. '
            'Can be file_descriptor or file_system')),

        'channels': scfg.Value(
            'native',
            help='list of channels needed by the model. '
            'Typically this can be inferred from the model'),

        'verbose': 1,
    }


class ClfPredictor(object):
    """
    Does classification prediction based on a pretrained model and input
    dataset or list of images.

    Ignore:
        >>> from bioharn import clf_fit
        >>> harn = clf_fit.setup_harn(cmdline=False, dataset='special:shapes128',
        >>>                           max_epoch=1, timeout=60)
        >>> deployed = harn.run()
        >>> config = {
        >>>     'deployed': deployed,
        >>> }
        >>> predictor = ClfPredictor(config)
        >>> predictor._ensure_model()
        >>> # =============================
        >>> # Test with a sampler input
        >>> sampler = harn.datasets['vali'].sampler
        >>> classifications = list(predictor.predict_sampler(sampler))
        >>> # =============================
        >>> # Test with a single image input
        >>> image = kwimage.grab_test_image()
    """
    def __init__(predictor, config=None, **kwargs):
        predictor.config = ClfPredictConfig(config)
        predictor.config.update(kwargs)
        predictor.xpu = None
        predictor.model = None
        predictor.raw_model = None
        predictor.coder = None

    @classmethod
    def demo(ClfPredictor):
        from bioharn import clf_fit
        harn = clf_fit.setup_harn(cmdline=False, dataset='special:shapes128',
                                  max_epoch=1, timeout=60)
        harn.initialize()
        if not harn.prev_snapshots():
            # generate a model if needed
            deployed = harn.run()
        else:
            deployed = harn.prev_snapshots()[-1]
        config = {
            'deployed': deployed,
        }
        predictor = ClfPredictor(config)
        return predictor

    @classmethod
    def _infer_native(cls, config):
        """
        Preforms whatever hacks are necessary to introspect the correct
        values of special "native" config options depending on the model.
        """
        # Set default fallback values
        native_defaults = {
            'input_dims': (224, 224),
            'min_dim': 64,
            'channels': 'rgb'
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
        return native

    def _ensure_model(predictor):
        # Just make sure the model is in memory (it might not be on the XPU yet)
        if predictor.model is None:
            xpu = nh.XPU.coerce(predictor.config['xpu'])
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
            if xpu != nh.XPU.from_data(model):
                print('Mount model on {}'.format(xpu))
                model = xpu.mount(model)
                predictor.model = model
                # The model must have a coder
                predictor.raw_model = predictor.xpu.raw(predictor.model)
                predictor.coder = predictor.raw_model.coder
            # hack to prevent multiple XPU data checks
            predictor.model._ensured_mount = True

    def predict(predictor, images):
        """
        Classify a sequence of images

        NOTE:
            This exists as a convinience, the prefered method is to predict
            using a sampler, which is better at preserving metadata.

        Yields:
            ClassificationResult
        """
        native = predictor._infer_native(predictor.config)
        dataset = ImageListDataset(images, input_dims=native['input_dims'],
                                   min_dim=native['min_dim'])
        loader = torch_data.DataLoader(dataset,
                                       batch_size=predictor.config['batch_size'],
                                       num_workers=predictor.config['workers'],
                                       drop_last=False,
                                       shuffle=False)
        prog = ub.ProgIter(loader, desc='clf predict',
                           verbose=predictor.config['verbose'])
        predictor._ensure_mounted_model()
        predictor.model.eval()
        with torch.no_grad():
            for raw_batch in prog:
                batch_result = predictor.predict_batch(raw_batch)
                for item in batch_result:
                    yield item

    def predict_sampler(predictor, sampler, aids=None):
        """
        Runs prediction on all positive instances in a sampler.

        Yields:
            ClassificationResult

        Example:
            >>> import ndsampler
            >>> predictor = ClfPredictor.demo()
            >>> sampler = ndsampler.CocoSampler.demo()
            >>> classifications = list(predictor.predict_sampler(sampler))
        """
        native = predictor._infer_native(predictor.config)
        dataset = ClfSamplerDataset(sampler, input_dims=native['input_dims'],
                                    min_dim=native['min_dim'], aids=aids)
        loader = torch_data.DataLoader(dataset,
                                       batch_size=predictor.config['batch_size'],
                                       num_workers=predictor.config['workers'],
                                       drop_last=False,
                                       shuffle=False)
        # Hack to fix: https://github.com/pytorch/pytorch/issues/973
        if predictor.config['sharing_strategy'] != 'default':
            torch.multiprocessing.set_sharing_strategy(
                predictor.config['sharing_strategy'])
        predictor._ensure_mounted_model()
        predictor.model.eval()
        prog = ub.ProgIter(loader, desc='clf predict sampler',
                           verbose=predictor.config['verbose'])
        with torch.no_grad():
            classes = predictor.raw_model.classes
            for raw_batch in prog:
                batch_result, class_probs = predictor.predict_batch(raw_batch)
                # Translate to row-based data structure
                # (Do we want a column based data structure option?)
                for (rx, row), prob in zip(batch_result.iterrows(), class_probs):
                    datakeys = set(Classification.__datakeys__) | set(row.keys())
                    clf_kwargs = row.copy()
                    clf_kwargs.update({
                        'prob': prob,
                        'classes': classes,
                        'datakeys': datakeys,
                    })
                    result = Classification(**clf_kwargs)
                    yield result

    def predict_batch(predictor, raw_batch):
        """
        Internal method, runs prediction on a single batch

        Returns:
            DataFrameArray: batched results
        """

        labels = raw_batch.get('labels', None)

        inputs = predictor.xpu.move(raw_batch['inputs'])
        outputs = predictor.model(inputs)
        # classes = predictor.raw_model.classes

        if predictor.coder is not None:
            decoded = predictor.coder.decode_batch(outputs)

            import kwarray
            class_probs = kwarray.ArrayAPI.numpy(decoded['class_probs'])
            pred_cxs = kwarray.ArrayAPI.numpy(decoded['pred_cxs'])
            pred_conf = kwarray.ArrayAPI.numpy(decoded['pred_conf'])

            data = {
                'cidx': pred_cxs,
                'conf': pred_conf,
            }
            if labels is not None:
                if 'aid' in labels:
                    data['aid'] = kwarray.ArrayAPI.numpy(labels['aid'])
                if 'gid' in labels:
                    data['gid'] = kwarray.ArrayAPI.numpy(labels['gid'])
                if 'cid' in labels:
                    data['true_cid'] = kwarray.ArrayAPI.numpy(labels['cid'])
                if 'class_idxs' in labels:
                    data['true_cidx'] = kwarray.ArrayAPI.numpy(labels['class_idxs'])
            batch_result = kwarray.DataFrameArray(data)
            return batch_result, class_probs
        else:
            # should there be a clf decoder? (probably for consistency)
            raise NotImplementedError


class Classification(ub.NiceRepr):
    """
    A data structure for a classification result.
    TODO: perhaps move to kwimage.structs / kwannot.

    Should this be vectorized to include multiple classifications by default?
    Probably.

    Attributes:
        prob: probability of each category
        cidx: indexes of the predicted category
        conf: confidence in prediction
        classes: a list of possible categories

    Example:
        >>> self = Classification(prob=[0.1, 0.2, 0.7], cidx=1, classes=['a', 'b', 'c'])
    """
    __datakeys__ = ['prob', 'cidx', 'conf']
    __metakeys__ = ['classes']
    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None, **kwargs):
        # Standardize input format
        if kwargs:
            if data or meta:
                raise ValueError('Cannot specify kwargs AND data/meta dicts')
            _datakeys = self.__datakeys__
            _metakeys = self.__metakeys__
            # Allow the user to specify custom data and meta keys
            if datakeys is not None:
                _datakeys = _datakeys + list(datakeys)
            if metakeys is not None:
                _metakeys = _metakeys + list(metakeys)
            # Perform input checks whenever kwargs is given
            data = {key: kwargs.pop(key) for key in _datakeys if key in kwargs}
            meta = {key: kwargs.pop(key) for key in _metakeys if key in kwargs}
            if kwargs:
                raise ValueError(
                    'Unknown kwargs: {}'.format(sorted(kwargs.keys())))
            if 'conf' not in data:
                # Infer conf if cidx and prob is given
                def _isvalid(d, k):
                    return d.get(k, None) is not None
                if _isvalid(data, 'prob') and _isvalid(data, 'cidx'):
                    data['conf'] = data['prob'][data['cidx']]

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data is explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def __nice__(self):
        attrs = ub.dict_union(self.data, self.meta)
        return ub.repr2(attrs, nl=1, precision=2)

    @property
    def prob(self):
        return self.data['prob']

    @property
    def cidx(self):
        return self.data['cidx']

    @property
    def conf(self):
        return self.data['conf']

    @property
    def classes(self):
        return self.meta['classes']

    @classmethod
    def random(cls, classes=None, rng=None):
        """
        Create a random classification

        Example:
            >>> self = Classification.random(classes=10, rng=0)
            >>> print('self = {!r}'.format(self))
        """
        if classes is None:
            classes = 3
        import ndsampler
        import kwarray
        classes = ndsampler.CategoryTree.coerce(classes)

        rng = kwarray.ensure_rng(rng)
        logits = torch.from_numpy(rng.rand(1, len(classes))).float()
        probs = classes.hierarchical_softmax(logits, dim=1).numpy()
        pred_idxs, pred_confs = classes.decision(probs, dim=1)
        cidx = pred_idxs[0]
        conf = pred_confs[0]
        prob = probs[0]
        self = cls(prob=prob, cidx=cidx, conf=conf, classes=classes)
        return self

    def draw_on(self, image, true_cidx=None):
        """
        Draws classification prediction on an image

        Example:
            >>> classes = ['class-A', 'class-B', 'class-C']
            >>> self = Classification.random(classes=classes, rng=0)
            >>> image = kwimage.grab_test_image(dsize=(300, 300))
            >>> canvas = self.draw_on(image, true_cidx=2)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
        """
        from kwimage.structs import _generic
        dtype_fixer = _generic._consistent_dtype_fixer(image)
        canvas = image
        canvas = kwimage.ensure_uint255(canvas)
        canvas = kwimage.atleast_3channels(canvas, copy=False)
        canvas = kwimage.draw_clf_on_image(
            canvas, classes=self.classes, probs=self.prob, tcx=true_cidx,
            pcx=self.cidx)
        canvas = dtype_fixer(canvas)
        return canvas


class ClfSamplerDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Wraps a ndsampler.Sampler for classification prediction

    Returns fixed-sized images centered around objects indexed by a
    CocoDataset, similar to, but simpler than the clf_dataset used in training.

    Example:
        >>> from bioharn.clf_predict import *  # NOQA
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo()
        >>> self = ClfSamplerDataset(sampler)
        >>> index = 0
        >>> item = self[index]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(item['inputs']['rgb'])
    """

    def __init__(self, sampler, input_dims=(224, 224), min_dim=64, aids=None):
        self.input_dims = input_dims
        self.sampler = sampler
        self.min_dim = min_dim

        if aids is None:
            # use all annotation if unspecified
            aids = list(sampler.dset.anns.keys())
        self.aids = aids

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sampler = self.sampler

        if self.aids is None:
            tr = sampler.regions.get_positive(index=index)
        else:
            tr = {'aid': self.aids[index]}
        tr = sampler._infer_target_attributes(tr)

        # always sample a square region with a minimum size
        dim = np.ceil((max(tr['width'], tr['height'])))
        dim = max(dim, self.min_dim)
        window_dims = (dim, dim)

        tr['height'], tr['width'] = window_dims
        sample = self.sampler.load_sample(tr, with_annots=False)

        image = kwimage.atleast_3channels(sample['im'])[:, :, 0:3]

        # Resize to input dimensinos
        if self.input_dims is not None:
            dsize = tuple(self.input_dims[::-1])
            image = kwimage.imresize(image, dsize=dsize, letterbox=True)

        im_chw = image.transpose(2, 0, 1) / 255.0
        class_id_to_idx = self.sampler.classes.id_to_idx
        inputs = {
            'rgb': torch.FloatTensor(im_chw),
        }
        labels = {
            'class_idxs': class_id_to_idx[tr['category_id']],
            'aid': tr['aid'],
            'gid': tr['gid'],
            'cid': tr['category_id'],
        }
        item = {
            'inputs': inputs,
            'labels': labels,
        }
        return item


class ImageListDataset(torch_data.Dataset):
    """
    TODO: implement me.
    """
    def __init__(self, images, input_dims=(224, 224), min_dim=64):
        self.images = images
        self.input_dims = input_dims
        self.min_dim = min_dim  # absolute minimum window size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        raise NotImplementedError


def _cached_predict(predictor, sampler, out_dpath='./cached_clf_out',
                    draw=False, enable_cache=True, async_buffer=False,
                    verbose=1, draw_truth=True):
    """

    Ignore:
        >>> from bioharn.clf_predict import *  # NOQA
        >>> import ndsampler
        >>> config = {}
        >>> config['deployed'] = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip')
        >>> predictor = ClfPredictor(config)
        >>> out_dpath = './cached_clf_out'
        >>> coco_fpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/all_pred.mscoco.json")
        >>> coco_dset = ndsampler.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset, workdir=None,
        >>>                                 backend=None)


        globals().update(xdev.get_func_kwargs(_cached_predict))

    """
    import kwarray
    import ndsampler
    from bioharn import util
    import tempfile
    coco_dset = sampler.dset

    aids = list(coco_dset.anns.keys())
    # predictor.config['verbose'] = 1

    det_outdir = ub.ensuredir((out_dpath, 'pred'))
    tmp_fpath = tempfile.mktemp()

    # if gids is None:
    #     gids = list(coco_dset.imgs.keys())

    print('enable_cache = {!r}'.format(enable_cache))
    # print('Found {} / {} existing predictions'.format(len(have_gids), len(gids)))

    # gids = ub.oset(gids) - have_gids
    pred_gen = predictor.predict_sampler(sampler, aids=aids)

    if async_buffer:
        desc = 'buffered detect'
        buffered_gen = util.AsyncBufferedGenerator(pred_gen,
                                                   size=coco_dset.n_images)
        gen = buffered_gen
    else:
        desc = 'unbuffered detect'
        gen = pred_gen

    gid_to_pred = {}
    prog = ub.ProgIter(gen, total=len(aids), desc=desc, verbose=verbose)
    for img_idx, (gid, dets) in enumerate(prog):
        gid_to_pred[gid] = dets

        img = coco_dset.imgs[gid]

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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/clf_predict.py

    Ignore:
        $HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
