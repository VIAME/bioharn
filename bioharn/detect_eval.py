from os.path import exists
from os.path import join
from os.path import dirname
import os
import ndsampler
import kwimage
import netharn as nh
import numpy as np
import ubelt as ub
import scriptconfig as scfg
from bioharn import detect_predict


class DetectEvaluateConfig(scfg.Config):
    default = {
        'deployed': scfg.Value(
            '/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip',
            help='deployed network filepath'),

        'batch_size': scfg.Value(4, help=(
            'number of images that run through the network at a time')),

        'input_dims': scfg.Value('native', help=(
            'size of input chip; or "native" which uses input_dims used in training')),

        'window_dims': scfg.Value('native', help=(
            'size of sliding window chip; or "full" which uses the entire frame; '
            'or "native", which uses window_dims specified in training')),

        'overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        'workers': scfg.Value(4, help='num CPUs for data loading'),

        'verbose': 1,

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.001,

        'xpu': scfg.Value('argv', help='a CUDA device or a CPU'),

        'dataset': scfg.Value('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json'),
        'workdir': scfg.Path('~/work/bioharn', help='Dump all results in your workdir'),

        'out_dpath': scfg.Path('./detect_eval_out/', help='folder to send the output'),

        'eval_in_train_dpath': scfg.Path(True, help='write eval results into the training directory if its known'),
    }


def evaluate_models(**kw):
    """
    Evaluate multiple models using a config file or CLI.

    /home/joncrall/work/bioharn/fit/nice/bioharn-det-v16-cascade/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP.zip

    Ignore:
        from bioharn.detect_eval import *  # NOQA
        kw = {}

        kw = {
            'deployed': '~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/manual-snapshots/_epoch_00000006.pt',
            'workers': 4,
            'batch_size': 10,
            'xpu': 0,
        }
        evaluate_models(**kw)
    """
    import itertools as it
    if 'config' in kw:
        config_fpath = kw['config']
        defaults = ub.dict_diff(kw, {'config'})
        multi_config = DetectEvaluateConfig(data=config_fpath, default=defaults, cmdline=True)
    else:
        multi_config = DetectEvaluateConfig(default=kw, cmdline=True)
    print('MultiConfig: {}'.format(ub.repr2(multi_config.asdict())))

    # Look for specific items in the base config where multiple values are
    # given. We will evaluate over all permutations of these values.
    base_config = multi_config.asdict()

    model_fpaths = base_config.pop('deployed')
    if not ub.iterable(model_fpaths):
        model_fpaths = [model_fpaths]
    model_fpaths = [ub.expandpath(p) for p in model_fpaths]

    for fpath in model_fpaths:
        if not exists(fpath):
            raise Exception('{} does not exist'.format(fpath))

    search_space = {}
    input_dims = base_config.pop('input_dims')
    if ub.iterable(input_dims) and len(input_dims) and ub.iterable(input_dims[0], strok=True):
        # multiple input dims to test are given as a list
        search_space['input_dims'] = input_dims
    else:
        # only one input given, need to wrap
        search_space['input_dims'] = [input_dims]

    keys = list(search_space.keys())
    basis = list(search_space.values())
    config_perms = [ub.dzip(keys, permval) for permval in it.product(*basis)]

    sampler = None
    metric_fpaths = []
    for model_fpath in ub.ProgIter(model_fpaths, desc='test model', verbose=3):
        print('model_fpath = {!r}'.format(model_fpath))

        predictor = None
        for perm in ub.ProgIter(config_perms, desc='test config', verbose=3):

            # Create the config for this detection permutation
            config = ub.dict_union(base_config, perm)
            config['deployed'] = model_fpath

            print('config = {}'.format(ub.repr2(config)))
            evaluator = DetectEvaluator(config)
            self = evaluator

            # Reuse the dataset / predictor when possible
            evaluator.predictor = predictor
            print('_init')
            evaluator._init()
            print('evaluate')
            metrics_fpath = evaluator.evaluate()
            print('metrics_fpath = {!r}'.format(metrics_fpath))
            metric_fpaths.append(metrics_fpath)

            # Save loaded predictor/sampler for the next run of this model/dataset
            predictor = evaluator.predictor
            sampler = evaluator.sampler


class DetectEvaluator(object):
    """
    Ignore:
        from bioharn.detect_eval import *  # NOQA
        config = {'xpu': 0, 'batch_size': 2}
        # config['deployed'] = ub.expandpath('~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000044.pt')
        # config['deployed'] = '/home/joncrall/work/bioharn/fit/nice/bioharn-det-v10-test-retinanet/deploy_MM_RetinaNet_daodqsmy_010_QRNNNW.zip'
        # config['deployed'] = '/home/joncrall/work/bioharn/fit/nice/bioharn-det-v11-test-cascade/deploy_MM_CascadeRCNN_ovphtcvk_037_HZUJKO.zip'
        config['deployed'] = '/home/joncrall/work/bioharn/fit/nice/bioharn-det-v16-cascade/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP'
        config['dataset'] = ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')

        evaluator = self = DetectEvaluator(config)
        self._init()

        self.evaluate()

        evaluator = self
        predictor = self.predictor
        self = predictor
        sampler = evaluator.sampler
    """

    def __init__(self, config=None):
        self.config = DetectEvaluateConfig(config)
        self.predictor = None
        self.sampler = None

    def _ensure_sampler(self):
        if self.sampler is None:
            print('loading dataset')
            coco_dset = ndsampler.CocoDataset(ub.expandpath(self.config['dataset']))
            print('loaded dataset')
            workdir = ub.expandpath(self.config['workdir'])
            sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                            backend=None)
            self.sampler = sampler
            # self.sampler.frames.prepare(workers=min(2, self.config['workers']))
            print('prepare frames')
            self.sampler.frames.prepare(workers=self.config['workers'])
            print('finished dataset load')

    def _init(self):
        self._ensure_sampler()

        # Load model
        deployed = nh.export.DeployedModel.coerce(self.config['deployed'])
        nice = deployed.train_info()['nice']
        if deployed.path is None:
            model_tag = nice + '_' + ub.augpath(deployed._info['snap_fpath'], dpath='', ext='', multidot=True)
        else:
            model_tag = nice + '_' + ub.augpath(deployed.path, dpath='', ext='', multidot=True)

        self.model_tag = model_tag
        self.dset_tag = self.sampler.dset.tag.rstrip('.json')

        # Load the trained model
        pred_keys = set(detect_predict.DetectPredictConfig.default.keys()) - {'verbose'}
        pred_cfg = ub.dict_subset(self.config, pred_keys)

        # if self.config['input_dims'] == 'native':
        #     # hack, this info exists, but not in an easy form
        #     train_config = eval(deployed.train_info()['extra']['config'], {})
        #     pred_cfg['input_dims'] = train_config['input_dims']

        native = detect_predict.DetectPredictor._infer_native(pred_cfg)
        pred_cfg.update(native)

        if self.predictor is None:
            # Only create the predictor if needed
            print('Needs initial init')
            self.predictor = detect_predict.DetectPredictor(pred_cfg)
            self.predictor._ensure_model()
        else:
            # Reuse loaded predictors from other evaluators.
            # Update the config in this case
            needs_reinit = self.predictor.config['deployed'] != pred_cfg['deployed']
            self.predictor.config.update(pred_cfg)
            print('needs_reinit = {!r}'.format(needs_reinit))
            if needs_reinit:
                self.predictor._ensure_model()
            else:
                print('reusing loaded model')

        self.classes = self.predictor.raw_model.classes

        # The parameters that influence the predictions
        pred_params = ub.dict_subset(self.predictor.config, [
            'input_dims',
            'window_dims',
        ])
        self.pred_cfg = nh.util.make_short_idstr(pred_params)
        self.predcfg_tag = self.pred_cfg

        self.paths = {}
        out_dpath = ub.ensuredir(self.config['out_dpath'])

        base_dpath = join(out_dpath, self.dset_tag, self.model_tag, self.pred_cfg)

        class UnknownTrainDpath(Exception):
            pass

        try:
            if self.config['eval_in_train_dpath']:
                # NOTE: the train_dpath in the info directory is wrt to the
                # machine the model was trained on. Used the deployed model to
                # grab that path instead wrt to the current machine.
                if deployed.path is None:
                    train_dpath = dirname(deployed.info['train_info_fpath'])
                else:
                    train_dpath = dirname(deployed.path)
                print('train_dpath = {!r}'.format(train_dpath))
                eval_dpath = join(train_dpath, 'eval', self.dset_tag,
                                  self.model_tag, self.pred_cfg)
                ub.ensuredir(eval_dpath)
                ub.ensuredir(dirname(base_dpath))
                if not os.path.islink(base_dpath) and exists(base_dpath):
                    ub.delete(base_dpath)
                ub.symlink(eval_dpath, base_dpath, overwrite=True)
            else:
                raise UnknownTrainDpath
        except UnknownTrainDpath:
            ub.ensuredir(base_dpath)

        self.paths['base'] = base_dpath
        self.paths['metrics'] = ub.ensuredir((self.paths['base'], 'metrics'))
        self.paths['viz'] = ub.ensuredir((self.paths['base'], 'viz'))

    def _run_predictions(self):
        self.predictor.config['verbose'] = 1
        sampler = self.sampler
        # pred_gen = self.predictor.predict_sampler(sampler)

        out_dpath = self.paths['base']
        gid_to_pred = detect_predict._cached_predict(
            self.predictor, sampler, out_dpath, gids=None, draw=10,
            enable_cache=True)

        return gid_to_pred

    def evaluate(self):
        # TODO
        self.predictor.config['verbose'] = 3
        gid_to_pred = self._run_predictions()

        # This can take awhile to accumulate, perhaps cache intermediate
        # results to disk, so we can restart efficiently?
        # gid_to_pred = {}
        # for i, (gid, pred) in enumerate(pred_gen):
        #     gid_to_pred[gid] = pred

        sampler = self.sampler

        # Build true dets
        classes = self.predictor.coder.classes

        gid_to_truth = {}
        for gid in gid_to_pred.keys():
            annots = sampler.load_annotations(gid)
            true_cids = [a['category_id'] for a in annots]
            true_cidx = np.array([classes.id_to_idx[c] for c in true_cids])
            true_dets = kwimage.Detections(
                boxes=kwimage.Boxes([a['bbox'] for a in annots], 'xywh'),
                class_idxs=true_cidx,
                classes=classes,
                # weights=np.ones(len(true_cidx)),
            ).numpy()
            gid_to_truth[gid] = true_dets

        from netharn.metrics import DetectionMetrics
        dmet = DetectionMetrics(classes=classes)
        for gid in ub.ProgIter(list(gid_to_pred.keys())):
            pred_dets = gid_to_pred[gid]
            true_dets = gid_to_truth[gid]
            dmet.add_predictions(pred_dets, gid=gid)
            dmet.add_truth(true_dets, gid=gid)

        # cfsn_vecs = dmet.confusion_vectors()

        voc_info = dmet.score_voc()
        print('voc_info = {}'.format(ub.repr2(voc_info, nl=1)))
        print('mAP = {}'.format(voc_info['mAP']))

        # print(dmet.score_voc())
        # print(dmet.score_coco(verbose=1))
        # dmet.score_netharn()

        print('self.predcfg_tag = {!r}'.format(self.predcfg_tag))

        # CascadeRCNN 512
        # 'voc_mAP':  0.8379016325674102,
        # 'coco_mAP': 0.8357345630911905
        # netharn_AP: 0.8218177807000042,

        # CascadeRCNN 1024
        # coco_mAP: 0.8661484309698254
        # voc_mAP: 0.8713472908407934
        # netharn_AP: 0.8175519079596175

        # CascadeRCNN 1024 - epoch 37
        # coco_mAP: 0.8665642753052862,
        # voc_mAP: 0.8712469728145056,
        # netharn_AP: 0.8172452008810529,

        # RetinaNet
        # voc_mAP: 0.8418748847551925,
        # coco_mAP: 0.8370107889677438

        # TODO: cache detections to a file on disk.
        # Give the DetectionMetrics code an entry point that just takes two
        # coco files and scores them.

        # import json
        # metrics = {
        #     'dset_tag': self.dset_tag,
        #     'model_tag': self.model_tag,
        #     'predcfg_tag': self.predcfg_tag,
        #     'coco_score': coco_score,
        #     'voc_score': voc_score,
        #     'netharn_score': netharn_score,
        # }
        # metrics_fpath = join(self.paths['metrics'], 'metrics.json')
        # with open(metrics_fpath, 'w') as file:
        #     json.dump(nh.hyperparams._ensure_json_serializable(metrics), file)


class CocoEvaluator(object):
    """
    Abstracts the evaluation process to execute on two coco datasets.

    This can be run as a standalone script where the user specifies the paths
    to the true and predited dataset explicitly, or this can be used by a
    higher level script that produces the predictions and then sends them to
    this evaluator.
    """

    def __init__(self, true_dataset, pred_dataset):

        def _coerce_dataset():
            # coerce the input into an ndsampler dataset or a dictionary of
            # detection objects.
            pass


def eval_coco(true_dataset, pred_dataset):
    """
    Ignore:
        >>> true_dataset = ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_test.mscoco.json')
        >>> pred_dataset = '/home/joncrall/work/bioharn/habcam_test_out/pred/detections.mscoco.json'
    """
    import ndsampler
    pred_coco = ndsampler.CocoDataset(pred_dataset)
    true_coco = ndsampler.CocoDataset(true_dataset)

    import netharn as nh
    dmet = nh.metrics.DetectionMetrics.from_coco(true_coco, pred_coco)
    print(dmet.score_coco(verbose=1))
    print(dmet.score_voc())
    print(dmet.score_voc(method='sklearn'))
    print(dmet.score_voc(method='voc2012'))
    print(dmet.score_voc(method='voc2007'))
    print(dmet.score_netharn())

    z = dmet.score_voc(method='sklearn')

    gid = 23
    gid = 61

    for gid in ub.ProgIter(true_coco.imgs.keys()):
        # gid = 24
        t = true_coco.subset([gid])
        p = pred_coco.subset([gid])

        if len(t.anns) > 10:
            continue

        dmet2 = nh.metrics.DetectionMetrics.from_coco(t, p)
        nh_info = dmet2.score_netharn()
        voc_info = dmet2.score_voc()

        voc_info['perclass']
        nh_info['peritem']

        cfsn_vecs = dmet2.confusion_vectors()
        ovr_vecs = cfsn_vecs.binarize_ovr(mode=1)
        ovr_vecs.cx_to_binvecs[0].data._pandas()

        ovr_vecs.precision_recall()
        dmet2.score_voc(method='sklearn')

        ap1 = nh_info['peritem']['ap']
        ap3 = voc_info['perclass'][0]['ap']

        print(dmet2.score_netharn())
        print(dmet2.score_voc(method='sklearn'))
        print(dmet2.score_voc(method='voc2012'))
        print(dmet2.score_voc(method='voc2007'))

        nh_info['peritem']['ppv'][::-1]
        nh_info['peritem']['tpr'][::-1]

        voc_info['perclass'][0]['prec']
        voc_info['perclass'][0]['rec']

        if abs(ap1 - ap3) > 0.1:
            break


#    pred_raw  pred  true  score  weight     iou  txs  pxs  gid
# 0         0     0     0 0.9797  0.9000  0.8887    2    1   61
# 1         0     0     0 0.9704  0.9000  0.8866    0    0   61
# 2         0     0    -1 0.8839  1.0000 -1.0000   -1    2   61
# 3         0     0    -1 0.7681  1.0000 -1.0000   -1    3   61
# 4         0     0    -1 0.4356  1.0000 -1.0000   -1    6   61
# 5         0     0    -1 0.2764  1.0000 -1.0000   -1    5   61
# 6         0     0    -1 0.1799  1.0000 -1.0000   -1    4   61
# 7        -1    -1     0 0.0000  0.9000 -1.0000    1   -1   61

if __name__ == '__main__':
    """
    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/manual-snapshots/_epoch_00000006.pt

    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000013.pt --batch_size=30 --xpu=0


    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000044.pt
    ~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/
    """

    evaluate_models()
