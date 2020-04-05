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
        'deployed': scfg.Value(None, help='deployed network filepath'),

        # Evaluation dataset
        'dataset': scfg.Value(None, help='path to an mscoco dataset'),
        'workdir': scfg.Path('~/work/bioharn', help='Dump all results in your workdir'),

        'batch_size': scfg.Value(4, help=(
            'number of images that run through the network at a time')),

        'input_dims': scfg.Value('native', help=(
            'size of input chip; or "native" which uses input_dims used in training')),

        'window_dims': scfg.Value('native', help=(
            'size of sliding window chip; or "full" which uses the entire frame; '
            'or "native", which uses window_dims specified in training')),

        'window_overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        'workers': scfg.Value(4, help='num CPUs for data loading'),

        'verbose': 1,

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.001,

        'xpu': scfg.Value('argv', help='a CUDA device or a CPU'),

        'channels': scfg.Value(
            'native',
            help='a specification of channels needed by this model. See ChannelSpec for details. '
            'Typically this can be inferred from the model'),

        'out_dpath': scfg.Path('./detect_eval_out/', help='folder to send the output'),

        'eval_in_train_dpath': scfg.Path(True, help='write eval results into the training directory if its known'),

        'draw': scfg.Value(10, help='number of images with predictions to draw'),
        'enable_cache': scfg.Value(True, help='writes predictions to disk'),
    }


def evaluate_models(cmdline=True, **kw):
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
        multi_config = DetectEvaluateConfig(data=config_fpath, default=defaults, cmdline=cmdline)
    else:
        multi_config = DetectEvaluateConfig(default=kw, cmdline=cmdline)
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

            # Reuse the dataset / predictor when possible
            evaluator.predictor = predictor
            evaluator.sampler = sampler
            print('_init')
            evaluator._init()
            print('evaluate')
            metrics_fpath = evaluator.evaluate()
            print('metrics_fpath = {!r}'.format(metrics_fpath))
            metric_fpaths.append(metrics_fpath)

            # Save loaded predictor/sampler for the next run of this model/dataset
            predictor = evaluator.predictor
            sampler = evaluator.sampler

    rows = []
    train_config_rows = []
    import json
    # import ast
    for fpath in ub.ProgIter(metric_fpaths, desc='gather summary'):
        metrics = json.load(open(fpath, 'r'))
        row = {}
        row['model_tag'] = metrics['model_tag']
        row['predcfg_tag'] = metrics['predcfg_tag']
        row['ap'] = metrics['pr_result']['ap']
        row['auc'] = metrics['roc_result']['auc']

        # Hack to get train config params
        # train_config = ast.literal_eval(metrics['train_info']['extra']['config'])
        train_config = eval(metrics['train_info']['extra']['config'],
                            {'inf': float('inf')}, {})
        train_config_rows.append(train_config)
        rows.append(row)

    import pandas as pd
    pd.set_option('max_colwidth', 256)
    df = pd.DataFrame(rows)
    print(df.to_string(float_format=lambda x: '%0.3f' % x))

    def find_varied_params(train_config_rows):
        all_keys = set()
        for c in train_config_rows:
            all_keys.update(set(c))
        ignore_keys = {
            'datasets', 'focus', 'max_epoch', 'nice', 'ovthresh', 'patience',
            'workdir', 'workers', 'xpu', 'sampler_backend', 'visible_thresh',
            'warmup_iters', 'pretrained', 'grad_norm_type', 'grad_norm_max',
            'warmup_ratio',
        }
        valid_keys = all_keys - ignore_keys
        key_basis = ub.ddict(set)
        for c in train_config_rows:
            for k in valid_keys:
                v = c.get(k, ub.NoParam)
                if isinstance(v, list):
                    v = tuple(v)
                key_basis[k].add(v)
        varied_basis = {}

        force_include_keys = {
            'window_overlap',  'batch_size',  'augment', 'init',  'bstep',
            'input_dims', 'lr', 'use_disparity',  'multiscale',
            'normalize_inputs', 'window_dims',
        }
        for k, vs in list(key_basis.items()):
            if len(vs) > 1 or k in force_include_keys:
                varied_basis[k] = set(vs)
        return varied_basis

    varied_basis = find_varied_params(train_config_rows)

    for row, config in zip(rows, train_config_rows):
        subcfg = ub.dict_subset(config, set(varied_basis), default=np.nan)
        row.update(subcfg)

    import pandas as pd
    pd.set_option('max_colwidth', 256)
    df = pd.DataFrame(rows)
    print(df.to_string(float_format=lambda x: '%0.3f' % x))


class DetectEvaluator(object):
    """
    Evaluation harness for a detection task.

    Creates an instance of :class:`bioharn.detect_predict.DetectPredictor`,
    executes prediction, compares the results to a groundtruth dataset, and
    outputs various metrics summarizing performance.

    Args:
        config (DetectEvaluateConfig):
            the configuration of the evaluator, which is a superset of
            :class:`bioharn.detect_predict.DetectPredictConfig`.

    Ignore:
        from bioharn.detect_eval import *  # NOQA
        config = {'xpu': 0, 'batch_size': 2, 'window_overlap': 0.5}
        config['deployed'] = '/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v27/dxziuzrv/deploy_MM_CascadeRCNN_dxziuzrv_019_GQDHOF.zip'
        config['dataset'] = ub.expandpath('/home/joncrall/data/private/_combo_cfarm/cfarm_test.mscoco.json')

        evaluator = evaluator = DetectEvaluator(config)
        evaluator._init()
        predictor = evaluator.predictor
        sampler = evaluator.sampler
        coco_dset = sampler.dset

        predictor.config['verbose'] = 1
        out_dpath = evaluator.paths['base']
        evaluator.evaluate()
    """

    def __init__(evaluator, config=None):
        evaluator.config = DetectEvaluateConfig(config)
        evaluator.predictor = None
        evaluator.sampler = None

    def _ensure_sampler(evaluator):
        if evaluator.sampler is None:
            print('loading dataset')
            coco_dset = ndsampler.CocoDataset(ub.expandpath(evaluator.config['dataset']))
            print('loaded dataset')
            workdir = ub.expandpath(evaluator.config['workdir'])
            sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                            backend=None)
            evaluator.sampler = sampler
            # evaluator.sampler.frames.prepare(workers=min(2, evaluator.config['workers']))
            print('prepare frames')
            evaluator.sampler.frames.prepare(workers=evaluator.config['workers'])
            print('finished dataset load')

    def _init(evaluator):
        evaluator._ensure_sampler()

        # Load model
        deployed = nh.export.DeployedModel.coerce(evaluator.config['deployed'])
        nice = deployed.train_info()['nice']

        evaluator.train_info = deployed.train_info()

        # hack together a model tag
        if hasattr(deployed, 'model_tag'):
            model_tag = deployed.model_tag
        else:
            if deployed.path is None:
                model_tag = nice + '_' + ub.augpath(deployed._info['snap_fpath'], dpath='', ext='', multidot=True)
            else:
                model_tag = nice + '_' + ub.augpath(deployed.path, dpath='', ext='', multidot=True)

        evaluator.model_tag = model_tag
        evaluator.dset_tag = evaluator.sampler.dset.tag.rstrip('.json')

        # Load the trained model
        pred_keys = set(detect_predict.DetectPredictConfig.default.keys()) - {'verbose'}
        pred_cfg = ub.dict_subset(evaluator.config, pred_keys)

        # if evaluator.config['input_dims'] == 'native':
        #     # hack, this info exists, but not in an easy form
        #     train_config = eval(deployed.train_info()['extra']['config'], {})
        #     pred_cfg['input_dims'] = train_config['input_dims']

        native = detect_predict.DetectPredictor._infer_native(pred_cfg)
        pred_cfg.update(native)

        if evaluator.predictor is None:
            # Only create the predictor if needed
            print('Needs initial init')
            evaluator.predictor = detect_predict.DetectPredictor(pred_cfg)
            evaluator.predictor._ensure_model()
        else:
            # Reuse loaded predictors from other evaluators.
            # Update the config in this case
            needs_reinit = evaluator.predictor.config['deployed'] != pred_cfg['deployed']
            evaluator.predictor.config.update(pred_cfg)
            print('needs_reinit = {!r}'.format(needs_reinit))
            if needs_reinit:
                evaluator.predictor._ensure_model()
            else:
                print('reusing loaded model')

        evaluator.classes = evaluator.predictor.raw_model.classes

        # The parameters that influence the predictions
        pred_params = ub.dict_subset(evaluator.predictor.config, [
            'input_dims',
            'window_dims',
            'window_overlap',
        ])
        evaluator.pred_cfg = nh.util.make_short_idstr(pred_params)
        evaluator.predcfg_tag = evaluator.pred_cfg

        evaluator.paths = {}
        out_dpath = ub.ensuredir(evaluator.config['out_dpath'])

        base_dpath = join(out_dpath, evaluator.dset_tag, evaluator.model_tag,
                          evaluator.pred_cfg)

        class UnknownTrainDpath(Exception):
            pass

        try:
            if evaluator.config['eval_in_train_dpath']:
                # NOTE: the train_dpath in the info directory is wrt to the
                # machine the model was trained on. Used the deployed model to
                # grab that path instead wrt to the current machine.
                if hasattr(deployed, 'train_dpath'):
                    train_dpath = deployed.train_dpath
                else:
                    if deployed.path is None:
                        train_dpath = dirname(deployed.info['train_info_fpath'])
                    else:
                        if os.path.isdir(deployed.path):
                            train_dpath = deployed.path
                        else:
                            train_dpath = dirname(deployed.path)
                print('train_dpath = {!r}'.format(train_dpath))
                eval_dpath = join(train_dpath, 'eval', evaluator.dset_tag,
                                  evaluator.model_tag, evaluator.pred_cfg)
                ub.ensuredir(eval_dpath)
                ub.ensuredir(dirname(base_dpath))
                if not os.path.islink(base_dpath) and exists(base_dpath):
                    ub.delete(base_dpath)
                ub.symlink(eval_dpath, base_dpath, overwrite=True, verbose=3)
            else:
                raise UnknownTrainDpath
        except UnknownTrainDpath:
            ub.ensuredir(base_dpath)

        evaluator.paths['base'] = base_dpath
        evaluator.paths['metrics'] = ub.ensuredir((evaluator.paths['base'], 'metrics'))
        evaluator.paths['viz'] = ub.ensuredir((evaluator.paths['base'], 'viz'))
        print('evaluator.paths = {!r}'.format(evaluator.paths))

    def _run_predictions(evaluator):

        predictor = evaluator.predictor
        sampler = evaluator.sampler
        # pred_gen = evaluator.predictor.predict_sampler(sampler)

        predictor.config['verbose'] = 1

        out_dpath = evaluator.paths['base']
        gid_to_pred, gid_to_pred_fpath = detect_predict._cached_predict(
            predictor, sampler, out_dpath, gids=None,
            draw=evaluator.config['draw'],
            enable_cache=0 and evaluator.config['enable_cache']
        )
        return gid_to_pred

    def evaluate(evaluator):
        if evaluator.predictor is None or evaluator.sampler is None:
            evaluator._init()
        # TODO
        evaluator.predictor.config['verbose'] = 3
        gid_to_pred = evaluator._run_predictions()

        # This can take awhile to accumulate, perhaps cache intermediate
        # results to disk, so we can restart efficiently?
        # gid_to_pred = {}
        # for i, (gid, pred) in enumerate(pred_gen):
        #     gid_to_pred[gid] = pred

        sampler = evaluator.sampler

        # Determine if truth and model classes are compatible
        model_classes = evaluator.predictor.coder.classes
        truth_classes = sampler.classes
        errors = []
        for node1, id1 in truth_classes.node_to_id.items():
            if id1 in model_classes.id_to_node:
                node2 = model_classes.id_to_node[id1]
                if node1 != node2:
                    errors.append(
                        'id={} exists in model and truth but have '
                        'different names, {}, {}'.format(id1, node1, node2))
            if node1 in model_classes.node_to_id:
                id2 = model_classes.node_to_id[node1]
                if id1 != id2:
                    errors.append(
                        'node={} exists in model and truth but have '
                        'different ids, {}, {}'.format(node1, id1, id2))

        graph2 = model_classes.graph.copy()
        for node1, id1 in truth_classes.node_to_id.items():
            if node1 not in model_classes.node_to_id:
                graph2.add_node(node1, id=id1)
        classes = ndsampler.CategoryTree(graph2)

        if errors:
            raise Exception('\n'.join(errors))

        # Build true dets
        gid_to_truth = {}
        for gid in gid_to_pred.keys():
            annots = sampler.load_annotations(gid)
            true_cids = [a['category_id'] for a in annots]
            true_cidx = np.array([classes.id_to_idx[c] for c in true_cids])
            true_sseg = [a.get('segmentation') for a in annots]
            true_weight = [a.get('weight', 1) for a in annots]

            true_dets = kwimage.Detections(
                boxes=kwimage.Boxes([a['bbox'] for a in annots], 'xywh'),
                segmentations=true_sseg,
                class_idxs=true_cidx,
                classes=classes,
                weights=np.array(true_weight),
            ).numpy()
            gid_to_truth[gid] = true_dets

        from netharn.metrics import DetectionMetrics
        dmet = DetectionMetrics(classes=classes)
        for gid in ub.ProgIter(list(gid_to_pred.keys())):
            pred_dets = gid_to_pred[gid]
            true_dets = gid_to_truth[gid]
            dmet.add_predictions(pred_dets, gid=gid)
            dmet.add_truth(true_dets, gid=gid)

        # Detection only scoring
        cfsn_vecs = dmet.confusion_vectors(ignore_class='ignore')

        # Get pure detection results
        binvecs = cfsn_vecs.binarize_peritem()
        roc_result = binvecs.roc()

        pr_result = binvecs.precision_recall(method='voc2012')
        # pr_result = binvecs.precision_recall(method='sklearn')
        print('roc_result = {!r}'.format(roc_result))
        print('pr_result = {!r}'.format(pr_result))

        if evaluator.config['draw']:
            import kwplot
            kwplot.autompl()

            fig = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True,
                                figtitle='{} {}\n{}'.format(
                                    evaluator.model_tag, evaluator.predcfg_tag,
                                    evaluator.dset_tag,))
            fig.set_size_inches((11, 6))
            pr_result.draw()

            # TODO: MCC / G-score / F-score vs threshold

            kwplot.figure(fnum=1, pnum=(1, 2, 2))
            roc_result.draw()

            fig_fpath = join(evaluator.paths['metrics'], 'pr_roc.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # Get per-class detection results
        ovr_binvecs = cfsn_vecs.binarize_ovr()

        ovr_roc_result = ovr_binvecs.roc()['perclass']
        ovr_pr_result = ovr_binvecs.precision_recall()['perclass']
        if evaluator.config['draw']:
            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle='{} {}\n{}'.format(
                                    evaluator.model_tag, evaluator.predcfg_tag,
                                    evaluator.dset_tag,))
            fig.set_size_inches((11, 6))
            ovr_roc_result.draw(fnum=2)

            # TODO: MCC / G-score / F-score vs threshold
            fig_fpath = join(evaluator.paths['metrics'], 'perclass_roc.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle='{} {}\n{}'.format(
                                    evaluator.model_tag, evaluator.predcfg_tag,
                                    evaluator.dset_tag,))
            fig.set_size_inches((11, 6))
            ovr_pr_result.draw(fnum=2)
            fig_fpath = join(evaluator.paths['metrics'], 'perclass_pr.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # print(dmet.score_voc())
        # print(dmet.score_coco(verbose=1))
        # dmet.score_netharn()

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

        import json
        metrics = {
            'dset_tag': evaluator.dset_tag,
            'model_tag': evaluator.model_tag,
            'predcfg_tag': evaluator.predcfg_tag,
            'roc_result': roc_result,
            'pr_result': pr_result,

            'ovr_roc_result': ovr_roc_result,
            'ovr_pr_result': ovr_pr_result,

            'eval_config': evaluator.config.asdict(),
            'train_info': evaluator.train_info,
        }

        if 0:
            voc_info = dmet.score_voc(ignore_class='ignore')
            print('voc_info = {!r}'.format(voc_info))
            metrics['voc_info'] = voc_info

        # Not sure why using only one doesnt work.
        metrics = nh.hyperparams._ensure_json_serializable(
            metrics, normalize_containers=True, verbose=0)
        metrics = nh.hyperparams._ensure_json_serializable(
            metrics, normalize_containers=False, verbose=0)
        metrics = nh.hyperparams._ensure_json_serializable(
            metrics, normalize_containers=True, verbose=0)

        metrics_fpath = join(evaluator.paths['metrics'], 'metrics.json')
        with open(metrics_fpath, 'w') as file:
            json.dump(metrics, file)

        return metrics_fpath


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


if __name__ == '__main__':
    """
    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/manual-snapshots/_epoch_00000006.pt

    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000013.pt --batch_size=30 --xpu=0


    python ~/code/bioharn/bioharn/detect_eval.py --deployed=~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000044.pt
    ~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/


        python ~/code/ndsampler/ndsampler/make_demo_coco.py

        dpath=/home/joncrall/work/bioharn/fit/nice/bioharn_shapes_example/torch_snapshots
        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=/home/joncrall/.cache/coco-demo/shapes256.mscoco.json
            --deployed=${dpath}/_epoch_00000000.pt,${dpath}/_epoch_00000009.pt,${dpath}/_epoch_00000011.pt \

        python ~/code/bioharn/bioharn/detect_eval.py --xpu=1 --workers=3 --batch_size=64 \
            --dataset=~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v3_test.mscoco.json \
            --deployed="[~/work/bioharn/fit/runs/bioharn-det-v14-cascade/iawztlag/deploy_MM_CascadeRCNN_iawztlag_032_ETMZBH.zip,\
                ~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/deploy_MM_CascadeRCNN_ogenzvgt_059_QBGWCT.zip,\
                ~/work/bioharn/fit/runs/bioharn-det-v16-cascade/hvayxfyx/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP, \
                $HOME/.cache/viame/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip,]" --profile

        python ~/code/bioharn/bioharn/detect_eval.py --xpu=1 --workers=4 --batch_size=64 \
            --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json \
            --deployed=/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v23/kmiqxzis

        python ~/code/bioharn/bioharn/detect_eval.py --xpu=1 --workers=4 --batch_size=64 \
            --dataset=~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v3_test.mscoco.json \
            --deployed=~/work/bioharn/fit/runs/bioharn-det-v16-cascade/hvayxfyx/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP --profile


~/work/bioharn/fit/runs/bioharn-test-yolo-v5/sxfhhhwy/deploy_Yolo2_sxfhhhwy_002_QTVZHQ.zip
~/work/bioharn/fit/runs/bioharn-det-v9-test-cascade/zjolejwz/deploy_MM_CascadeRCNN_zjolejwz_010_LUAKQJ.zip
~/work/bioharn/fit/runs/bioharn-det-v16-cascade/hvayxfyx/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP.zip
~/work/bioharn/fit/runs/bioharn-det-v6-test-retinanet/rioggtso/deploy_MM_RetinaNet_rioggtso_050_MLFGKZ.zip
~/work/bioharn/fit/runs/bioharn-det-v10-test-retinanet/daodqsmy/deploy_MM_RetinaNet_daodqsmy_010_QRNNNW.zip
~/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/ovphtcvk/deploy_MM_CascadeRCNN_ovphtcvk_037_HZUJKO.zip
~/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip
~/work/bioharn/fit/runs/bioharn-det-v14-cascade/iawztlag/deploy_MM_CascadeRCNN_iawztlag_032_ETMZBH.zip
~/work/bioharn/fit/runs/DEMO_bioharn-det-v13-cascade/ogenzvgt/deploy_MM_CascadeRCNN_ogenzvgt_006_IQLOXO.zip
~/work/bioharn/fit/runs/bioharn-det-v8-test-retinanet/opgoqmpg/deploy_MM_RetinaNet_opgoqmpg_000_MKJZNW.zip
~/work/bioharn/fit/runs/bioharn-det-v12-test-retinanet/mrepnniz/deploy_MM_RetinaNet_mrepnniz_094_ODCGUT.zip


        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/_dev/Habcam_2015_g027250_a00111034_c0016_v3_test.mscoco.json \
            --deployed=$HOME/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v22/rfdrszqa/deploy_MM_CascadeRCNN_rfdrszqa_048_DZYTDJ.zip  --workers=4 --batch_size=64

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected/annotations.test.json \
            --deployed="[$HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000000.pt,\
            $HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000030.pt,\
            $HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000046.pt,\
            $HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000050.pt,\
            $HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000055.pt,\
            $HOME/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/torch_snapshots/_epoch_00000056.pt,]"

        /home/joncrall/data/private/_combo_cfarm

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=/home/joncrall/data/private/_combo_cfarm/cfarm_test.mscoco.json \
            --overlap=0.5 --draw=1000 \
            --deployed=/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v27/dxziuzrv/deploy_MM_CascadeRCNN_dxziuzrv_019_GQDHOF.zip

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=~/data/public/Benthic/US_NE_2015_NEFSC_HABCAM/Corrected/annotations.test.json \
            --deployed="[$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/deploy_MM_CascadeRCNN_ddoxsxjs_048_QWTOJP.zip,$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/deploy_MM_CascadeRCNN_ddoxsxjs_078_FJXQLY.zip,\
            $HOME/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/torch_snapshots/_epoch_00000000.pt,\
            $HOME/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/torch_snapshots/_epoch_00000030.pt,\
            $HOME/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/torch_snapshots/_epoch_00000060.pt,\
            $HOME/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/torch_snapshots/_epoch_00000090.pt,\
            $HOME/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-v24/torch_snapshots/_epoch_00000106.pt,]"

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/private/_combos/test_cfarm_habcam_v1.mscoco.json \
            --deployed="[/home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/etvvhzni/deploy_MM_CascadeRCNN_etvvhzni_007_IPEIQA.zip,\
            /home/joncrall/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v29-balanced/gjxbpiei/deploy_MM_CascadeRCNN_gjxbpiei_002_LDATFJ.zip,]"

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json \
            --deployed=/home/joncrall/work/sealions/fit/runs/detect-sealion-cascade-v11/jwrqcohp/deploy_MM_CascadeRCNN_jwrqcohp_036_MHUOFO.zip

    """

    evaluate_models()
