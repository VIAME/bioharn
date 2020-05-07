from os.path import dirname
from os.path import basename
from os.path import exists
from os.path import join
import ubelt as ub


def hack_gather():
    import glob
    import json
    dpaths = list(map(ub.expandpath, [
        '$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-coi-v42_valitune/syavjxrl/',
        '$HOME/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-coi-v43_valitune/nfnqoxuu/',

        '$HOME/remote/namek/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgbd-fine-coi-v42/',
        '$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/',
    ]))

    rows = []
    for dpath in ub.ProgIter(dpaths, verbose=3):
        globstr = join(dpath, 'eval', '*', '*', '*', 'metrics', 'metrics.json')
        json_fpaths = glob.glob(globstr)

        for fpath in ub.ProgIter(json_fpaths, verbose=3):
            small_fpath = ub.augpath(fpath, prefix='small_')
            if not exists(small_fpath):
                print('fpath = {!r}'.format(fpath))
                with open(fpath, 'r') as file:
                    data = json.load(file)

                subkeys = ['dset_tag', 'model_tag', 'predcfg_tag',
                           'ignore_classes', 'eval_config', 'train_info']
                small_data = ub.dict_subset(data, subkeys)

                pr_result = data['pr_result']
                pr_result.pop('tpr', None)
                pr_result.pop('ppv', None)
                pr_result.pop('tp_count', None)
                pr_result.pop('fp_count', None)
                pr_result.pop('fn_count', None)
                pr_result.pop('thresholds', None)
                small_data['pr_result'] = pr_result

                roc_result = data['roc_result']
                roc_result.pop('tpr', None)
                roc_result.pop('fpr', None)
                roc_result.pop('tp_count', None)
                roc_result.pop('fp_count', None)
                roc_result.pop('thresholds', None)
                small_data['roc_result'] = roc_result

                ovr_roc_result = data['ovr_roc_result']
                for key, val in ovr_roc_result.items():
                    val.pop('tpr', None)
                    val.pop('fpr', None)
                    val.pop('tp_count', None)
                    val.pop('fp_count', None)
                    val.pop('thresholds', None)
                small_data['ovr_roc_result'] = ovr_roc_result

                ovr_pr_result = data['ovr_pr_result']
                for key, val in ovr_pr_result.items():
                    val.pop('tpr', None)
                    val.pop('ppv', None)
                    val.pop('tp_count', None)
                    val.pop('fp_count', None)
                    val.pop('fn_count', None)
                    val.pop('thresholds', None)
                small_data['ovr_pr_result'] = ovr_pr_result

                with open(small_fpath, 'w') as file:
                    json.dump(small_data, file, indent='    ')
            else:
                with open(small_fpath, 'r') as file:
                    small_data = json.load(file)
            rows.append(small_data)

    summary_rows = []
    for row in rows:
        if row['dset_tag'].startswith('may_priority_habcam_cfarm_v7_test.mscoc'):
            print(row['dset_tag'])
            print(row['model_tag'])
            print(row['predcfg_tag'])

            ov_pr_result = row['ovr_pr_result']
            summary_row = {}
            summary_row['model_tag'] = row['model_tag']
            for key, val in ov_pr_result.items():
                summary_row[key + '_ap'] = val['ap']
            summary_rows.append(summary_row)

    import pandas as pd
    pd.set_option('max_colwidth', 256)
    df = pd.DataFrame(summary_rows)

    ap_columns = [cname + '_ap' for cname, val in ov_pr_result.items()]
    id_columns = ['model_tag']

    map_df = df[id_columns].copy()
    map_df['mAP'] = df[ap_columns].mean(axis=1)
    map_df[ap_columns] = df[ap_columns]
    map_df = map_df.sort_values('mAP', ascending=False)
    print('\n=== mAP ==='.format(key))
    print(map_df.to_string(float_format=lambda x: '%0.3f' % x))

    for key in ap_columns:
        print('\n=== {!r} ==='.format(key))
        class_df = df.sort_values(key, ascending=False)
        print(class_df.to_string(float_format=lambda x: '%0.3f' % x))


def _SCORE_FISH_YOLO2():

    ### FIXUP GIDS

    import kwcoco
    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/svm_detections.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)

    true_dataset = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json')
    true_dset = kwcoco.CocoDataset(true_dataset)

    gid_mapping = {idx: img['id'] for idx, img in enumerate(true_dset.imgs.values())}
    def _remap_image_ids(dset, gid_mapping):
        dset.index.clear()
        for img in dset.dataset['images']:
            old_gid = img['id']
            img['id'] = gid_mapping.get(old_gid, old_gid)
        for ann in dset.dataset['annotations']:
            old_gid = ann['image_id']
            ann['image_id'] = gid_mapping.get(old_gid, old_gid)
    pred_dset._build_index()
    _remap_image_ids(pred_dset, gid_mapping)
    pred_dset.fpath = ub.augpath(pred_dset.fpath, suffix='_fixed')
    pred_dset.dump(pred_dset.fpath, newlines=True)

    #### SCORE

    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/svm_detections_fixed.json')
    true_dataset = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json')

    classes_of_interest = ['flatfish', 'roundfish', 'skate']

    model_tag = 'fish_yolo_v2'
    dset_tag = basename(true_dataset)
    metrics_dpath = ub.ensuredir('tmp_metrics_fish_yolo_v2')

    ignore_classes = None
    model_tag = basename(dirname(dirname(pred_dataset)))
    dset_tag = basename(dirname(dirname(dirname(pred_dataset))))

    # expt_title = '{} {}\n{}'.format(model_tag, predcfg_tag, dset_tag,)
    expt_title = '{}\n{}'.format(model_tag, dset_tag)

    config = {'draw': True}
    from bioharn.detect_eval import CocoEvaluator  # NOQA
    coco_eval = CocoEvaluator(true_dataset, pred_dataset, config)
    coco_eval._init()
    coco_eval.evaluate(classes_of_interest, ignore_classes, expt_title, metrics_dpath)


def _SCORE_SCALLOP_TK():

    ### FIXUP GIDS

    import kwcoco
    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/scallop_tk_detections.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)

    true_dataset = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json')
    true_dset = kwcoco.CocoDataset(true_dataset)

    gid_mapping = {idx: img['id'] for idx, img in enumerate(true_dset.imgs.values())}
    def _remap_image_ids(dset, gid_mapping):
        dset.index.clear()
        for img in dset.dataset['images']:
            old_gid = img['id']
            img['id'] = gid_mapping.get(old_gid, old_gid)
        for ann in dset.dataset['annotations']:
            old_gid = ann['image_id']
            ann['image_id'] = gid_mapping.get(old_gid, old_gid)
    _remap_image_ids(pred_dset, gid_mapping)
    pred_dset._build_index()
    pred_dset.rename_categories({'live_scallop': 'scallop'})
    pred_dset.fpath = ub.augpath(pred_dset.fpath, suffix='_fixed')
    pred_dset.dump(pred_dset.fpath, newlines=True)

    #### SCORE

    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/scallop_tk_detections_fixed.json')
    true_dataset = ub.expandpath('$HOME/remote/videonas/other/projects/noaa/scallop_detections/v1/truth_scallop_only.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)

    classes_of_interest = ['scallop']

    model_tag = 'scallop_tk'
    dset_tag = basename(true_dataset)
    metrics_dpath = ub.ensuredir('tmp_metrics_scallop_tk')

    ignore_classes = None
    model_tag = basename(dirname(dirname(pred_dataset)))
    dset_tag = basename(dirname(dirname(dirname(pred_dataset))))

    # expt_title = '{} {}\n{}'.format(model_tag, predcfg_tag, dset_tag,)
    expt_title = '{}\n{}'.format(model_tag, dset_tag)

    config = {'draw': True}
    from bioharn.detect_eval import CocoEvaluator  # NOQA
    coco_eval = CocoEvaluator(true_dataset, pred_dataset, config)
    coco_eval._init()
    coco_eval.evaluate(classes_of_interest, ignore_classes, expt_title, metrics_dpath)



def _SCORE_HACKS():
    """
    from bioharn.detect_eval import *  # NOQA
    from bioharn.detect_eval import _coerce_dataset

    ### BEST RGB MODEL VIA MAP
    pred_dataset = '/home/joncrall/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/pred'
    metrics_dpath = ub.ensuredir('tmp_metrics_rgb')


    ### BEST RGBD Model
    pred_dataset = '/home/joncrall/remote/namek/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgbd-fine-coi-v42/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgbd-fine-coi-v42__epoch_00000020/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/pred'
    metrics_dpath = ub.ensuredir('tmp_metrics_rgbd')


    metrics_dpath = ub.ensuredir('tmp_metrics_svm')
    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/svm_detections.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)

    import kwcoco
    metrics_dpath = ub.ensuredir('tmp_metrics_old_cfrnn')
    pred_dataset = ub.expandpath('$HOME/remote/videonas/other/projects/noaa/scallop_detections/v1/scallop_old_cfrnn_detections.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)

    import kwcoco
    true_dataset = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json')
    scallop_dset = kwcoco.CocoDataset(true_dataset)
    scallop_dset.rename_categories({
        'live sea scallop': 'scallop',
        'swimming sea scallop': 'scallop',
        'dead sea scallop': 'scallop',
        'clapper': 'scallop',
    })
    scallop_dset.remove_categories(set(scallop_dset.name_to_cat) - {'scallop'})
    scallop_dset.fpath = ub.expandpath('$HOME/remote/videonas/other/projects/noaa/scallop_detections/v1/truth_scallop_only.json')
    scallop_dset.dump(scallop_dset.fpath)

    true_dataset = ub.expandpath('$HOME/remote/videonas/other/projects/noaa/scallop_detections/v1/truth_scallop_only.json')
    true_dset = kwcoco.CocoDataset(true_dataset)

    # HACK

    gid_mapping = {idx: img['id'] for idx, img in enumerate(true_dset.imgs.values())}
    dset = pred_dset
    def _remap_image_ids(dset, gid_mapping):
        dset.index.clear()
        for img in dset.dataset['images']:
            old_gid = img['id']
            img['id'] = gid_mapping.get(old_gid, old_gid)
        for ann in dset.dataset['annotations']:
            old_gid = ann['image_id']
            ann['image_id'] = gid_mapping.get(old_gid, old_gid)
    pred_dset._build_index()
    _remap_image_ids(pred_dset, gid_mapping)
    pred_dset.fpath = ub.augpath(pred_dset.fpath, suffix='_fixed')
    pred_dset.dump(pred_dset.fpath, newlines=True)

    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/scallop_old_cfrnn_detections_fixed.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)
    predcfg_tag = ''
    model_tag = 'old_cfrnn'
    dset_tag = basename(true_dataset)

    true_dataset = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_test.mscoco.json')

    pred_dataset = ub.expandpath('$HOME/remote/namek/remote/videonas/other/projects/noaa/scallop_detections/v1/svm_detections_fixed.json')
    pred_dset = kwcoco.CocoDataset(pred_dataset)
    predcfg_tag = ''
    model_tag = 'svm'
    dset_tag = basename(true_dataset)
    metrics_dpath = ub.ensuredir('tmp_metrics_svm')

    classes_of_interest = ['live sea scallop', 'swimming sea scallop', 'flatfish', 'clapper']
    classes_of_interest = ['live_sea_scallop', 'swimming_sea_scallop', 'flatfish', 'clapper']
    ignore_classes = None

    # expt_title = ''
    # Hack title
    predcfg_tag = basename(dirname(pred_dataset))
    model_tag = basename(dirname(dirname(pred_dataset)))
    dset_tag = basename(dirname(dirname(dirname(pred_dataset))))


    # expt_title = '{} {}\n{}'.format(model_tag, predcfg_tag, dset_tag,)
    expt_title = '{}\n{}'.format(model_tag, dset_tag)

    config = {'draw': True}
    coco_eval = CocoEvaluator(true_dataset, pred_dataset, config)
    coco_eval._init()
    coco_eval.evaluate(classes_of_interest, ignore_classes, expt_title, metrics_dpath)

                         train    vali    test
lobster                      0       1       0
didemnum                     1       1       3
monkfish                     1       1       0
misc                         2       0       1
mollusk                      3       2       2
squid                        4       1       3
rock                         6       4       9
cerianthid                  11       6       1
eel                         13       2       5
convict worm                14      13      27
dust cloud                  61      58     119
waved whelk                135     134     271
snail                      178      42     110
skate                      203     194     391
flatfish                   338     191     404
dead sea scallop           421     178     379
crab                       478     371     756
clapper                    519     183     435
seastar                   1291     318     808
roundfish                 1340    1198    2418
swimming sea scallop      1659    1554    3129
live sea scallop         53083   31307   66247


         train   vali    test
n_anns   59761   35759  75518
n_imgs   11066    7872  16278


from ndsampler import coco_sampler
sampler = coco_sampler.CocoSampler.demo()
sampler.frames._lru.set_size(2)



    """
