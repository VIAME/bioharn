import ubelt as ub
import datetime
from pathlib import Path
# import pandas as pd
lines = ub.codeblock(
    '''
    # On namek I have
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v004/emrxfdav/deploy_ClfModel_emrxfdav_024_HUEOJO.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v005/fbrsggtn/deploy_ClfModel_fbrsggtn_000_RDPXVE.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v005/mwgqunsc/deploy_ClfModel_mwgqunsc_093_BBHBJV.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v006/lnhmnzai/deploy_ClfModel_lnhmnzai_011_IWKGTO.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v007/jhuqvxvl/deploy_ClfModel_jhuqvxvl_036_YLCYPG.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v008/davyvtss/deploy_ClfModel_davyvtss_048_BNFEFI.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v001/nrorbmcb/deploy_ClfModel_nrorbmcb_051_UFCIUU.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v003/scxnyjlc/deploy_ClfModel_scxnyjlc_000_GDERNV.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-coi-v1/jmvgwufh/deploy_MM_HRNetV2_w18_MaskRCNN_jmvgwufh_046_NGQMNZ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-coi-v1/rondpovs/deploy_MM_HRNetV2_w18_MaskRCNN_rondpovs_000_LUTDBR.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v2/bxytlhok/deploy_MM_HRNetV2_w18_MaskRCNN_bxytlhok_005_MZRCFD.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v2/rwcfnlgm/deploy_MM_HRNetV2_w18_MaskRCNN_rwcfnlgm_000_QRCMYT.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v4/sydbgnxj/deploy_MM_HRNetV2_w18_MaskRCNN_sydbgnxj_023_RLQZTH.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/lzwtqdrz/deploy_MM_HRNetV2_w18_MaskRCNN_lzwtqdrz_000_TAXAGV.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/odcvldiy/deploy_MM_HRNetV2_w18_MaskRCNN_odcvldiy_031_EUEOKJ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/pwgigbqa/deploy_MM_HRNetV2_w18_MaskRCNN_pwgigbqa_000_HZNLXY.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-disp-habcam-v7/sjgsnsan/deploy_MM_HRNetV2_w18_MaskRCNN_sjgsnsan_000_SBAYPR.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v3/edingcsl/deploy_MM_HRNetV2_w18_MaskRCNN_edingcsl_007_BAQMRH.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5-adapt/udmzrkmb/deploy_MM_HRNetV2_w18_MaskRCNN_udmzrkmb_003_FKJTWB.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v5/bcifnsvt/deploy_MM_HRNetV2_w18_MaskRCNN_bcifnsvt_029_KYHMWC.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-habcam-v6/hldsgogn/deploy_MM_HRNetV2_w18_MaskRCNN_hldsgogn_032_FURHIT.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-coi-v46/nngqryeh/deploy_MM_CascadeRCNN_nngqryeh_031_RVJZKO.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-coi-v46/soswhwmm/deploy_MM_HRNetV2_w18_MaskRCNN_soswhwmm_000_BDTIXJ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-coi-v42_valitune/syavjxrl/deploy_MM_CascadeRCNN_syavjxrl_005_BQFABN.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v41/ufkqjjuk/deploy_MM_CascadeRCNN_ufkqjjuk_016_UJJNDR.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v42/nfmnvqwq/deploy_MM_CascadeRCNN_nfmnvqwq_027_HGILPB.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v45/jsghbnij/deploy_MM_CascadeRCNN_jsghbnij_059_SXQKRF.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44/gvizryca/deploy_MM_CascadeRCNN_gvizryca_004_BBHIGU.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-flatfish-only-v44_valitune/mwctkynp/deploy_MM_CascadeRCNN_mwctkynp_005_MIMRDY.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v20/isipxrwf/deploy_MM_CascadeRCNN_isipxrwf_020_PGPMEW.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v21/lpsblbic/deploy_MM_CascadeRCNN_lpsblbic_048_UBJNAA.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v23/kmiqxzis/deploy_MM_CascadeRCNN_kmiqxzis_022_NROXRC.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/deploy_MM_CascadeRCNN_brekugqz_017_PHHQVT.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-fine-coi-v44/ovxflcrh/deploy_MM_CascadeRCNN_ovxflcrh_055_QTYTBG.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-flatfish-only-v45/pmyekaag/deploy_MM_CascadeRCNN_pmyekaag_000_BBASPJ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v27/dxziuzrv/deploy_MM_CascadeRCNN_dxziuzrv_019_GQDHOF.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v29-balanced/gjxbpiei/deploy_MM_CascadeRCNN_gjxbpiei_002_LDATFJ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/etvvhzni/deploy_MM_CascadeRCNN_etvvhzni_007_IPEIQA_multiclass.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/etvvhzni/deploy_MM_CascadeRCNN_etvvhzni_007_IPEIQA.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/mzhfneiy/deploy_MM_CascadeRCNN_mzhfneiy_000_IFTJDL.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/ogloumea/deploy_MM_CascadeRCNN_ogloumea_000_RUYLNO.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v31-bigger-balanced/uemfecck/deploy_MM_CascadeRCNN_uemfecck_000_OTFVYV.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v10-test-retinanet/daodqsmy/deploy_MM_RetinaNet_daodqsmy_010_QRNNNW.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/ovphtcvk/deploy_MM_CascadeRCNN_ovphtcvk_037_HZUJKO.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v12-test-retinanet/mrepnniz/deploy_MM_RetinaNet_mrepnniz_094_ODCGUT.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/deploy_MM_CascadeRCNN_ogenzvgt_059_QBGWCT.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v14-cascade/iawztlag/deploy_MM_CascadeRCNN_iawztlag_032_ETMZBH.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v16-cascade/hvayxfyx/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v6-test-retinanet/rioggtso/deploy_MM_RetinaNet_rioggtso_050_MLFGKZ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v8-test-retinanet/opgoqmpg/deploy_MM_RetinaNet_opgoqmpg_000_MKJZNW.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-det-v9-test-cascade/zjolejwz/deploy_MM_CascadeRCNN_zjolejwz_010_LUAKQJ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn_shapes_example3/bfuqgqpw/deploy_Yolo2_bfuqgqpw_000_HFVBON.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn-test-yolo-v5/sxfhhhwy/deploy_Yolo2_sxfhhhwy_002_QTVZHQ.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn_voc_example/fkduivnf/deploy_Yolo2_fkduivnf_000_GQNUQG.zip
    ~/remote/namek/work/bioharn/fit/runs/bioharn_voc_example/tfwqdvzg/deploy_Yolo2_tfwqdvzg_018_PPRKFG.zip
    ~/remote/namek/work/bioharn/fit/runs/DEMO_bioharn-det-v13-cascade/ogenzvgt/deploy_MM_CascadeRCNN_ogenzvgt_006_IQLOXO.zip
    ~/remote/namek/work/bioharn/fit/runs/untitled/gqhupaqk/deploy_MM_CascadeRCNN_gqhupaqk_004_CIVNPB.zip
    ~/remote/namek/work/bioharn/fit/runs/validate_demo/imgsoogc/deploy_MM_HRNetV2_w18_MaskRCNN_imgsoogc_000_CAKBMT.zip

    On the VIAME server I have: 
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-coi-v43_valitune/nfnqoxuu/deploy_MM_CascadeRCNN_nfnqoxuu_005_FPSVAR.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-fine-coi-v40/ntjzrxlb/deploy_MM_CascadeRCNN_ntjzrxlb_007_FVMWBU.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-fine-coi-v40/ntjzrxlb/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-fine-coi-v43/bvbvdplp/deploy_MM_CascadeRCNN_bvbvdplp_006_TXIDOF.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v22/rfdrszqa/deploy_MM_CascadeRCNN_rfdrszqa_048_DZYTDJ.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v24/ddoxsxjs/deploy_MM_CascadeRCNN_ddoxsxjs_048_QWTOJP.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v24/ddoxsxjs/deploy_MM_CascadeRCNN_ddoxsxjs_078_FJXQLY.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v31-bigger-balanced/moskmhld/deploy_MM_CascadeRCNN_moskmhld_015_SVBZIV.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-v18-cascade-mc-disp/uejaxygd/deploy_MM_CascadeRCNN_uejaxygd_046_JQSUFX.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-det-v19-cascade-mc-rgb/pqntuaya/deploy_MM_CascadeRCNN_pqntuaya_024_GLKWTT.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-flatfish-rgb-v10/foqjrwrr/deploy_MM_HRNetV2_w18_MaskRCNN_foqjrwrr_000_RJJNRV.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-flatfish-rgb-v10/svytnbjg/deploy_MM_HRNetV2_w18_MaskRCNN_svytnbjg_016_MYFSVM.zip
    ~/remote/viame/work/bioharn/fit/runs/bioharn-flatfish-rgb-v11/kqlgozei/deploy_MM_HRNetV2_w18_MaskRCNN_kqlgozei_003_MSOUGL.zip


    On Numenor I have: 
    ~/remote/numenor/work/bioharn/fit/runs/bioharn-fish-hr18-rgb-motion-v3/jrekfqik/deploy_bioharn-fish-hr18-rgb-motion-v3_jrekfqik_007_OJALSJ.zip
    ~/remote/numenor/work/bioharn/fit/runs/bioharn-fish-hrmask18-rgb-motion-v1/qgcndlju/deploy_bioharn-fish-hrmask18-rgb-motion-v1_qgcndlju_009_OIWZLX.zip
    ~/remote/numenor/work/bioharn/fit/runs/bioharn-fish-hrmask18-rgb-motion-v2/bwfnoeym/deploy_bioharn-fish-hrmask18-rgb-motion-v2_bwfnoeym_010_OREGPS.zip
    ~/remote/numenor/work/bioharn/fit/runs/test-basic-fullframe-clf/gtlwdhmv/deploy_test-basic-fullframe-clf_gtlwdhmv_013_TEKZIT.zip
    ~/remote/numenor/work/bioharn/fit/runs/test-basic-fullframe-clf-v2/shanlpep/deploy_test-basic-fullframe-clf-v2_shanlpep_000_QNGXYV.zip
    ~/remote/numenor/work/bioharn/fit/runs/test-basic-fullframe-clf-v4/hotylmxb/deploy_test-basic-fullframe-clf-v4_hotylmxb_007_PGGLST.zip
    ~/remote/numenor/work/bioharn/fit/runs/test-basic-fullframe-clf-v5/djxnjdoh/deploy_test-basic-fullframe-clf-v5_djxnjdoh_007_DGDGBI.zip
    ''')


deployed_fpaths = []
for line in ub.ProgIter(lines.split('\n')):
    line = line.strip()
    if not line.startswith('~'):
        continue
    fpath = ub.expandpath(line)
    deployed_fpaths.append(fpath)

    lines.split('\n')


@ub.memoize
def _memo_info(fpath):
    import torch_liberator
    deployed = torch_liberator.DeployedModel.coerce(fpath)
    info = deployed.train_info()
    return info


@ub.memoize
def _memo_exists(path):
    return path.exists()


@ub.memoize
def _memo_stat(path):
    return path.stat()


rows = []
for fpath in ub.ProgIter(deployed_fpaths, verbose=3):
    info = _memo_info(fpath)

    train_dpath = Path(fpath).parent

    import kwcoco
    cats = kwcoco.CategoryTree.coerce(info['hyper']['model'][1]['classes'])
    train_fpath = ub.argval('--train_dataset', argv=info['argv'])

    init_type, init_params = info['hyper']['initializer']
    if init_type.endswith('Pretrained'):
        starting_point = init_params['fpath']
        init_data = starting_point
    else:
        init_data = 'scratch'

    if any(['scallop' in c for c in cats]):

        eval_dpath = train_dpath / 'eval'
        has_eval = _memo_exists(eval_dpath)
        has_metrics = False
        metrics_cand = None
        # if has_eval:
        #     # after eval is / model_cfg / data_cfg / pred_cfg / metrics / *.json
        #     metrics_cand = list(eval_dpath.glob('*/*/*/metrics/*.json'))
        #     if metrics_cand:
        #         has_metrics = True

        try:
            epoch =  int(fpath.split('_')[-2])
        except Exception:
            epoch = None

        mtime = datetime.datetime.fromtimestamp(_memo_stat(Path(fpath)).st_mtime)

        row = {
            'name': Path(fpath).stem,
            'epoch': epoch,
            'total_epoch': None,
            'mtime': mtime,
            'model_path': fpath,
            'has_eval': has_eval,
            # 'has_metrics': has_metrics,
            # 'metrics_cand': metrics_cand,
            'cats': list(cats),
            'init': init_data,
            'train_fpath': train_fpath,
        }
        if row['init'] == 'scratch' and row['epoch'] is not None and row['epoch'] < 5:
            continue
        rows.append(row)
        print('row = {}'.format(ub.repr2(row, nl=1)))


lut = {row['name']: row for row in rows}


def none_to_num(x):
    if x is None:
        return 0
    else:
        return x


for row in rows:
    if row['init'] != 'scratch':
        total_epochs = none_to_num(row['epoch'])
        history = []
        prev = Path(row['init']).stem
        while prev in lut:
            prev_row = lut[prev]
            prev_epochs = none_to_num(prev_row['epoch'])
            total_epochs += prev_epochs
            history.append(prev)
            prev = Path(prev_row['init']).stem
        history.append(prev)
        row['history'] = history
        row['total_epoch'] = total_epochs
    else:
        row['total_epoch'] = row['epoch']


rows = sorted(rows, key=lambda x: none_to_num(x.get('total_epoch', -1)))
print('rows = {}'.format(ub.repr2(rows, nl=2, sort=0)))

cats_to_group = ub.group_items(rows, lambda x: tuple(sorted(x['cats'])))
print(ub.repr2(ub.map_vals(len, cats_to_group)))
# print('cats_to_group = {}'.format(ub.repr2(cats_to_group, nl=3)))

# for row in rows:
#     if row['metrics_cand']:
#         if len(row['metrics_cand']) > 1:
#             print(len(row['metrics_cand']))
#             for c in row['metrics_cand']:
#                 print('c = {!r}'.format(c))
#             from kwcoco.coco_evaluator import CocoResults
#             print('row = {}'.format(ub.repr2(row, nl=1)))

# df = pd.DataFrame(rows)
# df.train_fpath.unique()
