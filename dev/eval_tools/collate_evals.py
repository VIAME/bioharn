"""
Gather and collate evaluation metrics together in custom plots
"""


def tabulate_results2(all_single_results):
    import itertools as it
    import pandas as pd
    import kwarray

    # dups = ub.find_duplicates([(single.train_config['name'], single.epoch_num) for single in all_single_results])
    # # hack:
    # for x in sorted(ub.flatten([v[1:] for v in dups.values()]))[::-1]:
    #     del all_single_results[x]

    # x = all_single_results[2]
    # # .meta['train_info']['train_dpath']
    # y = all_single_results[12]
    # .meta['train_info']['train_dpath']

    rows = []
    for single in all_single_results:
        # if 'annotations_disp_flatfish.kwcoc' != single.meta['dset_tag']:
        #     print('cont')
        #     continue
        epoch_num = single.epoch_num
        if epoch_num is None:
            continue
        name = single.train_config['name']

        catname = 'nocls'
        measure = single.nocls_measures

        miter = it.chain([[catname, measure]], single.ovr_measures.items())
        for catname, measure in miter:

            row = {
                'name': name,
                'epoch': single.epoch_num,
                'catname': catname,
                'ap': measure['ap'],
                'auc': measure['auc'],
                'nsupport': measure['nsupport'],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    bestrows = []
    for catname, subdf in df.groupby('catname'):
        for name, subsubdf in subdf.groupby('name'):
            idx = subsubdf['ap'].argmax()
            cand = subsubdf.iloc[idx]
            bestrows.append(cand)
    best = pd.DataFrame(bestrows)
    best = best.sort_values(['catname', 'name'])

    relevant_classes = {'nocls', 'flatfish', 'live_sea_scallop'}
    relevant_flags = kwarray.isect_flags(best['catname'].values,  relevant_classes)
    relevant = best[relevant_flags].reset_index(drop=True)
    print(relevant)

    name_to_fulldf = dict(list(df.groupby('name')))

    relevant2_members = []
    for name, subdf in relevant.groupby('name'):
        # Get info for all classes for each "best" epoch
        subdf_full = name_to_fulldf[name]
        relevant_epochs = set(subdf['epoch'])
        flags1 = kwarray.isect_flags(subdf_full['epoch'].values, relevant_epochs)
        flags2 = kwarray.isect_flags(subdf_full['catname'].values,  relevant_classes)
        expanded = subdf_full[flags1 & flags2]
        relevant2_members.append(expanded)

    relevant2 = pd.concat(relevant2_members).sort_values(['catname', 'name', 'epoch']).reset_index(drop=True)
    print(relevant2)

    print(relevant2.pivot(index=['name', 'epoch'], columns='catname'))


def gather_evaluation_metrics(run_globs):
    """
    ls $HOME/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/*/eval
    """
    from os.path import join
    import glob
    import ubelt as ub
    # from kwcoco.metrics import confusion_vectors
    from kwcoco.coco_evaluator import CocoResults
    import json
    import parse
    import netharn as nh

    # run_globs = [
    #     # 'bioharn-flatfish-finetune-rgb-v21',
    #     # 'bioharn-allclass-rgb-v20',
    #     # '*-warmup*',
    #     # 'bioharn-flatfish-finetune-rgb-disp-v31',
    #     # 'bioharn-allclass-scratch-rgb-disp-v30',
    #     # 'bioharn-flatfish-finetune-rgb-v21',
    #     # 'bioharn-allclass-rgb-v20',

    #     # 'bioharn-allclass-rgb-v20',
    #     # 'bioharn-allclass-rgb-v20',
    #     # 'bioharn-allclass-partxfer-rgb-disp-v32',
    #     # 'bioharn-flatfish-finetune-rgb-disp-v33',
    #     'bioharn-allclass-partxfer-rgb-disp-cont-v34',
    # ]
    workdir = ub.expandpath('$HOME/data/dvc-repos/viame_dvc/work/bioharn')

    metric_fpaths = []
    for pat in run_globs:
        globpat = join(workdir, 'fit/runs', pat, '*/eval/*/*/*/metrics/metrics.json')
        metric_fpaths.extend(list(glob.glob(globpat)))

    all_results = []
    for fpath in ub.ProgIter(metric_fpaths, desc='load metrics'):
        with open(fpath, 'r') as file:
            data = json.load(file)
        results = CocoResults.from_json(data)
        results.metric_fpath = results
        for r in list(results.values()):
            r.metrics_fpath = fpath
        all_results.append(results)

    allkeys = set(ub.flatten(r.keys() for r in all_results))

    # Tensorboard keys of interest
    tbkeys = [
        'vali_epoch_loss',
        'vali_epoch_loss_cls_loss',
        'vali_epoch_loss_bbox_loss',
        'vali_epoch_loss_rpn_bbox_loss',
        'vali_epoch_loss_rpn_cls_loss',
        'train_epoch_loss',
        'train_epoch_loss_cls_loss',
        'train_epoch_loss_bbox_loss',
        'train_epoch_loss_rpn_bbox_loss',
        'train_epoch_loss_rpn_cls_loss',
    ]

    expt_to_single_results = {}

    for expt_key in allkeys:
        all_single_results = [r[expt_key] for r in all_results if expt_key in r]

        all_train_dpaths = {
            single.meta['train_info']['train_dpath']
            for single in all_single_results
        }
        train_dpath_to_tbscalars = {
            dpath: nh.util.read_tensorboard_scalars(dpath)
            for dpath in all_train_dpaths
        }

        for single in all_single_results:
            # if 'annotations.kwcoc' != single.meta['dset_tag']:
            #     continue
            # if 'annotations_disp_flatfish.kwcoc' != single.meta['dset_tag']:
            #     continue
            single.meta['train_info']['train_dpath']

            # hack to extract train config
            train_config = eval(single.meta['train_info']['extra']['config'], {}, {})
            deploy_fpath = single.meta['eval_config']['deployed']
            single.train_config = train_config

            result = parse.parse('{}_epoch_{num:d}.pt', deploy_fpath)
            # hack to extract train config
            train_config = eval(single.meta['train_info']['extra']['config'], {}, {})
            deploy_fpath = single.meta['eval_config']['deployed']
            single.train_config = train_config

            result = parse.parse('{}_epoch_{num:d}.pt', deploy_fpath)
            if result:
                epoch_num = result.named['num']
            else:
                # continue
                import torch_liberator
                deploy = torch_liberator.DeployedModel(deploy_fpath)
                snap = deploy.extract_snapshot()
                import netharn as nh
                state = nh.XPU('cpu').load(snap)
                epoch_num = int(state['epoch'])

            warmup_iters = train_config['warmup_iters']
            dpath = single.meta['train_info']['train_dpath']

            single.epoch_num = epoch_num
            print('deploy_fpath = {!r}'.format(deploy_fpath))

            row = {
                'name': train_config['name'],
                'epoch_num': epoch_num,
                'warmup_iters': warmup_iters,
                'flatfish_ap': single.ovr_measures['flatfish']['ap'],
                'flatfish_auc': single.ovr_measures['flatfish']['auc'],
                'nocls_ap': single.nocls_measures['ap'],
                'nocls_auc': single.nocls_measures['auc'],
            }

            # Add data from tensorboard
            try:
                tb_scalars = train_dpath_to_tbscalars[dpath]
                single.tb_scalars = tb_scalars
                for tbkey in tbkeys:
                    lossdata = tb_scalars[tbkey]
                    idx = lossdata['xdata'].index(epoch_num)
                    row[tbkey] = lossdata['ydata'][idx]
            except Exception:
                pass
            single.row = row

        expt_to_single_results[expt_key] = all_single_results
    return expt_to_single_results


def flatfish_plots():
    import pandas as pd
    # Plot AP versus epoch vs Loss
    import kwplot
    kwplot.autompl()
    import seaborn as sns
    sns.set()

    run_globs = [
        # 'bioharn-flatfish-finetune-rgb-v21',
        # 'bioharn-allclass-rgb-v20',
        # '*-warmup*',
        # 'bioharn-flatfish-finetune-rgb-disp-v31',
        # 'bioharn-allclass-scratch-rgb-disp-v30',
        # 'bioharn-flatfish-finetune-rgb-v21',
        # 'bioharn-allclass-rgb-v20',

        # 'bioharn-allclass-rgb-v20',
        # 'bioharn-allclass-rgb-v20',
        # 'bioharn-allclass-partxfer-rgb-disp-v32',
        # 'bioharn-flatfish-finetune-rgb-disp-v33',
        'bioharn-allclass-partxfer-rgb-disp-cont-v34',
        'bioharn-flatfish-finetune-rgb-v21',
        'bioharn-flatfish-finetune-rgb-disp-v33',
    ]

    expt_to_single_results = gather_evaluation_metrics(run_globs)

    for expt_key, all_single_results in expt_to_single_results.items():

        longform = []
        for single in all_single_results:
            row = single.row
            longform.append(row)

        # Expand out even futher
        longerform = []
        for row_ in longform:
            suffixes = ['ap', 'auc']
            prefixes = ['vali_epoch_loss', 'train_epoch_loss']

            for key, val in row_.items():
                for suffix in suffixes:
                    _suffix = '_' + suffix
                    if key.endswith(_suffix):
                        row = row_.copy()
                        row[suffix] = val
                        row[suffix + '_type'] = key.split(_suffix)[0]
                        longerform.append(row)

                for prefix in prefixes:
                    prefix_ = prefix + '_'
                    if key.endswith(prefix_):
                        row = row_.copy()
                        row[prefix] = val
                        row[prefix + '_type'] = key.split(prefix_)[1]
                        longerform.append(row)

        df1 = pd.DataFrame(longform)
        df2 = pd.DataFrame(longerform)

        df2['ap'].argmax()

        if 0:
            print(df1[df1.epoch_num < 10][['flatfish_ap', 'nocls_ap', 'epoch_num', 'name']])
            print(df2[df2.epoch_num < 10][['flatfish_ap', 'nocls_ap', 'epoch_num', 'name', 'ap', 'ap_type']])

        # Look at model average statistics plotted over time (epochs)
        ax1 = kwplot.figure(fnum=2, pnum=(1, 2, 1), doclf=True).gca()
        sns.lineplot(
            data=df2, x='epoch_num', y='ap', hue='name', style='ap_type',
            ax=ax1)
        ax1.set_title('AP')
        ax1.set_ylim(0, 1)

        ax2 = kwplot.figure(fnum=2, pnum=(1, 2, 2)).gca()
        sns.lineplot(
            data=df2, x='epoch_num', y='auc', hue='name', style='auc_type',
            ax=ax2)
        ax2.set_title('AUC')
        ax2.set_ylim(0, 1)

        name_to_best_epoch = {}
        for name, subdf in df1.groupby('name'):
            epoch_num = subdf.iloc[subdf['flatfish_ap'].argmax()].epoch_num
            name_to_best_epoch[name] = epoch_num

        chosen = []
        for single in all_single_results:
            name = single.train_config['name']
            target_epoch = name_to_best_epoch[name]
            if single.epoch_num == target_epoch:
                chosen.append(single)

        def measures_longform(self, name=None):
            # TODO: truncated
            from kwcoco.metrics.drawing import _realpos_label_suffix
            cfsn_keys = ['ppv', 'tpr', 'fpr']
            shortform = ub.dict_subset(self, cfsn_keys)
            mlongform = []
            label_suffix = _realpos_label_suffix(self)
            auc = self['auc']
            ap = self['ap']
            catname = self.catname
            auc_label = 'auc={:0.2f}: {} ({})'.format(auc, catname, label_suffix)
            ap_label = 'ap={:0.2f}: {} ({})'.format(ap, catname, label_suffix)
            if name is not None:
                auc_label += ' ' + name
                ap_label += ' ' + name

            for row_vals in zip(*shortform.values()):
                row = {}
                for key, val in zip(shortform.keys(), row_vals):
                    row[key] = val
                row['catname'] = self.catname
                row['auc_label'] = auc_label
                row['ap_label'] = ap_label
                mlongform.append(row)
            return mlongform

        chosen_ovr_longform = []
        for single in chosen:
            # TODO: add the above method to Measures that produces the longform
            # table.
            # self = single.ovr_measures['flatfish']
            # single.ovr_measures.draw(key='pr', ax=ax)
            name = single.train_config['name']
            if name is None:
                name = 'nocls'

            single_longform = []
            for catname, ovr in single.ovr_measures.items():
                if ovr['realpos_total'] > 0:
                    f = measures_longform(ovr, name=name + ' (epoch' + str(single.epoch_num) + ')')
                    single_longform.extend(f)

            f = measures_longform(single.nocls_measures, name=name)
            single_longform.extend(f)

            for row in single_longform:
                row['name'] = single.train_config['name'] + ' (epoch' + str(single.epoch_num) + ')'
                row['epoch_num'] = single.epoch_num
                if row['catname'] is None:
                    row['catname'] = 'nocls'

            chosen_ovr_longform.extend(single_longform)

        ovr_df = pd.DataFrame(chosen_ovr_longform)

        ax1 = kwplot.figure(fnum=4, pnum=(1, 2, 1), doclf=True).gca()
        sns.lineplot(
            data=ovr_df, x='tpr', y='ppv', hue='name', style='catname', legend='full', ax=ax1)
        ax1.set_title('PR Curves')
        ax2 = kwplot.figure(fnum=4, pnum=(1, 2, 2), doclf=False).gca()
        sns.lineplot(
            data=ovr_df, x='fpr', y='tpr', hue='name', style='catname', legend='full', ax=ax2)
        ax2.set_title('ROC Curves')
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        ax1 = kwplot.figure(fnum=5, pnum=(1, 2, 1), doclf=True).gca()
        sns.lineplot(
            data=ovr_df, x='tpr', y='ppv', hue='ap_label', legend='brief', ax=ax1)
        ax1.set_title('PR Curves')
        ax2 = kwplot.figure(fnum=5, pnum=(1, 2, 2), doclf=False).gca()
        sns.lineplot(
            data=ovr_df, x='fpr', y='tpr', hue='auc_label', legend='brief', ax=ax2)
        ax2.set_title('ROC Curves')
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        if 0:
            ax1 = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True).gca()
            sns.lineplot(
                data=df, x='epoch_num', y='flatfish_ap', hue='name', ax=ax1)
            ax1.set_title('Flatfish AP')

            ax1 = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True).gca()
            sns.lineplot(
                data=df, x='epoch_num', y='flatfish_ap', hue='name', ax=ax1)
            ax1.set_title('Flatfish AP')

            kwplot.figure(fnum=1).clf()
            ax1 = kwplot.figure(fnum=1, pnum=(1, 2, 1)).gca()
            ax2 = kwplot.figure(fnum=1, pnum=(1, 2, 2)).gca()
            sns.lineplot(data=df, x='epoch_num', y='ap', hue='name', style='ap_type', ax=ax1)
            sns.lineplot(data=df, x='epoch_num', y='auc', hue='name', style='auc_type', ax=ax2)

            kwplot.figure(fnum=1).clf()
            sns.lineplot(data=df, x='epoch_num', y='loss', hue='warmup_iters', style='loss_type')

            ax1 = kwplot.figure(fnum=1, pnum=(1, 2, 1)).gca()
            sns.lineplot(
                data=df, x='epoch_num', y='flatfish_ap', hue='warmup_iters', ax=ax1)
            ax1.set_title('AP')

            ax2 = kwplot.figure(fnum=1, pnum=(1, 2, 2)).gca()
            sns.lineplot(
                data=df, x='epoch_num', y='vali_epoch_loss_rpn_cls_loss', hue='warmup_iters', ax=ax2)
            ax2.set_title('Validation Loss')

            ax2 = kwplot.figure(fnum=1, pnum=(1, 2, 1)).gca()
            sns.lineplot(
                data=df, x='epoch_num', y='vali_epoch_loss_rpn_bbox_loss', hue='warmup_iters', ax=ax2)
            ax2.set_title('Validation Loss')


def check_complementaryness():
    """
    Find a case where one model detects something but the other doesn't
    """
    run_globs = [
        'bioharn-flatfish-finetune-rgb-v21',
        'bioharn-flatfish-finetune-rgb-disp-v33',
    ]
    expt_to_single_results = gather_evaluation_metrics(run_globs)

    import kwplot

    legend = kwplot.make_legend_img({
        'truth': 'green',
        'disp-v33': 'orange',
        'rgb-v21': 'purple',
    })
    legend = kwimage.ensure_float01(legend)

    for expt_key, all_single_results in expt_to_single_results.items():

        best_singles = {}
        for single in all_single_results:
            name = single.train_config['name']
            curr = best_singles.get(name, None)
            if curr is None or curr.row['flatfish_ap'] < single.row['flatfish_ap']:
                curr = single
            best_singles[name] = curr

        # Hack to find miss vs hit
        from os.path import dirname, join
        name_to_preds = {}
        for name, single in best_singles.items():
            pred_dpath = join(dirname(dirname(single.metrics_fpath)), 'pred')
            name_to_preds[name] = pred_dpath

        name_to_results = {}
        name_to_evaler = {}
        for name, pred_dpath in name_to_preds.items():
            # HACK FOR TRUTH PATH
            truth_fpath = '/home/khq.kitware.com/jon.crall/data/dvc-repos/viame_dvc/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_disp_flatfish.kwcoco.json'
            # import kwcoco
            # truth = kwcoco.CocoDataset(truth_fpath)
            from kwcoco import coco_evaluator
            coco_eval = coco_evaluator.CocoEvaluator({
                'true_dataset': truth_fpath,
                'pred_dataset': pred_dpath,
            })
            results = coco_eval.evaluate()
            main_results = results[expt_key]
            name_to_results[name] = main_results
            name_to_evaler[name] = coco_eval

        res1, res2 = name_to_results.values()
        evaler1, evaler2 = name_to_evaler.values()

        true_dset2 = evaler2.true_extra['coco_dset']
        true_dset1 = evaler1.true_extra['coco_dset']

        res1.cfsn_vecs
        res2.cfsn_vecs

        ff_idx1 = res1.cfsn_vecs.classes.node_to_idx['flatfish']
        ff_cid1 = res1.cfsn_vecs.classes.node_to_id['flatfish']
        ff_idx2 = res2.cfsn_vecs.classes.node_to_idx['flatfish']
        ff_cid2 = res2.cfsn_vecs.classes.node_to_id['flatfish']

        df1 = res1.cfsn_vecs.data.pandas()
        df2 = res2.cfsn_vecs.data.pandas()

        cases1 = df1[df1['true'] == ff_idx1]
        cases2 = df2[df2['true'] == ff_idx2]

        cases2 = cases2.sort_values('score')
        cases1 = cases1.sort_values('score')

        missed1 = cases2.gid[cases2['score'] < 0]
        missed2 = cases1.gid[cases1['score'] < 0]

        gids = sorted(set(missed2) - set(missed1))
        set(missed1) - set(missed2)

        gids = cases2.gids.values
        gid = gids[12]

        det2 = evaler2.gid_to_pred[gid]
        det1 = evaler1.gid_to_pred[gid]

        canvas = true_dset2.load_image(gid)
        gpath = true_dset2.get_image_fpath(gid)
        title = '/'.join(gpath.split('/')[-3:])

        true_det = true_dset2.annots(gid=gid).detections

        canvas = true_det.draw_on(canvas, color='green')

        # det2 = det2.compress(det2.scores > 0.6)
        # det1 = det1.compress(det1.scores > 0.6)

        canvas = det2.draw_on(canvas, color='orange')
        canvas = det1.draw_on(canvas, color='purple')

        import kwimage
        import numpy as np
        legend_transparent = np.zeros_like(canvas)
        legend_transparent = kwimage.ensure_alpha_channel(legend_transparent, alpha=0)
        legend_transparent[0:legend.shape[0], 0:legend.shape[1], 0:3] = legend
        legend_transparent[0:legend.shape[0], 0:legend.shape[1], 3] = 1
        canvas = kwimage.overlay_alpha_layers([canvas, legend_transparent])
        canvas = kwimage.overlay_alpha_layers([legend_transparent, canvas])

        import kwplot
        kwplot.imshow(canvas, title=title)



