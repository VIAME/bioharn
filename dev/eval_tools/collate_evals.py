"""
Gather and collate evaluation metrics together in custom plots
"""


def gather_evaluation_metrics():
    """
    ls $HOME/data/dvc-repos/viame_dvc/work/bioharn/fit/runs/*-warmup*/*/eval
    """
    from os.path import join
    import glob
    import ubelt as ub
    from kwcoco.metrics import confusion_vectors
    from kwcoco.coco_evaluator import CocoResults
    import json
    import parse

    run_globs = [
        'bioharn-allclass-rgb-v20',
        '*-warmup*'
    ]
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
        all_results.append(results)

    allkeys = set(ub.flatten(r.keys() for r in all_results))

    # Plot AP versus epoch vs Loss
    import kwplot
    kwplot.autompl()
    import seaborn as sns
    sns.set()

    import netharn as nh
    import pandas as pd

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

        longform = []

        for single in all_single_results:
            single.meta['train_info']['train_dpath']

            # hack to extract train config
            train_config = eval(single.meta['train_info']['extra']['config'], {}, {})
            deploy_fpath = single.meta['eval_config']['deployed']
            single.train_config = train_config

            result = parse.parse('{}_epoch_{num:d}.pt', deploy_fpath)
            if result:
                epoch_num = result.named['num']
                warmup_iters = train_config['warmup_iters']
                dpath = single.meta['train_info']['train_dpath']

                single.epoch_num = epoch_num

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
                tb_scalars = train_dpath_to_tbscalars[dpath]
                single.tb_scalars = tb_scalars
                for tbkey in tbkeys:
                    lossdata = tb_scalars[tbkey]
                    idx = lossdata['xdata'].index(epoch_num)
                    row[tbkey] = lossdata['ydata'][idx]

                longform.append(row)
            else:
                single.epoch_num = None


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

        ax1 = kwplot.figure(fnum=1, pnum=(1, 1, 1), doclf=True).gca()
        sns.lineplot(
            data=df1, x='epoch_num', y='flatfish_ap', hue='name', ax=ax1)
        ax1.set_title('Flatfish AP')

        ax1 = kwplot.figure(fnum=2, pnum=(1, 2, 1), doclf=True).gca()
        sns.lineplot(
            data=df2, x='epoch_num', y='ap', hue='name', style='ap_type',
            ax=ax1)
        ax1.set_title('AP')

        ax2 = kwplot.figure(fnum=2, pnum=(1, 2, 2)).gca()
        sns.lineplot(
            data=df2, x='epoch_num', y='auc', hue='name', style='auc_type',
            ax=ax2)
        ax2.set_title('AUC')

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
            # TODO: add to Measures
            # self = single.ovr_measures['flatfish']
            # single.ovr_measures.draw(key='pr', ax=ax)
            name = single.train_config['name']
            if name is None:
                name = 'nocls'

            single_longform = []
            for catname, ovr in single.ovr_measures.items():
                if ovr['realpos_total'] > 0:
                    f = measures_longform(ovr, name=name + ' (' + str(single.epoch_num) + ')')
                    single_longform.extend(f)

            f = measures_longform(single.nocls_measures, name=name)
            single_longform.extend(f)

            for row in single_longform:
                row['name'] = single.train_config['name'] + ' ' + str(single.epoch_num)
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

        ax1 = kwplot.figure(fnum=5, pnum=(1, 2, 1), doclf=True).gca()
        sns.lineplot(
            data=ovr_df, x='tpr', y='ppv', hue='ap_label', legend='brief', ax=ax1)
        ax1.set_title('PR Curves')
        ax2 = kwplot.figure(fnum=5, pnum=(1, 2, 2), doclf=False).gca()
        sns.lineplot(
            data=ovr_df, x='fpr', y='tpr', hue='auc_label', legend='brief', ax=ax2)
        ax2.set_title('ROC Curves')

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
