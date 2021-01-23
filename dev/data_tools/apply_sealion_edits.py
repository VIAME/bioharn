from os.path import basename
import numpy as np
from os.path import join
import pandas as pd
import ubelt as ub


def drop_2020_06_14():
    """
    # Download edits


    girder-client --api-url https://data.kitware.com/api/v1 list 5ee6a5149014a6d84ec02d66
    girder-client --api-url https://data.kitware.com/api/v1 download 5eea3dea9014a6d84ec78d09 $HOME/work/bioharn/_cache/sealion_edits/detections_edits_2011_v2.csv
    girder-client --api-url https://data.kitware.com/api/v1 download 5eea3de79014a6d84ec78d01 $HOME/work/bioharn/_cache/sealion_edits/detections_edits_2015_v2.csv
    girder-client --api-url https://data.kitware.com/api/v1 download 5eea41a29014a6d84ec78f1b $HOME/work/bioharn/_cache/sealion_edits/detections_edits_2014.csv


    girder-client --api-url https://data.kitware.com/api/v1 download 5ee6a5149014a6d84ec02d66 $HOME/work/bioharn/_cache/sealion_edits
    ls $HOME/work/bioharn/_cache/sealion_edits

    cd $HOME/work/bioharn/_cache/sealion_edits
    detections_edits_2011.csv  detections_edits_2015.csv  detections_edits_2016.csv  detections_errors_2011.xlsx

    """
    dl_root = ub.expandpath('$HOME/work/bioharn/_cache/sealion_edits')
    csv_fpaths = [
        join(dl_root, 'detections_edits_2011_v2.csv'),
        join(dl_root, 'detections_edits_2014.csv'),
        join(dl_root, 'detections_edits_2015_v2.csv'),
        join(dl_root, 'detections_edits_2016.csv'),
    ]

    df_lut = {}
    for csv_fpath in csv_fpaths:

        with open(csv_fpath) as file:
            print('csv_fpath = {!r}'.format(csv_fpath))
            lines = file.read().split('\n')
            nrows_set = set()
            bad_rows = []
            for rx, line in enumerate(lines, start=0):
                if line:
                    nrows = line.count(',')
                    nrows_set.add(nrows)
                    if nrows < 10:
                        bad_rows.append(rx)
                        print('rx = {!r}'.format(rx))
                        print(line)
                        print('nrows = {!r}'.format(nrows))
            print(nrows_set)

        columns = ['_aid', 'gname', '_gid', 'tl_x', 'tl_y', 'br_x', 'br_y', '7', '8', 'category', '10']
        df = pd.read_csv(csv_fpath, header=None, names=columns)
        assert not bad_rows
        if bad_rows:
            df.iloc[bad_rows]
            # Drop bad rows
            df = df[~pd.isnull(df['category'])]

        year = basename(csv_fpath).split('_')[2].split('.')[0]
        df['year'] = year
        df_lut[year] = df

    if 0:
        finalized_image_ranges = {
            '2011': (1, 530),    # 1-based as reported
            '2014': (218, 443),  # 1-based as reported
            '2015': (101, 528),  # 1-based as reported
            '2016': (331, 413),  # 1-based as reported
        }

        edited_dfs = {}
        for year, one_based_range in finalized_image_ranges.items():
            start, stop = np.array(one_based_range) - 1
            df = df_lut[year]
            flags = (df['_gid'] >= start) & (df['_gid'] <= stop)
            subdf = df[flags]
            edited_dfs[year] = subdf

        df = pd.concat(list(edited_dfs.values()))
    else:
        # Just use everything
        df = pd.concat(list(df_lut.values()))

    df['br_x'] = df['br_x'].astype(np.float)
    df['br_y'] = df['br_y'].astype(np.float)
    df['tl_y'] = df['tl_y'].astype(np.float)
    df['tl_x'] = df['tl_x'].astype(np.float)
    df['_gid'] = df['_gid'].astype(np.int)
    df['_aid'] = df['_aid'].astype(np.int)
    df_drop2 = df

    df = df_drop2
    base_fpath = '/home/joncrall/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6.mscoco.json'
    drop_source = 'refinement-2020-06-17'

    out_fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8.mscoco.json')
    return base_fpath, df, drop_source, out_fpath

def _s():
    if 0:
        a = df_drop1[df_drop1['year'] == '2014']
        b = df_drop2[df_drop2['year'] == '2014']
        set(a['gname']) | set(b['gname'])
        set(a['_gid']) | set(b['_gid'])


def drop_2020_03_18():
    base_fpath = '/home/joncrall/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v5.mscoco.json'

    csv_fpath1 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2014edit_img217.csv')
    csv_fpath2 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2015edit_img100.csv')
    csv_fpath3 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2016edit_img330.csv')

    columns = ['_aid', 'gname', '_gid', 'tl_x', 'tl_y', 'br_x', 'br_y', '7', '8', 'category', '10']
    df1 = pd.read_csv(csv_fpath1, header=None, names=columns)
    df1['year'] = '2014'
    df2 = pd.read_csv(csv_fpath2, header=None, names=columns)
    df2['year'] = '2015'
    df3 = pd.read_csv(csv_fpath3, header=None, names=columns)
    df3['year'] = '2016'
    df_drop1 = pd.concat([df1, df2, df3])

    drop_source = 'refinement-2020-03-18'
    df = df_drop1

    out_fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6.mscoco.json')
    return base_fpath, df, drop_source, out_fpath


def main():
    """
    Applies refinements to the sealion ground truth

    rsync -avpPR viame:data/./US_ALASKA_MML_SEALION/ $HOME/data
    rsync $HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6.mscoco.json viame:data/US_ALASKA_MML_SEALION/
    rsync $HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6_vali.mscoco.json viame:data/US_ALASKA_MML_SEALION/
    rsync $HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6_train.mscoco.json viame:data/US_ALASKA_MML_SEALION/

    Need to do special case for gids: 3562
    """
    import kwarray
    import kwimage
    import ndsampler
    import ubelt as ub
    import numpy as np
    from os.path import basename

    # Load the edits we are going to apply
    base_fpath, df, drop_source, out_fpath = drop_2020_06_14()

    # Load the COCO dataset we are goint to modify
    print('base_fpath = {!r}'.format(base_fpath))
    orig_coco_dset = ndsampler.CocoDataset(base_fpath)
    coco_dset = orig_coco_dset.copy()
    coco_dset._resolve_to_cat('Dead Pup')['alias'] = ['DeadPup', 'DeadNP']
    coco_dset.ensure_category('Ignore')
    coco_dset.ensure_category('furseal')
    coco_dset.ensure_category('unknown')
    if 'keypoint_categories' not in coco_dset.dataset:
        kp_names = set()
        for ann in coco_dset.anns.values():
            if 'keypoints' in ann:
                kpts = ann.get('keypoints', None)
                if isinstance(kpts, list):
                    for kp in kpts:
                        if 'category_name' in kp:
                            kp_names.add(kp['category_name'])
        coco_dset.dataset['keypoint_categories'] = [{
            'id': id,
            'name': name,
        } for id, name in enumerate(sorted(kp_names), start=1)]
        coco_dset._build_index()
    print('coco_dset = {!r}'.format(coco_dset))

    if 0:
        box_stats = coco_dset.boxsize_stats(anchors=4, verbose=1, clusterkw={'verbose': 0})
        print('box_stats = {}'.format(ub.repr2(box_stats, nl=4)))

    # Use image names to associate annotations between CSV and COCO files
    gid_to_basename = {img['id']: basename(img['file_name'])
                       for img in coco_dset.imgs.values()}
    basename_to_gids = ub.invert_dict(gid_to_basename, unique_vals=False)
    has_dups = max(map(len, basename_to_gids.values())) > 1
    assert not has_dups
    basename_to_gid = ub.map_vals(ub.peek, basename_to_gids)

    # '20150701' in gname_to_subdf
    # gname_to_subdf['20150701'].pandas()
    # gid_to_subdf.pop(3562, None)

    df_light = kwarray.DataFrameLight.from_pandas(df)

    gname_to_subdf = dict(df_light.groupby('gname'))
    gid_to_subdf = ub.map_keys(basename_to_gid, gname_to_subdf)

    # For each image, determine what should be modified
    stats = ub.ddict(list)

    toadd_anns = []
    tomodify_anns = []
    toremove_anns = []

    apply_timestamp = ub.timestamp()
    summaries = []

    for gid in ub.ProgIter(list(gid_to_subdf.keys()), desc='update images'):
        # Load the original COCO annotations
        orig_aids = sorted(coco_dset.gid_to_aids[gid])
        orig_anns = list(ub.take(coco_dset.anns, orig_aids))
        orig_annots = coco_dset.annots(orig_aids)
        orig_catnames = orig_annots.cnames
        orig_xywh = kwimage.Boxes(orig_annots.lookup('bbox'), 'xywh')
        self = orig_annots
        anns = [self._id_to_obj[aid] for aid in self.aids]
        orig_dets = kwimage.Detections.from_coco_annots(anns, dset=self._dset)

        # Load the edited CSV annotations
        subdf = gid_to_subdf[gid]
        sub_tlbr = subdf._getcols(['tl_x', 'tl_y', 'br_x', 'br_y'])
        sub_xywh = kwimage.Boxes(sub_tlbr, 'tlbr').to_xywh()
        sub_catnames = subdf['category']
        sub_catnames = ['unknown' if isinstance(n, float) and np.isnan(n)
                        else n for n in sub_catnames]
        sub_cats = [coco_dset._alias_to_cat(n) for n in sub_catnames]
        sub_catnames = [cat['name'] for cat in sub_cats]
        sub_cids = [cat['id'] for cat in sub_cats]
        sub_cidxs = list(ub.take(orig_dets.classes.id_to_idx, sub_cids))

        dets1 = orig_dets
        dets2 = kwimage.Detections(
            boxes=sub_xywh,
            class_idxs=sub_cidxs,
            classes=orig_dets.classes,
        )

        status, info, details = detection_delta(dets1, dets2)
        summaries.append(info)
        stats[status].append(gid)

        # Construct coco deltas that can be applied
        if status != 'same':
            add_msg = 'created: {}'.format(apply_timestamp)
            mod_msg = 'modified: {}'.format(apply_timestamp)

            # Mark annotations for addition
            add_dsets = dets2.take(details['add_idxs2'])
            add_anns2 = list(add_dsets.to_coco(
                style='new', dset=coco_dset, image_id=gid))
            for ann in add_anns2:
                ann['box_source'] = drop_source
                ann['changelog'] = [add_msg]
                toadd_anns.append(ann)

            # Mark annotations for removal
            remove_anns = list(ub.take(orig_anns, details['remove_idxs1']))
            toremove_anns.extend(remove_anns)

            # Mark annotaitons for modification
            modified_dets = dets2.take(details['modify_idxs2'])
            old_anns1 = list(ub.take(orig_anns, details['modify_idxs1']))
            delta_anns2 = list(modified_dets.to_coco(
                style='new', dset=coco_dset, image_id=gid))
            for old_ann, delta_ann in zip(old_anns1, delta_anns2):
                new_ann = old_ann.copy()
                new_ann.update(delta_ann)
                new_ann['box_source'] = drop_source
                new_ann['changelog'] = old_ann.get('changelog', []) + [mod_msg]
                tomodify_anns.append(new_ann)

    if 1:
        basic_sum = sum([pd.Series(info['basic']) for info in summaries])
        print('basic_sum =\n{!r}'.format(basic_sum))

        add_accum = ub.ddict(lambda: 0)
        for info in summaries:
            for k, v in info['added_catfreq'].items():
                add_accum[k] += v
        add_accum = ub.sorted_vals(add_accum)
        print('add_accum = {}'.format(ub.repr2(add_accum, nl=1)))

        remove_accum = ub.ddict(lambda: 0)
        for info in summaries:
            for k, v in info['removed_catfreq'].items():
                remove_accum[k] += v
        remove_accum = ub.sorted_vals(remove_accum)
        print('remove_accum = {}'.format(ub.repr2(remove_accum, nl=1)))

        modified_accum = ub.ddict(lambda: 0)
        for info in summaries:
            for k, v in info['modified_catfreq'].items():
                modified_accum[k] += v
        modified_accum = ub.sorted_vals(modified_accum)
        print('modified_accum = {}'.format(ub.repr2(modified_accum, nl=1)))

        coarsemodified_accum = ub.ddict(lambda: 0)
        for info in summaries:
            for k, v in info['modified_catfreq'].items():
                coarsemodified_accum[k[0]] += v
        coarsemodified_accum = ub.sorted_vals(coarsemodified_accum)
        print('coarsemodified_accum = {}'.format(ub.repr2(coarsemodified_accum, nl=1)))

    if __debug__:
        x = [ann['id'] for ann in toremove_anns]
        y = [ann['id'] for ann in tomodify_anns]
        assert not (set(x) & set(y))
        assert all('id' not in ann for ann in toadd_anns)

    # apply all changes
    toremove_aids = [ann['id'] for ann in toremove_anns]
    coco_dset.remove_annotations(toremove_aids)

    for new_ann in tomodify_anns:
        # Override all new properties, but leave old ones in tact
        old_ann = coco_dset.anns[new_ann['id']]
        old_ann.update(new_ann)

    for ann in toadd_anns:
        coco_dset.add_annotation(**ann)

    print('Modification summary: {}'.format(ub.map_vals(len, stats)))
    # partial: {'diff': 1152, 'adjust': 115}

    # full: {'diff': 1152, 'adjust': 190, 'same': 567, 'diffcats': 5}

    if False:
        # Debugging visualizations
        import kwplot
        kwplot.autompl()
        # gid = 2544
        # gid = 2404
        # gid = 2434
        # gids = kwarray.shuffle(list(gid_to_subdf.keys()))
        gids = sorted(stats['same'])

        gid = gids[0]
        det_before = orig_coco_dset.annots(gid=gid).detections
        det_after = coco_dset.annots(gid=gid).detections

        canvas = coco_dset.load_image(gid)
        det_before.draw_on(canvas, color='blue')
        det_after.draw_on(canvas, color='green')

        kwplot.imshow(canvas)

        import xdev
        for gid in xdev.InteractiveIter(gids):
            subdf = gid_to_subdf[gid]
            orig_aids = coco_dset.gid_to_aids[gid]
            orig_anns = list(ub.take(coco_dset.anns, orig_aids))

            orig_catnames = [coco_dset.cats[ann['category_id']]['name']
                             for ann in orig_anns]
            orig_xywh = kwimage.Boxes([ann['bbox'] for ann in orig_anns], 'xywh')

            sub_tlbr = subdf._getcols(['tl_x', 'tl_y', 'br_x', 'br_y'])
            sub_xywh = kwimage.Boxes(sub_tlbr, 'tlbr').to_xywh()
            sub_catnames = subdf['category']
            coco_dset.show_image(gid, fnum=1)

            sub_dets = kwimage.Detections.coerce(boxes=sub_xywh, cnames=sub_catnames)
            orig_dets = kwimage.Detections.coerce(boxes=orig_xywh, cnames=orig_catnames)
            orig_dets.draw(color='blue')
            sub_dets.draw(color='green')
            xdev.InteractiveIter.draw()

    if 0:
        coco_dset = ndsampler.CocoDataset(out_fpath)
    coco_dset.dump(out_fpath, newlines=True)
    coco_dset.fpath = out_fpath

    sealon_holdout_sets(coco_dset)


def detection_delta(dets1, dets2):
    """
    Charactarize the modification between two set of detections
    """
    # Determine what has changed if anything. We use the following codes:
    # diff - the number of boxes has changed
    # diffcats - the boxes are the same (up to 1 pixel) but categories are different
    # same - both boxes and categories are unmodified
    # adjust - boxes have been adjusted by more than 1 pixel. Categories might be different.
    import kwarray

    assert dets1.classes is dets2.classes
    cxywh1 = dets1.boxes.to_cxywh().data
    cxywh2 = dets2.boxes.to_cxywh().data

    pxl_thresh = 2
    match_iou_thresh = 0.1
    # match_iou_thresh = 0.4

    status = None
    info = {}
    details = {}

    if len(dets1) == len(dets2):
        # check if everything is the same.
        same_boxes = np.all(cxywh1 - cxywh2) < pxl_thresh
        same_classes = np.all(dets1.class_idxs == dets2.class_idxs)
        if same_boxes and same_classes:
            status = 'same'
        elif not same_boxes:
            status = 'adjust'
        elif not same_classes:
            status = 'diffcats'
        else:
            raise AssertionError
    else:
        status = 'diff'

    if status != 'same':
        # dets1 = det_before
        # dets2 = det_after

        # Solve an assignment problem between old and new boxes
        ious = dets1.boxes.ious(dets2.boxes)
        sameclass = dets1.class_idxs[:, None] == dets2.class_idxs[None, :]

        # Break ties between ious using classes
        eps = 1e-8

        # Only modify annotations that dont have drastic spatial differences If
        # for some reason an annotation changes spatial position by a large
        # amoumt it will be removed and readded.
        valid_match = (ious > match_iou_thresh).astype(np.float)
        class_affinity = (sameclass * eps)
        affinity = (ious + class_affinity) * valid_match

        assignment, _  = kwarray.maxvalue_assignment(affinity)
        if len(assignment):
            idxs1, idxs2 = map(list, zip(*assignment))
        else:
            idxs1, idxs2 = [], []

        all_idxs1 = ub.oset(range(len(dets1)))
        all_idxs2 = ub.oset(range(len(dets2)))

        assert all_idxs1.issuperset(idxs1)
        assert all_idxs2.issuperset(idxs2)
        unassigned_idxs1 = all_idxs1 - set(idxs1)  # removed
        unassigned_idxs2 = all_idxs2 - set(idxs2)  # added

        details = {
            'modify_idxs1': idxs1,
            'modify_idxs2': idxs2,
            'remove_idxs1': unassigned_idxs1,
            'add_idxs2': unassigned_idxs2,
        }

        removed_catfreq = ub.dict_hist(
            ub.take(dets1.classes, dets1.class_idxs[unassigned_idxs1]))
        added_catfreq = ub.dict_hist(
            ub.take(dets2.classes, dets2.class_idxs[unassigned_idxs2]))

        modifications = []
        for idx1, idx2 in zip(idxs1, idxs2):
            modification = []
            catname1 = dets1.classes[dets1.class_idxs[idx1]]
            catname2 = dets2.classes[dets2.class_idxs[idx2]]
            if catname1 != catname2:
                modification.append('{} -> {}'.format(catname1, catname2))

            box1 = cxywh1[idx1]
            box2 = cxywh2[idx2]
            box_delta = box1 - box2

            if np.any(box_delta > pxl_thresh):
                # characterize box change
                area1 = np.prod(box1[2:])
                area2 = np.prod(box2[2:])
                cxy1 = box1[0:2]
                cxy2 = box2[0:2]

                shift_dist = np.linalg.norm(cxy1 - cxy2)
                shift_ratio = shift_dist / np.sqrt(area2)
                area_ratio = area2 / area1
                box_mod = 'coords'
                # If there is a significant change in area or position mark that
                if area_ratio < 0.9:
                    box_mod = box_mod + '-shrink'
                elif area_ratio < 1.1:
                    box_mod = box_mod + '-grow'
                if shift_ratio > 0.5:
                    box_mod = box_mod + '-shift'
                modification.append(box_mod)

            if modification:
                modifications.append(tuple(modification))

        modified_catfreq = ub.dict_hist(modifications)
    else:
        unassigned_idxs1 = []
        unassigned_idxs2 = []
        modifications = []
        modified_catfreq = {}
        removed_catfreq = {}
        added_catfreq = {}

    basic_info = {
        'num_removed': len(unassigned_idxs1),
        'num_added': len(unassigned_idxs2),
        'num_modified': len(modifications),
    }
    info.update({
        'basic': basic_info,
        'modified_catfreq': modified_catfreq,
        'removed_catfreq': removed_catfreq,
        'added_catfreq': added_catfreq,
    })
    # print('info = {}'.format(ub.repr2(info, nl=2)))
    return status, info, details


def sealon_holdout_sets(coco_dset):
    import ubelt as ub
    year_to_imgs = ub.group_items(coco_dset.imgs.values(), lambda x: x['year_code'])
    print(ub.map_vals(len, year_to_imgs))
    vali_years = ['2010', '2016']
    split_gids = {}
    split_gids['vali'] = [img['id'] for img in ub.flatten(ub.dict_subset(year_to_imgs, vali_years).values())]
    split_gids['train'] = [img['id'] for img in ub.flatten(ub.dict_diff(year_to_imgs, vali_years).values())]
    for tag, gids in split_gids.items():
        subset = coco_dset.subset(gids)
        subset.fpath = ub.augpath(coco_dset.fpath, suffix='_{}'.format(tag), multidot=1)
        print('subset.fpath = {!r}'.format(subset.fpath))
        print('len(gids) = {}'.format(len(gids)))
        subset.dump(subset.fpath, newlines=True)


def take_manually_edited_subsets():
    import kwcoco
    in_fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8.mscoco.json')
    coco_dset = kwcoco.CocoDataset(in_fpath)

    modified_gids = []
    for ann in coco_dset.anns.values():
        changelog = ann.get('changelog', [])
        if ann['box_source'] == 'refinement-2020-06-17':
           gid = ann['image_id']
           modified_gids.append(gid)

        for log in changelog:
            if 'modified' in log:
               gid = ann['image_id']
               modified_gids.append(gid)

    modified_gids = sorted(set(modified_gids))

    out_fpath = ub.augpath(in_fpath, suffix='_manual', multidot=True)
    coco_dset = coco_dset.subset(modified_gids)
    coco_dset.fpath = out_fpath
    coco_dset.dump(coco_dset.fpath)
    print('modified_gids = {}'.format(ub.repr2(len(modified_gids), nl=1)))
    print('coco_dset = {!r}'.format(coco_dset))

    {img['year_code'] for img in coco_dset.imgs.values()}


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/apply_sealion_edits.py
    """
    main()
