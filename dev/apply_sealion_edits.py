

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
    import pandas as pd
    import ubelt as ub
    import numpy as np
    from os.path import basename

    fpath = '/home/joncrall/remote/namek/data/US_ALASKA_MML_SEALION/sealions_all_refined_v5.mscoco.json'
    print('fpath = {!r}'.format(fpath))
    orig_coco_dset = ndsampler.CocoDataset(fpath)
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

    # box_stats = coco_dset.boxsize_stats(anchors=4, verbose=1, clusterkw={'verbose': 0})
    # print('box_stats = {}'.format(ub.repr2(box_stats, nl=4)))

    csv_fpath1 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2014edit_img217.csv')
    csv_fpath2 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2015edit_img100.csv')
    csv_fpath3 = ub.expandpath('/home/joncrall/remote/namek/data/noaa/US_ALASKA_MML_SEALION/edits/detections_2016edit_img330.csv')

    columns = ['_aid', 'gname', '_gid', 'tl_x', 'tl_y', 'br_x', 'br_y', '7', '8', 'category', '10']
    df1 = pd.read_csv(csv_fpath1, header=None, names=columns)
    df2 = pd.read_csv(csv_fpath2, header=None, names=columns)
    df3 = pd.read_csv(csv_fpath3, header=None, names=columns)
    df = pd.concat([df1, df2, df3])

    gid_to_basename = {img['id']: basename(img['file_name'])
                       for img in coco_dset.imgs.values()}

    basename_to_gids = ub.invert_dict(gid_to_basename, unique_vals=False)
    has_dups = max(map(len, basename_to_gids.values())) > 1
    assert not has_dups
    basename_to_gid = ub.map_vals(ub.peek, basename_to_gids)

    df = kwarray.DataFrameLight(pd.concat([df1, df2, df3]))

    gname_to_subdf = dict(df.groupby('gname'))
    gid_to_subdf = ub.map_keys(basename_to_gid, gname_to_subdf)
    gid_to_subdf.pop(3562, None)

    gid_to_before = {}
    gid_to_after = {}

    stats = ub.ddict(list)
    for gid in ub.ProgIter(list(gid_to_subdf.keys()), desc='update images'):
        subdf = gid_to_subdf[gid]
        orig_aids = sorted(coco_dset.gid_to_aids[gid])
        orig_anns = list(ub.take(coco_dset.anns, orig_aids))
        orig_annots = coco_dset.annots(orig_aids)

        orig_catnames = orig_annots.cnames
        orig_xywh = kwimage.Boxes(orig_annots.lookup('bbox'), 'xywh')
        # orig_dets = orig_annots.detections.copy()

        self = orig_annots
        anns = [self._id_to_obj[aid] for aid in self.aids]
        orig_dets = kwimage.Detections.from_coco_annots(anns, dset=self._dset)

        sub_aids = subdf._getcol('_aid')
        sub_tlbr = subdf._getcols(['tl_x', 'tl_y', 'br_x', 'br_y'])
        sub_xywh = kwimage.Boxes(sub_tlbr, 'tlbr').to_xywh()
        sub_catnames = subdf['category']
        sub_catnames = ['unknown' if isinstance(n, float) and np.isnan(n)
                        else n for n in sub_catnames]
        sub_cats = [coco_dset._alias_to_cat(n) for n in sub_catnames]
        sub_catnames = [cat['name'] for cat in sub_cats]
        sub_cids = [cat['id'] for cat in sub_cats]

        case = None
        if len(sub_xywh) != len(orig_xywh):
            # print('assignment = {!r}'.format(assignment))
            case = 'diff'
        else:
            maxdiff = np.abs(sub_xywh.data - orig_xywh.data).max()
            if maxdiff < 2:
                if orig_catnames != sub_catnames:
                    np.array(orig_catnames) != np.array(sub_catnames)
                    case = 'diffcats'
                else:
                    case = 'same'
            else:
                case = 'adjust'

        stats[case].append(gid)
        if case in {'diff', 'adjust', 'diffcats'}:
            ious = orig_xywh.ious(sub_xywh)
            assignment, _  = kwarray.maxvalue_assignment(ious)
            if len(assignment):
                orig_idxs, sub_idxs = map(list, zip(*assignment))
            else:
                orig_idxs, sub_idxs = [], []

            all_orig_idxs = ub.oset(range(len(orig_aids)))
            all_sub_idxs = ub.oset(range(len(sub_aids)))

            assert all_orig_idxs.issuperset(orig_idxs)
            assert all_sub_idxs.issuperset(sub_idxs)
            unassigned_orig_idxs = all_orig_idxs - set(orig_idxs)
            unassigned_sub_idxs = all_sub_idxs - sub_idxs

            # Handle addition
            add_idxs = unassigned_sub_idxs
            num_add = len(add_idxs)
            add_sub_anns = kwarray.DataFrameLight({
                'bbox': sub_xywh.take(add_idxs).data.tolist(),
                'image_id': [gid] * num_add,
                'category_id': list(ub.take(sub_cids, add_idxs)),
                'box_source': ['refinement-2020-03-18'] * num_add,
                'changelog': [['created: {}'.format(ub.timestamp())]] * num_add,
            })
            new_anns = [new_ann for _, new_ann in add_sub_anns.iterrows()]
            new_aids = [coco_dset.add_annotation(**ann) for ann in new_anns]

            # Handle removal
            remove_idxs = unassigned_orig_idxs
            remove_aids = orig_annots.take(remove_idxs).aids
            coco_dset.remove_annotations(remove_aids)

            # Handle modification
            orig_match_annots = orig_annots.take(orig_idxs)
            orig_match_annots.cnames = ub.take(sub_catnames, sub_idxs)
            orig_match_annots.boxes = sub_xywh.take(sub_idxs)
            changelogs = orig_match_annots.get('changelog', default=None)
            changelogs = [list() if c is None else c for c in changelogs]
            for c in changelogs:
                c.append('modified: {}'.format(ub.timestamp()))

            modified_aids = sorted(coco_dset.gid_to_aids[gid])
            modified_dets = coco_dset.annots(modified_aids).detections.copy()
            gid_to_before[gid] = orig_dets
            gid_to_after[gid] = modified_dets

        elif case == 'same':
            pass
        else:
            raise Exception('case = {!r}'.format(case))

    if False:
        import kwplot
        kwplot.autompl()
        # gid = 2544
        # gid = 2404
        # gid = 2434
        # gids = kwarray.shuffle(list(gid_to_subdf.keys()))
        gids = sorted(stats['diff'])

        gid = gids[0]
        det_before = gid_to_before[gid]
        det_after = gid_to_after[gid]

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

    out_fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6.mscoco.json')
    if 0:
        coco_dset = ndsampler.CocoDataset(out_fpath)
    coco_dset.dump(out_fpath, newlines=True)
    coco_dset.fpath = out_fpath

    sealon_holdout_sets(coco_dset)


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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/apply_sealion_edits.py
    """
    main()
