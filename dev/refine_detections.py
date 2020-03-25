"""
Use pretrained networks to predict reasonable box initializations for images
images that only have "dot" annotations.

Associates detected boxes with dots, and uses several hueristics to refine
existing boxes.
"""
from os.path import join
import ubelt as ub
import numpy as np
import kwimage
import kwarray
import ndsampler
from sklearn import metrics


def refine_cascade_detecotr():
    """
    python -m bioharn.detect_fit \
        --nice=detect-sealion-cascade-v7 \
        --workdir=$HOME/work/sealions \
        --train_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_manual_v7.mscoco.json \
        --vali_dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_auto_v7.mscoco.json \
        --schedule=ReduceLROnPlateau-p2-c2 \
        --pretrained==$HOME/remote/viame/work/sealions/fit/runs/detect-sealion-cascade-v6/lvcppmtu/explit_checkpoints/_epoch_00000011_2020-03-23T221146+5.pt \
        --augment=complex \
        --init=noop \
        --arch=cascade \
        --optim=sgd --lr=1e-3 \
        --input_dims=window \
        --window_dims=512,512 \
        --window_overlap=0.5 \
        --multiscale=True \
        --normalize_inputs=True \
        --min_lr=1e-6 \
        --workers=4 --xpu=0 --batch_size=8 --bstep=1
    """


def run_cascade_detector():
    """
    notes:

    --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-singleclass-cascade-v3/neumtmpw/deploy_MM_CascadeRCNN_neumtmpw_031_FYQJLH \
        --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-singleclass-cascade-v3/neumtmpw/torch_snapshots/_epoch_00000028.pt \

    python -m bioharn.detect_predict \
        --dataset=~/data/US_ALASKA_MML_SEALION/sealions_all_v3.mscoco.json \
        --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-singleclass-cascade-v3/neumtmpw/deploy_MM_CascadeRCNN_neumtmpw_031_FYQJLH \
        --out_dpath=~/data/US_ALASKA_MML_SEALION/detections/cascade_v2 \
        --draw=10 \
        --input_dims=512,512 \
        --workers=10 \
        --xpu=0 --batch_size=256


    python -m bioharn.detect_predict \
        --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v6.mscoco.json \
        --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-sealion-cascade-v6/lvcppmtu/deploy_MM_CascadeRCNN_lvcppmtu_009_GDCTSU.zip \
        --out_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v6 \
        --draw=0 \
        --workers=4 \
        --workdir=$HOME/work/sealions \
        --sampler_backend=cog \
        --xpu=0 --batch_size=128

    python -m bioharn.detect_predict \
        --dataset=/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_auto_v7.mscoco.json \
        --deployed=/home/joncrall/work/sealions/fit/runs/detect-sealion-cascade-v7/cobrrkfz/explit_checkpoints/_epoch_00000006_2020-03-24T144631+5.pt \
        --out_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v7 \
        --draw=0 \
        --workers=4 \
        --workdir=$HOME/work/sealions \
        --sampler_backend=cog \
        --xpu=0 --batch_size=256
    """


def load_candidate_detections():
    """
    Hard coded paths to candidate detections
    """

    truth_fpath = ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7.mscoco.json')
    print('load true')
    true_dset = ndsampler.CocoDataset(truth_fpath)
    print('true_dset = {!r}'.format(true_dset))

    fpaths = {
        'cascade_v7': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v7/pred/detections.mscoco.json'),
        # 'cascade_v6': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v6/pred/detections.mscoco.json'),
        # 'cascade': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v2/pred/detections.mscoco.json'),
        # 'generic': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/detections_generic.json'),
        # 'swfsc': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/detections_swfsc.json'),
    }
    pred_dsets = {}
    for key, fpath in fpaths.items():
        print('load key = {!r}'.format(key))
        pred_dsets[key] = ndsampler.CocoDataset(fpath)

    # Use everything read so far
    # if 'cascade' not in pred_dsets:
    #     import glob
    #     patial_fpaths = list(glob.glob(ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v2/pred/single_image/*.json')))
    #     partial_dsets = []
    #     for fpath in ub.ProgIter(patial_fpaths):
    #         dset = ndsampler.CocoDataset(fpath)
    #         partial_dsets.append(dset)
    #     combo_dset = ndsampler.CocoDataset.union(*partial_dsets)
    #     pred_dsets['cascade'] = combo_dset

    return true_dset, pred_dsets


def _true_devcheck():
    truth_fpath = ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7.mscoco.json')
    print('load true')
    true_dset = ndsampler.CocoDataset(truth_fpath)

    allkeys = set()
    for img in true_dset.imgs.values():
        allkeys.update(img)

    # find fully manual annotation
    manual_gids = set()
    for ann in true_dset.anns.values():
        if ann['box_source'] == 'refinement-2020-03-18':
            manual_gids.add(ann['image_id'])

    manual_dset = true_dset.subset(manual_gids)

    # vali_years = ['2010', '2016']
    # split_gids = {}

    other_gids = sorted(set(true_dset.imgs.keys()) - manual_gids)
    # other_gids = kwarray.shuffle(other_gids, rng=432)[0:len(manual_dset.imgs)]

    other_dset = true_dset.subset(other_gids)

    manual_dset.fpath = true_dset.fpath.replace('_refined', '_manual')
    other_dset.fpath = true_dset.fpath.replace('_refined', '_auto')
    manual_dset.dump(manual_dset.fpath, newlines=True)
    other_dset.dump(other_dset.fpath, newlines=True)


def assign(true_dset, true_annots, stacked_dets):
    """
    Heuristic assignment function

    Ignore:
        import kwplot
        kwplot.autompl()

        true_dets = true_annots.detections

        assigned_true = true_dets.take([t[0] for t in assignment])
        assigned_pred = stacked_dets.take([t[1] for t in assignment])

        pt1 = np.array([
            kpts.xy[0]
            for kpts in assigned_true.data['keypoints']
        ])
        pt2 = assigned_pred.boxes.xy_center

        image = true_dset.load_image(true_img['id'])
        canvas = image.copy()
        kwplot.imshow(canvas, doclf=1)
        true_dets.draw(color='green')
        assigned_true.draw(color='blue', labels=False)
        assigned_pred.draw(color='purple', labels=False)
        kwplot.draw_line_segments(pt1, pt2)
    """
    if len(stacked_dets) == 0:
        return []

    boxstats = true_dset.boxsize_stats(
            aids=true_annots.aids,
            statskw=dict(median=True))

    true_xys = np.array([
        kp['xy']
        for kpts in true_annots.lookup('keypoints')
        for kp in kpts
    ])
    true_boxes = true_annots.boxes
    true_weight = np.ones(len(true_boxes))

    pred_boxes = stacked_dets.boxes
    pred_scores = stacked_dets.scores
    pred_cxys = pred_boxes.xy_center.reshape(-1, 2)
    pred_weight = np.ones(len(pred_boxes))

    if 'classes' in stacked_dets.meta:
        # Penalize predictions that have shapes very different than what we
        # would expect based on per-class priors
        pred_catnames = [stacked_dets.classes[idx]
                         for idx in stacked_dets.class_idxs]
        pred_size_prior = [
                boxstats['all']['stats']['med']
                if catname not in boxstats['perclass'] else
                boxstats['perclass'][catname]['stats']['med']
                for catname in pred_catnames]
        pred_size = np.hstack([pred_boxes.width, pred_boxes.height])
        pred_shape_delta = np.sqrt(((pred_size - pred_size_prior) ** 2).sum(axis=1))
        pred_weight *= (1 / pred_shape_delta)

        # Upweight truths that have very shapes very different than what we
        # would expect based on per-class priors. These boxes are more likely
        # to be bad, so prioritize finding an assignment for them.
        true_size_prior = [
                boxstats['all']['stats']['med']
                if catname not in boxstats['perclass'] else
                boxstats['perclass'][catname]['stats']['med']
                for catname in true_annots.cnames]
        true_size = np.hstack([true_boxes.width, true_boxes.height])
        true_shape_delta = np.sqrt(((true_size - true_size_prior) ** 2).sum(axis=1))
        true_weight *= true_shape_delta
    else:
        pred_catnames = None

    # add a small width and height to dots
    # true_dot_boxes = kwimage.Boxes([xy.tolist() + [1, 1]
    #                                 for xy in true_xys], 'cxywh')

    # compute distance between dots and predicted boxes
    dists = metrics.euclidean_distances(true_xys, pred_cxys)
    closeness = (1 / dists)
    # ious = true_boxes.ious(pred_boxes)

    # values = (1 / dists) * ious
    values = (
            # ious *
            closeness *
            pred_scores[None, :] *
            pred_weight[None, :] *
            np.log(true_weight[:, None] + 1)
    )

    try:
        # Bump the affinity between truth and predictions with the same label
        true_cidxs = np.array([stacked_dets.classes.index(cname)
                               for cname in true_annots.cnames])
        pred_cidxs = stacked_dets.class_idxs
        same_label = true_cidxs[:, None] == pred_cidxs[None, :]
        values[same_label] *= 2
    except Exception:
        pass

    # prevent ridiculous assignments
    # thresh = (true_boxes.width * 2)
    # values[dists > thresh] = -np.inf

    # area_ratio = true_boxes.area / pred_boxes.area.T
    # values[(area_ratio > 1.1) | (area_ratio < 0.10)] = -np.inf

    # The dots must be contained in the predicted box
    values[~pred_boxes.contains(true_xys).T] = -np.inf

    # Only assign predictions with a high enough score
    score_thresh = 0.4
    values[:, pred_scores < score_thresh] = -np.inf

    # flags = (ious > 0).astype(dists.dtype)

    # value = (1 / dists) * flags

    assignment, _  = kwarray.maxvalue_assignment(values)
    assignment = [(t[0], t[1], values[t[0], t[1]]) for t in assignment]
    return assignment


def associate_detections(true_dset, pred_dsets):
    """
    Given a set of known detections compute an association score between the true "dots" and the predicted boxes.
    Then find the max value assignment, and refine the truth annotations to use the assigned boxes over the heuristic truth ones.

    Args:
        true_dset (CocoDataset): the dataset with truth "dots"

        pred_dsets (Dict[str, CocoDataset]): a set of predictions that we will
            use to attempt to refine the truth boxes.
    """

    refined_dset = true_dset.copy()

    # We wont refine any images with manual annotations
    gids_with_manual_annotations = set()
    for ann in true_dset.anns.values():
        if 'box_source' in ann:
            if ann['box_source'] == 'refinement-2020-03-18':
                gids_with_manual_annotations.add(ann['image_id'])
    gids_with_heuristic_annotations = sorted(
            set(true_dset.imgs.keys()) - gids_with_manual_annotations)

    print('len(gids_with_manual_annotations) = {!r}'.format(len(gids_with_manual_annotations)))
    print('len(ids_with_heuristic_annotations) = {!r}'.format(len(gids_with_heuristic_annotations)))

    gid = gids_with_heuristic_annotations[15]
    VIZ = 60

    modified_time = ub.timestamp()

    for gx, gid in ub.ProgIter(enumerate(gids_with_heuristic_annotations)):
        true_img = true_dset.imgs[gid]
        true_annots = true_dset.annots(gid=true_img['id'])

        # Create collection all detection candidates
        key_to_dets = {}
        for key, pred_dset in pred_dsets.items():
            if true_img['file_name'] in pred_dset.index.file_name_to_img:
                pred_img = pred_dset.index.file_name_to_img[true_img['file_name']]
                pred_annots = pred_dset.annots(gid=pred_img['id'])
                key_to_dets[key] = pred_annots.detections

        if key_to_dets and len(true_annots):
            # Concat all candidates and assign to true annotations
            stacked_dets = kwimage.Detections.concatenate(list(key_to_dets.values()))
            assignment = assign(true_dset, true_annots, stacked_dets)
        else:
            assignment = []

        true_idxs = [t[0] for t in assignment]
        pred_idxs = [t[1] for t in assignment]

        assign_mapping = ub.dzip(true_idxs, pred_idxs)

        aids = true_annots.aids
        anns = list(ub.take(refined_dset.anns, aids))

        offset_keys = list(key_to_dets.keys())
        offset_vals = np.cumsum(list(map(len, key_to_dets.values())))

        # Refine the annotations
        for true_idx, ann in enumerate(anns):
            pred_idx = assign_mapping.get(true_idx, None)
            ann['bbox'] = true_dset.anns[ann['id']]['bbox']
            ann['key'] = 'true'
            if pred_idx is not None:
                new_bbox = stacked_dets.boxes.to_xywh().data[pred_idx]
                ann['bbox'] = new_bbox.tolist()
                key = offset_keys[np.where(pred_idx < offset_vals)[0][0]]
                ann['key'] = key
                ann['box_source'] = key
                if 'changelog' not in ann:
                    ann['changelog'] = []
                ann['changelog'].append('modified: {}'.format(modified_time))
                ann['score'] = stacked_dets.scores[pred_idx]

        if VIZ and gx < VIZ:
            import kwplot
            # drawkw = {
            #     'cascade_v6': dict(color='blue', thickness=6),
            #     'cascade': dict(color='blue', thickness=6),
            #     'generic': dict(color='purple', thickness=4),
            #     'swfsc': dict(color='red', thickness=3),
            #     'true': dict(color='green', thickness=2),
            # }
            # # drawkw = ub.dict_isect(drawkw, pred_dsets)

            # image = true_dset.load_image(true_img['id'])

            # refined_annots = refined_dset.annots(aids=aids)

            # canvas = image.copy()
            # # canvas = true_boxes.draw_on(canvas, color='green', thickness=4)

            # keys, groupxs = kwarray.group_indices(refined_annots.lookup('key'))
            # for key, idxs in zip(keys, groupxs):
            #     subdets = refined_annots.take(idxs).detections
            #     kw = drawkw[key].copy()
            #     kw.pop('thickness')
            #     canvas = subdets.draw_on(canvas, **kw)

            # cand_dpath = ub.ensuredir((viz_dpath, 'candidates'))
            # cand_fpath = join(cand_dpath, 'temp_{:04d}.jpg'.format(true_img['id']))
            # kwimage.imwrite(cand_fpath, canvas)

            true_dets = true_dset.annots(gid=true_img['id']).detections
            assigned_true = true_dets.take([t[0] for t in assignment])
            assigned_pred = stacked_dets.take([t[1] for t in assignment])
            pt1 = np.array([
                kpts.xy[0]
                for kpts in assigned_true.data['keypoints']
            ])
            pt2 = assigned_pred.boxes.xy_center

            image = true_dset.load_image(true_img['id'])
            canvas = image.copy()
            canvas = true_dets.draw_on(canvas, color='green')
            # canvas = stacked_dets.draw_on(canvas, color='red')
            canvas = assigned_true.draw_on(canvas, color='blue', labels=True)
            canvas = assigned_pred.draw_on(canvas, color='purple', labels=False)
            canvas = kwimage.draw_line_segments_on_image(canvas, pt1, pt2)
            viz_dpath = ub.ensuredir(
                    '/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/detections/refine7')
            assign_dpath = ub.ensuredir((viz_dpath, 'assign'))
            assign_fpath = join(assign_dpath, 'temp_{:04d}.jpg'.format(true_img['id']))
            kwimage.imwrite(assign_fpath, canvas)
            if 0:
                kwplot.autompl()
                kwplot.imshow(canvas, doclf=1)

    for ann in refined_dset.anns.values():
        key = ann.pop('key', 'true')

    refined_dset.fpath = true_dset.fpath.replace('_v7', '_v8')
    assert 'refined' in refined_dset.fpath
    refined_dset.dump(refined_dset.fpath, newlines=True)
    sealon_holdout_sets(refined_dset)


def sealon_holdout_sets(coco_dset):
    import ubelt as ub
    year_to_imgs = ub.group_items(coco_dset.imgs.values(), lambda x: x['year_code'])
    print(ub.map_vals(len, year_to_imgs))
    vali_years = ['2007', '2010']
    split_gids = {}
    split_gids['vali'] = [img['id'] for img in ub.flatten(ub.dict_subset(year_to_imgs, vali_years).values())]
    split_gids['train'] = [img['id'] for img in ub.flatten(ub.dict_diff(year_to_imgs, vali_years).values())]
    for tag, gids in split_gids.items():
        subset = coco_dset.subset(gids)
        subset.fpath = ub.augpath(coco_dset.fpath, suffix='_{}'.format(tag), multidot=1)
        print('subset.fpath = {!r}'.format(subset.fpath))
        print('len(gids) = {}'.format(len(gids)))
        subset.dump(subset.fpath, newlines=True)


def main():
    true_dset, pred_dsets = load_candidate_detections()
    associate_detections(true_dset, pred_dsets)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/refine_detections.py
    """
    main()
