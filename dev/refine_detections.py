"""
Use pretrained networks to predict reasonable box initializations for images
images that only have "dot" annotations.

Associates detected boxes with dots, and uses several hueristics to refine
existing boxes.
"""
from os.path import join
import scriptconfig as scfg
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

    mkdir -p $HOME/work/models
    girder-client --api-url https://data.kitware.com/api/v1 download \
        --parent-type file 5f0cbabc9014a6d84e1c5650 \
        $HOME/work/models/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip

    python -m bioharn.detect_predict \
        --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8.mscoco.json \
        --deployed=$HOME/work/models/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip \
        --out_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v8 \
        --draw=0 \
        --workers=4 \
        --workdir=$HOME/work/sealions \
        --xpu=0 --batch_size=128

    --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-singleclass-cascade-v3/neumtmpw/deploy_MM_CascadeRCNN_neumtmpw_031_FYQJLH \
        --deployed=$HOME/remote/viame/work/sealions/fit/runs/detect-singleclass-cascade-v3/neumtmpw/torch_snapshots/_epoch_00000028.pt \

    $HOME/remote/namek/work/sealions/fit/name/sealion-cascade-manual-coarse-v6/deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS.zip

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


class RefineConfig(scfg.Config):
    """
    This script assumes that:
        (1) all objects are covered by the truth boxes
        (2) the shape / size of the truth boxes might be incorrect
        (3) the predicted boxes are have better shapes / sizes
        (4) the predicted boxes might have false positives / negatives
        (5) "golden" manual boxes have a field bbox_source: "refinement-..."
    """
    default = {
        'true_fpath': scfg.Value(None, help='path to the coco truth file to refine'),
        'pred_fpaths': scfg.Value(None, help='One or more prediction datasets that will be used to refine truth decision'),

        'score_thresh': scfg.Value(0.4, help='Only assign predictions with a high enough score'),

        'out_fpath': scfg.Value(None, help='path to dump the refined dataset'),

        'viz_dpath': scfg.Value(None, help='Output path for visualizations'),
    }


def main():
    """
    python $HOME/code/bioharn/dev/refine_detections.py \
        --true_fpath=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8.mscoco.json \
        --pred_fpaths=[$HOME/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v8/pred/detections.mscoco.json,] \
        --out_fpath=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json \
        --viz_dpath=$HOME/remote/viame/data/US_ALASKA_MML_SEALION/detections/refine9 \
        --score_thresh=0.2

    # Convert to VIAME CSV
    See kwcoco_to_viame_csv.py
    """
    import kwcoco

    config = RefineConfig(cmdline=True)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    print('load true')
    true_dset = kwcoco.CocoDataset.coerce(config['true_fpath'])
    print('true_dset = {!r}'.format(true_dset))
    fpaths = {ub.augpath(p, base='', ext='', multidot=True): p for p in config['pred_fpaths']}
    pred_dsets = {}
    for key, fpath in fpaths.items():
        print('load key = {!r}'.format(key))
        pred_dsets[key] = kwcoco.CocoDataset(fpath)

    out_fpath = config['out_fpath']
    viz_dpath = config['viz_dpath']

    refined_dset = refine_detections(
        true_dset, pred_dsets, viz_dpath=viz_dpath,
        score_thresh=config['score_thresh'])

    refined_dset.fpath = out_fpath
    refined_dset.dump(refined_dset.fpath, newlines=True)

    sealon_holdout_sets(refined_dset)


def assign(true_dset, true_annots, stacked_dets, score_thresh=0.4):
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
    values[:, pred_scores < score_thresh] = -np.inf

    # flags = (ious > 0).astype(dists.dtype)

    # value = (1 / dists) * flags

    assignment, _  = kwarray.maxvalue_assignment(values)
    assignment = [(t[0], t[1], values[t[0], t[1]]) for t in assignment]
    return assignment


def refine_detections(true_dset, pred_dsets, viz_dpath=None, **kwargs):
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
            # if ann['box_source'] == 'refinement-2020-03-18':
            # This is currently a hueristic which indicates which boxes
            # were manually refined and which were the results of previous
            # detectors.
            if ann['box_source'].startswith('refinement-'):
                gids_with_manual_annotations.add(ann['image_id'])
    gids_with_heuristic_annotations = sorted(
            set(true_dset.imgs.keys()) - gids_with_manual_annotations)

    print('len(gids_with_manual_annotations) = {!r}'.format(len(gids_with_manual_annotations)))
    print('len(ids_with_heuristic_annotations) = {!r}'.format(len(gids_with_heuristic_annotations)))

    # gid = gids_with_heuristic_annotations[15]

    VIZ = 60
    if viz_dpath is not None:
        viz_dpath = ub.ensuredir(viz_dpath)

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
            assignment = assign(true_dset, true_annots, stacked_dets, **kwargs)
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

        if viz_dpath and VIZ and gx < VIZ:
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
            assign_dpath = ub.ensuredir((viz_dpath, 'assign'))
            assign_fpath = join(assign_dpath, 'temp_{:04d}.jpg'.format(true_img['id']))

            legimg = kwplot.make_legend_img(ub.invert_dict({
                'green': 'unassigned truth (to keep)',
                'blue': 'assigned truth (to remove)',
                'purple': 'assigned pred (to add)',
            }))
            canvas[0:legimg.shape[0], 0:legimg.shape[1], ...] = legimg

            kwimage.imwrite(assign_fpath, canvas)

            beforeafter_dpath = ub.ensuredir((viz_dpath, 'beforeafter'))

            after_canvas = image.copy()
            refined_dets = refined_dset.annots(gid=true_img['id']).detections
            refined_dets.data.pop('score', None)
            after_canvas = refined_dets.draw_on(after_canvas, color='green', labels=False)
            after_canvas = assigned_pred.draw_on(after_canvas, color='blue', labels=False)

            after_fpath = join(beforeafter_dpath, 'temp_{:04d}_after.jpg'.format(true_img['id']))
            kwimage.imwrite(after_fpath, after_canvas)

            before_canvas = image.copy()
            before_fpath = join(beforeafter_dpath, 'temp_{:04d}_before.jpg'.format(true_img['id']))
            before_canvas = true_dets.draw_on(before_canvas, color='green', labels=False)
            before_canvas = assigned_true.draw_on(before_canvas, color='blue', labels=True)
            kwimage.imwrite(before_fpath, before_canvas)

            if 0:
                kwplot.autompl()
                kwplot.imshow(canvas, doclf=1)

    for ann in refined_dset.anns.values():
        key = ann.pop('key', 'true')

    return refined_dset


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


#### --- old code

def load_hardcoded_candidate_detections():
    """
    Hard coded paths to candidate detections
    """

    truth_fpath = ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7.mscoco.json')
    print('load true')
    true_dset = ndsampler.CocoDataset(truth_fpath)
    print('load true')
    print('true_dset = {!r}'.format(true_dset))

    fpaths = {
        'cascade_v8': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v8/pred/detections.mscoco.json'),
        # 'cascade_v7': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v7/pred/detections.mscoco.json'),
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
    # UNUSED, devcode
    truth_fpath = ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v7.mscoco.json')
    print('load true')
    true_dset = ndsampler.CocoDataset(truth_fpath)

    allkeys = set()
    for ann in true_dset.anns.values():
        allkeys.update(set(ann.keys()))

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


def _devcheck_year_breakdown_reorg():
    """

    mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION
    sudo chown -R root:public /data/public/Aerial/US_ALASKA_MML_SEALION
    sudo chmod g+w -R /data/public/Aerial/US_ALASKA_MML_SEALION

    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2007/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2008/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2009/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2010/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2011/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2012/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2013/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2014/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2015/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2016/images

    # Copy contents of extracted directories

    -rlptgoD.

    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2007/ /data/public/Aerial/US_ALASKA_MML_SEALION/2007/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2008/ /data/public/Aerial/US_ALASKA_MML_SEALION/2008/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2008W/ /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2009/ /data/public/Aerial/US_ALASKA_MML_SEALION/2009/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2010/ /data/public/Aerial/US_ALASKA_MML_SEALION/2010/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2011/ /data/public/Aerial/US_ALASKA_MML_SEALION/2011/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2012/ /data/public/Aerial/US_ALASKA_MML_SEALION/2012/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2013/ /data/public/Aerial/US_ALASKA_MML_SEALION/2013/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2014/ /data/public/Aerial/US_ALASKA_MML_SEALION/2014/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2015/ /data/public/Aerial/US_ALASKA_MML_SEALION/2015/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2016/ /data/public/Aerial/US_ALASKA_MML_SEALION/2016/images
    """
    fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json')
    import kwcoco
    coco_dset = kwcoco.CocoDataset(fpath)

    {tuple([p for p in img['file_name'].split('/')[0:4] if not p.lower().endswith('.jpg')]) for img in coco_dset.imgs.values()}

    if 0:
        from kwcoco import coco_schema
        coco_schema.COCO_SCHEMA.validate(coco_dset.dataset)

    # Fixup broken keypoint category schemas
    for ann in coco_dset.anns.values():
        kpts = ann.get('keypoints', [])
        if kpts:
            for kp in kpts:
                if 'keypoint_category_id' in kp:
                    pass
                elif 'category' in kp:
                    kpcat = coco_dset._resolve_to_kpcat(kp.pop('category'))
                    kp['keypoint_category_id'] = kpcat['id']
                elif 'category_name' in kp:
                    kpcat = coco_dset._resolve_to_kpcat(kp.pop('category_name'))
                    kp['keypoint_category_id'] = kpcat['id']
                else:
                    raise Exception

    year_to_imgs = ub.group_items(coco_dset.imgs.values(), lambda x: x['year_code'])
    print(ub.map_vals(len, year_to_imgs))

    from os.path import relpath
    dest_root = '/data/public/Aerial/US_ALASKA_MML_SEALION'

    year_to_dset = {}

    for year, imgs in ub.ProgIter(list(year_to_imgs.items())):
        gids = [g['id'] for g in imgs]
        year_dset = coco_dset.subset(gids, copy=True, autobuild=False)

        # Munge the paths to the images
        # CAREFUL THIS IS CHANGING POINTERS IN THE NEW FILES AS WELL
        old_rel_path = join('BLACKEDOUT/extracted', str(year))
        old_rel_dot_path = join('/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted/', str(year))

        dest_dpath = join(dest_root, str(year))
        for img in year_dset.dataset['images']:
            new_filename = join('images', relpath(img['file_name'], old_rel_path))
            img['file_name'] = new_filename

            if 'dot_fpath' in img:
                new_filename = join('images', relpath(img['dot_fpath'], old_rel_dot_path))
                img['dot_fpath'] = new_filename

        # year_dset.index.build()
        year_dset.fpath = join(dest_dpath, 'sealions_{}_v9.kwcoco.json'.format(year))
        year_dset.dump(year_dset.fpath, newlines=True)
        year_dset.img_root = dest_dpath
        year_to_dset[year] = year_dset

    year_dset = year_to_dset['2007']

    for year, year_dset in year_to_dset.items():
        assert not list(year_dset.missing_images())


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/refine_detections.py
    """
    main()
