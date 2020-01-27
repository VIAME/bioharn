"""

Associate detected boxes with dots

"""
import ubelt as ub
import numpy as np
import kwimage
import kwarray
import ndsampler
from sklearn import metrics


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
    """


def load_candidate_detections():
    """
    Hard coded paths to candidate detections
    """

    fpaths = {
        'cascade': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/cascade_v2/pred/detections.mscoco.json'),
        'generic': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/detections_generic.json'),
        'swfsc': ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/detections/detections_swfsc.json'),
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

    truth_fpath = ub.expandpath('~/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_v3.mscoco.json')

    print('load true')
    true_dset = ndsampler.CocoDataset(truth_fpath)
    return true_dset, pred_dsets


def assign(true_annots, stacked_dets):
    """
    Heuristic assignment function
    """
    if len(stacked_dets) == 0:
        return []

    pred_boxes = stacked_dets.boxes
    pred_scores = stacked_dets.scores
    pred_xys = pred_boxes.xy_center.reshape(-1, 2)

    true_xys = np.array([
        kp['xy']
        for kpts in true_annots.lookup('keypoints')
        for kp in kpts
    ])

    true_boxes = true_annots.boxes

    # add a small width and height to dots
    # true_dot_boxes = kwimage.Boxes([xy.tolist() + [1, 1]
    #                                 for xy in true_xys], 'cxywh')

    # compute distance between dots and predicted boxes
    dists = metrics.euclidean_distances(true_xys, pred_xys)
    ious = true_boxes.ious(pred_boxes)

    values = (1 / dists) * ious
    values = values * pred_scores[None, :]

    # prevent ridiculous assignments
    thresh = (true_boxes.width * 2)
    values[dists > thresh] = -np.inf

    area_ratio = true_boxes.area / pred_boxes.area.T
    values[(area_ratio > 1.1) | (area_ratio < 0.10)] = -np.inf

    # The dot must be in the predicted box
    values[~pred_boxes.contains(true_xys).T] = -np.inf

    # The dots must be contained in the predicted box
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

    # for pred_dset in [dset1, dset2, dset3]:
    for true_img in ub.ProgIter(list(true_dset.imgs.values())):
        true_annots = true_dset.annots(gid=true_img['id'])

        key_to_dets = {}
        for key, pred_dset in pred_dsets.items():
            if true_img['file_name'] in pred_dset.index.file_name_to_img:
                pred_img = pred_dset.index.file_name_to_img[true_img['file_name']]
                pred_annots = pred_dset.annots(gid=pred_img['id'])
                pred_boxes = pred_annots.boxes
                pred_score = np.array(pred_annots.lookup('score'))
                key_to_dets[key] = kwimage.Detections(
                    boxes=pred_boxes,
                    scores=pred_score)

        if key_to_dets and len(true_annots):
            stacked_dets = kwimage.Detections.concatenate(list(key_to_dets.values()))
            assignment = assign(true_annots, stacked_dets)
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

        if 0:
            import kwplot
            kwplot.autompl()

            drawkw = {
                'cascade': dict(color='blue', thickness=6),
                'generic': dict(color='purple', thickness=4),
                'swfsc': dict(color='red', thickness=3),
                'true': dict(color='green', thickness=2),
            }

            image = true_dset.load_image(true_img['id'])

            true_boxes = true_dset.annots(gid=true_img['id']).boxes

            refined_annots = refined_dset.annots(aids=aids)

            canvas = image.copy()
            # canvas = true_boxes.draw_on(canvas, color='green', thickness=4)

            keys, groupxs = kwarray.group_indices(refined_annots.lookup('key'))
            for key, idxs in zip(keys, groupxs):
                subboxes = refined_annots.take(idxs).boxes
                kw = drawkw[key].copy()
                print('kw = {!r}'.format(kw))
                kw.pop('thickness')
                canvas = subboxes.draw_on(canvas, **kw)

            ub.ensuredir('/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/detections/refine3')
            kwimage.imwrite('/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/detections/refine3/temp_{:04d}.jpg'.format(true_img['id']), canvas)

            drawkw = {
                'cascade': dict(color='blue', thickness=6),
                'generic': dict(color='purple', thickness=4),
                'swfsc': dict(color='red', thickness=3),
            }
            canvas = image.copy()
            for key, dets in key_to_dets.items():
                canvas = key_to_dets[key].boxes.draw_on(canvas, **drawkw[key])
            canvas = true_boxes.draw_on(canvas, color='green', thickness=2)
            # kwplot.imshow(canvas)
            ub.ensuredir('/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/detections/draw3')
            kwimage.imwrite('/home/joncrall/remote/viame/data/US_ALASKA_MML_SEALION/detections/draw3/temp_{:04d}.jpg'.format(true_img['id']), canvas)

    for ann in refined_dset.anns.values():
        key = ann.pop('key', 'true')
        if key == 'true':
            ann['box_source'] = 'dot_heuristic'
        elif key == 'cascade':
            ann['box_source'] = 'cascade'
        elif key == 'swfsc':
            ann['box_source'] = 'seal_det'
        elif key == 'generic':
            ann['box_source'] = 'generic_det'
        else:
            assert False

    refined_dset.fpath = true_dset.fpath.replace('_v3', '_refined_v5')
    assert 'refined' in refined_dset.fpath
    refined_dset.dump(refined_dset.fpath, newlines=True)


def main():
    true_dset, pred_dsets = load_candidate_detections()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/associate_seals.py
    """
    main()
