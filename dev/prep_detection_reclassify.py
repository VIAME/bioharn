"""
Notes:

    Steps I manually took, need to automate these more effectively

    1. Find the best detection model we had. In this case `rgb-fine-coi-v40__epoch_00000007`

    2. Load the predictions file here, and dump them into an all_pred.mscoco.json file

    3. Load that file in clf_predict and predict on each box. Dump reclassified.mscoco.json

        python -m bioharn.clf_predict \
            --batch_size=16 \
            --workers=4 \
            --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip \
            --dataset=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/eval/habcam_cfarm_v8_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v40__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.5/all_pred.mscoco.json \
            --out_dpath=$HOME/tmp/cached_clf_out_cli

        # Produces: $HOME/tmp/cached_clf_out_cli/reclassified.mscoco.json

    4. Use kwcoco evalaute to compare reclassified.mscoco.json to the truth habcam_cfarm_v8_test.mscoco.json

        $HOME/tmp/cached_clf_out_cli/reclassified.mscoco.json


    5. Retrain the classification model

        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-hard-v004 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_hardbg1.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_hardbg1.mscoco.json \
            --schedule=ReduceLROnPlateau-p5-c5 \
            --max_epoch=400 \
            --augment=simple \
            --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --workers=8 \
            --xpu=auto \
            --batch_size=32 \
            --balance=classes

        # Re-retrain

        python -m bioharn.clf_fit \
            --name=bioharn-clf-rgb-hard-v005 \
            --train_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_train_hardbg1.mscoco.json \
            --vali_dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_hardbg1.mscoco.json \
            --schedule=step-10-20 \
            --max_epoch=400 \
            --augment=complex \
            --pretrained=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v004/emrxfdav/deploy_ClfModel_emrxfdav_024_HUEOJO.zip \
            --workdir=$HOME/work/bioharn \
            --arch=resnext101 \
            --channels="rgb" \
            --optim=sgd \
            --lr=1e-3 \
            --input_dims=256,256 \
            --normalize_inputs=True \
            --workers=8 \
            --xpu=auto \
            --batch_size=64 \
            --balance=classes

            /home/joncrall/

    6. Rerun classifications on the existing predicted data using the new model

        python -m bioharn.clf_predict \
            --batch_size=16 \
            --workers=4 \
            --deployed=$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-hard-v004/emrxfdav/deploy_ClfModel_emrxfdav_024_HUEOJO.zip \
            --dataset=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/eval/habcam_cfarm_v8_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v40__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.5/all_pred.mscoco.json \
            --out_dpath=$HOME/tmp/cached_clf_out_cli_hard

    7. Evaluate new predictions

        python -m kwcoco.coco_evaluator \
            --pred_dataset=$HOME/tmp/cached_clf_out_cli_hard/reclassified.mscoco.json \
            --true_dataset=$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json \
            --out_dpath=$HOME/tmp/reclassified_eval_v2



NEXT STEPS:

    * Run prediction on train / validation dataset

    * Create expanded set of true boxes with background boxes.

    * Train a classifier on that dataset.

    * Rerun the evaluation steps
"""

from os.path import dirname
from os.path import basename
from os.path import exists
from os.path import join
import glob
import ubelt as ub
import kwcoco


def prototype_prep_detection_reclassify():

    prediction_dpath = ub.expandpath('$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/eval/habcam_cfarm_v8_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v40__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.5/pred')

    # prediction_dpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/pred")

    prediction_fpaths = list(glob.glob(join(prediction_dpath, '*.mscoco.json')))
    fpaths =  prediction_fpaths[0:10]

    truth_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json')
    true_dset = kwcoco.CocoDataset(truth_fpath)

    # prediction_dsets = [
    #     kwcoco.CocoDataset(fpath)
    #     for fpath in ub.ProgIter(prediction_fpaths, desc='load detections')
    # ]

    pred_dsets = results = thread_based_multi_read(prediction_fpaths)
    pred_dset = kwcoco.CocoDataset.union(*pred_dsets)

    gid = ub.peek(pred_dset.imgs.keys())
    if not exists(pred_dset.get_image_fpath(gid)):
        print('need to reroot pred dset')
        orig_img_root = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos')

        self = pred_dset
        new_img_root = orig_img_root

        if exists(join(orig_img_root, pred_dset.imgs[gid]['file_name'])):
            print('the hacked root seems to work')
            for img in ub.ProgIter(list(pred_dset.imgs.values()), desc='checking all'):
                assert exists(join(orig_img_root, img['file_name']))
            pred_dset.reroot(orig_img_root, absolute=True, check=True, safe=True)
        else:
            raise Exception('the hacked root doesnt work')

    all_pred_fpath = join(dirname(prediction_dpath), 'all_pred.mscoco.json')
    pred_dset.dump(all_pred_fpath, newlines=True)

    truth_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos')


def thread_based_multi_read(fpaths, max_workers=8, verbose=0):
    # Can this be done better with asyncio?
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool('thread', max_workers=max_workers)
    for fpath in ub.ProgIter(fpaths, desc='submit load jobs', verbose=verbose):
        jobs.submit(kwcoco.CocoDataset, fpath)
    results = [f.result() for f in ub.ProgIter(jobs.as_completed(), desc='collect load jobs', total=len(jobs), verbose=verbose)]
    return results


def thread_based_multi_read2(fpaths, max_workers=8, verbose=0):
    # Can this be done better with asyncio?
    from concurrent import futures
    executor = futures.ThreadPoolExecutor(max_workers=max_workers)
    with executor:
        jobs = [
            executor.submit(kwcoco.CocoDataset, fpath)
            for fpath in ub.ProgIter(fpaths, desc='submit load jobs', verbose=verbose)
        ]
        prog = ub.ProgIter(futures.as_completed(jobs), total=len(jobs),
                           desc='collect load jobs', verbose=verbose)
        results = [f.result() for f in prog]
    return results


def async_based_multi_read(fpaths, max_workers=8):
    """
    TODO: figure out the best way to use asyncio here

    Benchmark:
        max_workers = 8
        import kwcoco
        import tempfile
        dpath = tempfile.mkdtemp()
        fpaths = [join(dpath, 'file_{:0d}.coco'.format(i)) for i in range(100)]
        for fpath in fpaths:
            with open(fpath, 'w') as file:
                kwcoco.CocoDataset.demo().dump(file)

        import timerit
        ti = timerit.Timerit(10, bestof=3, verbose=2)

        for timer in ti.reset('asyncio'):
            with timer:
                async_based_multi_read(fpaths)

        for timer in ti.reset('thread'):
            with timer:
                thread_based_multi_read(fpaths)
    """
    import asyncio
    import aiofiles
    import kwcoco
    import json

    # References:
    # https://asyncio.readthedocs.io/en/latest/producer_consumer.html
    async def produce(queue, fpaths):
        for fpath in fpaths:
            # put the item in the queue
            await queue.put(fpath)
        # indicate the producer is done
        await queue.put(None)

    async def consume(queue):
        items = []
        while True:
            # wait for an item from the producer
            fpath = await queue.get()
            if fpath is None:
                # the producer emits None to indicate that it is done
                break
            async with aiofiles.open(fpath, mode='r') as file:
                text = await file.read()
                dataset = json.loads(text)
            dset = kwcoco.CocoDataset(dataset, tag=basename(fpath))
            items.append(dset)
        return items

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=max_workers)
    producer_coro = produce(queue, fpaths)

    consumer_coro = consume(queue)

    gathered = asyncio.gather(producer_coro, consumer_coro)
    produce_got, consume_got = loop.run_until_complete(gathered)
    results = consume_got
    return results


def make_hardnegative_clf_dataset():
    """
        # Detect on the training set
        python -m bioharn.detect_predict \
            --dataset=$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json \
            --deployed=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip \
            --out_dpath=~/tmp/detect_habcam_v8_train \
            --xpu=auto --batch_size=10 --workers=4

        # Detect on the validation set
        python -m bioharn.detect_predict \
            --dataset=$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
            --deployed=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip \
            --out_dpath=~/tmp/detect_habcam_v8_vali \
            --xpu=auto --batch_size=32 --workers=8

        # Detect on the validation set
        python -m bioharn.detect_predict \
            --dataset=$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json \
            --deployed=$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip \
            --out_dpath=~/tmp/detect_habcam_v8_vali \
            --xpu=auto --batch_size=32 --workers=8
    """
    # from kwcoco import CocoDataset
    import kwcoco
    import glob
    pred_fpaths = list(glob.glob(ub.expandpath('~/tmp/detect_habcam_v8_vali/pred/*.mscoco.json')))
    true_coco = kwcoco.CocoDataset(ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json'))
    pred_coco = kwcoco.CocoDataset.from_coco_paths(pred_fpaths)
    print('true_coco = {!r}'.format(true_coco))
    print('pred_coco = {!r}'.format(pred_coco))

    hard_true_dset = build_hardneg_dset(true_coco, pred_coco)
    hard_true_dset.fpath = ub.augpath(true_coco.fpath, suffix='_hardbg1', multidot=True)
    hard_true_dset.dump(hard_true_dset.fpath, newlines=True)

    pred_fpaths = list(glob.glob(ub.expandpath('~/tmp/detect_habcam_v8_train/pred/*.mscoco.json')))
    true_coco = kwcoco.CocoDataset(ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_train.mscoco.json'))
    pred_coco = kwcoco.CocoDataset.from_coco_paths(pred_fpaths)
    print('true_coco = {!r}'.format(true_coco))
    print('pred_coco = {!r}'.format(pred_coco))

    hard_true_dset = build_hardneg_dset(true_coco, pred_coco)
    hard_true_dset.fpath = ub.augpath(true_coco.fpath, suffix='_hardbg1', multidot=True)
    hard_true_dset.dump(hard_true_dset.fpath, newlines=True)


def build_hardneg_dset(true_coco, pred_coco):

    from netharn.metrics import DetectionMetrics
    #hack
    bg_cid = true_coco.ensure_category('background', id=0)

    dmet = DetectionMetrics.from_coco(true_coco, pred_coco, verbose=1)
    gids = list(set(dmet.gid_to_pred_dets) & set(dmet.gid_to_true_dets))
    gids = list(ub.oset(gids) - {28487, 28491})

    # TODO: fix background inconsistencies
    cfsn_vecs = dmet.confusion_vectors(gids=gids, verbose=1, workers=0)

    # These predictions are unassigned, that means either they are unannotated
    # true boxes that are missing, or they are hard negatives
    unassigned_preds = (cfsn_vecs.data['true'] == -1)

    # Assume hard negative for now, other path needs manual or expert input
    unassigned_vecs = cfsn_vecs.data.compress(unassigned_preds)

    # Get the unassigned boxes
    candidate_hard_anns = []
    import kwarray
    unique_gids, groupxs = kwarray.group_indices(unassigned_vecs['gid'])
    for gid, idxs in ub.ProgIter(zip(unique_gids, groupxs), total=len(unique_gids), desc='extracting hard negatives'):
        pxs = unassigned_vecs['pxs'][idxs]
        scores = unassigned_vecs['score'][idxs]
        thresh = 0.1
        take_pxs = pxs[scores > thresh]
        pred_dets = dmet.gid_to_pred_dets[gid].take(take_pxs)

        for ann in pred_dets.to_coco():
            ann['category_name'] = 'background'
            ann['category_id'] = bg_cid
            ann['image_id'] = gid
            ann.pop('score', None)
            ann.pop('prob', None)
            candidate_hard_anns.append(ann)

        if 0:
            import kwplot
            kwplot.autompl()
            canvas = true_coco.load_image(gid)
            canvas = pred_dets.draw_on(canvas)
            kwplot.imshow(canvas)

    hard_true_dset = true_coco.copy()
    for ann in ub.ProgIter(candidate_hard_anns, desc='add hard negs'):
        hard_true_dset.add_annotation(**ann)
    return hard_true_dset

    for gid in unassigned_vecs['gid']:
        pred_dets = dmet.gid_to_pred_dets[gid]

    pass


def foo(fpaths):
    """
    Ignore:
        fpaths = ['file_{:0d}.txt'.format(i) for i in range(20)]
        for fpath in fpaths:
            with open(fpath, 'w') as file:
                file.write('fpath = {}'.format(fpath))
    """
    import asyncio
    import aiofiles
    async def multi_open(fpaths):
        async def async_open(fpath):
            print("Start loading {}".format(fpath))
            async with aiofiles.open(fpath, mode='r') as file:
                text = await file.read()
            print(f"Done reading {fpath}")
            return text
        return await asyncio.gather(*[async_open(fpath) for fpath in fpaths])
    coroutine = multi_open(fpaths)
    results = asyncio.run(coroutine)
    print('results = {!r}'.format(results))
