from os.path import join
import asyncio
import aiofiles
import glob
import ubelt as ub
import kwcoco


def prototype_prep_detection_reclassify():

    prediction_dpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/pred")

    prediction_fpaths = list(glob.glob(join(prediction_dpath, '*.mscoco.json')))

    truth_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json')
    true_dset = kwcoco.CocoDataset(truth_fpath)

    # prediction_dsets = [
    #     kwcoco.CocoDataset(fpath)
    #     for fpath in ub.ProgIter(prediction_fpaths, desc='load detections')
    # ]

    pred_dsets = results = pool_based_multi_read(prediction_fpaths)
    pred_dsets = [r.result() for r in pred_dsets]

    pred_dset = kwcoco.CocoDataset.union(*pred_dsets)

    pred_fpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/all_pred.mscoco.json")
    pred_dset.dump(pred_fpath, newlines=True)


def pool_based_multi_read(prediction_fpaths):
    # Can this be done better with asyncio?
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool('thread', max_workers=8)
    for fpath in ub.ProgIter(prediction_fpaths, desc='submit load jobs'):
        jobs.submit(kwcoco.CocoDataset, fpath)
    results = [f.result() for f in ub.ProgIter(jobs.as_completed(), desc='collect load jobs', total=len(jobs))]
    return results


def async_based_multi_read(prediction_fpaths):
    """
    TODO: figure out the best way to use asyncio here
    """
    async def read_coco_file(fpath):
        import json
        async with aiofiles.open(fpath, mode='r') as file:
            text = await file.read()
            dataset = json.loads(text)
            dset = kwcoco.CocoDataset(dataset)
            return dset

    async def read_multiple(fpaths):
        futures = [read_coco_file(fpath) for fpath in fpaths]
        results = list(asyncio.as_completed(futures))
        return results
        # for future in asyncio.as_completed(futures):
        #     await future

    def read_coco_async(fpaths):
        loop = asyncio.get_event_loop()
        coroutine = read_multiple(fpaths)
        result = loop.run_until_complete(coroutine)
        return
    fpath = prediction_fpaths[0]

    loop = asyncio.get_event_loop()

    x = read_coco_file(fpath)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(read_coco_file(fpath)))
    list(asyncio.as_completed([asyncio.Task(read_coco_file(fpath))]))

    fpaths = prediction_fpaths = prediction_fpaths[0:10]
    submitted = [read_coco_file(fpath) for fpath in prediction_fpaths]
    gathered = asyncio.gather(*submitted)

    z = loop.run_until_complete(gathered)
    print('z = {!r}'.format(z))
    z = asyncio.gather(x)
