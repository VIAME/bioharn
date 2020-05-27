from os.path import basename
from os.path import exists
from os.path import join
import glob
import ubelt as ub
import kwcoco


def prototype_prep_detection_reclassify():

    prediction_dpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/pred")

    prediction_fpaths = list(glob.glob(join(prediction_dpath, '*.mscoco.json')))
    fpaths =  prediction_fpaths[0:10]

    truth_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali.mscoco.json')
    true_dset = kwcoco.CocoDataset(truth_fpath)

    # prediction_dsets = [
    #     kwcoco.CocoDataset(fpath)
    #     for fpath in ub.ProgIter(prediction_fpaths, desc='load detections')
    # ]

    pred_dsets = results = thread_based_multi_read(prediction_fpaths)
    pred_dset = kwcoco.CocoDataset.union(*pred_dsets)

    if not exists(pred_dset.get_image_fpath(1)):
        print('need to reroot pred dset')
        orig_img_root = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos')
        pred_dset.img_root = orig_img_root
        if exists(pred_dset.get_image_fpath(1)):
            pred_dset.reroot(orig_img_root, absolute=True)
            assert exists(pred_dset.imgs[1]['file_name'])

    pred_fpath = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/all_pred.mscoco.json")
    pred_dset.dump(pred_fpath, newlines=True)

    truth_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos')


def thread_based_multi_read(fpaths, max_workers=8, verbose=0):
    # Can this be done better with asyncio?
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool('thread', max_workers=max_workers)
    for fpath in ub.ProgIter(fpaths, desc='submit load jobs', verbose=verbose):
        jobs.submit(kwcoco.CocoDataset, fpath)
    results = [f.result() for f in ub.ProgIter(jobs.as_completed(), desc='collect load jobs', total=len(jobs), verbose=verbose)]
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
