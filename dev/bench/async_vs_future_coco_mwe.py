"""
Based on regular MWE example, but uses coco
"""
from os.path import join
import ubelt as ub
import json
import tempfile
import random
import asyncio
import aiofiles
from concurrent import futures
import kwcoco


def load_json_worker(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data


def multi_read_threads(fpaths, max_workers=8, verbose=0):
    # Can this be done better with asyncio?
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool('thread', max_workers=max_workers)
    for fpath in ub.ProgIter(fpaths, desc='submit load jobs', verbose=verbose):
        jobs.submit(kwcoco.CocoDataset, fpath)
    gen = ub.ProgIter(jobs.as_completed(), desc='collect load jobs',
                      total=len(jobs), verbose=verbose)
    results = [f.result() for f in gen]
    return results


def multi_reads_async(fpaths, max_workers=8):
    """
    Is this the right way to use asyncio?
    """
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
                data = json.loads(text)
            items.append(data)
        return items

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=max_workers)
    producer_coro = produce(queue, fpaths)

    consumer_coro = consume(queue)

    gathered = asyncio.gather(producer_coro, consumer_coro)
    produce_got, consume_got = loop.run_until_complete(gathered)
    results = consume_got
    return results


def create_toydata(num_files=100):
    rng = random.Random(0)

    dpath = tempfile.mkdtemp()
    fpaths = [join(dpath, 'file_{:08d}.json'.format(idx))
              for idx in range(num_files)]

    for fpath in fpaths:
        with open(fpath, 'w') as file:
            kwcoco.CocoDataset.demo().dump(file)

    # display file size
    ub.cmd('du -sh {}'.format(dpath), verbose=1)
    return fpaths


def multi_read_serial(fpaths):
    results = [load_json_worker(fpath) for fpath in fpaths]
    return results


def benchmark():
    import timerit
    ti = timerit.Timerit(10, bestof=3, verbose=2)

    max_workers = 8
    num_files = 1000

    fpaths = create_toydata(num_files)

    for timer in ti.reset('asyncio'):
        with timer:
            datasets_async = multi_reads_async(fpaths, max_workers=max_workers)

    for timer in ti.reset('thread'):
        with timer:
            datasets_thread = multi_read_threads(fpaths, max_workers=max_workers)

    for timer in ti.reset('serial'):
        with timer:
            datasets_serial = multi_read_serial(fpaths)
