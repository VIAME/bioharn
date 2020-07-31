from os.path import join
import ubelt as ub
import json
import tempfile
import random
import asyncio
import aiofiles
from concurrent import futures


def load_json_worker(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data


def multi_read_threads(fpaths, max_workers=8, verbose=0):
    # Can this be done better with asyncio?
    executor = futures.ThreadPoolExecutor(max_workers=max_workers)
    with executor:
        jobs = [
            executor.submit(load_json_worker, fpath)
            for fpath in ub.ProgIter(fpaths, desc='submit read jobs', verbose=verbose)
        ]
        results = [
            f.result()
            for f in ub.ProgIter(
                futures.as_completed(jobs), total=len(jobs),
                desc='collect read jobs', verbose=verbose)
        ]
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


def random_json(rng, max_container_size=100, max_depth=1):

    max_str_size = 20

    max_depth = 2

    def _randstr():
        return str(rng.getrandbits(max_str_size))

    def _gen(depth=0):
        if depth >= max_depth:
            mode = rng.choice(['num', 'str'])
        else:
            mode = rng.choice(['list', 'dict'])
            # mode = rng.choice(['num', 'str', 'list', 'dict'])
        if mode == 'num':
            data = rng.randint(0, int(2 ** 31))
        elif mode == 'str':
            data = _randstr()
        elif mode == 'list':
            size = rng.randint(0, max_container_size)
            data = [_gen(depth=depth + 1) for _ in range(size)]
        elif mode == 'dict':
            size = rng.randint(0, max_container_size)
            data = {_randstr(): _gen(depth=depth + 1)
                    for _ in range(size)}
        return data

    data = _gen(depth=0)
    return data


def create_toydata(num_files=100, max_container_size=100):
    rng = random.Random(0)

    dpath = tempfile.mkdtemp()
    fpaths = [join(dpath, 'file_{:08d}.json'.format(idx))
              for idx in range(num_files)]

    for fpath in ub.ProgIter(fpaths, desc='make random data'):
        with open(fpath, 'w') as file:
            data = random_json(rng, max_container_size)
            json.dump(data, file)

    # if 0:
    # display file size
    ub.cmd('du -sh {}'.format(dpath), verbose=1)
    return fpaths


def multi_read_serial(fpaths):
    results = [load_json_worker(fpath) for fpath in fpaths]
    return results


def benchmark():

    max_workers = 8
    num_files = 400

    fpaths = create_toydata(num_files, max_container_size=200)

    import timerit
    ti = timerit.Timerit(10, bestof=3, verbose=2)

    for timer in ti.reset('asyncio'):
        with timer:
            datasets_async = multi_reads_async(fpaths, max_workers=max_workers)

    for timer in ti.reset('thread'):
        with timer:
            datasets_thread = multi_read_threads(fpaths, max_workers=max_workers)

    for timer in ti.reset('serial'):
        with timer:
            datasets_serial = multi_read_serial(fpaths)
