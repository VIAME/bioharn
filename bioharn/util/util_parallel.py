import ubelt as ub
import queue
from threading import Thread


class _AsyncConsumerThread(Thread):
    """
    Will fill the queue with content of the source in a separate thread.

    Example:
        >>> import queue
        >>> q = queue.Queue()
        >>> c = _AsyncConsumerThread(q, range(3))
        >>> c.start()
        >>> q.get(True, 1)
        0
        >>> q.get(True, 1)
        1
        >>> q.get(True, 1)
        2
        >>> q.get(True, 1) is ub.NoParam
        True
    """
    def __init__(self, queue, source):
        Thread.__init__(self)

        self._queue = queue
        self._source = source

    def run(self):
        for item in self._source:
            self._queue.put(item)
        # Signal the consumer we are done.
        self._queue.put(ub.NoParam)


class AsyncBufferedGenerator(object):
    """Buffers content of an iterator polling the contents of the given
    iterator in a separate thread.
    When the consumer is faster than many producers, this kind of
    concurrency and buffering makes sense.

    The size parameter is the number of elements to buffer.

    The source must be threadsafe.

    References:
        http://code.activestate.com/recipes/576999-concurrent-buffer-for-generators/
    """
    def __init__(self, source, size=100):
        self._queue = queue.Queue(size)

        self._poller = _AsyncConsumerThread(self._queue, source)
        self._poller.daemon = True
        self._poller.start()

    def __iter__(self):
        while True:
            item = self._queue.get(True)
            if item is ub.NoParam:
                return
            yield item


def atomic_move(src, dst):
    """
    Rename a file from ``src`` to ``dst``, atomically if possible.

    Args:
        src (str | PathLike): file path to an existing file
        dst (str | PathLike): file path to a destination file

    References:
        .. [1] https://alexwlchan.net/2019/03/atomic-cross-filesystem-moves-in-python/

    Notes:
        *   Moves must be atomic.  ``shutil.move()`` is not atomic.
            Note that multiple threads may try to write to the cache at once,
            so atomicity is required to ensure the serving on one thread doesn't
            pick up a partially saved image from another thread.

        *   Moves must work across filesystems.  Often temp directories and the
            cache directories live on different filesystems.  ``os.rename()`` can
            throw errors if run across filesystems.

        So we try ``os.rename()``, but if we detect a cross-filesystem copy, we
        switch to ``shutil.move()`` with some wrappers to make it atomic.

    Example:
        >>> import ubelt as ub
        >>> from os.path import join, exists
        >>> dpath = ub.ensure_app_cache_dir('ubelt')
        >>> fpath1 = join(dpath, 'foo')
        >>> fpath2 = join(dpath, 'bar')
        >>> ub.touch(fpath1)
        >>> atomic_move(fpath1, fpath2)
        >>> assert not exists(fpath1)
        >>> assert exists(fpath2)
    """
    import os
    try:
        os.rename(src, dst)
    except OSError as err:
        import errno
        import shutil
        if err.errno == errno.EXDEV:
            import uuid
            # Generate a unique ID, and copy `<src>` to the target directory
            # with a temporary name `<dst>.<ID>.tmp`.  Because we're copying
            # across a filesystem boundary, this initial copy may not be
            # atomic.  We intersperse a random UUID so if different processes
            # are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = "%s.%s.tmp" % (dst, copy_id)
            shutil.copyfile(src, tmp_dst)

            # Then do an atomic rename onto the new name, and clean up the
            # source image.
            os.rename(tmp_dst, dst)
            os.unlink(src)
        else:
            raise


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/util/util_parallel.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
