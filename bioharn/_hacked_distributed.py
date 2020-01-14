"""
Proof-of-concept for porting mmcv DataContainer concept to netharn. Depending
on how well this works these features might be useful as a standalone module or
to contribute to torch proper.

References:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/scatter_gather.py

FIXME 0 dimension tensors
"""
import torch.utils.data as torch_data
import torch
import ubelt as ub
import numpy as np  # NOQA
import re
import collections
import torch.nn.functional as F
# from torch.nn.parallel import DataParallel
from netharn.device import DataParallel, XPU, DataSerial
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel._functions import Scatter as OrigScatter

# if six.PY2:
#     import collections
#     container_abcs = collections
# elif six.PY3:
#     import collections.abc
#     container_abcs = collections.abc
# string_classes = six.string_types
# int_classes = six.integer_types
from torch._six import container_abcs
from torch._six import string_classes, int_classes
default_collate = torch_data.dataloader.default_collate


# numpy_type_map = torch_data.dataloader.numpy_type_map  # moved in torch 1.1.0
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class CollateException(Exception):
    pass


_DEBUG = False


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @classmethod
    def demo(cls, key='img', rng=None):
        """
        Create data for tests
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        if key == 'img':
            data = rng.rand(3, 512, 512)
            data = torch.from_numpy(data)
            self = cls(data, stack=True)
        elif key == 'labels':
            n = rng.randint(0, 10)
            data = rng.randint(0, 10, n)
            data = torch.from_numpy(data)
            self = cls(data, stack=False)
        else:
            raise KeyError(key)
        return self

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    def dim(self):
        return self.data.dim()


def container_collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes

    Ignore:
        >>> # DISABLE_DOCTSET
        >>> from bioharn.detect_dataset import *  # NOQA
        >>> from bioharn._hacked_distributed import *  # NOQA
        >>> dataset = DetectFitDataset.demo(key='shapes8', augment='complex', window_dims=(512, 512), gsize=(1920, 1080))

        >>> batch = [dataset[0], dataset[1], dataset[2]]
        >>> raw_batch = container_collate(batch)

        >>> target_gpus = [0]
        >>> inputs, kwargs = hack_scatter_kwargs(raw_batch, {}, target_gpus)

        >>> loader = torch.utils.data.DataLoader(dataset, collate_fn=container_collate, num_workers=0)


    Example:
        >>> item1 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item2 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item3 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> batch = items = [item1, item2, item3]
        >>> raw_batch = container_collate(items)

        >>> batch = [
        >>>     {'im': DataContainer.demo('img'), 'label': DataContainer.demo('labels')},
        >>>     {'im': DataContainer.demo('img'), 'label': DataContainer.demo('labels')},
        >>>     {'im': DataContainer.demo('img'), 'label': DataContainer.demo('labels')},
        >>> ]
        >>> raw_batch = container_collate(batch)
    """

    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [container_collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {
            key: container_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return _collate_else(batch, container_collate)


def _collate_else(batch, collate_func):
    """
    Handles recursion in the else case for these special collate functions

    This is duplicates all non-tensor cases from `torch_data.dataloader.default_collate`
    This also contains support for collating slices.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], slice):
        batch = default_collate([{
            'start': sl.start,
            'stop': sl.stop,
            'step': 1 if sl.step is None else sl.step
        } for sl in batch])
        return batch
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        # Hack the mapping collation implementation to print error info
        if _DEBUG:
            collated = {}
            try:
                for key in batch[0]:
                    collated[key] = collate_func([d[key] for d in batch])
            except Exception:
                print('\n!!Error collating key = {!r}\n'.format(key))
                raise
            return collated
        else:
            return {key: collate_func([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_func(samples) for samples in transposed]
    else:
        raise TypeError((error_msg.format(type(batch[0]))))


def list_collate(inbatch):
    """
    Collates batches containing items with non-uniform data sizes

    Used for detection datasets with boxes.

    Args:
        inbatch: a list of items returned by __getitem__ for each item in the
            batch

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for i in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 4, 4)  # dummy 4x4 image
        >>>     boxes = torch.LongTensor([[0, 0, 1, 1]] * i)
        >>>     item = (img, boxes)
        >>>     inbatch.append(item)
        >>> out_batch = list_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> batch_img, batch_boxes = out_batch
        >>> assert list(out_batch[0].shape) == [bsize, 3, 4, 4]
        >>> assert len(out_batch[1]) == bsize
        >>> assert len(out_batch[1][0]) == 0
        >>> assert len(out_batch[1][1]) == 1
        >>> assert len(out_batch[1][2]) == 2

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     boxes = torch.FloatTensor()
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = list_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert len(out_batch[1][0]) == bsize
    """
    try:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0 or num_items[0] == 0:
                    batch = inbatch
                else:
                    batch = default_collate(inbatch)
            else:
                batch = inbatch
        else:
            batch = _collate_else(inbatch, list_collate)
    except Exception as ex:
        if not isinstance(ex, CollateException):
            raise CollateException(
                'Failed to collate inbatch={}. Reason: {!r}'.format(inbatch, ex))
        else:
            raise
    return batch


def padded_collate(inbatch, fill_value=-1):
    """
    Used for detection datasets with boxes.

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 7
        >>> for i in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     n = 11 if i == 3 else rng.randint(0, 11)
        >>>     boxes = torch.rand(n, 4)
        >>>     item = (img, boxes)
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert list(out_batch[1].shape) == [bsize, 11, 4]

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     #boxes = torch.empty(0, 4)
        >>>     boxes = torch.FloatTensor()
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> #assert list(out_batch[1][0].shape) == [bsize, 0, 4]
        >>> assert list(out_batch[1][0].shape) in [[0], []]  # torch .3 a .4

    Example:
        >>> inbatch = [torch.rand(4, 4), torch.rand(8, 4),
        >>>            torch.rand(0, 4), torch.rand(3, 4),
        >>>            torch.rand(0, 4), torch.rand(1, 4)]
        >>> out_batch = padded_collate(inbatch)
        >>> assert list(out_batch.shape) == [6, 8, 4]
    """
    try:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0:
                    batch = torch.FloatTensor()
                elif num_items[0] == 0:
                    batch = torch.FloatTensor()
                else:
                    batch = default_collate(inbatch)
            else:
                max_size = max(num_items)
                real_tail_shape = None
                for item in inbatch:
                    if item.numel():
                        tail_shape = item.shape[1:]
                        if real_tail_shape is not None:
                            assert real_tail_shape == tail_shape
                        real_tail_shape = tail_shape

                padded_inbatch = []
                for item in inbatch:
                    n_extra = max_size - len(item)
                    if n_extra > 0:
                        shape = (n_extra,) + tuple(real_tail_shape)
                        if torch.__version__.startswith('0.3'):
                            extra = torch.Tensor(np.full(shape, fill_value=fill_value))
                        else:
                            extra = torch.full(shape, fill_value=fill_value,
                                               dtype=item.dtype)
                        padded_item = torch.cat([item, extra], dim=0)
                        padded_inbatch.append(padded_item)
                    else:
                        padded_inbatch.append(item)
                batch = inbatch
                batch = default_collate(padded_inbatch)
        else:
            batch = _collate_else(inbatch, padded_collate)
    except Exception as ex:
        if not isinstance(ex, CollateException):
            try:
                _debug_inbatch_shapes(inbatch)
            except Exception:
                pass
            raise CollateException(
                'Failed to collate inbatch={}. Reason: {!r}'.format(inbatch, ex))
        else:
            raise
    return batch


def _debug_inbatch_shapes(inbatch):
    import ubelt as ub
    print('len(inbatch) = {}'.format(len(inbatch)))
    extensions = ub.util_format.FormatterExtensions()

    @extensions.register((torch.Tensor, np.ndarray))
    def format_shape(data, **kwargs):
        return ub.repr2(dict(type=str(type(data)), shape=data.shape), nl=1, sv=1)

    print('inbatch = ' + ub.repr2(inbatch, extensions=extensions, nl=True))


# ----


def _fn_scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs.

    from mmcv.parallel._functions
    """
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            _fn_scatter(input[i], [devices[i // chunk_size]],
                          [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
            output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception('Unknown type {}.'.format(type(output)))


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


class HackedScatter(object):

    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]

        outputs = _fn_scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)

# ----


class Hacked_DataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return hack_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

# ----


def hack_scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    from mmcv.parallel.scatter_gather

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return HackedScatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def hack_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = hack_scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = hack_scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class Hacked_XPU(XPU):

    def mount(xpu, model):
        """
        Like move, but only for models.
        Note that this works inplace for non-Tensor objects.

        Args:
            model (torch.nn.Module): the model to mount

        Returns:
            DataSerial | DataParallel :
                the model mounted on the XPU (which may be multiple GPUs)

        Example:
            >>> model = torch.nn.Conv2d(1, 1, 1)
            >>> xpu = XPU()
        """
        # Unwrap the core model if necessary
        model = xpu.raw(model)
        model = xpu.move(model)
        if xpu._device_ids and len(xpu._device_ids) > 1:
            model = Hacked_DataParallel(
                model, device_ids=xpu._device_ids,
                output_device=xpu._main_device_id)
        else:
            model = DataSerial(model)
        return model
