from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import ubelt as ub
import kwarray


class ClfDataset(torch.utils.data.Dataset):
    """
    Efficient loader for classification training on coco samplers.

    This is a normal torch dataset that uses :module:`ndsampler` and
    :module:`imgaug` for data loading an augmentation.

    It also contains a ``make_loader`` method for creating a class balanced
    DataLoader. There is little netharn-specific about this class.

    Example:
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo()
        >>> self = ClfDataset(sampler)
        >>> index = 0
        >>> self[index]['inputs']['rgb'].shape
        >>> loader = self.make_loader(batch_size=8, shuffle=True, num_workers=0, num_batches=10)
        >>> for batch in ub.ProgIter(iter(loader), total=len(loader)):
        >>>     break
        >>> print('batch = {}'.format(ub.repr2(batch, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(batch['inputs']['rgb'][0])
    """
    def __init__(self, sampler, input_dims=(256, 256), augment=None):
        self.sampler = sampler
        self.augment = augment
        self.conditional_augmentors = None
        self.input_dims = input_dims
        self.classes = self.sampler.catgraph

        self.disable_augmenter = not bool(augment)
        self.augmenter = self._coerce_augmenter(augment)

    def __len__(self):
        return self.sampler.n_positives

    @ub.memoize_property
    def input_id(self):
        # TODO: reset memoize dict if augment / other param is changed
        def imgaug_json_id(aug):
            import imgaug
            if isinstance(aug, tuple):
                return [imgaug_json_id(item) for item in aug]
            elif isinstance(aug, imgaug.parameters.StochasticParameter):
                return str(aug)
            else:
                try:
                    info = ub.odict()
                    info['__class__'] = aug.__class__.__name__
                    params = aug.get_parameters()
                    if params:
                        info['params'] = [imgaug_json_id(p) for p in params]
                    if isinstance(aug, list):
                        children = aug[:]
                        children = [imgaug_json_id(c) for c in children]
                        info['children'] = children
                    return info
                except Exception:
                    # imgaug is weird and buggy
                    return str(aug)
        depends = [
            self.sampler._depends(),
            self.augmenter and imgaug_json_id(self.augmenter),
        ]
        _input_id = ub.hash_data(depends, hasher='sha512', base='abc')[0:40]
        return _input_id

    def __getitem__(self, index):
        import kwimage

        # Load sample image and category
        rng = kwarray.ensure_rng(None)
        tr = self.sampler.regions.get_positive(index, rng=rng)

        # always sample a square region with a minimum size
        dim = max(tr['width'], tr['height'])
        dim = max(dim, 64)

        if not self.disable_augmenter:
            if rng.rand() > 0.5:
                # sometimes add a random pad
                dim += int((rng.rand() * 0.2) * dim)

        tr['width'] = tr['height'] = dim
        sample = self.sampler.load_sample(tr, with_annots=False)

        image = kwimage.atleast_3channels(sample['im'])[:, :, 0:3]
        target = sample['tr']

        image = kwimage.ensure_uint255(image)
        if self.augmenter is not None and not self.disable_augmenter:
            det = self.augmenter.to_deterministic()
            image = det.augment_image(image)

        # Resize to input dimensinos
        if self.input_dims is not None:
            dsize = tuple(self.input_dims[::-1])
            image = kwimage.imresize(image, dsize=dsize, letterbox=True)

        class_id_to_idx = self.sampler.classes.id_to_idx
        cid = target['category_id']
        cidx = class_id_to_idx[cid]

        im_chw = image.transpose(2, 0, 1) / 255.0
        inputs = {
            'rgb': torch.FloatTensor(im_chw),
        }
        labels = {
            'class_idxs': cidx,
        }
        batch = {
            'inputs': inputs,
            'labels': labels,
        }
        return batch

    def _coerce_augmenter(self, augment):
        import netharn as nh
        import imgaug.augmenters as iaa
        import imgaug as ia
        if augment is True:
            augment = 'simple'
        if not augment:
            augmenter = None
        elif augment == 'simple':
            augmenter = iaa.Sequential([
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        elif augment == 'medium':
            augmenter = iaa.Sequential([
                iaa.Sometimes(0.2, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        elif augment == 'complex':
            gravity = 0
            self._geometric = iaa.Sequential([
                iaa.Sometimes(0.55, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-15, 15),
                    shear=ia.parameters.TruncatedNormal(0.0, 2.5, low=-16, high=16),
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
            ] + ([iaa.Rot90(k=[0, 1, 2, 3])] * int(1 - gravity))  +
                [
                    iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),
                 ],
            )
            self._rgb_intensity = iaa.Sequential([
                # Color, brightness, saturation, and contrast
                iaa.Sometimes(0.1, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Sometimes(.10, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(.10, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(.10, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Add((-10, 10), per_channel=0.5)),
                iaa.Sometimes(.1, iaa.Grayscale(alpha=(0, 1))),

                # Speckle noise
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
                iaa.Sometimes(.1, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (.09, .31), size_percent=(0.19, 0.055),
                        per_channel=0.15
                    ),
                ])),

                # Blurring
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.GaussianBlur((0, 2.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ])),

                # Sharpening
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.Sometimes(.1, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                    iaa.Sometimes(.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                ])),

                # Misc
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
            ], random_order=True)

            augmenter = []

            self._disp_intensity = iaa.Sequential([
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ], random_order=True)

            # TODO: better multi-channel augmentation
            augmenter = iaa.Sequential([
                self._rgb_intensity,
                self._geometric,
            ], random_order=False)
        else:
            raise KeyError('Unknown augmentation {!r}'.format(self.augment))
        return augmenter

    def make_loader(self, batch_size=16, num_batches='auto', num_workers=0,
                    shuffle=False, pin_memory=False, drop_last=False,
                    balance=None):

        if len(self) == 0:
            raise Exception('must have some data')

        def worker_init_fn(worker_id):
            for i in range(worker_id + 1):
                seed = np.random.randint(0, int(2 ** 32) - 1)
            seed = seed + worker_id
            kwarray.seed_global(seed)
            if self.augmenter:
                rng = kwarray.ensure_rng(None)
                self.augmenter.seed_(rng)

        loaderkw = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': worker_init_fn,
        }
        if balance is None:
            loaderkw['shuffle'] = shuffle
            loaderkw['batch_size'] = batch_size
            loaderkw['drop_last'] = drop_last
        elif balance == 'classes':
            from netharn.data.batch_samplers import BalancedBatchSampler
            index_to_cid = [
                cid for cid in self.sampler.regions.targets['category_id']
            ]
            batch_sampler = BalancedBatchSampler(
                index_to_cid, batch_size=batch_size,
                shuffle=shuffle, num_batches=num_batches)
            loaderkw['batch_sampler'] = batch_sampler
        else:
            raise KeyError(balance)

        loader = torch.utils.data.DataLoader(self, **loaderkw)
        return loader
