import ubelt as ub


class ChannelSpec(ub.NiceRepr):
    """
    Parse and extract information about network input channel specs for
    early or late fusion networks.

    Notes:
        The pipe ('|') character represents an early-fused input stream, and
        order matters (it is non-communative).

        The comma (',') character separates different inputs streams/branches
        for a multi-stream/branch network which will be lated fused. Order does
        not matter

    TODO:
        - [ ] : normalize representations? e.g: rgb = r|g|b?

    Example:
        >>> from bioharn.channel_spec import *  # NOQA
        >>> self = ChannelSpec('gray')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity,disparity')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb,disparity,flowx|flowy')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
    """

    _known = {
        'rgb': 'r|g|b'
    }

    def __init__(self, spec):
        self.spec = spec

    def __nice__(self):
        return self.spec

    def __json__(self):
        return self.spec

    @property
    def info(self):
        return {
            'spec': self.spec,
            'parsed': self.parse(),
            'unique': self.unique(),
            'normed': self.normalize(),
        }

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            self = data
        else:
            self = cls(data)
        return self

    def parse(self):
        """
        Build internal representation
        """
        # commas break inputs into multiple streams
        stream_specs = self.spec.split(',')
        parsed = {ss: ss.split('|') for ss in stream_specs}
        return parsed

    def normalize(self):
        spec = self.spec
        stream_specs = spec.split(',')
        parsed = {ss: ss for ss in stream_specs}
        for k1 in parsed.keys():
            for k, v in self._known.items():
                parsed[k1] = parsed[k1].replace(k, v)
        parsed = {k: v.split('|') for k, v in parsed.items()}
        return parsed

    def unique(self):
        """
        Returns the unique channels that will need to be given or loaded
        """
        return set(ub.flatten(self.parse().values()))

    def encode(self, item, axis=0, drop=False):
        """
        Given a dictionary containing preloaded components of the network
        inputs, build a concatenated network representations of each input
        stream.

        Args:
            item (dict): a batch item
            axis (int, default=0): concatenation dimension
            drop (bool, default=False): if True, drop the unprocessed
                components of the input.

        Returns:
            Dict[str, Tensor]: mapping between input stream and its early fused
                tensor input.

        Example:
            >>> import torch
            >>> dims = (4, 4)
            >>> item = {
            >>>     'rgb': torch.rand(3, *dims),
            >>>     'disparity': torch.rand(1, *dims),
            >>>     'flowx': torch.rand(1, *dims),
            >>>     'flowy': torch.rand(1, *dims),
            >>> }
            >>> # Complex Case
            >>> self = ChannelSpec('rgb,disparity,rgb|disparity|flowx|flowy,flowx|flowy')
            >>> inputs = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
            >>> print('input_shapes = {}'.format(ub.repr2(input_shapes, nl=1)))
            >>> # Simpler case
            >>> self = ChannelSpec('rgb|disparity')
            >>> inputs = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
            >>> print('input_shapes = {}'.format(ub.repr2(input_shapes, nl=1)))
        """
        import torch
        inputs = {}
        parsed = self.parse()
        unique = self.unique()
        if drop:
            components = {k: item.pop(k) for k in unique}
        else:
            components = {k: item[k] for k in unique}
        for key, parts in parsed.items():
            inputs[key] = torch.cat([components[k] for k in parts], dim=axis)
        return inputs
