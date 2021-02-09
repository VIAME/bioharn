import ubelt as ub


def _autogen_class_from_mm_config(fpath):
    """
    Autogen code corresonding to a mmdetection config.

    cd $HOME/code/mmdetection/configs
    find . -iname "*mask*"
    find . -iname "*mask*" | grep -i fcos
    find . -iname "*mask*" | grep -i cascade
    find . -iname "*mask*" | grep -i hrnet
    find . -iname "*mask*" | grep -i hrnet

    fpath = ub.expandpath('$HOME/code/mmdetection/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py')
    fpath = ub.expandpath('$HOME/code/mmdetection/configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py')
    _autogen_class_from_mm_config(fpath)
    """
    from mmcv import Config
    from os.path import dirname, exists, join
    mm_config = Config.fromfile(fpath)

    def _get_config_directory():
        """Find the predefined detector config directory."""
        try:
            # Assume we are running in the source mmdetection repo
            repo_dpath = dirname(dirname(dirname(__file__)))
        except NameError:
            # For IPython development when this __file__ is not defined
            import mmdet
            repo_dpath = dirname(dirname(mmdet.__file__))
        config_dpath = join(repo_dpath, 'configs')
        if not exists(config_dpath):
            raise Exception('Cannot find config path')
        return config_dpath

    from kwcoco.util.util_json import IndexableWalker
    mm_model = mm_config['model'].to_dict()
    train_cfg = mm_config['train_cfg'].to_dict()
    test_cfg = mm_config['test_cfg'].to_dict()

    mm_cfg = {
        'model': mm_model,
        'train_cfg': train_cfg,
        'test_cfg': test_cfg,
    }

    class ForwardReprRef(object):
        """
        Helper for ensuring text representation can contain forward
        references to variables that will be defined in the context of
        the template init
        """
        def __init__(self, value):
            self.value = value
        def __repr__(self):
            return str(self.value)
        def __str__(self):
            return str(self.value)

    walker = IndexableWalker(mm_cfg)
    pretrained_url = None
    for path, value in walker:
        if path[-1] == 'pretrained':
            pretrained_url = value
            walker[path] = None
        if isinstance(value, dict) and 'type' in value:
            if value['type'] == 'HRNet':
                value['in_channels'] = ForwardReprRef('in_channels')
        if path[-1] == 'num_classes':
            walker[path] = ForwardReprRef('len(classes)')

    mm_cfg_text = 'mm_cfg = mmcv.Config(' + ub.repr2(mm_cfg, nl=-1) + ')'

    template = ub.codeblock(
        '''
        class {classname}(MM_Detector):
            """
            Example:
                >>> # xdoctest: +REQUIRES(module:mmdet)
                >>> # xdoctest: +REQUIRES(--cuda)
                >>> self = {classname}(classes=3)
                >>> print(nh.util.number_of_parameters(self))
                >>> self.to(0)
                >>> batch = self.demo_batch()
                >>> outputs = self.forward(batch)
                >>> batch_dets = self.coder.decode_batch(outputs)
            """
            pretrained_url = {pretrained_url!r}

            def __init__(self, classes=None, input_stats=None, channels='rgb'):
                import mmcv
                import kwcoco
                classes = kwcoco.CategoryTree.coerce(classes)
                channels = ChannelSpec.coerce(channels)
                chann_norm = channels.normalize()
                assert len(chann_norm) == 1
                in_channels = len(ub.peek(chann_norm.values()))

        {mm_cfg_def}

                super().__init__(
                        mm_cfg['model'], train_cfg=mm_cfg['train_cfg'],
                        test_cfg=mm_cfg['test_cfg'],
                        classes=classes, input_stats=input_stats,
                        channels=channels)
        ''')

    fmtkw = {
        'classname': 'MM_{}_{}'.format(mm_model['backbone']['type'], mm_model['type']),
        'mm_cfg_def': ub.indent(mm_cfg_text, ' ' * 8),
        'pretrained_url': pretrained_url,
    }

    text = template.format(**fmtkw)
    print(text)
