from os.path import exists
from os.path import join


def upgrade_deployed_mmdet_model():
    """
    5dd3181eaf2e2eed3505827c
    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3eb8eaf2e2eed3508d604
    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3181eaf2e2eed3505827c

    girder-client --api-url https://data.kitware.com/api/v1 download 5eb9c21f9014a6d84e638b49 $HOME/tmp/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip

    girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604 $HOME/tmp/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip
    girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604 $HOME/tmp/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip
    """
    import ubelt as ub
    deploy_fpath = ub.expandpath('$HOME/tmp/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip')
    # deploy_fpath = ub.expandpath('$HOME/tmp/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip')
    from torch_liberator import deployer
    deployed = deployer.DeployedModel(deploy_fpath)

    # self = deployed
    # snap_fpath = self.info['snap_fpath']
    # # Extract the snapshot fpath to disk
    # from torch_liberator.util.util_zip import split_archive
    # import zipfile
    # archive_fpath, internal = split_archive(snap_fpath)
    # if archive_fpath is None:
    #     raise Exception('deployed snapshot is not in an archive')
    # with zipfile.ZipFile(archive_fpath, 'r') as myzip:
    #     myzip.extract(internal, extract_dpath)
    # temp_fpath = join(extract_dpath, internal)
    # print('temp_fpath = {!r}'.format(temp_fpath))
    # assert exists(temp_fpath)

    extract_dpath = ub.ensure_app_cache_dir('torch_liberator/extracted')
    temp_fpath = deployed.extract_snapshot(extract_dpath)

    import ndsampler
    model_cls, model_initkw = deployed.model_definition()
    old_classes = ndsampler.CategoryTree.coerce(model_initkw['classes'])
    num_classes = len(old_classes)

    # old mmdet has background as class 0, new has it as class K
    # https://mmdetection.readthedocs.io/en/latest/compatibility.html#codebase-conventions
    if 'background' in old_classes:
        num_classes_old = num_classes - 1
        new_classes = ndsampler.CategoryTree.from_mutex(list((ub.oset(list(old_classes)) - {'background'}) | {'background'}), bg_hack=False)
    else:
        num_classes_old = num_classes
        new_classes = old_classes

    import xinspect
    # model_src = print(inspect.getsource(model_cls.__init__))
    model_src = xinspect.dynamic_kwargs.get_func_sourcecode(model_cls.__init__, strip_def=True)

    import netharn as nh
    import torch
    xpu = nh.XPU.coerce('cpu')
    old_snapshot = xpu.load(temp_fpath)
    # Extract just the model state
    model_state = old_snapshot['model_state_dict']

    model_state_2 = {k.replace('module.detector.', ''): v for k, v in model_state.items()}

    # These are handled by the initkw
    model_state_2.pop('module.input_norm.mean', None)
    model_state_2.pop('module.input_norm.std', None)

    # Add major hacks to the config string to attempt to re-create what mmdet can handle
    config_strings = model_src.replace('mm_config', 'model')
    config_strings = 'from netharn.data.channel_spec import ChannelSpec\n' + config_strings
    config_strings = 'import ubelt as ub\n' + config_strings
    config_strings = 'classes = {!r}\n'.format(list(old_classes)) + config_strings
    if 'in_channels' in model_initkw:
        config_strings = 'in_channels = {!r}\n'.format(model_initkw['in_channels']) + config_strings
    if 'channels' in model_initkw:
        config_strings = 'channels = {!r}\n'.format(model_initkw['channels']) + config_strings
    if 'input_stats' in model_initkw:
        config_strings = 'input_stats = {!r}\n'.format(model_initkw['input_stats']) + config_strings
    config_strings = config_strings[:config_strings.find('_hack_mm_backbone_in_channels')]
    config_strings = config_strings.replace('self.', '')
    print(config_strings)

    checkpoint = {
        'state_dict': model_state_2,
        'meta': {
            # hack in mmdet metadata
            'mmdet_version': '1.0.0',
            'config': config_strings,
        },
    }

    config_strings = checkpoint['meta']['config']
    in_file = ub.augpath(temp_fpath, suffix='_prepared')
    torch.save(checkpoint, in_file)

    # checkpoint = torch.load(in_file)

    upgrade_fpath = ub.expandpath('~/code/mmdetection/tools/upgrade_model_version.py')
    upgrade_module = ub.import_module_from_path(upgrade_fpath)

    out_file = ub.augpath(temp_fpath, suffix='_upgrade2x')
    upgrade_module.convert(in_file, out_file, num_classes_old + 1)

    from bioharn.models import mm_models
    input_stats = model_initkw['input_stats']
    new_initkw = dict(classes=new_classes.__json__(), channels='rgb', input_stats=input_stats)
    new_model = mm_models.MM_CascadeRCNN(**new_initkw)
    new_model._initkw = new_initkw

    print(new_model.detector.roi_head.bbox_head[0].fc_cls.weight.shape)

    new_model_state = torch.load(out_file)
    print(model_state_2['bbox_head.0.fc_cls.weight'].shape)
    print(new_model_state['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].shape)

    _ = new_model.detector.load_state_dict(new_model_state['state_dict'])

    batch = {
        'inputs': {
            'rgb': torch.rand(1, 3, 256, 256),
        }
    }
    outputs = new_model.forward(batch, return_loss=False)
    batch_dets = new_model.coder.decode_batch(outputs)
    dets = batch_dets[0]

    import copy
    new_train_info = copy.deepcopy(deployed.train_info())
    new_train_info['__mmdet_conversion__'] = '1x_to_2x'

    new_train_info_fpath = join(extract_dpath, 'train_info.json')
    new_snap_fpath = join(extract_dpath, 'converted_deploy_snapshot.pt')
    import json
    with open(new_train_info_fpath, 'w') as file:
        json.dump(new_train_info, file, indent='    ')

    new_snapshot = {
        'model_state_dict': new_model.state_dict(),
        # {'detector.' + k: v for k, v in new_model_state['state_dict'].items()},
        'epoch': old_snapshot['epoch'],
    }
    torch.save(new_snapshot, new_snap_fpath)

    new_deployed = deployer.DeployedModel.custom(
        model=new_model, snap_fpath=new_snap_fpath,
        train_info_fpath=new_train_info_fpath, initkw=new_initkw)

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_mm2x')
    fpath = new_deployed.package(dpath=extract_dpath, name=new_name)
    print('fpath = {!r}'.format(fpath))
