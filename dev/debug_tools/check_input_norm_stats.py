
def check_deployed_input_norm(deployed_fpath):

    """
    deployed_fpath = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/name/bioharn-det-hrmask18-rgb-only-habcam-v5-adapt/deploy_MM_HRNetV2_w18_MaskRCNN_udmzrkmb_003_FKJTWB.zip')

    deployed_fpath = ub.expandpath('/home/joncrall/work/bioharn/fit/runs/bioharn-det-hrmask18-rgb-only-shapes-v3/hfksgkob/deploy_MM_HRNetV2_w18_MaskRCNN_hfksgkob_000_LPJXQI.zip')
    """

    import torch_liberator
    import ubelt as ub
    import kwarray
    import torch
    deployed = torch_liberator.DeployedModel.coerce(deployed_fpath)

    cls, initkw = deployed.model_definition()
    orig_input_stats = initkw['input_stats']
    for chan, chan_stats in orig_input_stats.items():
        orig_input_stats[chan] = ub.map_vals(kwarray.ArrayAPI.numpy, chan_stats)

    model = deployed.load_model()

    model_mean = model.detector.backbone.chan_backbones.rgb.input_norm.mean
    model_std = model.detector.backbone.chan_backbones.rgb.input_norm.std

    print('model_mean = {!r}'.format(model_mean))
    print('model_std = {!r}'.format(model_std))

    model_state = model.state_dict()
    keys = [k for k in model_state.keys() if 'input_norm' in k]
    saved_input_stats = ub.dict_isect(model_state, keys)
    saved_input_stats = ub.map_vals(kwarray.ArrayAPI.numpy, saved_input_stats)

    temp_fpath = deployed.extract_snapshot()

    disk_state = torch.load(temp_fpath)
    disk_model_state = disk_state['model_state_dict']
    keys = [k for k in disk_model_state.keys() if 'input_norm' in k]
    disk_input_stats = ub.dict_isect(disk_model_state, keys)
    disk_input_stats = ub.map_vals(kwarray.ArrayAPI.numpy, disk_input_stats)

    print('orig_input_stats = {}'.format(ub.repr2(orig_input_stats, nl=3)))
    print('saved_input_stats = {}'.format(ub.repr2(saved_input_stats, nl=2)))
    print('disk_input_stats = {}'.format(ub.repr2(disk_input_stats, nl=2)))
