from os.path import exists
from os.path import join
import scriptconfig as scfg
import torch
import ubelt as ub
import copy
import json


class UpdateBioharnConfig(scfg.Config):
    default = {
        'deployed': scfg.Path(None, help='path to torch_liberator zipfile to convert'),
        'use_cache': scfg.Path(False, help='do nothing if we already converted'),
    }


def update_deployed_bioharn_model(config):
    """
    Simply put old weights in a new model.
    If the code hasn't functionally changed then this should work.

    CLI:
        python -m bioharn.compat.update_bioharn_model \
            --deployed=$HOME/.cache/bioharn/deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS_sealion_coarse.zip
    """
    from torch_liberator import deployer
    import netharn as nh
    from bioharn.models import mm_models

    config = UpdateBioharnConfig(config)

    deploy_fpath = config['deployed']

    extract_dpath = ub.ensure_app_cache_dir('torch_liberator/extracted')

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_bio3x')
    new_fpath = join(extract_dpath, new_name)

    print('Upgrade deployed model config: config = {!r}'.format(config))

    if config['use_cache']:
        if exists(new_fpath):
            print('Returning cached new_fpath = {!r}'.format(new_fpath))
            return new_fpath

    deployed = deployer.DeployedModel(deploy_fpath)

    print('Extracting old snapshot to: {}'.format(extract_dpath))
    temp_fpath = deployed.extract_snapshot(extract_dpath)

    model_cls, model_initkw = deployed.model_definition()

    new_initkw = model_initkw
    new_model = mm_models.MM_CascadeRCNN(**new_initkw)
    new_model._initkw = new_initkw

    xpu = nh.XPU.coerce('cpu')
    new_model_state = xpu.load(temp_fpath)

    # print(new_model.detector.roi_head.bbox_head[0].fc_cls.weight.shape)
    # print(model_state_2['bbox_head.0.fc_cls.weight'].shape)
    # print(new_model_state['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].shape)

    from netharn.initializers.functional import load_partial_state
    load_info = load_partial_state(new_model, new_model_state['model_state_dict'], verbose=3)
    del load_info
    # new_model_state['model_state_dict']['input_norm.mean']
    # _ = new_model.load_state_dict(new_model_state['model_state_dict'])

    TEST_FORWARD = 0
    if TEST_FORWARD:
        batch = {
            'inputs': {
                'rgb': torch.rand(1, 3, 256, 256),
            }
        }
        outputs = new_model.forward(batch, return_loss=False)
        batch_dets = new_model.coder.decode_batch(outputs)
        dets = batch_dets[0]
        print('dets = {!r}'.format(dets))

    new_train_info = copy.deepcopy(deployed.train_info())
    new_train_info['__mmdet_conversion__'] = '1x_to_2x'
    new_train_info['__bioharn_model_vesion__'] = new_model.__bioharn_model_vesion__

    new_train_info_fpath = join(extract_dpath, 'train_info.json')
    new_snap_fpath = temp_fpath
    with open(new_train_info_fpath, 'w') as file:
        json.dump(new_train_info, file, indent='    ')

    new_snapshot = {
        'model_state_dict': new_model.state_dict(),
        'epoch': new_model_state['epoch'],
        '__mmdet_conversion__': '1x_to_2x',
    }
    torch.save(new_snapshot, new_snap_fpath)

    new_deployed = deployer.DeployedModel.custom(
        model=new_model, snap_fpath=new_snap_fpath,
        train_info_fpath=new_train_info_fpath, initkw=new_initkw)

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_bio3x')
    fpath = new_deployed.package(dpath=extract_dpath, name=new_name)
    print('fpath = {!r}'.format(fpath))
    return fpath


def main():
    config = UpdateBioharnConfig(cmdline=True)
    update_deployed_bioharn_model(config)


if __name__ == '__main__':
    """
    CommandLine:
        python -m bioharn.compat.update_bioharn_model
    """
    main()
