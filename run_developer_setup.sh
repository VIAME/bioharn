#!/bin/bash 

# Install dependency packages
pip install -r requirements/runtime.txt
pip install -r requirements/tests.txt

# Note: mmcv and mmdet are weird to install
#pip install -r requirements/optional.txt


MMCV_FULL_VERSION=$(python -c "
from distutils.version import LooseVersion
import torch


if LooseVersion(torch.__version__) >= LooseVersion('1.5.0'):
    torch_part = '1.5.0'
elif LooseVersion(torch.__version__) >= LooseVersion('1.4.0'):
    torch_part = '1.4.0'
elif LooseVersion(torch.__version__) >= LooseVersion('1.3.0'):
    torch_part = '1.3.0'
else:
    raise ValueError('unsupported torch version')


if LooseVersion(torch.version.cuda) >= LooseVersion('10.2'):
    cuda_part = 'cu102'
elif LooseVersion(torch.version.cuda) >= LooseVersion('10.1'):
    cuda_part = 'cu101'
elif LooseVersion(torch.version.cuda) >= LooseVersion('10.0'):
    raise ValueError('unsupported cuda version')
elif LooseVersion(torch.version.cuda) >= LooseVersion('9.2'):
    cuda_part = 'cu92'
else:
    raise ValueError('unsupported torch version')

mmcv_part = '1.0.4'
#mmcv_part = 'latest'

mmcv_full_version = '+'.join([mmcv_part, 'torch' + torch_part, cuda_part])
print(mmcv_full_version)
")
echo "MMCV_FULL_VERSION = $MMCV_FULL_VERSION"

pip install mmcv-full==$MMCV_FULL_VERSION -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

pip install git+https://github.com/open-mmlab/mmdetection.git@595bf86e69ad7452498f32166ece985d9cc012be

#pip install mmdet


# Install irharn in developer mode
pip install -e .
