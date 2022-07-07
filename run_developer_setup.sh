#!/bin/bash 

# Install dependency packages
pip install --no-deps imgaug
# Handle imgaug deps
pip install six scipy Pillow matplotlib scikit-image shapely numba

pip install -r requirements/runtime.txt
pip install -r requirements/tests.txt

# Note: mmcv and mmdet are weird to install
#pip install -r requirements/optional.txt


__for_developer__(){
    # What is my local cuda version
    cat "$HOME/.local/cuda/version.txt"

    # We need to ensure our torch version agrees with our cuda version
    python -c "import torch; print(torch.cuda.is_available())"
    python -c "import torch; print(torch.version.cuda)"
    
    pip uninstall torch torchvision

    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

    python -c "import torch; print(torch.cuda.is_available())"

    cd "$HOME/code/netharn"
    pip install -r requirements/super_setup.txt

    ./super_setup.py ensure
    ./super_setup.py develop

    _PYCODE="import subprocess, re; print(re.match('.*release ([0-9]*.[0-9]*),.*', str(subprocess.check_output(['nvcc', '--version']))).groups()[0].replace('.', ''))"
    CUDA_VERSION=$(python -c "$_PYCODE")
    echo "$CUDA_VERSION"
}

__gdal_from_source(){
    #curl https://github.com/OSGeo/gdal/archive/v3.0.2.zip
    #cd $HOME/tmp
    #curl -LJO https://github.com/OSGeo/gdal/releases/download/v3.0.2/gdal-3.0.2.tar.gz
    #tar -xzf gdal-3.0.2.tar.gz
    #cd gdal-3.0.2/
    #./configure

    cd "$HOME/tmp"
    cd ~/code
    if [ ! -d "$HOME/code/fletch-for-gdal" ]; then
        git clone https://github.com/Erotemic/fletch.git ~/code/fletch-for-gdal
        cd ~/code/fletch-for-gdal
    fi

    # Setup a build directory and build fletch
    cd ~/code/fletch-for-gdal

    CMAKE_INSTALL_PREFIX=$HOME/.local

    FLETCH_BUILD=$HOME/code/fletch-for-gdal/build-gdal-minimal-test

    mkdir -p "$FLETCH_BUILD"
    cd "$FLETCH_BUILD"

    cmake -G "Unix Makefiles" \
        -D CMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
        -D GDAL_SELECT_VERSION=2.2.2 \
        -D fletch_ENABLE_GDAL=True \
        -D fletch_ENABLE_PROJ4=True \
        ..

    BUILD_PREFIX=$FLETCH_BUILD/install
    "$BUILD_PREFIX/bin/gdal-config" --version
    TARGET_GDAL_VERSION=$($BUILD_PREFIX/bin/gdal-config --version)
    echo "TARGET_GDAL_VERSION = $TARGET_GDAL_VERSION"

    make -j10

    # Install to the prefix
    cat CMakeCache.txt | grep INSTALL

    #export PATH=$BUILD_PREFIX/bin:$PATH
    PATH=$BUILD_PREFIX/bin:$PATH pip download "GDAL==$TARGET_GDAL_VERSION" --no-deps
    GDAL_SRC_FPATH=$(find . -maxdepth 1 -iname "GDAL*.tar.gz")
    make install

    # Weird, hack it
    chmod +x "$CMAKE_INSTALL_PREFIX"/bin/gdal*
    
    pip install --global-option=build_ext --prefix="$CMAKE_INSTALL_PREFIX" "$GDAL_SRC_FPATH"  --verbose

        #--global-option="-I$CMAKE_INSTALL_PREFIX/gdal" \
        #--prefix=$BUILD_PREFIX \
        #$GDAL_SRC_FPATH  --verbose

    #PATH=$BUILD_PREFIX/bin:$PATH pip install --global-option=build_ext \
    #    --global-option="-I$BUILD_PREFIX/gdal" \
    #    --prefix=$BUILD_PREFIX \
    #    $GDAL_SRC_FPATH  --verbose


    #-D fletch_BUILD_WITH_PYTHON:BOOL=False \
    #-D fletch_PYTHON_MAJOR_VERSION=3 \
    #-D fletch_ENABLE_libjpeg-turbo=True \
    #-D fletch_ENABLE_ZLib=True \
    #-D fletch_ENABLE_libtiff=True \
    #-D fletch_ENABLE_libgeotiff=True \
    #-D fletch_ENABLE_PNG=True \
    #-D fletch_ENABLE_libxml2=True \
    #-D fletch_ENABLE_GEOS=True \

    #make -j10

    #    -D fletch_ENABLE_Boost=True \
    #    -D fletch_ENABLE_FFmpeg=True \
    #    -D fletch_ENABLE_Eigen=False \
    #    -D fletch_ENABLE_GLog=False \
    #    -D fletch_ENABLE_SuiteSparse=True \
    #    -D fletch_ENABLE_Ceres=True \
    #    -D fletch_ENABLE_OpenCV=False \
    #    -D fletch_ENABLE_PDAL=True \
    #    ..
    
}

python -c "import torch; print(torch.version.cuda)"


MMCV_URL=$(python -c "
from distutils.version import LooseVersion
import torch

torch_version = LooseVersion(torch.__version__)
if torch_version < LooseVersion('1.3.0'):
    raise ValueError('unsupported torch version')
else:
    # should be 1.3.0, 1.4.0, 1.5.0, 1.6.0 etc..
    torch_part = '.'.join(list(map(str, list(torch_version.version[0:2]) + [0])))

if torch.version.cuda is None:
    cuda_part = 'cpu'
elif LooseVersion(torch.version.cuda) >= LooseVersion('11.5'):
    cuda_part = 'cu115'
elif LooseVersion(torch.version.cuda) >= LooseVersion('11.3'):
    cuda_part = 'cu113'
elif LooseVersion(torch.version.cuda) >= LooseVersion('11.1'):
    cuda_part = 'cu111'
elif LooseVersion(torch.version.cuda) >= LooseVersion('10.2'):
    cuda_part = 'cu102'
elif LooseVersion(torch.version.cuda) >= LooseVersion('10.1'):
    cuda_part = 'cu101'
elif LooseVersion(torch.version.cuda) >= LooseVersion('10.0'):
    raise ValueError('unsupported cuda version')
elif LooseVersion(torch.version.cuda) >= LooseVersion('9.2'):
    cuda_part = 'cu92'
else:
    raise ValueError('unsupported torch version')

# See https://github.com/open-mmlab/mmcv

dl_url_fmt = 'https://download.openmmlab.com/mmcv/dist/{cuda_part}/torch{torch_part}/index.html'

mmcv_url = dl_url_fmt.format(cuda_part=cuda_part, torch_part=torch_part)
print(mmcv_url)
")
echo "MMCV_URL = $MMCV_URL"


#MMCV_VERSION=1.3.0
MMCV_VERSION=1.4.8
MMDET_VERSION=2.23.0

_devcheck(){
    #pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    #pip install mmcv-full==1.3.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    #pip install mmdet==2.11.0

    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
    pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    pip install mmdet==2.23.0

    python -c "from mmcv import ops"

    #pip install torch==1.8.0 torchvision==0.9.0

    python -c "import torch; print(torch.version.cuda)"
    python -c "import torch; print(torch.__version__)"
    #pip install mmcv-full==1.0.5+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
    # pip install git+https://github.com/open-mmlab/mmdetection.git@595bf86e69ad7452498f32166ece985d9cc012be
}


# See: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md

#pip install mmcv-full==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
# mmcv is weird about resolving this
#pip install pycocotools  
pip install mmcv-full==$MMCV_VERSION -f "$MMCV_URL"
pip install mmdet==$MMDET_VERSION

fix_opencv_conflicts(){
    __doc__="
    Check to see if the wrong opencv is installed, and perform steps to clean
    up the incorrect libraries and install the desired (headless) ones.
    "
    # Fix opencv issues
    pip freeze | grep "opencv-python=="
    HAS_OPENCV_RETCODE="$?"
    pip freeze | grep "opencv-python-headless=="
    HAS_OPENCV_HEADLESS_RETCODE="$?"

    # VAR == 0 means we have it
    if [[ "$HAS_OPENCV_HEADLESS_RETCODE" == "0" ]]; then
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            pip uninstall opencv-python opencv-python-headless -y
            pip install opencv-python-headless
        fi
    else
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            pip uninstall opencv-python -y
        fi
        pip install opencv-python-headless
    fi
}

try_with_mim(){
    pip install openmim
    
    mim list
    mim install mmcv-full
    mim install mmcv-full

    python -c "from mmcv import ops"

    python -c "import mmcv"

    python -c "from mmcv.cnn import resnet"
    python -c "from mmdet import models"
}


fix_opencv_conflicts


#pip install mmdet
# Install bioharn in developer mode
pip install -e .
