#!/bin/bash 

# Install dependency packages
pip install -r requirements/runtime.txt
pip install -r requirements/tests.txt

# Note: mmcv and mmdet are weird to install
#pip install -r requirements/optional.txt


__for_developer__(){
    # What is my local cuda version
    cat $HOME/.local/cuda/version.txt

    # We need to ensure our torch version agrees with our cuda version
    python -c "import torch; print(torch.cuda.is_available())"
    python -c "import torch; print(torch.version.cuda)"
    
    pip uninstall torch torchvision

    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

    python -c "import torch; print(torch.cuda.is_available())"

    cd $HOME/code/netharn
    pip install -r requirements/super_setup.txt

    ./super_setup.py ensure
    ./super_setup.py develop

    _PYCODE="import subprocess, re; print(re.match('.*release ([0-9]*.[0-9]*),.*', str(subprocess.check_output(['nvcc', '--version']))).groups()[0].replace('.', ''))"
    CUDA_VERSION=$(python -c "$_PYCODE")
    echo $CUDA_VERSION
}

__gdal_from_source(){
    #curl https://github.com/OSGeo/gdal/archive/v3.0.2.zip
    #cd $HOME/tmp
    #curl -LJO https://github.com/OSGeo/gdal/releases/download/v3.0.2/gdal-3.0.2.tar.gz
    #tar -xzf gdal-3.0.2.tar.gz
    #cd gdal-3.0.2/
    #./configure

    cd $HOME/tmp
    cd ~/code
    if [ ! -d "$HOME/code/fletch-for-gdal" ]; then
        git clone https://github.com/Erotemic/fletch.git ~/code/fletch-for-gdal
        cd ~/code/fletch-for-gdal
    fi

    # Setup a build directory and build fletch
    cd ~/code/fletch-for-gdal

    CMAKE_INSTALL_PREFIX=$HOME/.local

    FLETCH_BUILD=$HOME/code/fletch-for-gdal/build-gdal-minimal-test

    mkdir -p $FLETCH_BUILD
    cd $FLETCH_BUILD

    cmake -G "Unix Makefiles" \
        -D CMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \
        -D GDAL_SELECT_VERSION=2.2.2 \
        -D fletch_ENABLE_GDAL=True \
        -D fletch_ENABLE_PROJ4=True \
        ..

    BUILD_PREFIX=$FLETCH_BUILD/install
    $BUILD_PREFIX/bin/gdal-config --version
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
    chmod +x $CMAKE_INSTALL_PREFIX/bin/gdal*
    
    pip install --global-option=build_ext --prefix=$CMAKE_INSTALL_PREFIX $GDAL_SRC_FPATH  --verbose

        --global-option="-I$CMAKE_INSTALL_PREFIX/gdal" \
        --prefix=$BUILD_PREFIX \
        $GDAL_SRC_FPATH  --verbose

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


MMCV_VERSION=1.2.5
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


# mmcv is weird about resolving this
#pip install pycocotools  
pip install mmcv-full==$MMCV_VERSION -f $MMCV_URL
#pip install mmcv-full==1.0.5+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
# pip install git+https://github.com/open-mmlab/mmdetection.git@595bf86e69ad7452498f32166ece985d9cc012be
pip install mmdet==2.7.0


#pip install mmdet
# Install bioharn in developer mode
pip install -e .
