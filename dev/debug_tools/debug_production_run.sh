#!/bin/bash
__doc__="
Reproduce an issue in VIAME

Current available public files are listed here on
https://github.com/VIAME/VIAME Ones that we use are registered in the script

Requirements:
    pip install gdown girder-client
"

install-viame-release(){
    INSTALL_PREFIX=$HOME/tmp/viame_debug
    mkdir -p "$INSTALL_PREFIX"
    cd "$INSTALL_PREFIX"

    ### Install the VIAME release binaries, hosted on google drive
    declare -A KNOWN_VIAME_GRDIVE_IDS=(
        ["VIAME-v0.19.2-Linux-64Bit.tar.gz"]="11KiTxcHj448fJJfuP3Hb5O80K4DgMPU4"
    )
    VIAME_GDRIVE_ID="${KNOWN_VIAME_GRDIVE_IDS[VIAME-v0.19.2-Linux-64Bit.tar.gz]}"
    gdown "$VIAME_GDRIVE_ID"

    # Download the sealion models
    declare -A KNOWN_MODEL_GIRDER_IDS=(
        ["sealion-models"]="62c5b770db3a8ed8f9409cb1"
    )
    MODEL_GIRDER_ID="${KNOWN_MODEL_GIRDER_IDS[sealion-models]}"
    girder-client --api-url https://viame.kitware.com/api/v1 download "$MODEL_GIRDER_ID"

    # Extract VIAME binaries
    tar -xvf VIAME-v0.19.2-Linux-64Bit.tar.gz
    export VIAME_INSTALL=$(realpath viame)
    echo "VIAME_INSTALL = $VIAME_INSTALL"

    # Extract sealion models and move them into VIAME_INSTALL
    unzip -o VIAME-Sea-Lion-Models-v2.7.zip -d "$VIAME_INSTALL" 
}


test-viame-release(){

    # Download test data and setup a input_image.txt
    girder-client --api-url https://viame.kitware.com/api/v1 download "605befaa8ba6bae828a05fe4" 
    echo "
    20070609_SLAP5809_Orig.JPG
    " > input_image.txt

    # Setup VIAME Paths (no need to run multiple times if you already ran it)
    # INSERT DIR TO EXTRACTED VIAME BINS
    export VIAME_INSTALL=$(realpath viame)
    source "${VIAME_INSTALL}/setup_viame.sh"

    # Run pipeline
    kwiver runner "${VIAME_INSTALL}/configs/pipelines/detector_sea_lion_v3_mask_rcnn.pipe" \
                  -s input:video_filename=input_image.txt
}


main(){
    install-viame-release
    test-viame-release
}


# bpkg convention
# https://github.com/bpkg/bpkg
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
	# We are sourcing the library
	#export -f transcrypt_main
	echo "Sourcing the library"
	#export -p
else
	# Executing file as a script
	main "${@}"
	exit $?
fi
