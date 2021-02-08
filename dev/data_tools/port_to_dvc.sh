## DVC Quickstart



dvc_setup_and_checkout(){
    # -------------------
    # Step 0: Environment
    # -------------------
    # The following quickstart guide depends on having
    # these environemnt variables defined

    AD_USERNAME=joncrall
    REMOTE_URI=viame

    # Your Kitware active-directory username
    #AD_USERNAME=jon.crall

    # This is the remote machine that is hosting the data cache
    #REMOTE_URI=viame.kitware.com

    # We also assume you have python and pip installed
    # (and ideally are in a virtual environment)


    # ----------------------
    # Step 1: Setup SSH Keys
    # (If you already have ssh keys, skip to step 3).
    # ----------------------

    PRIVATE_KEY_FPATH="$HOME/.ssh/id_${AD_USERNAME}_ed25519"

    if [ -f $PRIVATE_KEY_FPATH ]; then
        echo "Found PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"
    else
        echo "Create PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"

        ssh-keygen -t ed25519 -b 256 -f $PRIVATE_KEY_FPATH -N ""
        echo $PRIVATE_KEY_FPATH

        # Ensure permissions correct and the new key is registered with the ssh-agent
        chmod 700 ~/.ssh
        chmod 400 ~/.ssh/id_*
        chmod 644 ~/.ssh/id_*.pub
        eval "$(ssh-agent -s)"
        ssh-add $PRIVATE_KEY_FPATH

        # -----------------------------------------
        # Step 2: Register SSH Keys with dvc remote
        # -----------------------------------------
        # Run ssh-copy-id to let the remote know about your ssh keys
        # You will have to enter your active-directory password here
        ssh-copy-id $AD_USERNAME@$REMOTE_URI
    fi


    # -------------------
    # Step 3: Install DVC
    # (with ssh packages)
    # -------------------
    pip install dvc[ssh]

    # -----------------------
    # Step 4: Clone this Repo
    # -----------------------
    cd $HOME/data
    gt clone git@gitlab.kitware.com:$REMOTE_URI/viame_dvc.git
    cd viame_dvc


    ## ALTERNATIVE
    #dvc remote add --global viame ssh://$REMOTE_URI/data/dvc-repos/viame_dvc -f
    #dvc remote modify --global viame url ssh://$REMOTE_URI/data/dvc-repos/viame_dvc
    #dvc remote modify --global viame user $AD_USERNAME
    #cat $HOME/.config/dvc/config

    #rm .dvc/config
    #dvc remote add --local viame ssh://$REMOTE_URI/data/dvc-repos/viame_dvc -f
    #dvc remote modify --local viame url ssh://$REMOTE_URI/data/dvc-repos/viame_dvc
    #dvc remote modify --local viame user $AD_USERNAME 
    #cat .dvc/config

    # Point to whatever remote you want
    dvc remote add --global viame ssh://$REMOTE_URI/data/dvc-caches/viame_dvc -f
    dvc remote modify --global viame url ssh://$REMOTE_URI/data/dvc-caches/viame_dvc 
    dvc remote modify --global viame user $AD_USERNAME
    cat $HOME/.config/dvc/config

    dvc remote add --local viame ssh://$REMOTE_URI/data/dvc-caches/viame_dvc -f
    dvc remote modify --local viame url ssh://$REMOTE_URI/data/dvc-caches/viame_dvc
    dvc remote modify --local viame user $AD_USERNAME 
    cat .dvc/config
    cat .dvc/config.local

    cp .dvc/config.local .dvc/config
    cat .dvc/config

    dvc pull -r viame

    dvc cache dir

    # -----------------------
    # Step 5: Start Using DVC
    # -----------------------

    # Query the default remote
    dvc config core.remote
    dvc remote list

}


dvc_create_new_dvc_server(){

    # Ensure that the external repo exists
    # https://gitlab.kitware.com/viame/viame_dvc

    # SSH INTO THE SERVER

    # Creating a new DVC Repo
    sudo mkdir -p /data/dvc-caches
    sudo chmod -R 777 /data/dvc-caches
    sudo chown -R root:private /data/dvc-caches

    sudo mkdir -p /data/dvc-repos
    sudo chmod -R 777 /data/dvc-repos
    sudo chown -R root:private /data/dvc-repos

    cd /data/dvc-repos
    ls -al /data/dvc-repos
    git clone git@gitlab.kitware.com:viame/viame_dvc.git

    cd /data/dvc-repos/viame_dvc
    dvc init
    git commit -m "Init DVC repo"
    git push

    # Set cache-type strategy preferences
    dvc config cache.type reflink,symlink,copy
    dvc config cache.type copy

    dvc checkout --relink

    # Set up an ssh remote on viame.kitware.com
    dvc remote add --default viame /data/dvc-caches/viame_dvc

    cat .dvc/config

    DVC_CACHE_DPATH=/data/dvc-caches
    echo "DVC_CACHE_DPATH = $DVC_CACHE_DPATH"
    sudo find $DVC_CACHE_DPATH -type d -exec chmod 0775 {} \;
    sudo find $DVC_CACHE_DPATH -type f -exec chmod 0444 {} \;

    tree -L 3 $DVC_CACHE_DPATH
}


copy_viame_datas(){
    echo "DATA_TO_COPY = $DATA_TO_COPY"
    tree -L 2 $DATA_TO_COPY

    symlinks -rd /data/dvc-repos/viame_dvc/
    rsync -avPR \
        /data/public/Aerial/./_ORIG_US_ALASKA_MML_SEALION \
        /data/dvc-repos/viame_dvc/

    cd /data/dvc-caches/viame_dvc/US_ALASKA_MML_SEALION
    mv _ORIG_US_ALASKA_MML_SEALION/* .

    DVC_REPO=/data/dvc-repos/viame_dvc
    cd $DVC_REPO

    tree -L 3

    # Add the raw data to DVC
    YEARS=(2007 2008 2008W 2009 2010 2011 2012 2014 2015 2016)
    for YEAR in "${YEARS[@]}"
    do
        echo "YEAR = $YEAR"
        dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/sealions_${YEAR}_v9.kwcoco.json
        dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/sealions_${YEAR}_v9.viame.csv
        dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/images
    done

    YEAR=2013
    dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/sealions_${YEAR}_v3.kwcoco.json
    dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/sealions_${YEAR}_v3.viame.csv
    dvc add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/images

    ls _ORIG_US_ALASKA_MML_SEALION/*/*.dvc

    # Add the raw data to DVC
    YEARS=(2007 2008 2008W 2009 2010 2011 2012 2013 2014 2015 2016)
    for YEAR in "${YEARS[@]}"
    do
        echo "YEAR = $YEAR"
        git add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/*.dvc
    done

    dvc add _ORIG_US_ALASKA_MML_SEALION/*/*.dvc

    #dvc add _ORIG_US_ALASKA_MML_SEALION/2008
    #dvc add _ORIG_US_ALASKA_MML_SEALION/2008/images
    #dvc add _ORIG_US_ALASKA_MML_SEALION/2008/sealions_2008_v9.kwcoco.json
    #dvc add _ORIG_US_ALASKA_MML_SEALION/2008/sealions_2008_v9.viame.csv

}

extra(){
     ln -s /data/dvc-repos /home/joncrall/data/dvc-repos

     cd /data/dvc-repos/viame_dvc

    # Add the raw data to DVC
    YEARS=(2007 2008 2008W 2009 2010 2011 2012 2013 2014 2015 2016)
    for YEAR in "${YEARS[@]}"
    do
        echo "YEAR = $YEAR"
        FPATH=$(ls --color=never /data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/${YEAR}/sealions_${YEAR}_v*.kwcoco.json)
        echo "FPATH = $FPATH"
        cat $FPATH | head -n 100
        kwcoco reroot --absolute=True --src $FPATH --dst $FPATH.abs --new_prefix=/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/${YEAR}
        cat $FPATH.abs | head -n 100
        #git add _ORIG_US_ALASKA_MML_SEALION/${YEAR}/*.dvc
    done

     cd $HOME/data/dvc-repos/viame_dvc/
     kwcoco union \
         --dst _ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json.abs \
         --src _ORIG_US_ALASKA_MML_SEALION/2007/sealions_2007_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2008/sealions_2008_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2008W/sealions_2008W_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2009/sealions_2009_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2011/sealions_2011_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2012/sealions_2012_v3.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2013/sealions_2013_v3.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2014/sealions_2014_v9.kwcoco.json.abs \
             _ORIG_US_ALASKA_MML_SEALION/2015/sealions_2015_v9.kwcoco.json.abs 

     kwcoco union \
         --src \
         _ORIG_US_ALASKA_MML_SEALION/2010/sealions_2010_v9.kwcoco.json.abs \
         _ORIG_US_ALASKA_MML_SEALION/2016/sealions_2016_v9.kwcoco.json.abs \
         --dst _ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json.abs

    cd $HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION
    kwcoco reroot sealions_train_v9.kwcoco.json --absolute=False --dst sealions_train_v9.kwcoco.json.rel
    kwcoco reroot sealions_vali_v9.kwcoco.json --absolute=False --dst sealions_vali_v9.kwcoco.json.rel

    dvc unprotect sealions_train_v9.kwcoco.json sealions_vali_v9.kwcoco.json
    cp sealions_train_v9.kwcoco.json.rel sealions_train_v9.kwcoco.json
    cp sealions_vali_v9.kwcoco.json.rel sealions_vali_v9.kwcoco.json
    dvc add sealions_train_v9.kwcoco.json sealions_vali_v9.kwcoco.json
	git add sealions_vali_v9.kwcoco.json.dvc sealions_train_v9.kwcoco.json.dvc
    git commit -am "update sealion coco files"
    git push



    dvc add *.json

    kwcoco stats $HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json
    kwcoco stats $HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json

    cat sealions_vali_v9.kwcoco.json  | head -n 100
    cat 2007/sealions_2007_v9.kwcoco.json  | head -n 100

    kwcoco reroot sealions_train_v9.kwcoco.json --absolute=False --dst tmp_sealions_train_v9.kwcoco.json
    kwcoco reroot sealions_vali_v9.kwcoco.json --absolute=False --dst tmp_sealions_vali_v9.kwcoco.json

    ## Restructure ##
    mkdir -p US_ALASKA_MML_SEALION
    dvc move _ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json US_ALASKA_MML_SEALION
    dvc move _ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json US_ALASKA_MML_SEALION

    mkdir -p US_ALASKA_MML_SEALION/2007
    mkdir -p US_ALASKA_MML_SEALION/2008
    mkdir -p US_ALASKA_MML_SEALION/2008W
    mkdir -p US_ALASKA_MML_SEALION/2009
    mkdir -p US_ALASKA_MML_SEALION/2010
    mkdir -p US_ALASKA_MML_SEALION/2011
    mkdir -p US_ALASKA_MML_SEALION/2012
    mkdir -p US_ALASKA_MML_SEALION/2013
    mkdir -p US_ALASKA_MML_SEALION/2014
    mkdir -p US_ALASKA_MML_SEALION/2015
    mkdir -p US_ALASKA_MML_SEALION/2016


    dvc move _ORIG_US_ALASKA_MML_SEALION/2007/*.json US_ALASKA_MML_SEALION/2007
    dvc move _ORIG_US_ALASKA_MML_SEALION/2007/*.csv US_ALASKA_MML_SEALION/2007
    dvc move _ORIG_US_ALASKA_MML_SEALION/2007/images US_ALASKA_MML_SEALION/2007

    dvc move _ORIG_US_ALASKA_MML_SEALION/2008/*.json US_ALASKA_MML_SEALION/2008
    dvc move _ORIG_US_ALASKA_MML_SEALION/2008/*.csv US_ALASKA_MML_SEALION/2008
    dvc move _ORIG_US_ALASKA_MML_SEALION/2008/images US_ALASKA_MML_SEALION/2008

    dvc move _ORIG_US_ALASKA_MML_SEALION/2008W/*.json US_ALASKA_MML_SEALION/2008W
    dvc move _ORIG_US_ALASKA_MML_SEALION/2008W/*.csv US_ALASKA_MML_SEALION/2008W
    dvc move _ORIG_US_ALASKA_MML_SEALION/2008W/images US_ALASKA_MML_SEALION/2008W

    dvc move _ORIG_US_ALASKA_MML_SEALION/2009/*.json US_ALASKA_MML_SEALION/2009
    dvc move _ORIG_US_ALASKA_MML_SEALION/2009/*.csv US_ALASKA_MML_SEALION/2009
    dvc move _ORIG_US_ALASKA_MML_SEALION/2009/images US_ALASKA_MML_SEALION/2009

    dvc move _ORIG_US_ALASKA_MML_SEALION/2010/*.json US_ALASKA_MML_SEALION/2010
    dvc move _ORIG_US_ALASKA_MML_SEALION/2010/*.csv US_ALASKA_MML_SEALION/2010
    dvc move _ORIG_US_ALASKA_MML_SEALION/2010/images US_ALASKA_MML_SEALION/2010

    dvc move _ORIG_US_ALASKA_MML_SEALION/2011/*.json US_ALASKA_MML_SEALION/2011
    dvc move _ORIG_US_ALASKA_MML_SEALION/2011/*.csv US_ALASKA_MML_SEALION/2011
    dvc move _ORIG_US_ALASKA_MML_SEALION/2011/images US_ALASKA_MML_SEALION/2011

    dvc move _ORIG_US_ALASKA_MML_SEALION/2012/*.json US_ALASKA_MML_SEALION/2012
    dvc move _ORIG_US_ALASKA_MML_SEALION/2012/*.csv US_ALASKA_MML_SEALION/2012
    dvc move _ORIG_US_ALASKA_MML_SEALION/2012/images US_ALASKA_MML_SEALION/2012

    dvc move _ORIG_US_ALASKA_MML_SEALION/2013/*.json US_ALASKA_MML_SEALION/2013
    dvc move _ORIG_US_ALASKA_MML_SEALION/2013/*.csv US_ALASKA_MML_SEALION/2013
    dvc move _ORIG_US_ALASKA_MML_SEALION/2013/images US_ALASKA_MML_SEALION/2013

    dvc move _ORIG_US_ALASKA_MML_SEALION/2014/*.json US_ALASKA_MML_SEALION/2014
    dvc move _ORIG_US_ALASKA_MML_SEALION/2014/*.csv US_ALASKA_MML_SEALION/2014
    dvc move _ORIG_US_ALASKA_MML_SEALION/2014/images US_ALASKA_MML_SEALION/2014

    dvc move _ORIG_US_ALASKA_MML_SEALION/2015/*.json US_ALASKA_MML_SEALION/2015
    dvc move _ORIG_US_ALASKA_MML_SEALION/2015/*.csv US_ALASKA_MML_SEALION/2015
    dvc move _ORIG_US_ALASKA_MML_SEALION/2015/images US_ALASKA_MML_SEALION/2015

    dvc move _ORIG_US_ALASKA_MML_SEALION/2016/*.json US_ALASKA_MML_SEALION/2016
    dvc move _ORIG_US_ALASKA_MML_SEALION/2016/*.csv US_ALASKA_MML_SEALION/2016
    dvc move _ORIG_US_ALASKA_MML_SEALION/2016/images US_ALASKA_MML_SEALION/2016

    find . -type f -iname "*.dvc" -exec git add  {} \;
    find . -type f -iname ".gitignore" -exec git add  {} \;


    python -m bioharn.detect_fit \
        --name=sealion-cascade-v10 \
        --workdir=$HOME/work/sealions \
        --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache \
        --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \


    python -m bioharn.detect_fit \
        --name=sealion-cascade-v10 \
        --workdir=$HOME/work/sealions \
        --sampler_workdir=$HOME/data/dvc-repos/viame_dvc/.ndsampler/_cache \
        --train_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_train_v9.kwcoco.json \
        --vali_dataset=$HOME/data/dvc-repos/viame_dvc/_ORIG_US_ALASKA_MML_SEALION/sealions_vali_v9.kwcoco.json \
        --schedule=ReduceLROnPlateau-p5-c5 \
        --max_epoch=400 \
        --augment=complex \
        --init=noop \
        --arch=cascade \
        --channels="rgb" \
        --optim=sgd \
        --lr=1e-3 \
        --window_dims=512,512 \
        --input_dims=window \
        --window_overlap=0.5 \
        --multiscale=False \
        --normalize_inputs=imagenet \
        --workers=8 \
        --xpu=auto \
        --batch_size=4 \
        --sampler_backend=None \
        --num_batches=1000 \
        --balance=None \
        --bstep=16
     
}
