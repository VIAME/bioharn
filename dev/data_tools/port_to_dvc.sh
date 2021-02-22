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



#### 

# Setup the private Viame DVC repo

create_gitlab_dvc_project(){
    load_secrets
    HOST=https://gitlab.kitware.com
    PRIVATE_GITLAB_TOKEN=$(git_token_for $HOST)
    if [[ "$PRIVATE_GITLAB_TOKEN" == "ERROR" ]]; then
        false || echo "Failed to load authentication key"
    fi

    TMP_DIR=$(mktemp -d -t ci-XXXXXXXXXX)

    # Find the GroupID (aka namespace ID) for the gitlab group
    GROUP_NAME=viame
    curl --header "PRIVATE-TOKEN: $PRIVATE_GITLAB_TOKEN" "$HOST/api/v4/groups" > $TMP_DIR/all_group_info
    GROUP_ID=$(cat $TMP_DIR/all_group_info | jq ". | map(select(.name==\"$GROUP_NAME\")) | .[0].id")
    echo "GROUP_ID = $GROUP_ID"

    # Create a new GitLab Project
    # https://docs.gitlab.com/ee/api/projects.html#create-project-for-user
    curl --request POST \
        --header "PRIVATE-TOKEN: $PRIVATE_GITLAB_TOKEN" \
        --data-urlencode "name=viame_private_dvc" \
        --data-urlencode "description=\"A DVC Repo for Private VIAME data\"" \
        --data-urlencode "namespace_id=$GROUP_ID" \
        --data-urlencode "visibility=private" \
        "$HOST/api/v4/projects" > $TMP_DIR/new_project

    cat $TMP_DIR/new_project | jq .
}

cd /data/dvc-repos
git clone git@gitlab.kitware.com:viame/viame_private_dvc.git
cd /data/dvc-repos/viame_private_dvc
dvc init

# Set cache-type strategy preferences
dvc config cache.type reflink,symlink,copy

# Setup cache dir on viame
dvc cache dir --local /data/dvc-caches/viame_private_dvc

# Set up an ssh remote on viame.kitware.com
dvc remote add --default viame ssh://viame.kitware.com:/data/dvc-caches/viame_private_dvc

git add .dvc .dvcignore
git commit -am "Init DVC repo"
git push

cat .dvc/config

rsync -avrLP /data/private/./Benthic /data/dvc-repos/viame_private_dvc

tree /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM/ | wc
tree /data/private/Benthic/US_NE_2017_CFARM_HABCAM/ | wc

DVC_REPO=/data/dvc-repos/viame_private_dvc
cd $DVC_REPO
mkdir -p $DVC_REPO/Benthic/US_NE_2017_CFARM_HABCAM/_assets
mv $DVC_REPO/Benthic/US_NE_2017_CFARM_HABCAM/Corrected_Old $DVC_REPO/Benthic/US_NE_2017_CFARM_HABCAM/_assets/images
cd $DVC_REPO/Benthic/US_NE_2017_CFARM_HABCAM/_assets
dvc add images
dvc move images Corrected_Old
cd $DVC_REPO/Benthic/US_NE_2017_CFARM_HABCAM/
dvc add "HabCam 2017 dataset1 annotations.csv"

DVC_REPO=/data/dvc-repos/viame_private_dvc
cd $DVC_REPO/Benthic

tree $DVC_REPO/Benthic/US_NE_2018_CFARM_HABCAM/ | wc
tree /data/private/Benthic/US_NE_2018_CFARM_HABCAM/ | wc
cd $DVC_REPO/Benthic/US_NE_2018_CFARM_HABCAM
mkdir -p _assets
mv Left_Old _assets/
dvc add _assets/*
git add _assets/Left_Old.dvc _assets/.gitignore
dvc add annotations.csv
git add annotations.csv.dvc .gitignore
git commit -am "Add US_NE_2018_CFARM_HABCAM"

cd $DVC_REPO/Benthic/US_NE_2019_CFARM_HABCAM
tree /data/private/Benthic/US_NE_2019_CFARM_HABCAM/ | wc
cd $DVC_REPO/Benthic/US_NE_2019_CFARM_HABCAM
mkdir _assets
mv Left_Old _assets
mv Processed _assets
mv sample-3d-results _assets
dvc add *.csv _assets/*

DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_2019_CFARM_HABCAM_PART2
tree | wc
tree /data/private/Benthic/US_NE_2019_CFARM_HABCAM_PART2/ | wc
mkdir _assets
mv Left_Old _assets
dvc add *.csv _assets/*


DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH
tree | wc
tree /data/private/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH/ | wc
mkdir -p _assets
mv Corrected _assets/
mv Disparity _assets/
mv Left _assets/
mv Raw _assets/
dvc add *csv _assets/*


# POINTER <-

DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_NEFSC_2015_HABCAM_GEORGES_BANK
mkdir _assets
mv Disparity  Left  Raw  Rectified _assets
dvc add _assets/* --desc "Add assets from US_NE_NEFSC_2015_HABCAM_GEORGES_BANK"

DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_NEFSC_2015_HABCAM_MID_ATLANTIC
mkdir _assets
mv Disparity  Left  Raw  Rectified _assets
dvc add _assets/* --desc "Add assets from US_NE_NEFSC_2015_HABCAM_MID_ATLANTIC"

DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_NEFSC_2016_HABCAM_GEORGES_BANK
mkdir _assets
mv Disparity  Left  Raw  Rectified _assets
dvc add _assets/* --desc "Add assets from US_NE_NEFSC_2015_HABCAM_MID_ATLANTIC"

DVC_REPO=/data/dvc-repos/viame_private_dvc
ls $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_NEFSC_2016_HABCAM_MID_ATLANTIC
mkdir _assets
mv Disparity  Left  Raw  Rectified _assets
dvc add _assets/* --desc "Add assets from US_NE_NEFSC_2015_HABCAM_MID_ATLANTIC"

# -----------------

cd /data/dvc-repos/viame_dvc
dvc move US_ALASKA_MML_SEALION public/US_ALASKA_MML_SEALION
git mv US_ALASKA_MML_SEALION public/US_ALASKA_MML_SEALION

cd /data/dvc-repos/viame_dvc

find /data/public/./Benthic -iname "*csv" -exec grep -l flatfish {} \;
find /data/private/./Benthic -iname "*csv" -exec grep -l flatfish {} \;
rsync -avrLP /data/public/./Benthic /data/dvc-repos/viame_dvc 

# These are the ones with flatfish
rsync -avrLP /data/public/./Benthic/US_NE_2019_CFF_HABCAM_PART2 /data/dvc-repos/viame_dvc && \
rsync -avrLP /data/public/./Benthic/US_NE_2019_CFF_HABCAM//data/dvc-repos/viame_dvc && \
rsync -avrLP /data/public/./Benthic/US_NE_2018_CFF_HABCAM /data/dvc-repos/viame_dvc && \
rsync -avrLP /data/public/./Benthic/US_NE_2017_CFF_HABCAM /data/dvc-repos/viame_dvc && \
rsync -avrLP /data/public/./Benthic/US_NE_2015_NEFSC_HABCAM /data/dvc-repos/viame_dvc


# PUBLIC SIDE

cd /data/dvc-repos/viame_dvc/Benthic/AQ_2020_SEFSC_PENGUIN_HEADCAM
mkdir _assets/images
mv *.png _assets/images

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2015_NEFSC_HABCAM
mkdir _assets/
mv Cog  Corrected  Disparities _assets
find . -iname "*csv" -exec  mv {} . \;
find . -iname "*json" -exec  mv {} . \;
find . -iname "*txt" -exec  mv {} . \;

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2017_CFF_HABCAM
mkdir _assets/
mv Left Raws _assets
find . -iname "*csv" 
mv ./_assets/Left/annotations.csv .

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2018_CFF_HABCAM
mkdir _assets/
mv Left Raws _assets
mv ./_assets/Left/annotations.csv .

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM
mkdir _assets/
mv Left Raws _assets
mv ./_assets/Left/annotations.csv .

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM_PART2
mkdir _assets/
mv Left Raws _assets
mv ./_assets/Left/annotations.csv .


cd /data/dvc-repos/viame_dvc
dvc add \
    Benthic/US_NE_2019_CFF_HABCAM_PART2/_assets/Left \
    Benthic/US_NE_2019_CFF_HABCAM_PART2/_assets/Raws \
    Benthic/US_NE_2019_CFF_HABCAM_PART2/_assets/annotations.csv \
    Benthic/US_NE_2019_CFF_HABCAM/_assets/Left \
    Benthic/US_NE_2019_CFF_HABCAM/_assets/Raws \
    Benthic/US_NE_2019_CFF_HABCAM/_assets/annotations.csv \
    Benthic/US_NE_2018_CFF_HABCAM/_assets/Left \
    Benthic/US_NE_2018_CFF_HABCAM/_assets/Raws \
    Benthic/US_NE_2018_CFF_HABCAM/_assets/annotations.csv \
    Benthic/US_NE_2017_CFF_HABCAM/_assets/Left \
    Benthic/US_NE_2017_CFF_HABCAM/_assets/Raws \
    Benthic/US_NE_2017_CFF_HABCAM/_assets/annotations.csv \
    Benthic/US_NE_2015_NEFSC_HABCAM/_assets/Corrected \
    Benthic/US_NE_2015_NEFSC_HABCAM/_assets/Disparities \
    Benthic/US_NE_2015_NEFSC_HABCAM/_assets/annotations.csv \
    Benthic/US_NE_2015_NEFSC_HABCAM/_assets/annotations.habcam_csv \
    Benthic/US_NE_2015_NEFSC_HABCAM/_assets/metadata.txt 


DVC_REPO=/data/dvc-repos/viame_dvc
ls $DVC_REPO/Benthic/
tree -L 2 $DVC_REPO/Benthic/
cd $DVC_REPO/Benthic/US_NE_2017_CFF_HABCAM


# Move data from private over to the repo, reorganize into data bundles so
# assets are in a subdirectory and annotations are findable via ls

# TODO: will likely need to reconvert CFARM 2017-2019

find /data/private -iname "*cog_rgb*"
find /data/public -iname "*cog_rgb*"
find . -iname "*cog_rgb*"
find /data/public -iname "*cog_rgb*"


# Get rid of assets folder
cd /data/dvc-repos/viame_dvc

tree -L 3 /data/dvc-repos/viame_dvc
tree -L 2 /data/dvc-repos/viame_private_dvc
tree -L 3 /data/dvc-repos/viame_private_dvc

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NW_2017_NWFSC_PYROSOME_TRAIN
find . -type f | awk -F. '!a[$NF]++{print $NF}'
mkdir -p images
mv *.pgm images

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NW_2017_NWFSC_PYROSOME_TEST
mkdir -p images
mv *.pgm images


cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2014_HABCAM_FLATFISH
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM_PART2
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2018_CFARM_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2016_HABCAM_MID_ATLANTIC
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2016_HABCAM_GEORGES_BANK
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2015_HABCAM_MID_ATLANTIC
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_NEFSC_2015_HABCAM_GEORGES_BANK
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets

find . -iname "*.json"

cat ./US_NE_2019_CFARM_HABCAM_PART2/US_NE_2019_CFARM_HABCAM_PART2_v4.kwcoco.json | head -n 20


cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM_PART2
kwcoco reroot US_NE_2019_CFARM_HABCAM_PART2_v4.kwcoco.json --absolute=False  --dst=./tmp.json --old_prefix="/home/joncrall/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM_PART2/_assets/" --new_prefix=""
mv tmp.json US_NE_2019_CFARM_HABCAM_PART2_v4.kwcoco.json
#jq ".images[0]" US_NE_2019_CFARM_HABCAM_PART2_v4.kwcoco.json
#jq ".images[0]" tmp.json

cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM
kwcoco reroot US_NE_2019_CFARM_HABCAM_v4.kwcoco.json --absolute=False  --dst=./tmp.json --old_prefix="/home/joncrall/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2019_CFARM_HABCAM/_assets/" --new_prefix=""
jq ".images[0]" US_NE_2019_CFARM_HABCAM_v4.kwcoco.json
jq ".images[0]" tmp.json
mv tmp.json US_NE_2019_CFARM_HABCAM_v4.kwcoco.json


cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM
kwcoco reroot US_NE_2017_CFARM_HABCAM_v4.kwcoco.json --absolute=False  --dst=./tmp.json --old_prefix="/home/joncrall/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2017_CFARM_HABCAM/_assets/" --new_prefix=""
jq ".images[0]" US_NE_2017_CFARM_HABCAM_v4.kwcoco.json
jq ".images[0]" tmp.json
mv tmp.json US_NE_2017_CFARM_HABCAM_v4.kwcoco.json


cd /data/dvc-repos/viame_private_dvc/Benthic/US_NE_2018_CFARM_HABCAM
kwcoco reroot US_NE_2018_CFARM_HABCAM_v4.kwcoco.json --absolute=False  --dst=./tmp.json --old_prefix="/home/joncrall/data/dvc-repos/viame_private_dvc/Benthic/US_NE_2018_CFARM_HABCAM/_assets/" --new_prefix=""
jq ".images[0]" US_NE_2018_CFARM_HABCAM_v4.kwcoco.json
jq ".images[0]" tmp.json
mv tmp.json US_NE_2018_CFARM_HABCAM_v4.kwcoco.json

find . -iname "*.json" -exec jq ".images[0]" {} \;


cd /data/dvc-repos/viame_dvc/Benthic/

cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM_PART2
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets
cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2019_CFF_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets
cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2018_CFF_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets
cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2017_CFF_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets
cd /data/dvc-repos/viame_dvc/Benthic/US_NE_2015_NEFSC_HABCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets
cd /data/dvc-repos/viame_dvc/Benthic/AQ_2020_SEFSC_PENGUIN_HEADCAM
mv _assets/* . && mv _assets/.gitignore . && rmdir _assets


# TODO: Add this in US_NE_NEFSC_2014_HABCAM_FLATFISH
