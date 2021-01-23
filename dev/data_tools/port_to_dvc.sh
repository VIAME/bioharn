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
    git clone git@gitlab.kitware.com:$REMOTE_URI/viame_dvc.git
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
    #dvc config cache.type reflink,symlink,copy
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
