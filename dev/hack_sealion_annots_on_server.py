from os.path import join, basename, dirname
import ubelt as ub
import glob
import copy


def hack():
    """
    cd /data/public/Aerial
    /US_ALASKA_MML_SEALION

    sudo chmod g+w -R /data/public/Aerial/
    sudo chown -R root:public /data/public/Aerial/US_ALASKA_MML_SEALION_FLAT/
    sudo chown -R root:public *
    """
    dpath = '/data/public/Aerial/_ORIG_US_ALASKA_MML_SEALION'

    flat_dpath = ub.ensuredir('/data/public/Aerial/US_ALASKA_MML_SEALION')

    csv_fpaths = glob.glob(join(dpath, '*', '*.csv'))
    # csv_fpaths = glob.glob(join(dpath, '2013', '*.csv'))

    new_csv_fpaths = []
    to_copy = []
    for csv_fpath in ub.ProgIter(csv_fpaths):

        old_root = dirname(csv_fpath)
        year = basename(old_root)  # Hack to get the year code
        new_dpath = ub.ensuredir((flat_dpath, year))
        new_csv_fpath = ub.augpath(csv_fpath, dpath=new_dpath)
        new_csv_fpaths.append(new_csv_fpath)

        with open(csv_fpath, 'r') as file:
            text = file.read()
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line and not line.startswith('#')]

        # HACK FOR FRAME NUMS
        gname_to_frame_num = {}

        new_lines = []
        seen_ = set()
        for line in lines:
            parts = line.split(',')
            old_gname = parts[1]
            old_gpath = join(old_root, old_gname)

            # Remove the prefix from the file name
            new_gname = basename(old_gname)
            new_gpath = join(new_dpath, new_gname)

            # Mark the file to copy
            if old_gpath not in seen_:
                to_copy.append((old_gpath, new_gpath))
                # Each annot references the file, so prevent duplicate
                # copies
                seen_.add(old_gpath)

            # HACK FOR FRAME NUMS
            if old_gname not in gname_to_frame_num:
                gname_to_frame_num[old_gname] = len(gname_to_frame_num)
            frame_num = gname_to_frame_num[old_gname]

            # Modify the CSV
            new_parts = copy.copy(parts)
            new_parts[1] = new_gname
            new_parts[2] = str(frame_num)
            new_lines.append(','.join(new_parts))

        # Rewrite the modified CSV to the same directory as the images
        new_text = '\n'.join(new_lines)
        with open(new_csv_fpath, 'w') as file:
            text = file.write(new_text)

        ub.symlink(new_csv_fpath, ub.augpath(new_csv_fpath, base='annotations'))

    dst_fpaths = [d for s, d in to_copy]

    len(dst_fpaths)
    len(set(dst_fpaths))
    assert not ub.find_duplicates(dst_fpaths), 'should have no dups'

    # Symlink instead of copying to save space
    for src, dst in to_copy:
        ub.symlink(src, dst)


def _devcheck_year_breakdown_reorg():
    """

    mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION
    sudo chown -R root:public /data/public/Aerial/US_ALASKA_MML_SEALION
    sudo chmod g+w -R /data/public/Aerial/US_ALASKA_MML_SEALION

    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2007/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2008/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2009/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2010/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2011/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2012/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2013/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2014/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2015/images
    sudo mkdir -p /data/public/Aerial/US_ALASKA_MML_SEALION/2016/images

    # Copy contents of extracted directories

    -rlptgoD.

    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2007/ /data/public/Aerial/US_ALASKA_MML_SEALION/2007/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2008/ /data/public/Aerial/US_ALASKA_MML_SEALION/2008/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2008W/ /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2009/ /data/public/Aerial/US_ALASKA_MML_SEALION/2009/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2010/ /data/public/Aerial/US_ALASKA_MML_SEALION/2010/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2011/ /data/public/Aerial/US_ALASKA_MML_SEALION/2011/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2012/ /data/public/Aerial/US_ALASKA_MML_SEALION/2012/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2013/ /data/public/Aerial/US_ALASKA_MML_SEALION/2013/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2014/ /data/public/Aerial/US_ALASKA_MML_SEALION/2014/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2015/ /data/public/Aerial/US_ALASKA_MML_SEALION/2015/images
    rsync -rP /data/projects/viame/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2016/ /data/public/Aerial/US_ALASKA_MML_SEALION/2016/images


    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2007/sealions_2007_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2007/sealions_2007_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2008/sealions_2008_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2008/sealions_2008_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/sealions_2008W_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2008W/sealions_2008W_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2009/sealions_2009_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2009/sealions_2009_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2010/sealions_2010_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2010/sealions_2010_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2011/sealions_2011_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2011/sealions_2011_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2014/sealions_2014_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2014/sealions_2014_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2015/sealions_2015_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2015/sealions_2015_v9.viame.csv

    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/US_ALASKA_MML_SEALION/2016/sealions_2016_v9.kwcoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2016/sealions_2016_v9.viame.csv

    tree -L 2 /data/public/Aerial/US_ALASKA_MML_SEALION

    """
    fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json')
    import kwcoco
    coco_dset = kwcoco.CocoDataset(fpath)

    {tuple([p for p in img['file_name'].split('/')[0:4] if not p.lower().endswith('.jpg')]) for img in coco_dset.imgs.values()}

    if 0:
        from kwcoco import coco_schema
        coco_schema.COCO_SCHEMA.validate(coco_dset.dataset)

    # Fixup broken keypoint category schemas
    for ann in coco_dset.anns.values():
        kpts = ann.get('keypoints', [])
        if kpts:
            for kp in kpts:
                if 'keypoint_category_id' in kp:
                    pass
                elif 'category' in kp:
                    kpcat = coco_dset._resolve_to_kpcat(kp.pop('category'))
                    kp['keypoint_category_id'] = kpcat['id']
                elif 'category_name' in kp:
                    kpcat = coco_dset._resolve_to_kpcat(kp.pop('category_name'))
                    kp['keypoint_category_id'] = kpcat['id']
                else:
                    raise Exception

    year_to_imgs = ub.group_items(coco_dset.imgs.values(), lambda x: x['year_code'])
    print(ub.map_vals(len, year_to_imgs))

    from os.path import relpath
    dest_root = '/data/public/Aerial/US_ALASKA_MML_SEALION'

    year_to_dset = {}
    for year, imgs in ub.ProgIter(list(year_to_imgs.items())):
        gids = [g['id'] for g in imgs]
        year_dset = coco_dset.subset(gids, copy=True, autobuild=False)

        # Munge the paths to the images
        # CAREFUL THIS IS CHANGING POINTERS IN THE NEW FILES AS WELL
        old_rel_path = join('BLACKEDOUT/extracted', str(year))
        old_rel_dot_path = join('/home/joncrall/data/raid/noaa/sealions/BLACKEDOUT/extracted/', str(year))

        dest_dpath = join(dest_root, str(year))
        for img in year_dset.dataset['images']:
            new_filename = join('images', relpath(img['file_name'], old_rel_path))
            img['file_name'] = new_filename

            if 'dot_fpath' in img:
                new_filename = join('images', relpath(img['dot_fpath'], old_rel_dot_path))
                img['dot_fpath'] = new_filename

        # year_dset.index.build()
        year_dset.fpath = join(dest_dpath, 'sealions_{}_v9.kwcoco.json'.format(year))
        year_dset.dump(year_dset.fpath, newlines=True)
        year_dset.img_root = dest_dpath
        year_to_dset[year] = year_dset

    year_dset = year_to_dset['2007']

    for year, year_dset in year_to_dset.items():
        assert not list(year_dset.missing_images())


def move_to_public_folder():
    """
    cp /data/projects/viame/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.mscoco.json \
       /data/public/Aerial/US_ALASKA_MML_SEALION/2012

    kwcoco reroot \
        --src /data/projects/viame/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.mscoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2012/sealions_2012_v3.kwcoco.json \
        --old_prefix=BLACKEDOUT/extracted/2012/ \
        --new_prefix=images \
        --absolute=False

    """
    import kwcoco
    old_year_fpath = '/data/projects/viame/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.mscoco.json'
    dset = kwcoco.CocoDataset(old_year_fpath)
    pass
