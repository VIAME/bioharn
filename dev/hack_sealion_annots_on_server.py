
def hack():
    """
    cd /data/public/Aerial/US_ALASKA_MML_SEALION
    sudo chmod g+w -R /data/public/Aerial/
    sudo chown -R root:public /data/public/Aerial/US_ALASKA_MML_SEALION_FLAT/
    """
    from os.path import join, basename, dirname
    import ubelt as ub
    import glob
    import copy

    dpath = '/data/public/Aerial/US_ALASKA_MML_SEALION'

    flat_dpath = ub.ensuredir('/data/public/Aerial/US_ALASKA_MML_SEALION_FLAT')

    csv_fpaths = glob.glob(join(dpath, '*', '*.csv'))

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

            # Modify the CSV
            new_parts = copy.copy(parts)
            new_parts[1] = new_gname
            new_lines.append(','.join(new_parts))

        # Rewrite the modified CSV to the same directory as the images
        new_text = '\n'.join(new_lines)
        with open(new_csv_fpath, 'w') as file:
            text = file.write(new_text)

    dst_fpaths = [d for s, d in to_copy]

    len(dst_fpaths)
    len(set(dst_fpaths))
    assert not ub.find_duplicates(dst_fpaths), 'should have no dups'

    # Symlink instead of copying to save space
    for src, dst in to_copy:
        ub.symlink(src, dst)
