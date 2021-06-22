"""
Need to fixup 2012 and 2013 annotations
"""

from os.path import relpath
from os.path import basename
from os.path import join
import io
import struct
import ubelt as ub
import glob


class NoAnnots(Exception):
    pass


def make_correspondence():
    """
    Requires:
        pip install xlrd
    """
    import pandas as pd
    from pandas import ExcelWriter  # NOQA
    from pandas import ExcelFile  # NOQA

    # sheets = pd.read_excel('/home/joncrall/Downloads/filenames.xlsx', sheet_name=None)
    sheets = pd.read_excel(ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION/correspondence.xlsx'), sheet_name=None)

    # for year, df in sorted(sheets.items()):
    #     pass

    blackedout_root = ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted')
    blackedout_dpaths = list(glob.glob(join(blackedout_root, '20*')))
    year_to_blackedout_dpath = {}
    for dpath in blackedout_dpaths:
        year = basename(dpath)[0:4]
        year_to_blackedout_dpath[year] = dpath

    counted_root = ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION/COUNTED/extracted')
    counted_dpaths = list(glob.glob(join(counted_root, '20*_raw')))
    year_to_counted_dpath = {}
    for dpath in counted_dpaths:
        year = basename(dpath)[0:4]
        year_to_counted_dpath[year] = dpath

    corresponding = []

    years = ['2014', '2015', '2016']
    # years = ['2012', '2013']

    for year in years:
        df = sheets[year]

        counted_dpath = year_to_counted_dpath[year]
        blackedout_dpath = year_to_blackedout_dpath[year]
        counted_fpaths = recursive_find_imgs(counted_dpath)
        blackedout_fpaths = recursive_find_imgs(blackedout_dpath)

        counted_fpaths = [fpath for fpath in counted_fpaths if '[Originals]' not in fpath]

        def normalize_counted(x):
            x = ub.augpath(x, dpath='')
            x = x.upper().replace(' ', '').replace('-', '_')
            if x.endswith('.JPG'):
                x = x[:-4]
            if x.endswith('.'):
                x = x[:-1]
            return x

        def normalize_blacked(x):
            x = ub.augpath(x, dpath='', ext='')
            x = x.upper()
            if x.endswith('_C'):
                x = x[:-2]
            return x

        name_to_path1 = dict(ub.group_items(blackedout_fpaths, normalize_blacked))
        name_to_path2 = dict(ub.group_items(counted_fpaths, normalize_counted))

        for v in name_to_path1.values():
            assert len(v) == 1
        for v in name_to_path2.values():
            assert len(v) == 1

        def find_closest(query, candidates):
            """
            query = fn2
            candidates = list(name_to_path2.keys())
            """
            import Levenshtein
            import numpy as np
            dists = np.array(
                [Levenshtein.distance(query, cand) for cand in candidates])
            top_idxs = np.argsort(dists)
            top_dists = dists[top_idxs]
            k = max((top_dists < 6).sum(), 1)
            if 0:
                print('Query: {!r}'.format(query))
                for dist, idx in zip(top_dists, top_idxs[:k]):
                    print('    {}, {!r}'.format(dist, candidates[idx]))
            return top_dists[0], candidates[top_idxs[0]]

        print('-----')

        missing1 = []
        missing2 = []
        for fn1, fn2 in zip(df['BLACKEDOUT_filename'], df['COUNTEDIMAGE_Filename']):
            fn1_ = normalize_blacked(fn1)
            fn2_ = normalize_counted(fn2)

            fpath1 = name_to_path1[fn1_][0]
            try:
                fpath2 = name_to_path2[fn2_][0]
            except Exception:
                query = fn2_
                candidates = list(name_to_path2.keys())
                dist, cand = find_closest(query, candidates)
                assert dist in [4, 10]
                fn2_ = normalize_counted(cand)
                fpath2 = name_to_path2[fn2_][0]

            if fn1_ not in name_to_path1:
                missing1.append(fn1)
            if fn2_ not in name_to_path2:
                missing2.append(fn2)
            corresponding.append((fpath1, fpath2))

        print('year = {!r}'.format(year))
        print('missing blackedout {}'.format(len(missing1)))
        print('missing counted {}'.format(len(missing2)))

    fpath_to_annots = {}
    fpath2_to_fpath1 = {fpath2: fpath1 for fpath1, fpath2 in corresponding}
    failed_fpaths = ub.ddict(list)

    for fpath1, fpath2 in ub.ProgIter(corresponding):
        try:
            annots = parse_photoshop_count_annots(fpath2)
            fpath_to_annots[fpath2] = annots
        except NoAnnots as ex:
            failed_fpaths[ex.args].append(fpath2)

    print('Num Parsed {}: '.format(len(fpath_to_annots)))
    print('Num Total {}: '.format(len(corresponding)))
    print('Annots Parsed {}: '.format(sum(map(len, map(list, fpath_to_annots.values())))))
    print('Num Ignored {}'.format(ub.map_vals(len, failed_fpaths)))

    catnames = set()
    for fpath, annots in fpath_to_annots.items():
        for ann in annots:
            ann['category_name'] = str(ann['category_name']).replace('b\'', '').replace('\'', '')
            catnames.add(ann['category_name'])
    catnames = list(catnames)

    import kwcoco
    dset = kwcoco.CocoDataset(img_root=dpath)
    for name in catnames:
        dset.add_category(name)

    img_root = '/home/joncrall/data/noaa/sealions/BLACKEDOUT/extracted'

    for fpath, annots in fpath_to_annots.items():
        fpath1 = fpath2_to_fpath1[fpath]
        gid = dset.add_image(
            relpath(fpath1, img_root),
            counted_filename=basename(fpath2),
            blackedout_filename=basename(fpath1)
        )
        for ann in annots:
            cid = dset.name_to_cat[ann['category_name']]['id']
            cx, cy = ann['keypoints'][0]['xy']
            if ann['category_name'] == 'Bull':
                w = h = 128
            elif ann['category_name'] == 'Fem':
                w = h = 96
            else:
                w = h = 64
            x = round(cx - w / 2, 1)
            y = round(cy - h / 2, 1)
            bbox = [x, y, w, h]
            ann['keypoints'][0]['xy'] = [round(cx, 2), round(cy, 2)]
            # if 'bbox' not in ann:
            ann['bbox'] = bbox
            aid = dset.add_annotation(gid, cid, **ann)

    dset.dump('sealions_photoshop_annots_v1.mscoco.json', newlines=True)

    if 0:
        dset.img_root = img_root
        dset.show_image(800)


def recursive_find_imgs(dpath):
    fpaths = []
    fpaths += list(glob.glob(join(dpath, '**/*.JPG'), recursive=True))
    fpaths += list(glob.glob(join(dpath, '**/*.jpg'), recursive=True))
    fpaths += list(glob.glob(join(dpath, '**/*.jpeg'), recursive=True))
    fpaths += list(glob.glob(join(dpath, '**/*.JPEG'), recursive=True))
    return fpaths


def nonrecursive_find_imgs(dpath):
    import os
    fpaths = [join(dpath, f) for f in os.listdir(dpath) if f.lower().endswith(('.jpg', '.jpeg'))]
    return fpaths


def hack_extract_all():
    import glob
    dpath = '/home/joncrall/data/noaa/KITWARE/CountedImgs/2016_Counted_Processed_Hi-res'

    dpaths = [
        '/home/joncrall/data/noaa/sealions/2014',
        '/home/joncrall/data/noaa/sealions/2015',
        '/home/joncrall/data/noaa/sealions/2016',
        '/home/joncrall/data/noaa/sealions/2013',
    ]
    fpaths = []
    for dpath in dpaths:
        fpaths = []
        dpath = '/home/joncrall/data/noaa/sealions-2/'
        fpaths += recursive_find_imgs(dpath)

    fpath_to_annots = {}
    failed_fpaths = ub.ddict(list)

    for fpath in ub.ProgIter(fpaths):
        try:
            annots = parse_photoshop_count_annots(fpath)
            fpath_to_annots[fpath] = annots
        except NoAnnots as ex:
            failed_fpaths[ex.args].append(fpath)

    print('Num Parsed {}: '.format(len(fpath_to_annots)))
    print('Num Total {}: '.format(len(fpaths)))
    print('Annots Parsed {}: '.format(sum(map(len, map(list, fpath_to_annots.values())))))
    print('Num Ignored {}'.format(ub.map_vals(len, failed_fpaths)))

    catnames = set()
    for fpath, annots in fpath_to_annots.items():
        for ann in annots:
            catnames.add(ann['category_name'])
    catnames = list(catnames)

    import kwcoco
    dset = kwcoco.CocoDataset(img_root=dpath)
    for name in catnames:
        dset.add_category(name)

    for fpath, annots in fpath_to_annots.items():
        gid = dset.add_image(fpath)
        for ann in annots:
            cid = dset.name_to_cat[ann['category_name']]['id']
            aid = dset.add_annotation(gid, cid, **ann)

    print(ub.map_vals(len, failed_fpaths))

    miss_photoshop = failed_fpaths[('no photoshop',)]
    miss_contobj = failed_fpaths[('no countObject',)]

    for fpath in miss_contobj:
        from PIL import Image
        img = Image.open(open(fpath, 'rb'))
        text = ub.repr2(img.info['photoshop'])
        if len(text) > 1000:
            print('fpath = {!r}'.format(fpath))
            print(text)
            print(len(text))

    has_code = 0
    for fpath in ub.ProgIter(fpaths):
        info = ub.cmd('head -n 10000 {} | strings | grep -i count'.format(fpath), shell=True)
        # info = ub.cmd('strings {} | grep -i count'.format(fpath), shell=True)
        # info = ub.cmd('strings {} | grep -i objc'.format(fpath), shell=True)
        # assert info['ret'] == 0
        if info['out'] != '':
            has_code += 1
    print('has_code = {!r}'.format(has_code))


def test_photoshop_parsable():
    """
    Attempt to figure out which sealion years have photoshop annotations
    """
    dpath = ub.expandpath('$HOME/data/noaa/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/')
    IMAGE_EXTENSIONS = (
        '.bmp', '.pgm', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
        '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif', '.r0',
        '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
    )
    import os
    for root, ds, fs in os.walk(dpath):
        if '2013' not in root and '2012' not in root:
            continue
        print('root = {!r}'.format(root))
        num_files = len(fs)
        print('num_files = {!r}'.format(num_files))

        fpaths = [join(root, f) for f in fs if f.lower().endswith(IMAGE_EXTENSIONS)]

        fpath_to_annots = {}
        failed_fpaths = ub.ddict(list)
        for fpath in ub.ProgIter(fpaths):
            try:
                annots = parse_photoshop_count_annots(fpath)
                fpath_to_annots[fpath] = annots
            except NoAnnots as ex:
                failed_fpaths[ex.args].append(fpath)

        failed_counts = ub.map_vals(len, failed_fpaths)
        success_counts = len(fpath_to_annots)
        print('failed_counts = {!r}'.format(failed_counts))
        print('success_counts = {!r}'.format(success_counts))


def parse_photoshop_count_annots(fpath):
    """
    fpath = ub.expandpath('~/Downloads/20160704_SSLS0811_C.JPG')

    fpath = '/home/joncrall/remote/namek/data/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2013/20130620_SSLC0532_C.JPG'
    fpath = '/home/joncrall/remote/namek/data/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted/2012/20120625_SSLC0056_C.jpg'

    parse_photoshop_count_annots(fpath)
    import xdev
    xdev.profile_now(parse_photoshop_count_annots)(fpath)
    """
    from PIL import Image
    img = Image.open(open(fpath, 'rb'))
    annot_block = None
    # print(img.info.keys())

    if 'photoshop' not in img.info:
        raise NoAnnots('no photoshop')

    # See also:
    # https://github.com/python-pillow/Pillow/blob/master/src/PIL/JpegImagePlugin.py#L112-L144
    for k, v in img.info['photoshop'].items():
        if b'countObject' in v:
            annot_block = v
            break

    if annot_block is None:
        raise NoAnnots('no countObject')

    def parse_known_part(stream, key=None):
        if key is None:
            key = stream.read(4)

        if key == b'VlLs':
            len_payload = stream.read(4)
            parsed = struct.unpack('>l', len_payload)
            return key, len_payload, parsed
        elif key == b'bool':
            payload = stream.read(1)
            parsed = struct.unpack('>b', payload)[0]
            return key, payload, parsed
        elif key == b'long':
            payload = stream.read(4)
            parsed = struct.unpack('>l', payload)[0]
            return key, payload, parsed
        elif key == b'doub':
            payload = stream.read(8)
            parsed = struct.unpack('>d', payload)[0]
            return key, payload, parsed
        elif key == b'Objc':
            objc_version = stream.read(4)  # Should be 1
            unknown = stream.read(2)  # hack
            return (key, objc_version, unknown)
        elif key == b'TEXT':
            len_payload = stream.read(4)
            num_chars = struct.unpack('>l', len_payload)[0]
            num_bytes = num_chars * 2
            text_payload = stream.read(num_bytes)
            # Not sure if this is exactly right
            decoded = text_payload[1::2].rstrip(b'\x00')
            return (key, (len_payload, text_payload), decoded)
        else:
            info = parse_identifier(stream, key=key)

            if info[3] == b'countObject':
                # HACK
                info = info + ('HACK: ', stream.read(4),)
            return info
            # raise Exception

    def parse_identifier(stream, key=None):
        if key is None:
            key = stream.read(4)
        len_parsed = struct.unpack('>l', key)[0]
        if len_parsed == 0:
            len_parsed = 4
        len_parsed = max(4, len_parsed)
        identifier = stream.read(len_parsed)
        return ('ID', key, len_parsed, identifier)

    parts = []
    stream = io.BytesIO(annot_block)
    stream.read(annot_block.find(b'Vrsn') + 12)

    while True:
        key = stream.read(4)
        if not len(key):
            break

        part = parse_known_part(stream, key)
        parts.append(part)
        # parse_known_part(stream)

    class State:
        def __init__(state):
            state.objects = []
            state.curr = None
            state.attrkey = None

        def accept(state):
            # print(ub.repr2(state.curr))
            if state.curr is not None:
                state.objects.append(state.curr)
            state.attrkey = None
            state.curr = None

    state = State()

    _iter = iter(parts)
    for part in _iter:
        if part[0] == b'VlLs':
            state.accept()
            # print('--------------')
        if part[0] == b'Objc':
            state.accept()
            # print('***')
            state.curr = {}
            next(_iter)
        if part[0] == 'ID':
            if state.curr is not None:
                state.attrkey = part[3]
                state.curr[state.attrkey] = None
        else:
            if state.curr is not None and state.attrkey is not None:
                if state.curr[state.attrkey] is None:
                    state.curr[state.attrkey] = part[2]
        # print(part)
    # print(ub.repr2(objects, nl=3))

    annots = []
    category = None
    for obj in state.objects:
        if b'Nm  ' in obj:
            category = obj[b'Nm  ']

        if b'X   ' in obj:
            ann = {
                'category_name': category,
                'keypoints': [{
                    'xy': (obj[b'X   '], obj[b'Y   ']),
                    'keypoint_category_name': 'dot',
                }]
            }
            annots.append(ann)

    return annots


def make_correspondence_2012_2013():
    """
    Going back and redoing 2012 / 2013 data

    Requires:
        pip install xlrd
    """
    import pandas as pd
    from pandas import ExcelWriter  # NOQA
    from pandas import ExcelFile  # NOQA

    sheets = pd.read_excel(
            ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION/correspondence.xlsx'), sheet_name=None)

    data_root = ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION')
    blackedout_root = ub.expandpath('$HOME/data/US_ALASKA_MML_SEALION/BLACKEDOUT/extracted')
    blackedout_dpaths = list(glob.glob(join(blackedout_root, '20*')))
    year_to_blackedout_dpath = {}
    for dpath in blackedout_dpaths:
        year = basename(dpath)[0:4]
        year_to_blackedout_dpath[year] = dpath

    # years = ['2014', '2015', '2016']
    years = ['2012', '2013']

    for year in years:
        df = sheets[year]
        # counted_dpath = year_to_counted_dpath[year]

        blackedout_dpath = year_to_blackedout_dpath[year]
        blackedout_fpaths = nonrecursive_find_imgs(blackedout_dpath)

        if year in {'2013', '2012'}:
            counted_dpath = join(blackedout_dpath, '[Originals]')
            counted_fpaths = nonrecursive_find_imgs(counted_dpath)

        def normalize_counted(x):
            x = ub.augpath(x, dpath='')
            x = x.upper().replace(' ', '').replace('-', '_')
            if x.endswith('.JPG'):
                x = x[:-4]
            if x.endswith('.'):
                x = x[:-1]
            return x

        def normalize_blacked(x):
            x = ub.augpath(x, dpath='', ext='')
            x = x.upper()
            if x.endswith('_C'):
                x = x[:-2]
            return x

        name_to_path1 = dict(ub.group_items(blackedout_fpaths, normalize_blacked))
        name_to_path2 = dict(ub.group_items(counted_fpaths, normalize_counted))

        for v in name_to_path1.values():
            assert len(v) == 1
        for v in name_to_path2.values():
            assert len(v) == 1

        def find_closest(query, candidates):
            """
            query = fn2
            candidates = list(name_to_path2.keys())
            !pip install python-Levenshtein
            """
            import Levenshtein
            import numpy as np
            dists = np.array(
                [Levenshtein.distance(query, cand) for cand in candidates])
            top_idxs = np.argsort(dists)
            top_dists = dists[top_idxs]
            k = max((top_dists < 6).sum(), 1)
            if 0:
                print('Query: {!r}'.format(query))
                for dist, idx in zip(top_dists, top_idxs[:k]):
                    print('    {}, {!r}'.format(dist, candidates[idx]))
            return top_dists[0], candidates[top_idxs[0]]

        print('-----')

        missing1 = []
        missing2 = []
        non_exact = []
        exact_match = []
        failure = []
        corresponding = []
        """
        The reason for this terrible block of code is the following

        There is a spreadsheet that provides a correspondence between the
        BLACKEDOUT_filenames and the COUNTEDIMAGE_filenames (the former should
        be good for learning, and the latter actually contain the annotations).

        However, the actual COUNTEDIMAGE filenames in the directory are
        different than the names listed in the excel spreadsheet, but not by
        much. Sometimes, its the difference of adding and underscore or
        ommiting a prefix, but its not consistent, so we have to do some hacks.
        """
        for fn1, fn2 in zip(df['BLACKEDOUT_filename'], df['COUNTEDIMAGE_Filename']):
            fn1_ = normalize_blacked(fn1)
            fn2_ = normalize_counted(fn2)

            fpath1 = name_to_path1[fn1_][0]
            try:
                fpath2 = name_to_path2[fn2_][0]
            except Exception:
                query = fn2_
                candidates = list(name_to_path2.keys())
                dist, cand = find_closest(query, candidates)
                non_exact.append((fn2, cand))
                if dist not in [4, 10]:
                    failure.append((fn1_, fn2_, cand, dist))
                    continue
                assert dist in [4, 10]
                fn2_ = normalize_counted(cand)
                fpath2 = name_to_path2[fn2_][0]
            else:
                exact_match.append((fn1_, fn2_))

            if fn1_ not in name_to_path1:
                missing1.append(fn1)
            if fn2_ not in name_to_path2:
                missing2.append(fn2)
            corresponding.append((fpath1, fpath2))

        print('year = {!r}'.format(year))
        print('missing blackedout {}'.format(len(missing1)))
        print('missing counted {}'.format(len(missing2)))
        print('len(exact_match) = {}'.format(len(exact_match)))
        print('len(non_exact) = {}'.format(len(non_exact)))
        print('len(failure) = {}'.format(len(failure)))

        if year in {'2013', '2012'}:
            import parse
            corresponding2 = []
            for fpath1, fpath2 in corresponding:
                b1 = basename(fpath1)

                if b1 == '20130628_SSLC0469_C.jpg':
                    # HACK FOR BAD CORRESONDENCE.
                    continue

                b2 = basename(fpath2)
                p1 = b1.split('_')[0]
                p2 = b2.split('_')[0]

                if len(p1) == len(p2) and p1 != p2:
                    print('SKIP')
                    print('fpath1 = {!r}'.format(fpath1))
                    print('fpath2 = {!r}'.format(fpath2))
                    continue

                parsed1 = parse.parse('{}SSL{char}{num:d}{}', fpath1)
                parsed2 = parse.parse('{}SSL{char}{num:d}{}', fpath2)
                if parsed1 and parsed2:
                    n1 = parsed1.named['num']
                    n2 = parsed2.named['num']
                    if n1 != n2 or parsed1.named['char'] != parsed2.named['char']:
                        print('SKIP')
                        print('fpath1 = {!r}'.format(fpath1))
                        print('fpath2 = {!r}'.format(fpath2))
                        continue
                else:
                    print('SKIP')
                    print('fpath1 = {!r}'.format(fpath1))
                    print('fpath2 = {!r}'.format(fpath2))
                    continue
                corresponding2.append((fpath1, fpath2))
        else:
            corresponding2 = corresponding

        assert not ub.find_duplicates([t[0] for t in corresponding2])
        assert not ub.find_duplicates([t[1] for t in corresponding2])
        fpath2_to_fpath1 = {fpath2: fpath1 for fpath1, fpath2 in corresponding2}

        fpath_to_annots = {}
        failed_fpaths = ub.ddict(list)
        for fpath1, fpath2 in ub.ProgIter(corresponding2):
            try:
                annots = parse_photoshop_count_annots(fpath2)
                fpath_to_annots[fpath2] = annots
            except NoAnnots as ex:
                failed_fpaths[ex.args].append(fpath2)

        print('Num Parsed {}: '.format(len(fpath_to_annots)))
        print('Num Total {}: '.format(len(corresponding)))
        print('Annots Parsed {}: '.format(sum(map(len, map(list, fpath_to_annots.values())))))
        print('Num Ignored {}'.format(ub.map_vals(len, failed_fpaths)))

        import kwcoco
        dset = kwcoco.CocoDataset(img_root=data_root)

        catnames = set()
        for fpath, annots in fpath_to_annots.items():
            for ann in annots:
                ann['category_name'] = str(ann['category_name']).replace('b\'', '').replace('\'', '')
                catnames.add(ann['category_name'])
        catnames = list(catnames)

        for name in catnames:
            dset.add_category(name)

        for fpath, annots in fpath_to_annots.items():
            # Use the blackedout image as the real data image
            fpath1 = fpath2_to_fpath1[fpath]
            gid = dset.add_image(
                relpath(fpath1, data_root),
                counted_filename=basename(fpath2),
                blackedout_filename=basename(fpath1)
            )
            for ann in annots:
                cid = dset.name_to_cat[ann['category_name']]['id']
                cx, cy = ann['keypoints'][0]['xy']
                if ann['category_name'] == 'Bull':
                    w = h = 128
                elif ann['category_name'] == 'Fem':
                    w = h = 96
                else:
                    w = h = 64
                x = round(cx - w / 2, 1)
                y = round(cy - h / 2, 1)
                bbox = [x, y, w, h]
                ann['keypoints'][0]['xy'] = [round(cx, 2), round(cy, 2)]
                # if 'bbox' not in ann:
                ann['bbox'] = bbox
                dset.add_annotation(gid, cid, **ann)

        dset.dataset['keypoint_categories'] = [
            {'name': 'dot', 'id': 1},
        ]
        for ann in dset.anns.values():
            for kp in ann.get('keypoints', []):
                kpcat = kp.pop('keypoint_category_name')
                kpcid = dset._resolve_to_kpcat(kpcat)['id']
                kp['keypoint_category_id'] = kpcid

        fname = 'sealions_photoshop_annots_{}_v2.kwcoco.json'.format(year)
        dset.fpath = join(data_root, fname)
        dset.dump(dset.fpath, newlines=True)

    """

    # Predict on these new annotations with existing models

    mkdir -p $HOME/work/models
    girder-client --api-url https://data.kitware.com/api/v1 download \
        --parent-type file 5f0cbabc9014a6d84e1c5650 \
        $HOME/work/models/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip

    python -m bioharn.detect_predict \
        --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_photoshop_annots_2012_v2.kwcoco.json \
        --deployed=$HOME/work/models/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip \
        --out_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v8_2012 \
        --draw=4 \
        --workers=4 \
        --workdir=$HOME/work/sealions \
        --xpu=0 --batch_size=128

    python -m bioharn.detect_predict \
        --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_photoshop_annots_2013_v2.kwcoco.json \
        --deployed=$HOME/work/models/deploy_MM_CascadeRCNN_bzmjrthj_037_CYEOPQ.zip \
        --out_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v8_2013 \
        --draw=4 \
        --workers=4 \
        --workdir=$HOME/work/sealions \
        --xpu=0 --batch_size=32


    # Do refinement preocess to pick the new detection boxes when appropriate
    python $HOME/code/bioharn/dev/refine_detections.py \
        --true_fpath=$HOME/data/US_ALASKA_MML_SEALION/sealions_photoshop_annots_2012_v2.kwcoco.json \
        --pred_fpaths=[$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v8_2012/pred/detections.mscoco.json,] \
        --out_fpath=$HOME/data/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.mscoco.json \
        --viz_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/refine9_2012 \
        --score_thresh=0.2

    python $HOME/code/bioharn/dev/refine_detections.py \
        --true_fpath=$HOME/data/US_ALASKA_MML_SEALION/sealions_photoshop_annots_2013_v2.kwcoco.json \
        --pred_fpaths=[$HOME/data/US_ALASKA_MML_SEALION/detections/cascade_v8_2013/pred/detections.mscoco.json,] \
        --out_fpath=$HOME/data/US_ALASKA_MML_SEALION/sealions_2013_refined_v3.mscoco.json \
        --viz_dpath=$HOME/data/US_ALASKA_MML_SEALION/detections/refine9_2013 \
        --score_thresh=0.2

    # Copy to VIAME server
    cp $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.kwcoco.json \
       $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.kwcoco.json

    # Copy to VIAME server
    cp $HOME/remote/namek/data/US_ALASKA_MML_SEALION/sealions_2013_refined_v3.mscoco.json \
       $HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_2013_refined_v3.mscoco.json


    # On VIAME server: Move to the non-flat folder in PUBLIC
    kwcoco reroot \
        --src /data/projects/viame/US_ALASKA_MML_SEALION/sealions_2012_refined_v3.mscoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2012/sealions_2012_v3.kwcoco.json \
        --old_prefix=BLACKEDOUT/extracted/2012/ \
        --new_prefix=images \
        --absolute=False

    kwcoco reroot \
        --src /data/projects/viame/US_ALASKA_MML_SEALION/sealions_2013_refined_v3.mscoco.json \
        --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2013/sealions_2013_v3.kwcoco.json \
        --old_prefix=BLACKEDOUT/extracted/2013/ \
        --new_prefix=images \
        --absolute=False


    # On VIAME server: Convert to VIAME CSV
    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/_ORIG_US_ALASKA_MML_SEALION/2012/sealions_2012_v3.kwcoco.json \
        --dst /data/public/Aerial/_ORIG_US_ALASKA_MML_SEALION/2012/sealions_2012_v3.viame.csv


    python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
        --src /data/public/Aerial/_ORIG_US_ALASKA_MML_SEALION/2013/sealions_2013_v3.kwcoco.json \
        --dst /data/public/Aerial/_ORIG_US_ALASKA_MML_SEALION/2013/sealions_2013_v3.viame.csv


    # For final step see the hack_sealion_annots_on_server script
    """
