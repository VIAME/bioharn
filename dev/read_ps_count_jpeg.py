from os.path import join
import io
import struct
import ubelt as ub


class NoAnnots(Exception):
    pass


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
        fpaths += list(glob.glob(join(dpath, '*.JPG')))
        fpaths += list(glob.glob(join(dpath, '*.jpg')))
        fpaths += list(glob.glob(join(dpath, '*.jpeg')))
        fpaths += list(glob.glob(join(dpath, '*.JPEG')))

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

    import ndsampler
    dset = ndsampler.CocoDataset(img_root=dpath)
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


def parse_photoshop_count_annots(fpath):
    """
    fpath = ub.expandpath('~/Downloads/20160704_SSLS0811_C.JPG')
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
                    'category_name': 'dot',
                }]
            }
            annots.append(ann)

    return annots
