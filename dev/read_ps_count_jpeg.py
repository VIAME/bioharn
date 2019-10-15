from os.path import join
import io
import struct
import ubelt as ub


def hack_extract_all():
    import glob
    dpath = '/home/joncrall/data/noaa/KITWARE/CountedImgs/2016_Counted_Processed_Hi-res'
    fpaths = list(glob.glob(join(dpath, '*.JPG')))

    fpath_to_annots = {}

    for fpath in ub.ProgIter(fpaths):

        try:
            annots = parse_photoshop_count_annots(fpath)
            fpath_to_annots[fpath] = annots
        except AssertionError:
            # info = ub.cmd('strings {} | grep -i count'.format(fpath), shell=True)
            info = ub.cmd('strings {} | grep -i count'.format(fpath), shell=True)
            # assert info['ret'] == 0
            assert info['out'] == ''
            assert info['err'] == ''

        sum(map(len, map(list, fpath_to_annots.values())))

    has_objc = 0
    for fpath in ub.ProgIter(fpaths):
        # info = ub.cmd('strings {} | grep -i count'.format(fpath), shell=True)
        info = ub.cmd('strings {} | grep -i objc'.format(fpath), shell=True)
        # assert info['ret'] == 0
        if info['out'] != '':
            has_objc += 1


def parse_photoshop_count_annots(fpath):
    # fpath = ub.expandpath('~/Downloads/20160704_SSLS0811_C.JPG')
    from PIL import Image
    img = Image.open(open(fpath, 'rb'))
    annot_block = None

    assert 'photoshop' in img.info

    for k, v in img.info['photoshop'].items():
        if b'countObject' in v:
            annot_block = v
            break

    assert annot_block is not None

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

    objects = []
    # Using global because of IPython, and this script is already hacky as is
    global curr
    global attrkey
    attrkey = None
    curr = None

    def accept():
        global curr
        global attrkey

        print(ub.repr2(curr))
        global attrkey
        if curr is not None:
            objects.append(curr)
        attrkey = None
        curr = None

    _iter = iter(parts)
    attrkey = None
    for part in _iter:
        if part[0] == b'VlLs':
            accept()
            print('--------------')
        if part[0] == b'Objc':
            accept()
            print('***')
            curr = {}
            next(_iter)
        if part[0] == 'ID':
            if curr is not None:
                attrkey = part[3]
                curr[attrkey] = None
        else:
            if curr is not None and attrkey is not None:
                if curr[attrkey] is None:
                    curr[attrkey] = part[2]
        print(part)
    # print(ub.repr2(objects, nl=3))

    annots = []
    category = None
    for obj in objects:
        if b'Nm  ' in obj:
            category = obj[b'Nm  ']

        if b'X   ' in obj:
            ann = {
                'category': category,
                'keypoints': {
                    'xy': (obj[b'X   '], obj[b'Y   ']),
                }
            }
            annots.append(ann)

    return annots

    if False:
        list(map(chr, annot_block[stream.tell():stream.tell() + 64]))

        annot_block[stream.tell():stream.tell() + 128]

        pay = annot_block[stream.tell():stream.tell() + 8]
        struct.unpack('>l', pay)[0]


def first_try():
    fpath = ub.expandpath('~/Downloads/20160704_SSLS0811_C.JPG')
    file = open(fpath, 'rb')

    data = file.read()

    # JPEGS must start with this
    JPEG_START_IMAGE = b'\xFF\xD8'
    assert data[0:2] == JPEG_START_IMAGE

    pos = data.find(b'countGroupListVlLs')
    data[pos:pos + 10]
    pos = data.find(b'countGroupListVlLs')
    data[pos - 100:pos + 100]

    import logging
    logging.basicConfig()

    # Open the file and read all the EXIF metadata
    import exifread
    exifread.logger.setLevel(10)
    f1 = open(fpath, 'rb')
    tags = exifread.process_file(f1)  # NOQA

    # Any photoshop count data is between the exif data and the image data.

    remain = f1.read()

    # Not robust: look for the first start of jpeg marker and get all data
    # between. This might not work if the bytes of some data field happens
    # to be the jpeg start image
    # hack_block = remain[:remain.find(JPEG_START_IMAGE)]

    from PIL import Image
    img = Image.open(open(fpath, 'rb'))
    annot_block = None
    for k, v in img.info['photoshop'].items():
        if b'countObject' in v:
            annot_block = v
            break

    # https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/

    tokens = [
        # printProofSetupObjc
        # printSixteenBit
        # BckgObjc
        # vectorData
        # cropWhenPrinting
        # cropRectRight
        # cropRectTop
        # Vrsn

        b'countObjectListVlLs',
        b'countGroupListVlLs',
        b'countObject',
        b'countGroup',

        b'long',
        b'TEXT',
        b'doub',
        b'Objc',
        b'bool',
        b'Vsbl',
        b'Y   ',
        b'X   ',
    ]

    # 'obj ' = Reference
    # 'Objc' = Descriptor
    # 'VlLs' = List
    # 'doub' = Double
    # Type: OSType key
    # 'obj ' = Reference
    # 'Objc' = Descriptor
    # 'VlLs' = List
    # 'doub' = Double
    # 'UntF' = Unit float
    # 'TEXT' = String
    # 'enum' = Enumerated
    # 'long' = Integer
    # 'comp' = Large Integer
    # 'bool' = Boolean
    # 'GlbO' = GlobalObject same as Descriptor
    # 'type' = Class
    # 'GlbC' = Class
    # 'alis' = Alias
    # 'tdta' = Raw Data

    class StopParse(Exception):
        pass

    def parse_next(rest):
        _parts = []
        tok = None
        pos = None
        for t in tokens:
            cand = rest.find(t)
            if cand == -1:
                continue
            if pos is None or cand < pos:
                if rest[cand:cand + len(t)] == t:
                    tok = t
                    pos = cand
        if pos is None:
            raise StopParse

        left_part = rest[:pos]
        tok_part = rest[pos:pos + len(tok)]
        right_part = rest[pos + len(tok):]

        _parts.append(left_part)
        _parts.append(tok_part)

        if tok == b'TEXT':
            length_bytes = right_part[0:4]

            # All values defined as Unicode string consist of: A 4-byte length
            # field, representing the number of characters in the string (not
            # bytes).  The string of Unicode values, two bytes per character.
            num_chars = struct.unpack('>l', length_bytes)[0]
            num_bytes = num_chars * 2
            right_part = right_part[4:]
            text_bytes = right_part[:num_bytes]
            # print(text_bytes.decode('utf16'))
            _parts.append(((length_bytes, num_chars), text_bytes))
            right_part = right_part[num_bytes:]
        if tok in [b'doub']:

            tok_data = right_part[0:8]
            double_data = struct.unpack('>d', tok_data)
            _parts.append((tok_data, double_data))
            right_part = right_part[8:]

        elif tok in [b'long']:
            tok_data = right_part[0:4]
            # dat_littleendian = struct.unpack('<l', tok_data)
            long_data = struct.unpack('>l', tok_data)
            _parts.append((right_part[0:4], long_data))
            right_part = right_part[4:]
        rest = right_part
        _parts = [p for p in _parts if p]
        return _parts, rest

    parts = []
    rest = annot_block

    pos = rest.find(b'Vrsn')
    head = rest[:pos]
    rest = rest[pos:]
    parts.append(head)
    parts.append(rest[0:4])

    _parts, rest = parse_next(rest[4:])
    parts.extend(_parts)

    while True:
        try:
            _parts, rest = parse_next(rest)
            parts.extend(_parts)
        except StopParse:
            break

    annot_block[0:4]

    def parse_number(stream, key=None):
        pass

    remain[remain.find(JPEG_START_IMAGE):remain.find(JPEG_START_IMAGE) + 100]
    remain[remain.find(JPEG_START_IMAGE):].find(b'count')
