"""
Basic script to convert VIAME-CSV to kwcoco

References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html
"""

import scriptconfig as scfg


class ConvertConfig(scfg.Config):
    default = {
        'src': scfg.PathList('in.viame.csv'),
        'dst': scfg.Value('out.kwcoco.json'),
    }


def main(cmdline=True, **kw):
    import kwcoco
    config = ConvertConfig(default=kw, cmdline=cmdline)

    dset = kwcoco.CocoDataset()

    # TODO: ability to map image ids to agree with another coco file
    csv_fpaths = config['src']
    for csv_fpath in csv_fpaths:
        from bioharn.io.viame_csv import ViameCSV
        csv = ViameCSV(csv_fpath)
        csv.extend_coco(dset=dset)

    dset.fpath = config['dst']
    print('dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)


if __name__ == '__main__':
    main()
