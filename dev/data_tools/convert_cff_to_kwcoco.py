r"""
Usage
-----

.. code::

    cd ~/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH

    python ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py \
        --in_fpath habcam-2020-2021.csv \
        --out_fpath data.kwcoco.zip

    kwcoco-explorer --data data.kwcoco.zip

"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
import pandas as pd
import kwcoco
import kwutil
import numpy as np
import kwimage


class ConvertCffToKwcocoCLI(scfg.DataConfig):
    """
    Convert habcam-2020-2021.csv data to kwcoco
    """
    in_fpath = scfg.Value('habcam-2020-2021.csv', help='input csv file', position=1)
    out_fpath = scfg.Value('data.kwcoco.zip', help='output kwcoco file', position=2)

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from convert_cff_to_kwcoco import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = ConvertCffToKwcocoCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        main(**config)

__cli__ = ConvertCffToKwcocoCLI


def main(in_fpath, out_fpath):
    """
    """
    data = pd.read_csv('habcam-2020-2021.csv')

    image_cols = [
        'Date', 'Time',
        'Altitude', 'Water_depth', 'Heading', 'Pitch', 'Roll', 'Vehicle_depth',
        'Lat_ddmm', 'Long_ddmm', 'Temperature', 'Conductivity', 'Lat_decdeg',
        'Long_decdeg', 'Speed', 'FOV'
    ]

    if 0:
        # Developer inspection
        data['Shape'].value_counts()
        data['Organism'].value_counts()
        data['orgID'].value_counts()
        data['shapeID'].value_counts()
        data['subID'].value_counts()
        data['Substrate'].value_counts()

    for idx, row in enumerate(data.to_dict('records')):
        try:
            float(row['x1'])
        except Exception:
            print(idx, f'row = {ub.urepr(row, nl=1)}')

    # hack: remove bad data
    data = data[data['x1'] != 'Astropecten spp']

    # Ensure coordinate data type is float
    coord_cols = ['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4']
    for c in coord_cols:
        data[c] = data[c].apply(float)

    # Efficiently mark if a row has coordinates
    data['has_ann_coords'] = ~data['x1'].isna()

    imgname_to_subrows = dict(list(data.groupby('Imagename')))

    dset = kwcoco.CocoDataset()
    prog = ub.ProgIter(imgname_to_subrows.items(),
                       total=len(imgname_to_subrows), desc='Convert images')
    for imagename, subrows in prog:
        img = {
            'file_name': f'jpgs/L{imagename}',
        }

        # Determine if data is image-level or annotation-level
        anns = []

        ann_rows = subrows[subrows['has_ann_coords']]
        img_rows = subrows[~subrows['has_ann_coords']]

        for row in ann_rows.to_dict('records'):
            ann = process_annotation_row(row)
            anns.append(ann)

        image_infos = img_rows.to_dict('records')
        if image_infos:
            img['substrates'] = list({r['Substrate'] for r in image_infos})

        # Handle the rest of image level information
        for col in image_cols:
            assert ub.allsame(subrows[col]), 'sanity'
        image_data = ub.udict(row) & image_cols
        try:
            # Try to convert to a standard datetime format if possible
            datetime = kwutil.datetime.coerce(image_data.get('Date') + ' ' + image_data.get('Time'))
            image_data['datetime'] = datetime.isoformat()
        except Exception:
            ...
        img.update(image_data)

        # Add information to the coco datasest
        image_id = dset.add_image(**img)
        for ann in anns:
            ann['image_id'] = image_id
            ann['category_id'] = dset.ensure_category(ann.pop('category_name'))
            dset.add_annotation(**ann)

    dset.fpath = out_fpath
    dset.validate()
    dset.dump()


def process_annotation_row(row):
    """
    Four shapes were used for annotations
        1.	Lines – for all live scallops (shell length or width), probable live scallops, swimming scallops, and probable swimming scallops. X1,Y1 and X2,Y2 are end points.
        2.	Points – for scallops too small to measure. Coordinates X1,Y1.
        3.	Rectangles – one option for all other organisms. X1,Y1 is the start of the drag and X2,Y2 is the rectangle width and height.
        4.	Extreme rectangle – other option for all other organisms. X1,Y1 through X4,Y4 are the rectangle corner coordinates
    """
    poly = None
    if row['Shape'] == 'Rectangle':
        box = kwimage.Box.coerce([row['x1'], row['y1'], row['x2'], row['y2']], format='xywh')
    elif row['Shape'] == 'Extreme_rectangle':
        poly = kwimage.Polygon.coerce(np.array([
            [row['x1'], row['y1']],
            [row['x2'], row['y2']],
            [row['x3'], row['y3']],
            [row['x4'], row['y4']],
        ]))
        box = poly.to_box()
    elif row['Shape'] == 'Line':
        # hack a line into a box
        xy1 = np.array([row['x1'], row['y1']])
        xy2 = np.array([row['x2'], row['y2']])
        centroid = (xy1 + xy2) / 2
        diameter = np.sqrt(((xy2 - xy1) ** 2).sum())
        radius = diameter / 2
        poly = kwimage.Polygon.circle(centroid, r=radius)
        box = poly.to_box()
    elif row['Shape'] == 'Point':
        # hack the point into a bbox
        box = kwimage.Box.coerce([row['x1'], row['y1'], 1, 1], format='xywh')
    else:
        raise KeyError(row['Shape'])

    catname = row['Organism']

    ann = {}
    ann['bbox'] = box.to_coco()
    ann['category_name'] = catname
    ann['category_id'] = row['orgID']
    if poly is not None:
        ann['poly'] = poly.to_coco('new')
    return ann


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/bioharn/dev/data_tools/convert_cff_to_kwcoco.py
        python -m convert_cff_to_kwcoco
    """
    __cli__.main()
