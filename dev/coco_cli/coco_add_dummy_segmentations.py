#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoDummyAddSegmentationsCLI:
    name = 'DummyAddSegmentations'

    class CLIConfig(scfg.Config):
        """
        Compute summary statistics about a COCO dataset
        """
        default = {
            'src': scfg.Value(['special:shapes8'], nargs='+', help='path to dataset'),

            'dst': scfg.Value(None, help=(
                'Save the dataset to a new file')),

            'embed': scfg.Value(False, help='embed into interactive shell'),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        if isinstance(config['src'], str):
            fpaths = [config['src']]
        else:
            fpaths = config['src']

        datasets = []
        for fpath in ub.ProgIter(fpaths, desc='reading datasets', verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)
            datasets.append(dset)

        if config['embed']:
            # Hidden hack
            import xdev
            xdev.embed()

        # Hacks
        add_sseg_to_cats = [
            'live sea scallop',
            'swimming sea scallop',
            'dead sea scallop',
        ]

        for dset in datasets:
            import kwimage
            for ann in ub.ProgIter(dset.anns.values(), desc='add sseg'):
                catname = dset._resolve_to_cat(ann['category_id'])['name']
                if catname in add_sseg_to_cats:
                    bbox = kwimage.Boxes([ann['bbox']], 'xywh').to_cxywh()
                    xy = bbox.xy_center[0]
                    w = bbox.width[0, 0]
                    r = w / 2.0
                    circle = kwimage.Polygon.circle(xy, r, resolution=32)
                    ann['segmentation'] = circle.to_coco(style='new')

        dset.fpath = config['dst']
        print('dump dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)


_CLI = CocoDummyAddSegmentationsCLI

if __name__ == '__main__':
    _CLI.main()
