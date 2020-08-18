"""
Basic script to convert kwcoco to VIAME-CSV

References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html


Alternative Method:
    # Convert to VIAME CSV

    VIAME_PREFIX=$HOME/code/VIAME/build-py3.8/install
    source $VIAME_PREFIX/setup_viame.sh

    # Write out the required input list
    python -c "\
import kwcoco
coco_dset = kwcoco.CocoDataset('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json')
text = chr(10).join(sorted([img['file_name'] for img in coco_dset.imgs.values()]))
open('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/input_list.txt', 'w').write(text)
"

    cat $VIAME_PREFIX/examples/detection_file_conversions/pipelines/coco_json_to_viame_csv.pipe

    kwiver runner $VIAME_PREFIX/examples/detection_file_conversions/pipelines/coco_json_to_viame_csv.pipe \
        -s detection_reader:file_name=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json \
        -s detection_writer:file_name=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.kwiver.viame.csv \
        -s image_reader:video_filename=$HOME/data/US_ALASKA_MML_SEALION/input_list.txt

"""

import scriptconfig as scfg


class ConvertConfig(scfg.Config):
    default = {
        'src': scfg.Value('in.mscoco.json'),
        'dst': scfg.Value('out.viame.csv'),
    }


def main(**kw):
    """
    Ignore:
        kw = {
            'src': ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.mscoco.json'),
            'dst': ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v9.csv'),
        }
    """
    config = ConvertConfig(default=kw, cmdline=True)

    import kwcoco
    import kwimage
    import ubelt as ub
    coco_dset = kwcoco.CocoDataset(config['src'])

    # all_annots = coco_dset.annots()
    # TODO: eventually, kwcoco should maintain this index
    # tid_to_aids = ub.group_items(all_annots.lookup('id'), all_annots.lookup('track_id', None))
    # tid_to_aids.pop(None)
    # all_ann_keys = {key for ann in coco_dset.anns.values() for key in ann.keys()}
    # for ann in coco_dset.anns.values():
    #     if 'score' in ann:
    #         break

    csv_rows = []
    for gid, img in ub.ProgIter(coco_dset.imgs.items(), total=coco_dset.n_images):
        gname = img['file_name']
        aids = coco_dset.gid_to_aids[gid]

        frame_index = img.get('frame_index', 0)
        # vidid = img.get('video_id', None)

        for aid in aids:
            ann = coco_dset.anns[aid]
            cat = coco_dset.cats[ann['category_id']]
            catname = cat['name']

            # just use annotation id if no tracks
            tid = ann.get('track_id', aid)
            # tracked_aids = tid_to_aids.get(tid, [aid])
            # track_len = len(tracked_aids)

            tl_x, tl_y, br_x, br_y = kwimage.Boxes([ann['bbox']], 'xywh').toformat('tlbr').data[0].tolist()

            score = ann.get('score', 1)

            row = [
                 tid,             # 1 - Detection or Track Unique ID
                 gname,           # 2 - Video or Image String Identifier
                 frame_index,     # 3 - Unique Frame Integer Identifier
                 round(tl_x, 3),  # 4 - TL-x (top left of the image is the origin: 0,0
                 round(tl_y, 3),  # 5 - TL-y
                 round(br_x, 3),  # 6 - BR-x
                 round(br_y, 3),  # 7 - BR-y
                 score,           # 8 - Auxiliary Confidence (how likely is this actually an object)
                 -1,              # 9 - Target Length
                 catname,         # 10+ - category name
                 score,           # 11+ - category score
            ]

            # Optional fields
            for kp in ann.get('keypoints', []):
                if 'keypoint_category_id' in kp:
                    cname = coco_dset._resolve_to_kpcat(kp['keypoint_category_id'])['name']
                elif 'category_name' in kp:
                    cname = kp['category_name']
                elif 'category' in kp:
                    cname = kp['category']
                else:
                    raise Exception(str(kp))
                kp_x, kp_y = kp['xy']
                row.append('(kp) {} {} {}'.format(
                    cname, round(kp_x, 3), round(kp_y, 3)))

            note_fields = [
                'box_source',
                'changelog',
                'color',
            ]
            for note_key in note_fields:
                if note_key in ann:
                    row.append('(note) {}: {}'.format(note_key, repr(ann[note_key]).replace(',', '<comma>')))

            row = list(map(str, row))
            for item in row:
                if ',' in row:
                    print('BAD row = {!r}'.format(row))
                    raise Exception('comma is in a row field')

            row_str = ','.join(row)
            csv_rows.append(row_str)

    csv_text = '\n'.join(csv_rows)
    dst_fpath = config['dst']
    print('dst_fpath = {!r}'.format(dst_fpath))
    with open(dst_fpath, 'w') as file:
        file.write(csv_text)
