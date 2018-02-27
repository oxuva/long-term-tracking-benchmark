from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from oxuva import util
from oxuva import data


TRACK_FIELDS = [
    'video_id', 'object_id',
    'class_id', 'class_name', 'contains_cuts', 'always_visible',
    'frame_num', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax',
]


def load_annotations_csv(fp):
    reader = csv.DictReader(fp, fieldnames=TRACK_FIELDS)
    rows = [row for row in reader]

    # Group rows by object.
    rows_by_track = {}
    for row in rows:
        vid_id = row['video_id']
        obj_id = row['object_id']
        rows_by_track.setdefault((vid_id, obj_id), []).append(row)

    tracks = {}
    for vid_obj in rows_by_track.keys():
        vid_id, obj_id = vid_obj
        frames = []
        for row in rows_by_track[vid_obj]:
            present = _parse_is_present(row['object_presence'])
            # TODO: Support 'exemplar' field in CSV format?
            t = int(row['frame_num'])
            annot = data.make_frame_label(
                present=present,
                xmin=float(row['xmin']) if present else None,
                xmax=float(row['xmax']) if present else None,
                ymin=float(row['ymin']) if present else None,
                ymax=float(row['ymax']) if present else None,
            )
            frames.append((t, annot))
        assert len(frames) >= 2
        first_row = rows_by_track[vid_obj][0]
        tracks.setdefault(vid_id, {})[obj_id] = data.make_track_label(
            category=first_row['class_name'],
            frames=frames,
            contains_cuts=first_row['contains_cuts'],
            always_visible=first_row['always_visible'],
        )

    return tracks


def dump_annotations_csv(tracks, fp):
    writer = csv.DictWriter(fp, fieldnames=TRACK_FIELDS)
    for vid in sorted(tracks.keys()):
        # Sort objects by their first frame.
        for obj, track in sorted(tracks[vid].items(), key=_obj_sort_key):
            for frame_num, frame in track['frames']:
                assert frame_num == int(frame_num)
                frame_num = int(frame_num)
                assert frame_num % 30 == 0
                # timestamp = frame_num // 30
                class_name = track.get('category', '')
                class_id = CLASS_ID_LOOKUP[class_name] if class_name else ''
                row = {
                    'video_id':   vid,
                    'object_id':  obj,
                    'class_id':   class_id,
                    'class_name': class_name,
                    'contains_cuts': _str_contains_cuts(track.get('contains_cuts', None)),
                    'always_visible': _str_always_visible(track.get('always_visible', None)),
                    'frame_num':  frame_num,
                    'object_presence': 'present' if frame['present'] else 'absent',
                    'xmin': frame['xmin'],
                    'xmax': frame['xmax'],
                    'ymin': frame['ymin'],
                    'ymax': frame['ymax'],
                }
                writer.writerow(row)


def _obj_sort_key(obj_track):
    obj, track = obj_track
    return (_first_frame(track), obj)

def _first_frame(track):
    assert len(track['frames']) > 0
    t, _ = track['frames'][0]
    return t


def _parse_is_present(s):
    if s == 'present':
        return True
    elif s == 'absent':
        return False
    else:
        raise ValueError('unknown value for presence: {}'.format(s))

def _str_is_present(present):
    if present:
        return 'present'
    else:
        return 'absent'

def _str_contains_cuts(contains_cuts):
    if contains_cuts == True:
        return 'true'
    elif contains_cuts == False:
        return 'false'
    else:
        return 'unknown'

def _parse_contains_cuts(s):
    return util.str2bool_or_none(s)

def _str_always_visible(always_visible):
    if always_visible == True:
        return 'true'
    elif always_visible == False:
        return 'false'
    else:
        return 'unknown'

def _parse_always_visible(s):
    return util.str2bool_or_none(s)
