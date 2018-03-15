from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from oxuva import util


def make_track_label(category, frames, contains_cuts=None, always_visible=None):
    return {
        'frames': frames,
        'category': category,
        'contains_cuts': contains_cuts,
        'always_visible': always_visible,
    }


def make_frame_label(present=None, exemplar=None, xmin=None, ymin=None, xmax=None, ymax=None):
    return {
        'present': util.default_if_none(present, False),
        'exemplar': util.default_if_none(exemplar, False),
        'xmin': util.default_if_none(xmin, 0.0),
        'xmax': util.default_if_none(xmax, 0.0),
        'ymin': util.default_if_none(ymin, 0.0),
        'ymax': util.default_if_none(ymax, 0.0),
    }


def remove_empty_tracks(tracks):
    output = {}
    for vid_name, objs in tracks.iteritems():
        objs = {
            obj_name: obj for obj_name, obj in objs.items()
            if any(frame['present'] for t, frame in obj['frames'])
        }
        if len(objs) == 0:
            continue
        output[vid_name] = objs
    return output


def trim_absent_frames(tracks):
    '''
    In-place.
    '''
    for vid_name, objs in tracks.iteritems():
        for obj_name, obj in objs.items():
            exemplar = next(
                t for t, frame in obj['frames']
                if frame['present'] and frame['exemplar']
            )
            obj['frames'] = [
                (t, frame) for t, frame in obj['frames']
                if t >= exemplar
            ]
    return tracks
