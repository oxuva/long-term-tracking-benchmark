from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from oxuva import util


def make_track_label(category, frames, contains_cuts=None, always_visible=None):
    '''Creates a track annotation dictionary.'''
    return {
        'frames': frames,
        'category': category,
        'contains_cuts': contains_cuts,
        'always_visible': always_visible,
    }


def make_frame_label(present=None, exemplar=None, xmin=None, ymin=None, xmax=None, ymax=None):
    '''Creates a frame annotation dictionary (for ground-truth).'''
    return {
        'present': util.default_if_none(present, False),
        'exemplar': util.default_if_none(exemplar, False),
        'xmin': util.default_if_none(xmin, 0.0),
        'xmax': util.default_if_none(xmax, 0.0),
        'ymin': util.default_if_none(ymin, 0.0),
        'ymax': util.default_if_none(ymax, 0.0),
    }
