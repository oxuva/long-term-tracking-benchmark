from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

from oxuva import util


class Task(object):
    '''Describes a tracking task with optional ground-truth annotations.'''

    def __init__(self, init_time, init_rect, labels=None, last_time=None, attributes=None):
        '''Create a trasking task.

        Args:
            init_time -- Time of supervision (in frames).
            init_rect -- Rectangle dict.
            labels -- SparseTimeSeries of frame annotation dicts.
                Does not include first frame.
            last_time -- Time of last frame of interest, inclusive (optional).
                Consider frames init_time <= t <= last_time.
            attributes -- Dictionary with extra attributes.

        If last_time is None, then the last frame of labels will be used.
        '''
        self.init_time = init_time
        self.init_rect = init_rect
        if labels:
            if init_time in labels:
                raise ValueError('labels should not contain init time')
        self.labels = labels
        if last_time is None and labels is not None:
            self.last_time = labels.sorted_keys()[-1]
        else:
            self.last_time = last_time
        self.attributes = attributes or {}

    def len(self):
        return self.last_time - self.init_time + 1


def make_task_from_track(track):
    '''Creates a tracking task from a track annotation dict (oxuva.annot.make_track_label).

    The first frame is adopted as initialization.
    The remaining frames become the ground-truth rectangles.
    '''
    frames = list(track['frames'].sorted_items())
    init_time, init_annot = frames[0]
    labels = util.SparseTimeSeries(frames[1:])
    # TODO: Check that init_annot['exemplar'] is True.
    init_rect = {k: init_annot[k] for k in ['xmin', 'xmax', 'ymin', 'ymax']}
    attributes = {k: v for k, v in track.items() if k not in {'frames'}}
    return Task(init_time, init_rect, labels=labels, attributes=attributes)
