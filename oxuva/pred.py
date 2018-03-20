from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from oxuva import util


def make_prediction(present=None, score=None, xmin=None, ymin=None, xmax=None, ymax=None):
    '''Describes the output of a tracker in one frame.'''
    return {
        'present': util.default_if_none(present, True),
        'score': util.default_if_none(score, 0.0),
        'xmin': util.default_if_none(xmin, 0.0),
        'xmax': util.default_if_none(xmax, 0.0),
        'ymin': util.default_if_none(ymin, 0.0),
        'ymax': util.default_if_none(ymax, 0.0),
    }
