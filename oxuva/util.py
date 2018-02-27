from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def str2bool(x):
    x = x.strip().lower()
    if x in ['t', 'true', 'y', 'yes', '1']:
        return True
    if x in ['f', 'false', 'n', 'no', '0']:
        return False
    raise ValueError('warning: unclear value: {}'.format(x))

def str2bool_or_none(x):
    try:
        return str2bool(x)
    except ValueError:
        return None

def bool2str(x):
    return str(x).lower()

def default_if_none(x, value):
    return value if x is None else x

def harmonic_mean(*args):
    assert all([x >= 0 for x in args])
    if any([x == 0 for x in args]):
        return 0.
    return np.asscalar(1. / np.mean(1. / np.asfarray(args)))

def geometric_mean(*args):
    assert all([x >= 0 for x in args])
    if any([x == 0 for x in args]):
        return 0.
    return np.asscalar(np.exp(np.mean(np.log(args))))
