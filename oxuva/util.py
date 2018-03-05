from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import numpy as np
import os
import pickle


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


def cache(protocol, filename, func, makedir=True, ignore_existing=False):
    '''Caches the result of a function in a file.

    Args:
        func -- Function with no arguments.
        makedir -- Create parent directory if it does not exist.
        ignore_existing -- Ignore existing cache file and call function.
            If it existed, the old cache file will be over-written.
    '''
    if (not ignore_existing) and os.path.exists(filename):
        with open(filename, 'r') as r:
            result = protocol.load(r)
    else:
        dir = os.path.dirname(filename)
        if makedir and (not os.path.exists(dir)):
            os.makedirs(dir)
        result = func()
        # Write to a temporary file and then perform atomic rename.
        # This guards against partial cache files.
        tmp = filename + '.tmp'
        with open(tmp, 'w') as w:
            protocol.dump(result, w)
        os.rename(tmp, filename)
    return result

cache_json = functools.partial(cache, json)
cache_pickle = functools.partial(cache, pickle)
