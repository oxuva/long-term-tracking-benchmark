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


def cache(protocol, filename, func, makedir=True):
    '''Caches the result of a function in a file.

    Args:
        func -- Function with no arguments.
    '''
    if os.path.exists(filename):
        with open(filename, 'r') as r:
            result = protocol.load(r)
    else:
        if makedir:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        result = func()
        try:
            with open(filename, 'w') as w:
                protocol.dump(result, w)
        except:
            os.remove(filename)
            raise
    return result

cache_json = functools.partial(cache, json)
cache_pickle = functools.partial(cache, pickle)
