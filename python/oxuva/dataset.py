from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

from oxuva import util


class VideoObjectDict(object):
    '''Represents map video -> object -> element.
    Behaves as a dictionary with keys of (video, object) tuples.

    Example:
        for key in tracks.keys():
            print(tracks[key])

        tracks = VideoObjectDict()
        ...
        for vid in tracks.videos():
            for obj in tracks.objects(vid):
                print(tracks[(vid, obj)])
    '''

    def __init__(self, elems=None):
        if elems is None:
            self._elems = dict()
        elif isinstance(elems, VideoObjectDict):
            self._elems = dict(elems._elems)
        else:
            self._elems = dict(elems)

    def videos(self):
        return set([vid for vid, obj in self._elems.keys()])

    def objects(self, vid):
        # TODO: This is somewhat inefficient if called for all videos.
        return [obj_i for vid_i, obj_i in self._elems.keys() if vid_i == vid]

    def __len__(self):
        return len(self._elems)

    def __getitem__(self, key):
        return self._elems[key]

    def __setitem__(self, key, value):
        self._elems[key] = value

    def __delitem__(self, key):
        del self._elems[key]

    def keys(self):
        return self._elems.keys()

    def values(self):
        return self._elems.values()

    def items(self):
        return self._elems.items()

    def __iter__(self):
        for k in self._elems.keys():
            yield k

    def to_nested_dict(self):
        elems = {}
        for (vid, obj), elem in self._elems.items():
            elems.setdefault(vid, {})[obj] = elem
        return elems

    def update_from_nested_dict(self, elems):
        for vid, vid_elems in elems.items():
            for obj, elem in vid_elems.items():
                self._elems[(vid, obj)] = elem
