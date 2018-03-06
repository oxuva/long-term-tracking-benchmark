from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from oxuva import assess
from oxuva import util


class TestSubsetUsingPrevious(unittest.TestCase):

    def test_missing(self):
        source = util.TimeSeries({1: 'one', 3: 'three'})
        times = [1, 2, 3]
        got = assess.subset_using_previous_if_missing(source, times)
        want = {1: 'one', 2: 'one', 3: 'three'}
        self.assertEqual(dict(got.items()), want)

    def test_beyond_end(self):
        source = util.TimeSeries({1: 'one', 3: 'three'})
        times = [2, 4, 6]
        got = assess.subset_using_previous_if_missing(source, times)
        want = {2: 'one', 4: 'three', 6: 'three'}
        self.assertEqual(dict(got.items()), want)

    def test_before_start(self):
        source = util.TimeSeries({3: 'three'})
        times = [1, 2, 3]
        self.assertRaises(Exception, lambda:
                          assess.subset_using_previous_if_missing(source, times))


class TestIOU(unittest.TestCase):

    def test_same(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        iou = assess.iou(p, p)
        np.testing.assert_almost_equal(iou, 1.0)

    def test_empty_intersection(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        q = {'xmin': 0.4, 'xmax': 0.8, 'ymin': 0.6, 'ymax': 0.8}
        iou = assess.iou(p, q)
        np.testing.assert_almost_equal(iou, 0.0)

    def test_commutative(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        q = {'xmin': 0.2, 'xmax': 0.6, 'ymin': 0.7, 'ymax': 0.8}
        pq = assess.iou(p, q)
        qp = assess.iou(q, p)
        np.testing.assert_almost_equal(pq, qp)


if __name__ == '__main__':
    unittest.main()
