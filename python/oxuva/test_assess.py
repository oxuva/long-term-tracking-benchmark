from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import math
import numpy as np

from oxuva.assess import subset_using_previous_if_missing
from oxuva.assess import iou
from oxuva.assess import max_geometric_mean_line
from oxuva.assess import posthoc_threshold
from oxuva import util


class TestSubsetUsingPrevious(unittest.TestCase):

    def test_missing(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [1, 2, 3]
        got = subset_using_previous_if_missing(source, times)
        want = [(1, 'one'), (2, 'one'), (3, 'three')]
        self.assertEqual(list(got.sorted_items()), want)

    def test_beyond_end(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [2, 4, 6]
        got = subset_using_previous_if_missing(source, times)
        want = [(2, 'one'), (4, 'three'), (6, 'three')]
        self.assertEqual(list(got.sorted_items()), want)

    def test_before_start(self):
        source = util.SparseTimeSeries({3: 'three'})
        times = [1, 2, 3]
        self.assertRaises(
            Exception, lambda: subset_using_previous_if_missing(source, times))

    def test_idempotent(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [1, 2, 3, 4]
        once = subset_using_previous_if_missing(source, times)
        twice = subset_using_previous_if_missing(once, times)
        self.assertEqual(list(once.sorted_items()), list(twice.sorted_items()))


class TestIOU(unittest.TestCase):

    def test_simple(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.5, 'ymax': 0.9}
        q = {'xmin': 0.3, 'xmax': 0.5, 'ymin': 0.5, 'ymax': 1.0}
        # vol(p) is 0.3 * 0.4 = 0.12
        # vol(q) is 0.2 * 0.5 = 0.1
        # intersection is 0.1 * 0.4 = 0.04
        # union is (0.12 + 0.1) - 0.04 = 0.22 - 0.04 = 0.18
        want = 0.04 / 0.18
        np.testing.assert_almost_equal(iou(p, q), want)
        np.testing.assert_almost_equal(iou(q, p), want)

    def test_same(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        np.testing.assert_almost_equal(iou(p, p), 1.0)

    def test_empty_intersection(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        q = {'xmin': 0.4, 'xmax': 0.8, 'ymin': 0.6, 'ymax': 0.8}
        np.testing.assert_almost_equal(iou(p, q), 0.0)


class TestMaxGeometricMeanLineSeg(unittest.TestCase):

    def test_simple(self):
        x1, y1 = 0, 1
        x2, y2 = 1, 0
        got = max_geometric_mean_line(x1, y1, x2, y2)
        want = 0.5
        np.testing.assert_almost_equal(got, want)

    def test_bound(self):
        x1, y1 = 0.3, 0.9
        x2, y2 = 0.7, 0.4
        max_val = max_geometric_mean_line(x1, y1, x2, y2)
        self.assertLessEqual(util.geometric_mean(x1, y1), max_val)
        self.assertLessEqual(util.geometric_mean(x2, y2), max_val)
        xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        self.assertLessEqual(util.geometric_mean(xm, ym), max_val)


class TestPosthocThreshold(unittest.TestCase):

    def test_main(self):
        assessments = [
            {'TP': 1, 'FP': 0, 'TN': 0, 'FN': 0, 'score': 40},  # true positive
            {'TP': 1, 'FP': 0, 'TN': 0, 'FN': 0, 'score': 30},  # false positive
            {'TP': 0, 'FP': 1, 'TN': 0, 'FN': 0, 'score': 20},  # true positive
            {'TP': 1, 'FP': 0, 'TN': 0, 'FN': 0, 'score': 10},  # true positive
            # Treat score zero as threshold for this imaginary classifier.
            {'TP': 0, 'FP': 0, 'TN': 1, 'FN': 0, 'score': -10},  # true negative
            {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 1, 'score': -20},  # false negative
            {'TP': 0, 'FP': 0, 'TN': 1, 'FN': 0, 'score': -30},  # true negative
        ]
        got = posthoc_threshold(assessments)
        want = [
            {'TP': 0, 'FP': 0, 'TN': 3, 'FN': 4},
            {'TP': 1, 'FP': 0, 'TN': 3, 'FN': 3},
            {'TP': 2, 'FP': 0, 'TN': 3, 'FN': 2},
            {'TP': 2, 'FP': 1, 'TN': 2, 'FN': 2},
            {'TP': 3, 'FP': 1, 'TN': 2, 'FN': 1},
        ]
        self.assertEqual(len(got), len(want))
        for point_got, point_want in zip(got, want):
            for k in ['TP', 'FP', 'TN', 'FN']:
                np.testing.assert_almost_equal(point_got[k], point_want[k])


if __name__ == '__main__':
    unittest.main()
