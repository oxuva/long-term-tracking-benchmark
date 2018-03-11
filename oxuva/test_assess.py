from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from oxuva import assess
from oxuva import util


class TestSubsetUsingPrevious(unittest.TestCase):

    def test_missing(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [1, 2, 3]
        got = assess.subset_using_previous_if_missing(source, times)
        want = [(1, 'one'), (2, 'one'), (3, 'three')]
        self.assertEqual(list(got.sorted_items()), want)

    def test_beyond_end(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [2, 4, 6]
        got = assess.subset_using_previous_if_missing(source, times)
        want = [(2, 'one'), (4, 'three'), (6, 'three')]
        self.assertEqual(list(got.sorted_items()), want)

    def test_before_start(self):
        source = util.SparseTimeSeries({3: 'three'})
        times = [1, 2, 3]
        self.assertRaises(
            Exception, lambda: assess.subset_using_previous_if_missing(source, times))

    def test_idempotent(self):
        source = util.SparseTimeSeries({1: 'one', 3: 'three'})
        times = [1, 2, 3, 4]
        once = assess.subset_using_previous_if_missing(source, times)
        twice = assess.subset_using_previous_if_missing(once, times)
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
        np.testing.assert_almost_equal(assess.iou(p, q), want)
        np.testing.assert_almost_equal(assess.iou(q, p), want)

    def test_same(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        iou = assess.iou(p, p)
        np.testing.assert_almost_equal(iou, 1.0)

    def test_empty_intersection(self):
        p = {'xmin': 0.1, 'xmax': 0.4, 'ymin': 0.6, 'ymax': 0.8}
        q = {'xmin': 0.4, 'xmax': 0.8, 'ymin': 0.6, 'ymax': 0.8}
        iou = assess.iou(p, q)
        np.testing.assert_almost_equal(iou, 0.0)


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
        got = assess.posthoc_threshold(assessments)
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
