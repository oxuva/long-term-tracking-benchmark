'''
Examples:

    To evaluate the prediction of a tracker for one tracking task:

        assessment = assess.assess_sequence(task.labels, prediction, iou_threshold=0.5)

    The function assess_sequence() calls subset_using_previous_if_missing() internally.
    This function may alternatively be called before assess_sequence().
    The result will be the same because subset_using_previous_if_missing() is idempotent.

        prediction_subset = assess.subset_using_previous_if_missing(
            prediction, task.labels.sorted_keys())
        assessment = assess.assess_sequence(
            task.labels, prediction_subset, iou_threshold=0.5)

    Since assessment is a SparseTimeSeries of frame assessments,
    we can consider a subset of frames:

        assessment_subset = util.select_interval(
            assessment, min_time, max_time, init_time=task.init_time)

    To accumulate per-frame assessments into a summary for the sequence:

        sequence_assessment = assess.assessment_sum(frame_assessments)

    This can also be used to accumulate sequence summaries for a dataset:

        dataset_assessment = assess.assessment_sum(sequence_assessments)

    To obtain the performance metrics from the summary:

        stats = assess.quality_metrics(dataset_assessment)

    Full example:

        assessments = {}
        for key in tasks:
            assessments[key] = assess.assess_sequence(
                tasks[key].labels, predictions[key], iou_threshold=0.5)

        sequence_assessments = {
            vid_obj: assess.assessment_sum(assessments[vid_obj].values())
            for vid_obj in assessments}
        dataset_assessment = assess.assessment_sum(sequence_assessments.values())
        return assess.quality_metrics(dataset_assessment)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from oxuva import util


def quality_metrics(assessment):
    '''Computes the TPR, TNR from TP, FP, etc.

    Args:
        assessment -- Dictionary with TP, FP, TN, FN.
    '''
    metrics = {}
    num_pos = assessment['TP'] + assessment['FN']
    num_neg = assessment['TN'] + assessment['FP']
    if num_pos > 0:
        metrics['TPR'] = float(assessment['TP']) / num_pos
    # else:
    #     raise ValueError('unable to compute TPR (no positives)')
    if num_neg > 0:
        metrics['TNR'] = float(assessment['TN']) / num_neg
    # else:
    #     raise ValueError('unable to compute TNR (no negatives)')
    # TODO: Add some errorbars?
    if num_pos > 0 and num_neg > 0:
        metrics['GM'] = util.geometric_mean(metrics['TPR'], metrics['TNR'])
        metrics['MaxGM'] = max_geometric_mean_line(metrics['TNR'], metrics['TPR'], 1, 0)
    # Include the raw totals.
    metrics.update(assessment)
    return metrics


def subset_using_previous_if_missing(series, times):
    '''Extracts a subset of values at the given times.
    If there is no data for a particular time, then the last value is used.

    Args:
        series: SparseTimeSeries of data.
        times: List of times.

    Returns:
        Time series sampled at specified times.

    Examples:
        >> subset_using_previous_if_missing([(2, 'hi'), (4, 'bye')], [2, 3, 4, 5])
        ['hi', 'hi', 'bye', 'bye']

    Raises an exception if asked for a time before the first element in series.
    '''
    assert isinstance(series, util.SparseTimeSeries)
    series = list(series.sorted_items())
    subset = [None for _ in times]
    t_curr, x_curr = None, None
    for i, t in enumerate(times):
        # Read from series until we have read all elements <= t.
        read_all = False
        while not read_all:
            if len(series) == 0:
                read_all = True
            else:
                t_next, x_next = series[0]
                if t_next > t:
                    # We have gone past t.
                    read_all = True
                else:
                    # Keep going.
                    t_curr, x_curr = t_next, x_next
                    series = series[1:]
        if t_curr is None:
            raise ValueError('no value for time: {}'.format(t))
        subset[i] = x_curr
    return util.SparseTimeSeries(zip(times, subset))


def assess_sequence(gt, pred, iou_threshold):
    '''Evaluate predicted track against ground-truth annotations.

    Args:
        gt: SparseTimeSeries of annotation dicts.
        pred: SparseTimeSeries of prediction dicts.
        iou_threshold: Threshold for determining true positive.

    Returns:
        An assessment of each frame with ground-truth.
        This is a TimeSeries of frame assessment dicts.
    '''
    times = gt.sorted_keys()
    # if pred.sorted_keys() != times:
    pred = subset_using_previous_if_missing(pred, times)
    return util.SparseTimeSeries({t: assess_frame(gt[t], pred[t], iou_threshold) for t in times})


def make_assessment(num_frames=0, tp=0, fp=0, tn=0, fn=0):
    return {'num_frames': num_frames, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def assessment_sum(assessments):
    return util.dict_sum_strict(assessments, make_assessment())


def assess_frame(gt, pred, iou_threshold):
    '''Turn prediction into TP, FP, TN, FN.

    Args:
        gt, pred: Dictionaries with fields: present, xmin, xmax, ymin, ymax.

    Returns:
        Frame assessment dict with TP, FP, etc.
    '''
    # TODO: Some other metrics may require knowledge of aspect ratio.
    # (This is not an issue for now since IOU is invariant).

    # TP: gt "present" and pred "present" and box correct
    # FN: gt "present" and not (pred "present" and box correct)
    # TN: gt "absent" and pred "absent"
    # FP: gt "absent" and pred "present"
    result = make_assessment(num_frames=1)
    if gt['present']:
        if pred['present'] and iou(gt, pred) >= iou_threshold:
            result['TP'] += 1
        else:
            result['FN'] += 1
    else:
        if pred['present']:
            result['FP'] += 1
        else:
            result['TN'] += 1
    if 'score' in pred:
        result['score'] = pred['score']
    return result


def iou(a, b):
    i = vol(intersect(a, b))
    u = vol(a) + vol(b) - i
    return float(i) / float(u)


def vol(r):
    # Any inverted rectangle is silently considered empty.
    # (Allows for empty intersection.)
    xsize = max(0, r['xmax'] - r['xmin'])
    ysize = max(0, r['ymax'] - r['ymin'])
    return xsize * ysize


def intersect(a, b):
    return {
        'xmin': max(a['xmin'], b['xmin']),
        'ymin': max(a['ymin'], b['ymin']),
        'xmax': min(a['xmax'], b['xmax']),
        'ymax': min(a['ymax'], b['ymax']),
    }


def posthoc_threshold(assessments):
    '''Trace curve of operating points by varying score threshold.

    Args:
        assessments: List of sequence assessments.
            A sequence assessment is a TimeSeries of frame assessments.
    '''
    # Group all "present" predictions (TP, FP) by score.
    by_score = {}
    for ass in assessments:
        if ass['TP'] or ass['FP']:
            by_score.setdefault(float(ass['score']), []).append(ass)

    # Trace threshold from min to max.
    # Initially everything is labelled absent (negative).
    num_present = sum(ass['TP'] + ass['FN'] for ass in assessments)
    num_absent = sum(ass['TN'] + ass['FP'] for ass in assessments)
    total = {'TP': 0, 'FP': 0, 'TN': num_absent, 'FN': num_present}

    # Start switching the highest scores from "absent" to "present".
    points = []
    points.append(dict(total))
    if float('nan') in by_score:
        raise ValueError('score is nan but prediction is "present"')
    for score in sorted(by_score.keys(), reverse=True):
        # Iterate through the next block of points with the same score.
        for assessment in by_score[score]:
            total['TP'] += assessment['TP']
            total['FN'] -= assessment['TP']
            total['FP'] += assessment['FP']
            total['TN'] -= assessment['FP']
        points.append(dict(total))
    return points


def max_geometric_mean_line(x1, y1, x2, y2):
    # Obtained using Matlab symbolic toolbox.
    # >> syms x1 x2 y1 y2 th
    # >> x = (1-th)*x1 + th*x2
    # >> y = (1-th)*y1 + th*y2
    # >> f = x * y
    # >> coeffs(f, th)
    # [ x1*y1, - y1*(x1 - x2) - x1*(y1 - y2), (x1 - x2)*(y1 - y2)]
    a = (x1 - x2) * (y1 - y2)
    b = - y1 * (x1 - x2) - x1 * (y1 - y2)
    # Maximize the quadratic on [0, 1].
    # Check endpoints.
    candidates = [0.0, 1.0]
    if a < 0:
        # Concave quadratic. Check if peak is in bounds.
        th_star = -b / (2 * a)
        if 0 <= th_star <= 1:
            candidates.append(th_star)
    g = lambda x, y: math.sqrt(x * y)
    h = lambda th: g((1-th)*x1 + th*x2, (1-th)*y1 + th*y2)
    return max([h(th) for th in candidates])
