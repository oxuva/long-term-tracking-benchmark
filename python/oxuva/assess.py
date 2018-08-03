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

import functools
import itertools
import json
import math
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from oxuva import dataset
from oxuva import io_pred
from oxuva import util


FRAME_RATE = 30


def quality_metrics(assessment):
    '''Computes the TPR, TNR from TP, FP, etc.

    Args:
        assessment -- Dictionary with TP, FP, TN, FN.
    '''
    metrics = {}
    num_pos = assessment['TP'] + assessment['FN']
    num_neg = assessment['TN'] + assessment['FP']
    with np.errstate(invalid='ignore'):
        # Allow nan values in cases of 0 / 0.
        metrics['TPR'] = np.asfarray(assessment['TP']) / num_pos
        metrics['TNR'] = np.asfarray(assessment['TN']) / num_neg
    # TODO: Add some errorbars?
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
                # Peek at next element.
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
        if t_curr != t:
            logger.warning('no prediction for time %d: use prediction for time %s', t, t_curr)
        subset[i] = x_curr
    return util.SparseTimeSeries(zip(times, subset))


def load_predictions_and_select_frames(tasks, tracker_pred_dir, permissive=False, log_prefix=''):
    '''Loads all predictions of a tracker and takes the subset of frames with ground truth.

    Args:
        tasks: VideoObjectDict of Tasks.
        tracker_pred_dir: Directory that contains files video_object.csv

    Returns:
        VideoObjectDict of SparseTimeSeries of frame assessments.
    '''
    logger.info('load predictions from "%s"', tracker_pred_dir)
    preds = dataset.VideoObjectDict()
    for track_num, vid_obj in enumerate(tasks.keys()):
        vid, obj = vid_obj
        track_name = vid + '_' + obj
        logger.debug(log_prefix + 'object {}/{} {}'.format(track_num + 1, len(tasks), track_name))
        pred_file = os.path.join(tracker_pred_dir, '{}.csv'.format(track_name))
        try:
            with open(pred_file, 'r') as fp:
                pred = io_pred.load_predictions_csv(fp)
        except IOError as exc:
            if permissive:
                logger.warning('exclude track %s: %s', track_name, str(exc))
            else:
                raise
        pred = subset_using_previous_if_missing(pred, tasks[vid_obj].labels.sorted_keys())
        preds[vid_obj] = pred
    return preds


def assess_sequence(gt, pred, iou_threshold):
    '''Evaluate predicted track against ground-truth annotations.

    Args:
        gt: SparseTimeSeries of annotation dicts.
        pred: SparseTimeSeries of prediction dicts.
        iou_threshold: Threshold for determining true positive.

    Returns:
        An assessment of each frame with ground-truth.
        This is a TimeSeries of per-frame assessment dicts.
    '''
    times = gt.sorted_keys()
    # if pred.sorted_keys() != times:
    pred = subset_using_previous_if_missing(pred, times)
    return util.SparseTimeSeries({t: assess_frame(gt[t], pred[t], iou_threshold) for t in times})


def make_assessment(num_frames=0, tp=0, fp=0, tn=0, fn=0, num_present=0, num_absent=0):
    return {'num_frames': num_frames, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'num_present': num_present, 'num_absent': num_absent}


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
    result = make_assessment(num_frames=1,
                             num_present=(1 if gt['present'] else 0),
                             num_absent=(0 if gt['present'] else 1))
    if gt['present']:
        if pred['present'] and iou_clip(gt, pred) >= iou_threshold:
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


def iou_clip(a, b):
    bounds = unit_rect()
    a = intersect(a, bounds)
    b = intersect(b, bounds)
    return iou(a, b)


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


def unit_rect():
    return {'xmin': 0.0, 'ymin': 0.0, 'xmax': 1.0, 'ymax': 1.0}


def posthoc_threshold(assessments):
    '''Trace curve of operating points by varying score threshold.

    Args:
        assessments: List of TimeSeries of per-frame assessments.
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
    h = lambda th: g((1 - th) * x1 + th * x2, (1 - th) * y1 + th * y2)
    return max([h(th) for th in candidates])


def make_dataset_assessment(totals, quantized_totals, frame_assessments=None):
    '''Sufficient to produce all plots.

    This is what will be returned to the user by the evaluation server.
    '''
    return {
        'totals': totals,
        'quantized_totals': quantized_totals,
        'frame_assessments': frame_assessments,  # Ignored by dump_xxx functions!
    }


def union_dataset_assessment(x, y):
    '''Combines the tracks of two datasets.'''
    if y is None:
        return x
    if x is None:
        return y
    return {
        'totals': dataset.VideoObjectDict(dict(itertools.chain(
            x['totals'].items(),
            y['totals'].items()))),
        'quantized_totals': dataset.VideoObjectDict(dict(itertools.chain(
            x['quantized_totals'].items(),
            y['quantized_totals'].items()))),
    }


def dump_dataset_assessment_json(x, f):
    data = {
        # Convert to list of items because JSON does not support tuple as keys.
        'totals': sorted(x['totals'].items()),
        # Convert to list of items because JSON does not support tuple as keys.
        # Extract elements of each QuantizedAssessment.
        'quantized_totals': [(vid_obj, value.elems)
                             for vid_obj, value in sorted(x['quantized_totals'].items())],
    }
    json.dump(data, f, sort_keys=True)


def load_dataset_assessment_json(f):
    data = json.load(f)
    return make_dataset_assessment(
        totals=dataset.VideoObjectDict({
            tuple(vid_obj): total for vid_obj, total in data['totals']}),
        quantized_totals=dataset.VideoObjectDict({
            tuple(vid_obj): QuantizedAssessment({
                tuple(interval): total for interval, total in quantized_totals})
            for vid_obj, quantized_totals in data['quantized_totals']}))


def assess_dataset(tasks, predictions, iou_threshold, resolution_seconds=30):
    '''
    Args:
        tasks: VideoObjectDict of tasks. Each task must include annotations.
        predictions: VideoObjectDict of predictions.

    Returns:
        Enough information to produce the plots.
    '''
    frame_assessments = dataset.VideoObjectDict({
        key: assess_sequence(tasks[key].labels, predictions[key], iou_threshold)
        for key in tasks.keys()})
    return make_dataset_assessment(
        totals=dataset.VideoObjectDict({
            key: assessment_sum(frame_assessments[key].values())
            for key in frame_assessments.keys()}),
        quantized_totals=dataset.VideoObjectDict({
            key: quantize_sequence_assessment(frame_assessments[key],
                                              init_time=tasks[key].init_time,
                                              resolution=(FRAME_RATE * resolution_seconds))
            for key in frame_assessments.keys()}),
        frame_assessments=frame_assessments)


class QuantizedAssessment(object):
    '''Describes the assessment of intervals of a sequence.

    This is sufficient to construct the temporal plots
    without revealing whether each individual prediction is correct or not.
    '''

    def __init__(self, elems):
        '''
        Args:
            elems: Map from (a, b) to assessment dict.
        '''
        if isinstance(elems, dict):
            elems = list(elems.items())
        elems = sorted(elems)
        self.elems = elems

    def get(self, min_time=None, max_time=None):
        '''Get cumulative assessment of interval [min_time, max_time].'''
        # Include all bins within [min_time, max_time].
        subset = []
        for interval, value in self.elems:
            u, v = interval
            # if min_time <= u <= v <= max_time:
            if (min_time is None or min_time <= u) and (max_time is None or v <= max_time):
                subset.append(value)
            elif (min_time < u < max_time) or (min_time < v < max_time):
                # If interval is not within [min_time, max_time],
                # then require that it is entirely outside [min_time, max_time].
                raise ValueError('interval {} straddles requested {}'.format(
                    str((u, v)), str((min_time, max_time))))
        return assessment_sum(subset)

    # def get_vector(self, intervals):
    #     return _to_vector_dict([self.get(min_time, max_time)
    #                             for min_time, max_time in intervals])


def quantize_sequence_assessment(assessment, resolution, init_time):
    '''
    Args:
        assessment: SparseTimeSeries of assessment dicts.
        resolution: Integer specifying temporal resolution.
        init_time: Absolute time at which tracker was started.

    Returns:
        Ordered list of ((a, b), value) elements where a, b are integers.
    '''
    if int(resolution) != resolution:
        logger.warning('resolution is not integer: %g', resolution)
    resolution = int(resolution)

    subsets = {}
    for abs_time, frame in assessment.items():
        t = abs_time - init_time
        i = int(math.ceil(t / float(resolution)))
        interval = resolution * (i - 1), resolution * i
        subsets.setdefault(interval, []).append(frame)
    sums = {interval: assessment_sum(subsets[interval]) for interval in subsets.keys()}
    return QuantizedAssessment(sorted(sums.items()))


def _to_vector_dict(list_of_dicts):
    keys = _get_keys_and_assert_equal(list_of_dicts)
    vector_dict = {}
    for key in keys:
        vector_dict[key] = np.array([x[key] for x in list_of_dicts])
    return vector_dict


def dataset_quality(totals, enable_bootstrap=True, num_trials=None, base_seed=0):
    '''
    Args:
        totals: VideoObjectDict of per-sequence assessment dicts.
    '''
    quality = summarize(totals.values())
    if enable_bootstrap:
        if num_trials is None:
            raise ValueError('must specify number of trials for bootstrap sampling')
        quality.update(bootstrap(summarize, totals, num_trials, base_seed=base_seed))
    quality = {k: np.asarray(v).tolist() for k, v in quality.items()}
    return quality


def dataset_quality_interval(quantized_assessments, min_time=None, max_time=None,
                             enable_bootstrap=True, num_trials=None, base_seed=0):
    '''
    Args:
        totals: VideoObjectDict of per-sequence assessment dicts.
    '''
    interval_totals = dataset.VideoObjectDict({
        track: quantized_assessments[track].get(min_time, max_time)
        for track in quantized_assessments.keys()})
    quality = summarize(interval_totals.values())
    if enable_bootstrap:
        if num_trials is None:
            raise ValueError('must specify number of trials for bootstrap sampling')
        quality.update(bootstrap(summarize, interval_totals, num_trials, base_seed=base_seed))
    quality = {k: np.asarray(v).tolist() for k, v in quality.items()}
    return quality


def dataset_quality_filter(totals, require_none_absent=False, require_some_absent=False,
                           enable_bootstrap=True, num_trials=None, base_seed=0):
    # Apply filter after bootstrap sampling dataset.
    summarize_func = functools.partial(summarize_filter,
                                       require_none_absent=require_none_absent,
                                       require_some_absent=require_some_absent)
    quality = summarize_func(totals.values())
    if enable_bootstrap:
        if num_trials is None:
            raise ValueError('must specify number of trials for bootstrap sampling')
        quality.update(bootstrap(summarize_func, totals, num_trials, base_seed=base_seed))
    quality = {k: np.asarray(v).tolist() for k, v in quality.items()}
    return quality


def summarize(totals):
    '''Obtain dataset quality from per-sequence assessments.

    Args:
        totals: List of assessment dicts.
    '''
    return quality_metrics(assessment_sum(totals))


def summarize_filter(totals, require_none_absent=False, require_some_absent=False):
    totals = [x for x in totals if
              (not require_none_absent or x['num_absent'] == 0) and
              (not require_some_absent or x['num_absent'] > 0)]
    return summarize(totals)


def bootstrap(func, data, num_trials, base_seed=0):
    '''
    Args:
        func: Maps list of per-track elements to a dictionary of metrics.
            This will be called num_trials times.
        data: VideoObjectDict of elements.

    The function will be called func(x) where x is a list of the values in data.
    It would normally be called func(data.values()).

    VideoObjectDict is required because sampling is performed on videos not tracks.
    '''
    metrics = []
    for i in range(num_trials):
        sample = _bootstrap_sample_by_video(data, seed=(base_seed + i))
        logger.debug('bootstrap trial %d: num sequences %d', i + 1, len(sample))
        metrics.append(func(sample))
    return _stats_from_repetitions(metrics)


def _bootstrap_sample_by_video(tracks, seed):
    '''Samples videos with replacement and returns a list of all tracks.

    Args:
        tracks: VideoObjectDict
    '''
    assert isinstance(tracks, dataset.VideoObjectDict)
    by_video = tracks.to_nested_dict()
    rand = np.random.RandomState(seed)
    names = list(by_video.keys())
    names_sample = rand.choice(names, len(by_video), replace=True)
    return list(itertools.chain.from_iterable(by_video[name].values() for name in names_sample))


def _stats_from_repetitions(xs):
    '''Maps a list of dictionaries to the mean and variance of the values.

    Appends '_mean' and '_var' to the original keys.
    '''
    # Check that all dictionaries have the same keys.
    fields = _get_keys_and_assert_equal(xs)
    stats = {}
    stats.update({field + '_mean': np.mean([x[field] for x in xs], axis=0) for field in fields})
    stats.update({field + '_var': np.var([x[field] for x in xs], axis=0) for field in fields})
    return stats


def _get_keys_and_assert_equal(xs):
    '''Asserts that all dictionaries have the same keys and returns the set of keys.'''
    assert len(xs) > 0
    fields = None
    for x in xs:
        curr_fields = set(x.keys())
        if fields is None:
            fields = curr_fields
        else:
            if curr_fields != fields:
                raise ValueError('fields differ: {} and {}'.format(fields, curr_fields))
    return fields
