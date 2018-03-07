from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from oxuva import util


# def measure_quality(gt, pred, iou_threshold, log_prefix='', **kwargs):
#     '''Measures the quality of a single track.
#
#     The quality is the total number of TP, TN, FP, FN.
#
#     Args:
#         kwargs: For assess_sequence().
#     '''
#     assessment = assess_sequence(gt, pred, iou_threshold, log_prefix=log_prefix, **kwargs)
#     return summarize_sequence(assessment)


def summarize_sequence(assessment):
    '''Converts a per-frame assessment into a cumulative sequence assessment.

    Args:
        assessment -- Dictionary that maps t to a frame assessment dict.
    '''
    quality = {
        'len': len(assessment),
        'TP': sum(x['TP'] for t, x in assessment.items()),
        'TN': sum(x['TN'] for t, x in assessment.items()),
        'FP': sum(x['FP'] for t, x in assessment.items()),
        'FN': sum(x['FN'] for t, x in assessment.items()),
    }
    # Compute per-sequence statistics.
    # However, many sequences do not have any negative (absent frames).
    # Hence this will be noisy at best and divide by zero at worst.
    # quality['TPR'] = float(quality['TP']) / (quality['TP'] + quality['FN'])
    # quality['FNR'] = float(quality['FN']) / (quality['TP'] + quality['FN'])
    # quality['TNR'] = float(quality['TN']) / (quality['TN'] + quality['FP'])
    # quality['FPR'] = float(quality['FP']) / (quality['TN'] + quality['FP'])
    # quality['recall']    = float(quality['TP']) / (quality['TP'] + quality['FN'])
    # quality['precision'] = float(quality['TP']) / (quality['TP'] + quality['FP'])
    return quality


def statistics(tracks_quality):
    '''Computes the TPR and TNR from the quality of many sequences.

    Args:
        tracks_quality -- List of quality for each track.
    '''
    tracks_quality = list(tracks_quality)  # In case tracks_quality is a generator.
    total = {
        k: sum([q[k] for q in tracks_quality])
        for k in ['TP', 'FN', 'FP', 'TN']
    }
    num_pos = total['TP'] + total['FN']
    num_neg = total['TN'] + total['FP']
    if num_pos > 0:
        total['TPR'] = float(total['TP']) / num_pos
    # else:
    #     raise ValueError('unable to compute TPR (no positives)')
    if num_neg > 0:
        total['TNR'] = float(total['TN']) / num_neg
    # else:
    #     raise ValueError('unable to compute TNR (no negatives)')
    # TODO: Add some errorbars?
    total['GM'] = util.geometric_mean(total.get('TPR', 0.0), total.get('TNR', 0.0))
    return total


def subset_using_previous_if_missing(series, times):
    '''Extracts a subset of values at the given times.
    If there is no data for a particular time, then the last value is used.

    Args:
        series: TimeSeries of data.
        times: List of times.

    Returns:
        Time series sampled at specified times.

    Examples:
        >> subset_using_previous_if_missing([(2, 'hi'), (4, 'bye')], [2, 3, 4, 5])
        ['hi', 'hi', 'bye', 'bye']

    Raises an exception if asked for a time before the first element in series.
    '''
    assert isinstance(series, util.TimeSeries)
    series = list(series.items())
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
    return util.TimeSeries(zip(times, subset))


def assess_sequence(gt, pred, iou_threshold, log_prefix='', exclude_first=True):
    '''Evaluate predicted track against ground-truth annotations.

    Args:
        gt: TimeSeries of annotation dicts.
            Each annotation is a dictionary with:
                annot['present']: bool
                annot['xmin']: float
                annot['ymin']: float
                annot['xmax']: float
                annot['ymax']: float
        pred: TimeSeries of prediction dicts.
            Each prediction is a dictionary with:
                pred['present']: bool (or int)
                pred['xmin']: float
                pred['ymin']: float
                pred['xmax']: float
                pred['ymax']: float
        iou_threshold: Threshold for determining true positive.

    Returns:
        An assessment of each frame with ground-truth.
        This is a dictionary that maps time to an assessment dictionary,
        which contains TP, FP, etc.
    '''
    if not isinstance(gt, util.TimeSeries):
        raise TypeError('type is not TimeSeries: {}'.format(type(gt)))
    if not isinstance(pred, util.TimeSeries):
        raise TypeError('type is not TimeSeries: {}'.format(type(gt)))
    times = gt.keys()
    if exclude_first:
        times = times[1:]
    pred = subset_using_previous_if_missing(pred, times)
    return util.TimeSeries({t: assess_frame(gt[t], pred[t], iou_threshold) for t in times})


def assess_frame(gt, pred, iou_threshold):
    '''Turn prediction into TP, FP, TN, FN.

    Args:
        gt, pred: Dictionaries with fields: present, xmin, xmax, ymin, ymax.
    '''
    # TODO: Some other metrics may require knowledge of aspect ratio.
    # (This is not an issue for now since IOU is invariant).

    # TP: gt "present" and pred "present" and box correct
    # FN: gt "present" and not (pred "present" and box correct)
    # TN: gt "absent" and pred "absent"
    # FP: gt "absent" and pred "present"
    result = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
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
