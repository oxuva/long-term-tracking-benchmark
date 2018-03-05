from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from oxuva import util


def measure_quality(gt, pred, iou_threshold, log_prefix='', **kwargs):
    '''Measures the quality of a single track.

    The quality is the total number of TP, TN, FP, FN.

    Args:
        kwargs: For assess_predictions().
    '''
    result = assess_predictions(gt, pred, iou_threshold, log_prefix=log_prefix, **kwargs)
    quality = {
        'len': len(result),
        'TP': sum(x['TP'] for _, x in result.items()),
        'TN': sum(x['TN'] for _, x in result.items()),
        'FP': sum(x['FP'] for _, x in result.items()),
        'FN': sum(x['FN'] for _, x in result.items()),
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
    tracks_quality = list(tracks_quality) # In case tracks_quality is a generator.
    total = {
        k: sum([q[k] for q in tracks_quality])
        for k in ['TP', 'FN', 'FP', 'TN']
    }
    num_pos = total['TP'] + total['FN']
    num_neg = total['TN'] + total['FP']
    if num_pos > 0:
        total['TPR'] = float(total['TP']) / num_pos
    else:
        raise ValueError('unable to compute TPR (no positives)')
    if num_neg > 0:
        total['TNR'] = float(total['TN']) / num_neg
    else:
        raise ValueError('unable to compute TNR (no negatives)')
    # TODO: Add some errorbars?
    total['GM'] = util.geometric_mean(total['TPR'], total['TNR'])
    return total


def subset_using_previous_if_missing(data, times):
    '''Extracts a subset of values at the given times.
    If there is no data for a particular time, then the last value is used.

    Args:
        data: List of (time, x) pairs.
        times: List of times.

    Example:
        subset_using_previous_if_missing([(2, 'hi'), (4, 'bye')], [2, 3, 4, 5])
        => ['hi', 'hi', 'bye', 'bye']
    '''
    # Assume that data is sorted by time.
    subset = [None for _ in times]
    t_curr, x_curr = None, None
    for i, t in enumerate(times):
        # Read from data until we have read all elements <= t.
        read_all = False
        while not read_all:
            if len(data) == 0:
                read_all = True
            else:
                t_next, x_next = data[0]
                if t_next > t:
                    # We have gone past t.
                    read_all = True
                else:
                    # Keep going.
                    t_curr, x_curr = t_next, x_next
                    data = data[1:]
        if t_curr is None:
            raise ValueError('no value for time: {}'.format(t))
        subset[i] = x_curr
    return subset


def assess_predictions(gt, pred, iou_threshold, min_time=None, max_time=None, log_prefix=''):
    '''Compare predicted track to ground-truth annotations.

    Args:
        gt: List of (frame, annotation) pairs ordered by frame.
            Each annotation is a dictionary with:
                annot['present']: bool
                annot['xmin']: float
                annot['ymin']: float
                annot['xmax']: float
                annot['ymax']: float
        pred: List of (frame, prediction) pairs ordered by frame.
            Each prediction is a dictionary with:
                pred['present']: bool (or int)
                pred['xmin']: float
                pred['ymin']: float
                pred['xmax']: float
                pred['ymax']: float
                pred['imwidth']: float
                pred['imheight']: float
        iou_threshold: Threshold for determining true positive.
        min_time, max_time: Consider frames in [min_time, max_time] (inclusive).
            Use None to disable limit.

    Returns:
        An assessment of each frame with ground-truth.
    '''
    # pred = dict(pred)
    t_first, gt_first = gt[0]
    # Check that first GT frame (exemplar) has object "present".
    if not gt_first['present']:
        raise AssertionError('{}object not present in first frame; re-generate JSON file'.format(log_prefix))

    # Exclude first frame from evaluation.
    gt_subset = gt[1:]
    times = [t for t, _ in gt_subset]
    pred = subset_using_previous_if_missing(pred, times)

    assess = {}
    for (t, gt_t), pred_t in zip(gt_subset, pred):
        if min_time is not None and (t - t_first) < min_time:
            continue
        if max_time is not None and (t - t_first) > max_time:
            continue
        assess[t] = assess_frame(gt_t, pred_t, iou_threshold)
    # # Convert from list of dictionaries to dictionary of numpy arrays.
    # return {
    #     k: np.array([assess[t][k] for t, _ in gt['frames'])
    #     for k in assess[0].keys()
    # }
    return assess


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
