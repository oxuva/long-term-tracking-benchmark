from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import json
import numpy as np
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oxuva import assess
from oxuva import io
from oxuva import util

FRAME_RATE = 30

MARKERS = ['o', 'v', '^', '<', '>', 's', 'd']  # '*'
CMAP_PREFERENCE = ['tab10', 'tab20', 'hsv']
GRID_COLOR = plt.rcParams['grid.color']  # '#cccccc'
CLEARANCE = 1.1  # Axis range is CLEARANCE * max_value, rounded up.

ARGS_FORMATTER = argparse.ArgumentDefaultsHelpFormatter  # Show default values

INTERVAL_TYPES = ['before', 'after', 'between']
INTERVAL_AXIS_LABEL = {
    'before': 'Frames before time (min)',
    'after': 'Frames after time (min)',
    'between': 'Frames in interval (min)',
}


def _add_arguments(parser):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data', default='dev', help='{dev,test}')
    common.add_argument('--challenge', default='constrained',
                        help='{open,constrained,all}')
    common.add_argument('--verbose', '-v', action='store_true')
    common.add_argument('--permissive', action='store_true',
                        help='Silently exclude tracks which caused an error')
    common.add_argument('--ignore_cache', action='store_true')
    common.add_argument('--cache_dir', default='cache/')
    common.add_argument('--iou_thresholds', nargs='+', type=float, default=[0.5],
                        help='List of IOU thresholds to use', metavar='IOU')

    plot_args = argparse.ArgumentParser(add_help=False)
    plot_args.add_argument('--width_inches', type=float, default=5.0)
    plot_args.add_argument('--height_inches', type=float, default=4.0)

    subparsers = parser.add_subparsers(dest='subcommand', help='Analysis mode')

    # table: Produce a table (one column per IOU threshold)
    table_parser = subparsers.add_parser(
        'table', parents=[common], formatter_class=ARGS_FORMATTER)

    # plot: Produce a figure (one figure per IOU threshold)
    plot_parser = subparsers.add_parser(
        'plot', parents=[common, plot_args], formatter_class=ARGS_FORMATTER)
    plot_parser.add_argument('--no_level_sets', action='store_false', dest='level_sets')

    # interval_plot: Produce a figure for interval ranges (0, t) and (t, inf).
    interval_parser = subparsers.add_parser(
        'interval_plot', parents=[common, plot_args], formatter_class=ARGS_FORMATTER)
    # interval_parser.add_argument('--min_time', type=float, default=0, help='seconds')
    interval_parser.add_argument('--max_time', type=int, default=600, help='seconds')
    interval_parser.add_argument('--time_step', type=int, default=60, help='seconds')
    interval_parser.add_argument('--no_same_axes', action='store_false', dest='same_axes')


def main():
    parser = argparse.ArgumentParser(formatter_class=ARGS_FORMATTER)
    _add_arguments(parser)
    global args
    args = parser.parse_args()

    tracks_file = os.path.join('annotations', args.data + '.csv')
    tasks = _load_tasks(tracks_file)
    tracker_names = _load_tracker_names()

    # Assign colors and markers alphabetically to achieve invariance across plots.
    trackers = sorted(tracker_names.keys(), key=lambda s: s.lower())
    color_list = _generate_colors(len(trackers))
    tracker_colors = dict(zip(trackers, color_list))
    tracker_markers = dict(zip(trackers, itertools.cycle(MARKERS)))

    # Each element preds[tracker] is a VideoObjectDict of TimeSeries of prediction dicts.
    # Only predictions for frames with ground-truth labels are kept.
    # This is much smaller than the predictions for all frames, and is therefore cached.
    predictions = {}
    pred_dir = os.path.join('predictions', args.data)
    for tracker_ind, tracker in enumerate(trackers):
        log_context = 'tracker {}/{} {}'.format(
            tracker_ind + 1, len(trackers), tracker)
        cache_file = os.path.join(
            args.data, 'predictions', '{}.pickle'.format(tracker))
        predictions[tracker] = util.cache_pickle(
            os.path.join(args.cache_dir, 'analyze', cache_file),
            lambda: _load_predictions_and_select_frames(
                tasks, os.path.join(pred_dir, tracker),
                log_prefix=log_context + ': '),
            ignore_existing=args.ignore_cache,
            verbose=args.verbose)

    # Each element assessment[tracker][iou] is a VideoObjectDict
    # of TimeSeries of frame assessment dicts.
    # TODO: Is it unsafe to use float (iou) as dictionary key?
    assessment = {tracker: {iou: util.VideoObjectDict({
        key: assess.assess_sequence(tasks[key].labels, predictions[tracker][key], iou)
        for key in tasks}) for iou in args.iou_thresholds} for tracker in trackers}
    # Each element quality[tracker][iou] is a VideoObjectDict of sequence summary dicts.
    quality = {tracker: {iou: util.VideoObjectDict(
        _map_dict(assess.summarize_sequence, assessment[tracker][iou]))
        for iou in args.iou_thresholds} for tracker in trackers}

    if args.subcommand == 'table':
        _print_statistics(quality, tracker_names)
    elif args.subcommand == 'plot':
        for iou in args.iou_thresholds:
            _plot_statistics(assessment, quality, trackers, iou,
                             tracker_names, tracker_colors, tracker_markers)
    elif args.subcommand == 'interval_plot':
        for iou in args.iou_thresholds:
            _plot_intervals(
                tasks, assessment, trackers, iou,
                tracker_names, tracker_colors, tracker_markers)


def _load_tracker_names():
    if args.challenge == 'all':
        challenges = ['constrained', 'open']
    else:
        challenges = [args.challenge]

    union = {}
    for c in challenges:
        with open('trackers_{}.json'.format(c), 'r') as f:
            tracker_names = json.load(f)
        union.update(tracker_names)
    return union


def _load_tasks(fname):
    with open(fname, 'r') as fp:
        # if fname.endswith('.json'):
        #     tracks = json.load(fp)
        if fname.endswith('.csv'):
            tracks = io.load_annotations_csv(fp)
        else:
            raise ValueError('unknown extension: {}'.format(fname))
    return util.VideoObjectDict(_map_dict(util.make_task_from_track, tracks))


def _load_predictions_and_select_frames(tasks, tracker_pred_dir, log_prefix=''):
    '''Loads all predictions of a tracker and takes the subset of frames with ground truth.

    Args:
        tasks -- VideoObjectDict of Tasks.
        tracker_pred_dir -- Directory that contains files video_object.csv

    Returns:
        VideoObjectDict of SparseTimeSeries of frame assessments.
    '''
    preds = util.VideoObjectDict()
    for track_num, vid_obj in enumerate(tasks.keys()):
        vid, obj = vid_obj
        task = tasks[vid_obj]
        track_name = vid + '_' + obj
        log_context = '{}object {}/{} {}'.format(
            log_prefix, track_num + 1, len(tasks), track_name)
        if args.verbose:
            print(log_context, file=sys.stderr)
        pred_file = os.path.join(tracker_pred_dir, '{}.csv'.format(track_name))
        try:
            with open(pred_file, 'r') as fp:
                pred = io.load_predictions_csv(fp)
        except IOError, exc:
            if args.permissive:
                print('warning: exclude track {}: {}'.format(track_name, str(exc)), file=sys.stderr)
            else:
                raise
        pred = assess.subset_using_previous_if_missing(pred, task.labels.sorted_keys())
        preds[vid_obj] = pred
    return preds


def _print_statistics(quality, names=None):
    names = names or {}
    table_dir = os.path.join('analysis', args.data, args.challenge)
    _ensure_dir_exists(table_dir)
    table_file = os.path.join(table_dir, 'table.txt')
    if args.verbose:
        print('write table to {}'.format(table_file), file=sys.stderr)
    with open(table_file, 'w') as f:
        fieldnames = (['tracker', 'tnr'] +
                      ['tpr_{}'.format(iou) for iou in args.iou_thresholds])
        print(','.join(fieldnames), file=f)
        for tracker in sorted(quality.keys()):
            tprs = []
            stats = {
                iou: assess.statistics(quality[tracker][iou].values())
                for iou in args.iou_thresholds}
            first_iou = args.iou_thresholds[0]
            row = ([names.get(tracker, tracker),
                    '{:.6g}'.format(stats[first_iou]['TNR'])] +
                   ['{:.6g}'.format(stats[iou]['TPR']) for iou in args.iou_thresholds])
            print(','.join(row), file=f)


def _plot_statistics(assessments, quality, trackers, iou_threshold,
                     names=None, colors=None, markers=None):
    names = names or {}
    colors = colors or {}
    markers = markers or {}

    stats = {tracker: assess.statistics(quality[tracker][iou_threshold].values())
             for tracker in trackers}
    trackers = sorted(trackers, key=lambda t: _stats_sort_key(stats[t]), reverse=True)

    plt.figure(figsize=(args.width_inches, args.height_inches))
    plt.xlabel('True Negative Rate (Absent)')
    plt.ylabel('True Positive Rate (Present)')
    if args.level_sets:
        _plot_level_sets()
    for tracker in trackers:
        plt.plot(
            [stats[tracker]['TNR']], [stats[tracker]['TPR']],
            label=names.get(tracker, tracker),
            marker=markers.get(tracker, None),
            color=colors.get(tracker, None),
            markerfacecolor='none', markeredgewidth=2, clip_on=False)
    max_tpr = max([stats[tracker]['TPR'] for tracker in trackers])
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=_ceil_multiple(CLEARANCE * max_tpr, 0.1))
    plt.grid(color=GRID_COLOR)
    legend = lambda: plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0))
    legend()
    plot_dir = os.path.join('analysis', args.data, args.challenge)
    _ensure_dir_exists(plot_dir)
    base_name = 'stats_iou_{}'.format(iou_threshold)
    _save_fig(os.path.join(plot_dir, base_name + '.pdf'))
    plt.gca().legend().set_visible(False)
    _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))

    # Add posthoc-threshold curves to figure.
    legend()
    for tracker in trackers:
        _plot_posthoc_curve(assessments[tracker][iou_threshold],
                            marker='', color=colors.get(tracker, None))
    _save_fig(os.path.join(plot_dir, base_name + '_posthoc.pdf'))
    plt.gca().legend().set_visible(False)
    _save_fig(os.path.join(plot_dir, base_name + '_posthoc_no_legend.pdf'))


def _plot_posthoc_curve(assessments, **kwargs):
    frames = list(itertools.chain(*[series.values() for series in assessments.values()]))
    operating_points = assess.posthoc_threshold(frames)
    metrics = map(assess.quality_metrics, operating_points)
    plt.plot([point['TNR'] for point in metrics],
             [point['TPR'] for point in metrics], **kwargs)


def _plot_intervals(tasks, assessment, trackers, iou_threshold,
                    names=None, colors=None, markers=None):
    names = names or {}
    colors = colors or {}
    markers = markers or {}
    times_sec = range(0, args.max_time + 1, args.time_step)

    # Get overall stats for order in legend.
    quality = {
        tracker: _map_dict(assess.summarize_sequence, assessment[tracker][iou_threshold])
        for tracker in trackers}
    stats = {
        tracker: assess.statistics(quality[tracker].values())
        for tracker in trackers}
    trackers = sorted(trackers, key=lambda t: _stats_sort_key(stats[t]), reverse=True)

    intervals = {}
    points = {}
    for mode in INTERVAL_TYPES:
        intervals[mode], points[mode] = _make_intervals(times_sec, mode)
    stats = {mode: {tracker: _interval_stats(
        tasks, assessment[tracker][iou_threshold], intervals[mode])
        for tracker in trackers} for mode in INTERVAL_TYPES}
    tpr = {mode: {tracker: [
        s.get('TPR', None) for s in stats[mode][tracker]]
        for tracker in trackers} for mode in INTERVAL_TYPES}

    # Find maximum TPR value over all plots (to have same axes).
    max_tpr = {mode: max(
        val for tracker in trackers for val in tpr[mode][tracker] if val is not None)
        for mode in INTERVAL_TYPES}

    for mode in INTERVAL_TYPES:
        plt.figure(figsize=(args.width_inches, args.height_inches))
        plt.xlabel(INTERVAL_AXIS_LABEL[mode])
        plt.ylabel('True Positive Rate')
        for tracker in trackers:
            plt.plot(1 / 60.0 * np.asfarray(points[mode]), tpr[mode][tracker],
                     label=names.get(tracker, tracker),
                     marker=markers.get(tracker, None),
                     color=colors.get(tracker, None),
                     markerfacecolor='none', markeredgewidth=2, clip_on=False)
        plt.xlim(xmin=0, xmax=args.max_time / 60.0)
        ymax = max(max_tpr.values()) if args.same_axes else max_tpr[mode]
        plt.ylim(ymin=0, ymax=_ceil_multiple(CLEARANCE * ymax, 0.1))
        plt.grid(color=GRID_COLOR)
        plot_dir = os.path.join('analysis', args.data, args.challenge)
        _ensure_dir_exists(plot_dir)
        base_name = 'interval_{}_iou_{}'.format(mode, iou_threshold)
        _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))
        plt.legend()
        _save_fig(os.path.join(plot_dir, base_name + '.pdf'))


def _make_intervals(values, interval_type):
    '''Produces intervals and points at which to plot them.

    Returns:
        intervals, points

    Example:
        >> _make_intervals([0, 1, 2, 3], 'before')
        [(0, 1), (0, 2), (0, 3)], [1, 2, 3]

        >> _make_intervals([0, 1, 2, 3], 'after')
        [(0, inf), (1, inf), (2, inf), (3, inf)], [1, 2, 3]

        >> _make_intervals([0, 1, 2, 3], 'between')
        [(0, 1), (1, 2), (2, 3)], [0.5, 1.5, 2.5]
    '''
    if interval_type == 'before':
        intervals = [(0, x) for x in values if x > 0]
        points = [x for x in values if x > 0]
    elif interval_type == 'after':
        intervals = [(x, float('inf')) for x in values]
        points = list(values)
    elif interval_type == 'between':
        intervals = zip(values, values[1:])
        points = [0.5 * (a + b) for a, b in intervals]
    return intervals, points


def _interval_stats(tasks, assessment, intervals):
    '''Computes the quality statistics for each interval.

    Args:
        tasks -- VideoObjectDict of Tasks.
            This is required to get the initial frame number of each track.
        assessment -- VideoObjectDict of SparseTimeSeries of frame assessment dicts.
        intervals -- List of tuples [(a, b), ...]

    Returns:
        List that contains statistics for each interval.
    '''
    stats = [None for _ in intervals]
    for interval_index, (a_sec, b_sec) in enumerate(intervals):
        quality = util.VideoObjectDict()
        for vid_obj in tasks:
            t0 = tasks[vid_obj].init_time  # in number of frames
            subseq = _select_interval(
                assessment[vid_obj],
                t0 + FRAME_RATE * a_sec,
                t0 + FRAME_RATE * b_sec)
            quality[vid_obj] = assess.summarize_sequence(subseq)
        stats[interval_index] = assess.statistics(quality.values())
    return stats


def _select_interval(frames, a, b):
    return util.SparseTimeSeries({t: x for t, x in frames.sorted_items() if a <= t <= b})


def _map_dict(f, x):
    return {k: f(v) for k, v in x.items()}


def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _generate_colors(n):
    # return [colorsys.hsv_to_rgb(i / n, s, v) for i in range(n)]
    for cmap_name in CMAP_PREFERENCE:
        cmap = matplotlib.cm.get_cmap(cmap_name)
        if n <= cmap.N:
            break
    return [cmap(float(i) / n) for i in range(n)]


def _save_fig(plot_file):
    if args.verbose:
        print('write plot to {}'.format(plot_file), file=sys.stderr)
    plt.savefig(plot_file)


def _plot_level_sets(n=10, num_points=100):
    x = np.linspace(0, 1, num_points + 1)[1:]
    for gm in np.asfarray(range(1, n)) / n:
        # gm = sqrt(x*y); y = gm^2 / x
        y = gm**2 / x
        plt.plot(x, y, color=GRID_COLOR, linewidth=1, linestyle='dashed')


def _ceil_multiple(x, step):
    return math.ceil(x / step) * step


def _stats_sort_key(stats):
    return (stats['GM'], stats['TPR'], stats['TNR'])


if __name__ == '__main__':
    main()
