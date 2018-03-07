from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import colorsys
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

from itertools import cycle

FRAME_RATE = 30

MARKERS = ['o', 'v', '^', '<', '>', 's', 'd']  # '*'
GRID_COLOR = plt.rcParams['grid.color']  # '#cccccc'
CLEARANCE = 1.2  # Axis range is CLEARANCE * max_value, rounded up.

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
    # plot_parser.add_argument('--posthoc', action='store_true')

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
    annotations = _load_annotations(tracks_file)
    tracker_names = _load_tracker_names()

    # Assign colors and markers alphabetically to achieve invariance across plots.
    trackers = sorted(tracker_names.keys(), key=lambda s: s.lower())
    tracker_colors = {tracker: color
                      for tracker, color in zip(trackers, _generate_colors(len(trackers), v=0.9))}
    tracker_markers = {tracker: marker
                       for tracker, marker in zip(trackers, cycle(MARKERS))}

    # Each element assessment[tracker][iou] is a VideoObjectDict
    # of TimeSeries of frame assessments.
    # TODO: Is it unsafe to use float (iou) as dictionary key?
    assessment = {}
    pred_dir = os.path.join('predictions', args.data)
    for tracker_ind, tracker in enumerate(trackers):
        for iou_ind, iou in enumerate(args.iou_thresholds):
            log_context = 'tracker {}/{} {}: iou {}/{} {}'.format(
                tracker_ind + 1, len(trackers), tracker,
                iou_ind + 1, len(args.iou_thresholds), iou)
            cache_file = os.path.join(
                args.data, 'assessment', '{}_{}.pickle'.format(tracker, iou))
            assessment.setdefault(tracker, {})[iou] = util.cache_pickle(
                os.path.join(args.cache_dir, 'analyze', cache_file),
                lambda: _load_predictions_and_assess(
                    annotations, iou,
                    tracker_pred_dir=os.path.join(pred_dir, tracker),
                    log_prefix=log_context + ': '),
                ignore_existing=args.ignore_cache,
                verbose=args.verbose)

    # Each element quality[tracker][iou] is a VideoObjectDict of sequence summary dicts.
    quality = {tracker: {iou:
                         util.VideoObjectDict(_map_dict(assess.summarize_sequence, assessment[tracker][iou]))
                         for iou in args.iou_thresholds} for tracker in trackers}

    if args.subcommand == 'table':
        _print_statistics(quality, tracker_names)
    elif args.subcommand == 'plot':
        for iou in args.iou_thresholds:
            _plot_statistics(quality, trackers, iou,
                             tracker_names, tracker_colors, tracker_markers)
    elif args.subcommand == 'interval_plot':
        for iou in args.iou_thresholds:
            _plot_intervals(
                annotations, assessment, trackers, iou,
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


def _load_annotations(fname):
    with open(fname, 'r') as fp:
        # if fname.endswith('.json'):
        #     tracks = json.load(fp)
        if fname.endswith('.csv'):
            tracks = io.load_annotations_csv(fp)
        else:
            raise ValueError('unknown extension: {}'.format(fname))
    return tracks


def _load_predictions_and_assess(annotations, iou_threshold, tracker_pred_dir, log_prefix=''):
    '''Loads all predictions of a tracker.

    Args:
        annotations -- VideoObjectDict of track annotations.
        tracker_pred_dir -- Directory that contains files video_object.csv

    Returns:
        VideoObjectDict of TimeSeries of frame assessments.
    '''
    assessment = util.VideoObjectDict()
    for track_num, vid_obj in enumerate(annotations):
        vid, obj = vid_obj
        annot = annotations[vid_obj]
        track_name = vid + '_' + obj
        log_context = '{}object {}/{} {}'.format(
            log_prefix, track_num + 1, len(annotations), track_name)
        if args.verbose:
            print(log_context, file=sys.stderr)
        pred_file = os.path.join(tracker_pred_dir, '{}.csv'.format(track_name))
        try:
            with open(pred_file, 'r') as fp:
                pred = io.load_predictions_csv(fp)
            assessment[vid_obj] = assess.assess_sequence(
                annot['frames'], pred, iou_threshold=iou_threshold,
                log_prefix='{}: '.format(log_context))
        except IOError, exc:
            if args.permissive:
                print('warning: exclude track {}: {}'.format(track_name, str(exc)), file=sys.stderr)
            else:
                raise
    return assessment


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


def _plot_statistics(quality, trackers, iou_threshold,
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
        plt.plot([stats[tracker]['TNR']], [stats[tracker]['TPR']],
                 label=names.get(tracker, tracker),
                 marker=markers.get(tracker, None),
                 color=colors.get(tracker, None),
                 markerfacecolor='none', markeredgewidth=2, clip_on=False)
    max_tpr = max([stats[tracker]['TPR'] for tracker in trackers])
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=_ceil_multiple(CLEARANCE * max_tpr, 0.1))
    plt.grid(color=GRID_COLOR)
    plot_dir = os.path.join('analysis', args.data, args.challenge)
    _ensure_dir_exists(plot_dir)
    base_name = 'stats_iou_{}'.format(iou_threshold)
    _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))
    plt.legend(loc='upper right')
    _save_fig(os.path.join(plot_dir, base_name + '.pdf'))


def _plot_intervals(annotations, assessment, trackers, iou_threshold,
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
    stats = {mode: {tracker:
                    _interval_stats(annotations, assessment[tracker][iou_threshold], intervals[mode])
                    for tracker in trackers} for mode in INTERVAL_TYPES}
    tpr = {mode: {tracker:
                  [s.get('TPR', None) for s in stats[mode][tracker]]
                  for tracker in trackers} for mode in INTERVAL_TYPES}

    # Find maximum TPR value over all plots (to have same axes).
    max_tpr = {mode:
               max(val for tracker in trackers for val in tpr[mode][tracker] if val is not None)
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


def _interval_stats(annotations, assessment, intervals):
    '''Computes the quality statistics for each interval.

    Args:
        annotations -- VideoObjectDict of TimeSeries of frame annotation dicts.
            This is required to get the initial frame number of each track.
        assessment -- VideoObjectDict of TimeSeries of frame assessment dicts.
        intervals -- List of tuples [(a, b), ...]

    Returns:
        List that contains statistics for each interval.
    '''
    stats = [None for _ in intervals]
    for interval_index, (a_sec, b_sec) in enumerate(intervals):
        quality = util.VideoObjectDict()
        for vid_obj in annotations:
            t0 = _start_time(annotations[vid_obj])  # in number of frames
            subseq = _select_interval(
                assessment[vid_obj],
                t0 + FRAME_RATE * a_sec,
                t0 + FRAME_RATE * b_sec)
            quality[vid_obj] = assess.summarize_sequence(subseq)
        stats[interval_index] = assess.statistics(quality.values())
    return stats


def _select_interval(frames, a, b):
    return {t: x for t, x in frames.items() if a <= t <= b}


def _start_time(annotation):
    frames = annotation['frames']
    if not isinstance(frames, util.TimeSeries):
        raise TypeError('type is not TimeSeries: {}'.format(type(frames)))
    return frames.keys()[0]


def _map_dict(f, x):
    return {k: f(v) for k, v in x.items()}


def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _generate_colors(n, s=1.0, v=1.0):
    return [colorsys.hsv_to_rgb(i / n, s, v) for i in range(n)]


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
