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

MARKERS = ['o', 'v', '^', '<', '>', 's', 'd'] # '*'
GRID_COLOR = plt.rcParams['grid.color'] # '#cccccc'
CLEARANCE = 1.2 # Axis range is CLEARANCE * max_value, rounded up.


def _add_arguments(parser):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data', default='dev', help='{dev,test}')
    common.add_argument('--challenge', default='constrained',
                        help='{open,constrained,all}')
    common.add_argument('--verbose', '-v', action='store_true')
    common.add_argument('--permissive', action='store_true',
                        help='Silently exclude tracks which caused an error')
    common.add_argument('--cache_dir', default='cache/')
    common.add_argument('--iou_thresholds', nargs='+', type=float, metavar='IOU',
                        default=[0.3, 0.5, 0.7])

    subparsers = parser.add_subparsers(dest='subcommand', help='Analysis mode')
    # report: Produce a table (one column per IOU threshold)
    report_parser = subparsers.add_parser('report', parents=[common])
    # plot: Produce a figure (one figure per IOU threshold)
    plot_parser = subparsers.add_parser('plot', parents=[common])
    # plot_parser.add_argument('--posthoc', action='store_true')
    plot_parser.add_argument('--width_inches', type=float, default=5.0)
    plot_parser.add_argument('--height_inches', type=float, default=4.0)
    plot_parser.add_argument('--level_sets', action='store_true')


def main():
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    global args
    args = parser.parse_args()

    tracks_file = os.path.join('annotations', args.data+'.csv')
    pred_dir = os.path.join('predictions', args.data)

    tracks = _load_annotations(tracks_file)
    tracks = {(video_name, obj_name): track
        for video_name, video_objs in tracks.items()
        for obj_name, track in video_objs.items()}

    tracker_names = _get_tracker_names()
    trackers = sorted(tracker_names.keys())

    tracker_colors = {t: color
        for t, color in zip(trackers, _generate_colors(len(trackers), v=0.9))}
    tracker_markers = {t: marker
        for t, marker in zip(trackers, cycle(MARKERS))}

    # Nested dict with elements quality[tracker][iou][(vid_name, obj_name)]
    quality = {}
    for tracker_ind, tracker in enumerate(trackers):
        for iou_ind, iou in enumerate(args.iou_thresholds):
            log_context = 'iou {}/{} {}: tracker {}/{} {}'.format(
                iou_ind+1, len(args.iou_thresholds), iou,
                tracker_ind+1, len(trackers), tracker)
            cache_file = os.path.join(
                args.data, 'quality', '{}_{}.pickle'.format(tracker, iou))
            quality.setdefault(tracker, {})[iou] = util.cache_pickle(
                os.path.join(args.cache_dir, cache_file),
                lambda: _load_predictions_and_measure_quality(
                    tracks, iou,
                    tracker_pred_dir=os.path.join(pred_dir, tracker),
                    log_prefix=log_context + ': '))

    if args.subcommand == 'report':
        _print_statistics(quality)
    elif args.subcommand == 'plot':
        for iou in args.iou_thresholds:
            _plot_statistics(quality, trackers, iou,
                             tracker_names, tracker_colors, tracker_markers)


def _get_tracker_names():
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
        if fname.endswith('.json'):
            tracks = json.load(fp)
        elif fname.endswith('.csv'):
            tracks = io.load_annotations_csv(fp)
        else:
            raise ValueError('unknown extension: {}'.format(fname))
    return tracks


def _load_predictions_and_measure_quality(tracks, iou_threshold, tracker_pred_dir,
                                          log_prefix=''):
    '''Loads and assesses all tracks for a tracker.

    Calls assess.measure_quality().

    Args:
        tracker_pred_dir -- Directory that contains files video_object.csv

    Returns:
        Dictionary that maps iou threshold to
        dictionary that maps (video_name, object_name) to track quality.
    '''
    quality = {}
    for track_ind, (vid_obj, annot) in enumerate(tracks.items()):
        video_name, obj_name = vid_obj
        track_name = video_name + '_' + obj_name
        log_context = '{}object {}/{} {}'.format(
            log_prefix, track_ind+1, len(tracks), track_name)
        if args.verbose:
            print(log_context, file=sys.stderr)
        pred_file = os.path.join(tracker_pred_dir, '{}.csv'.format(track_name))
        try:
            with open(pred_file, 'r') as fp:
                pred = io.load_predictions_csv(fp)
            quality[vid_obj] = assess.measure_quality(
                annot['frames'], pred, iou_threshold=iou_threshold,
                log_prefix='{}: '.format(log_context))
        except IOError, exc:
            if args.permissive:
                print('warning: exclude track {}: {}'.format(track_name, str(exc)), file=sys.stderr)
            else:
                raise
    return quality


def _print_statistics(quality):
    fieldnames = (['tracker', 'tnr'] +
                  ['tpr_{}'.format(iou) for iou in args.iou_thresholds])
    print(','.join(fieldnames))
    for tracker in sorted(quality.keys()):
        tprs = []
        stats = {
            iou: assess.statistics(quality[tracker][iou].values())
            for iou in args.iou_thresholds}
        first_iou = args.iou_thresholds[0]
        row = ([tracker, '{:.6g}'.format(stats[first_iou]['TNR'])] +
               ['{:.6g}'.format(stats[iou]['TPR']) for iou in args.iou_thresholds])
        print(','.join(row))


def _plot_statistics(quality, trackers, iou_threshold,
                     names=None, colors=None, markers=None):
    names = names or {}
    colors = colors or {}
    markers = markers or {}

    stats = {
        tracker: assess.statistics(quality[tracker][iou_threshold].values())
        for tracker in trackers}
    sort_key = lambda s: (s['GM'], s['TPR'], s['TNR'])
    trackers = sorted(trackers, key=lambda t: sort_key(stats[t]), reverse=True)

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
    plt.ylim(ymin=0, ymax=_ceil_multiple(CLEARANCE*max_tpr, 0.1))
    plt.grid(color=GRID_COLOR)
    plt.legend(loc='upper right')
    plot_dir = os.path.join('analysis', args.data, args.challenge)
    plot_file = os.path.join(plot_dir, 'stats_iou_{}.pdf'.format(iou_threshold))
    _ensure_dir_exists(plot_dir)
    if args.verbose:
        print('write plot to {}'.format(plot_file), file=sys.stderr)
    plt.savefig(plot_file)


def _generate_colors(n, s=1.0, v=1.0):
    return [colorsys.hsv_to_rgb(i/n, s, v) for i in range(n)]


def _plot_level_sets(n=10, num_points=100):
    x = np.linspace(0, 1, num_points+1)[1:]
    for gm in np.asfarray(range(1, n)) / n:
        # gm = sqrt(x*y); y = gm^2 / x
        y = gm**2 / x
        plt.plot(x, y, color=GRID_COLOR, linewidth=1, linestyle='dashed')


def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def _ceil_multiple(x, step):
    return math.ceil(x / step) * step


if __name__ == '__main__':
    main()
