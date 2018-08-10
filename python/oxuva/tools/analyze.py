from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import functools
import json
import numpy as np
import math
import os
import subprocess
import sys

import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import oxuva

# <REPO_DIR>/python/oxuva/tools/analyze.py
TOOLS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.realpath(os.path.join(TOOLS_DIR, '..', '..', '..'))

FRAME_RATE = 30
MARKERS = ['o', 'v', '^', '<', '>', 's', 'd']  # '*'
CMAP_PREFERENCE = ['tab10', 'tab20', 'hsv']
GRID_COLOR = '0.85'  # plt.rcParams['grid.color']
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
    common.add_argument('--data', default='dev', help='{dev,test,devtest}')
    common.add_argument('--challenge', default='open',
                        help='Assess trackers from which challenge?',
                        choices=['constrained', 'open', 'open_minus_constrained'])
    # common.add_argument('--verbose', '-v', action='store_true')
    common.add_argument('--loglevel', default='info', choices=['info', 'debug', 'warning'])
    common.add_argument('--permissive', action='store_true',
                        help='Silently exclude tracks which caused an error')
    # common.add_argument('--ignore_cache', action='store_true')
    common.add_argument('--no_use_summary', dest='use_summary', action='store_false',
                        help='Do not load/dump assessment summaries')
    common.add_argument('--iou_thresholds', nargs='+', type=float, default=[0.5],
                        help='List of IOU thresholds to use', metavar='IOU')
    common.add_argument('--top', type=int, default=10,
                        help='Only show top n trackers (zero to show all)')
    common.add_argument('--no_bootstrap', dest='bootstrap', action='store_false',
                        help='Disable results that require bootstrap sampling')
    common.add_argument('--bootstrap_trials', type=int, default=100,
                        help='Number of trials for bootstrap sampling')
    common.add_argument('--errorbar_size', type=float,
                        default=1.64485,  # scipy.stats.norm.ppf(0.5 + 0.9 / 2)
                        help='Number of standard deviations')
    common.add_argument('--convert_to_png', action='store_true',
                        help='Convert PDF figures to PNG for web')
    common.add_argument('--png_resolution', type=int, default=150,
                        help='Dots-per-inch for PNG conversion')

    plot_args = argparse.ArgumentParser(add_help=False)
    plot_args.add_argument('--width_inches', type=float, default=4.2)
    plot_args.add_argument('--height_inches', type=float, default=4.0)
    plot_args.add_argument('--legend_inches', type=float, default=1.3)

    tpr_tnr_args = argparse.ArgumentParser(add_help=False)
    tpr_tnr_args.add_argument('--no_level_sets', dest='level_sets', action='store_false')
    tpr_tnr_args.add_argument('--no_lower_bounds', dest='lower_bounds', action='store_false')
    tpr_tnr_args.add_argument('--asterisk', action='store_true')

    subparsers = parser.add_subparsers(dest='subcommand', help='Analysis mode')
    subparsers.required = True  # https://bugs.python.org/issue9253#msg186387
    # table: Produce a table (one column per IOU threshold)
    subparser = subparsers.add_parser('table', formatter_class=ARGS_FORMATTER, parents=[common])
    # plot_tpr_tnr: Produce a figure (one figure per IOU threshold)
    subparser = subparsers.add_parser('plot_tpr_tnr', formatter_class=ARGS_FORMATTER,
                                      parents=[common, plot_args, tpr_tnr_args])
    # plot_tpr_tnr_intervals: Produce a figure (one figure per IOU threshold)
    subparser = subparsers.add_parser('plot_tpr_tnr_intervals', formatter_class=ARGS_FORMATTER,
                                      parents=[common, plot_args, tpr_tnr_args])
    subparser.add_argument('--times', type=int, default=[0, 60, 120, 240], help='seconds')
    # plot_tpr_time: Produce a figure for interval ranges (0, t) and (t, inf).
    subparser = subparsers.add_parser('plot_tpr_time', formatter_class=ARGS_FORMATTER,
                                      parents=[common, plot_args])
    subparser.add_argument('--max_time', type=int, default=600, help='seconds')
    subparser.add_argument('--time_step', type=int, default=60, help='seconds')
    subparser.add_argument('--no_same_axes', dest='same_axes', action='store_false')
    # plot_present_absent: Produce a figure (one figure per IOU threshold)
    subparser = subparsers.add_parser('plot_present_absent', formatter_class=ARGS_FORMATTER,
                                      parents=[common, plot_args])


def main():
    parser = argparse.ArgumentParser(formatter_class=ARGS_FORMATTER)
    _add_arguments(parser)
    global args
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    dataset_names = _get_datasets(args.data)
    # Load tasks without annotations.
    dataset_tasks = {
        dataset: _load_tasks(os.path.join(REPO_DIR, 'dataset', 'tasks', dataset + '.csv'))
        for dataset in dataset_names}
    # Take union of all datasets.
    tasks = {key: task for dataset in dataset_names
             for key, task in dataset_tasks[dataset].items()}

    tracker_names = _load_tracker_names()
    trackers = set(tracker_names.keys())
    dataset_assessments = {}
    for dataset in dataset_names:
        dataset_assessments[dataset] = _get_assessments(dataset, trackers)
        # Take subset of trackers for which it was possible to load results.
        trackers = set(dataset_assessments[dataset].keys())
    if len(trackers) < 1:
        raise RuntimeError('could not obtain assessment of any trackers')

    # Assign colors and markers alphabetically to achieve invariance across plots.
    trackers = sorted(trackers, key=lambda s: s.lower())
    color_list = _generate_colors(len(trackers))
    tracker_colors = dict(zip(trackers, color_list))
    tracker_markers = dict(zip(trackers, itertools.cycle(MARKERS)))

    # Merge tracks from all datasets.
    # TODO: Ensure that none have same key?
    assessments = {}
    for tracker in trackers:
        assessments[tracker] = {}
        for iou in args.iou_thresholds:
            assessments[tracker][iou] = functools.reduce(
                oxuva.union_dataset_assessment,
                (dataset_assessments[dataset][tracker][iou] for dataset in dataset_names),
                None)

    # Use simple metrics to get ranking.
    rank_quality = {
        tracker: oxuva.dataset_quality(assessments[tracker][0.5]['totals'], enable_bootstrap=False)
        for tracker in trackers}
    trackers = _sort_quality(rank_quality)
    top_trackers = trackers[:args.top] if args.top else trackers

    if args.subcommand == 'table':
        _print_statistics(assessments, trackers, tracker_names)
    elif args.subcommand == 'plot_tpr_tnr':
        _plot_tpr_tnr_overall(assessments, top_trackers,
                              tracker_names, tracker_colors, tracker_markers)
    elif args.subcommand == 'plot_tpr_tnr_intervals':
        _plot_tpr_tnr_intervals(assessments, top_trackers,
                                tracker_names, tracker_colors, tracker_markers)
    elif args.subcommand == 'plot_tpr_time':
        for iou in args.iou_thresholds:
            for bootstrap in ([False, True] if args.bootstrap else [False]):
                _plot_intervals(assessments, top_trackers, iou, bootstrap,
                                tracker_names, tracker_colors, tracker_markers)
    elif args.subcommand == 'plot_present_absent':
        for iou in args.iou_thresholds:
            for bootstrap in ([False, True] if args.bootstrap else [False]):
                _plot_present_absent(assessments, top_trackers, iou, bootstrap,
                                     tracker_names, tracker_colors, tracker_markers)


def _get_assessments(dataset, trackers):
    '''
    Args:
        dataset: String that identifies dataset ("dev" or "test").
        trackers: List of tracker names.

    Returns:
        Dictionary that maps [tracker][iou] to dataset assessment.
        Only returns assessments for subset of trackers that were successful.
    '''
    # Create functions to load tasks with annotations on demand.
    # (Do not need annotations if using cached assessments.)
    # TODO: Code would be easier to read using a class with lazy-cached elements as members?
    get_annotations = oxuva.LazyCacheCaller(functools.partial(
        _load_tasks_with_annotations,
        os.path.join(REPO_DIR, 'dataset', 'annotations', dataset + '.csv')))

    assessments = {}
    for tracker_ind, tracker in enumerate(trackers):
        try:
            log_context = 'tracker {}/{} {}'.format(tracker_ind + 1, len(trackers), tracker)
            tracker_assessments = {}
            # Load predictions at most once for all IOU thresholds (can be slow).
            get_predictions = oxuva.LazyCacheCaller(
                lambda: oxuva.load_predictions_and_select_frames(
                    get_annotations(),
                    os.path.join('predictions', dataset, tracker),
                    permissive=args.permissive,
                    log_prefix=log_context + ': '))
            # Obtain results at all IOU thresholds in order to make axes equal in all graphs.
            # TODO: Is it unsafe to use float (iou) as dictionary key?
            for iou in args.iou_thresholds:
                logger.info('assess tracker "%s" with iou %g', tracker, iou)
                assess_func = lambda: oxuva.assess_dataset(get_annotations(), get_predictions(),
                                                           iou, resolution_seconds=30)
                if args.use_summary:
                    tracker_assessments[iou] = oxuva.cache(
                        oxuva.Protocol(
                            dump=oxuva.dump_dataset_assessment_json,
                            load=oxuva.load_dataset_assessment_json, binary=False),
                        os.path.join(
                            'assess', dataset, tracker, 'iou_{}.json'.format(oxuva.float2str(iou))),
                        assess_func)
                else:
                    # When it is not cached, it will include frame_assessments.
                    # TODO: Could cache (selected frames of) predictions to file if this is slow.
                    tracker_assessments[iou] = assess_func()
        except IOError as ex:
            logger.warning('could not obtain assessment of tracker "%s" on dataset "%s": %s',
                           tracker, dataset, ex)
        else:
            assessments[tracker] = tracker_assessments
    return assessments


def _load_tracker_names():
    with open('trackers.json', 'r') as f:
        trackers = json.load(f)
    trackers = {key: tracker for key, tracker in trackers.items()
                if ((args.challenge == 'open') or
                    (args.challenge == 'constrained' and tracker['constrained']) or
                    (args.challenge == 'open_minus_constrained' and not tracker['constrained']))}
    tracker_names = {key: tracker['name'] for key, tracker in trackers.items()}
    return tracker_names


def _get_datasets(name):
    if name == 'devtest':
        return ['dev', 'test']
    else:
        return [name]


def _load_tasks(fname):
    logger.debug('load tasks without annotations from "%s"', fname)
    with open(fname, 'r') as fp:
        return oxuva.load_dataset_tasks_csv(fp)


def _load_tasks_with_annotations(fname):
    logger.debug('load tasks with annotations from "%s"', fname)
    with open(fname, 'r') as fp:
        # if fname.endswith('.json'):
        #     tracks = json.load(fp)
        if fname.endswith('.csv'):
            tracks = oxuva.load_dataset_annotations_csv(fp)
        else:
            raise ValueError('unknown extension: {}'.format(fname))
    return oxuva.map_dict(oxuva.make_task_from_track, tracks)


def _print_statistics(assessments, trackers, names=None):
    fields = ['TPR', 'TNR', 'GM', 'MaxGM']
    if args.bootstrap:
        # Include xxx_var keys too.
        fields = list(itertools.chain.from_iterable([key, key + '_var'] for key in fields))
    names = names or {}
    stats = {tracker: {iou: (
        oxuva.dataset_quality(assessments[tracker][iou]['totals'],
                              enable_bootstrap=args.bootstrap, num_trials=args.bootstrap_trials))
        for iou in args.iou_thresholds} for tracker in trackers}
    table_dir = os.path.join('analysis', args.data, args.challenge)
    _ensure_dir_exists(table_dir)
    table_file = os.path.join(table_dir, 'table.csv')
    logger.info('write table to %s', table_file)
    with open(table_file, 'w') as f:
        fieldnames = ['tracker'] + [
            metric + '_' + str(iou) for iou in args.iou_thresholds for metric in fields]
        print(','.join(fieldnames), file=f)
        for tracker in trackers:
            row = [names.get(tracker, tracker)] + [
                '{:.6g}'.format(stats[tracker][iou][metric])
                for iou in args.iou_thresholds for metric in fields]
            print(','.join(row), file=f)


def _plot_tpr_tnr_overall(assessments, trackers,
                          names=None, colors=None, markers=None):
    bootstrap_modes = [False, True] if args.bootstrap else [False]

    for iou in args.iou_thresholds:
        for bootstrap in bootstrap_modes:
            for posthoc in ([False] if bootstrap else [False, True]):
                _plot_tpr_tnr(('tpr_tnr_iou_' + oxuva.float2str(iou) +
                               ('_posthoc' if posthoc else '') +
                               ('_bootstrap' if bootstrap else '')),
                              assessments, trackers, iou, bootstrap, posthoc,
                              names=names, colors=colors, markers=markers,
                              min_time_sec=None, max_time_sec=None, include_score=True)


def _plot_tpr_tnr_intervals(assessments, trackers,
                            names=None, colors=None, markers=None):
    modes = ['before', 'after']
    intervals_sec = {}
    for mode in modes:
        intervals_sec[mode], _ = _make_intervals(args.times, mode)

    bootstrap_modes = [False, True] if args.bootstrap else [False]
    for bootstrap in bootstrap_modes:
        for iou in args.iou_thresholds:
            # Order by performance on all frames.
            stats = {tracker: oxuva.dataset_quality(assessments[tracker][iou]['totals'],
                                                    enable_bootstrap=bootstrap,
                                                    num_trials=args.bootstrap_trials)
                     for tracker in trackers}
            order = _sort_quality(stats, use_bootstrap_mean=False)

            tpr_key = 'TPR_mean' if bootstrap else 'TPR'
            # Get stats for all plots to establish axis range.
            # Note: This means that dataset_quality_interval() is called twice.
            max_tpr = max([max([max([
                oxuva.dataset_quality_interval(
                    assessments[tracker][iou]['quantized_totals'],
                    min_time=None if min_time is None else FRAME_RATE * min_time,
                    max_time=None if max_time is None else FRAME_RATE * max_time,
                    enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)[tpr_key]
                for tracker in trackers])
                for min_time, max_time in intervals_sec[mode]])
                for mode in modes])

            for mode in modes:
                for min_time_sec, max_time_sec in intervals_sec[mode]:
                    base_name = '_'.join(
                        ['tpr_tnr', 'iou_' + oxuva.float2str(iou),
                         'interval_{}_{}'.format(oxuva.float2str(min_time_sec),
                                                 oxuva.float2str(max_time_sec))] +
                        (['bootstrap'] if bootstrap else []))
                    _plot_tpr_tnr(base_name, assessments, trackers, iou, bootstrap, posthoc=False,
                                  min_time_sec=min_time_sec, max_time_sec=max_time_sec,
                                  max_tpr=max_tpr, order=order,
                                  names=names, colors=colors, markers=markers)


def _plot_tpr_tnr(base_name, assessments, trackers, iou_threshold, bootstrap, posthoc,
                  min_time_sec=None, max_time_sec=None, include_score=False,
                  max_tpr=None, order=None,
                  names=None, colors=None, markers=None, legend_kwargs=None):
    names = names or {}
    colors = colors or {}
    markers = markers or {}
    legend_kwargs = legend_kwargs or {}

    tpr_key = 'TPR_mean' if bootstrap else 'TPR'
    tnr_key = 'TNR_mean' if bootstrap else 'TNR'

    for iou in args.iou_thresholds:
        stats = {
            tracker: oxuva.dataset_quality_interval(
                assessments[tracker][iou_threshold]['quantized_totals'],
                min_time=None if min_time_sec is None else FRAME_RATE * min_time_sec,
                max_time=None if max_time_sec is None else FRAME_RATE * max_time_sec,
                enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)
            for tracker in trackers}
        if order is None:
            order = _sort_quality(stats, use_bootstrap_mean=False)

        plt.figure(figsize=(args.width_inches, args.height_inches))
        plt.xlabel('True Negative Rate (Absent)')
        plt.ylabel('True Positive Rate (Present)')
        if args.level_sets:
            _plot_level_sets()

        for tracker in order:
            if bootstrap:
                plot_func = functools.partial(
                    _errorbar,
                    xerr=args.errorbar_size * np.sqrt([stats[tracker]['TNR_var']]),
                    yerr=args.errorbar_size * np.sqrt([stats[tracker]['TPR_var']]),
                    capsize=3)
            else:
                plot_func = plt.plot
            plot_func([stats[tracker][tnr_key]], [stats[tracker][tpr_key]],
                      label=_tracker_label(names.get(tracker, tracker), include_score,
                                           stats[tracker], use_bootstrap_mean=False),
                      color=colors.get(tracker, None),
                      marker=markers.get(tracker, None),
                      markerfacecolor='none', markeredgewidth=2, clip_on=False)
            if args.lower_bounds:
                plt.plot(
                    [stats[tracker][tnr_key], 1], [stats[tracker][tpr_key], 0],
                    color=colors.get(tracker, None),
                    linestyle='dashed', marker='')

        if posthoc:
            num_posthoc = 0
            for tracker in order:
                # Add posthoc-threshold curves to figure.
                # TODO: Plot distribution of post-hoc curves when using bootstrap sampling?
                if assessments[tracker][iou_threshold].get('frame_assessments', None) is None:
                    logger.warning('cannot do posthoc curve for tracker "%s" at iou %g',
                                   tracker, iou_threshold)
                else:
                    _plot_posthoc_curve(assessments[tracker][iou_threshold]['frame_assessments'],
                                        marker='', color=colors.get(tracker, None))
                    num_posthoc += 1
            if num_posthoc == 0:
                logger.warning('skip posthoc plot: zero trackers')
                return

        if max_tpr is None:
            max_tpr = max([stats[tracker][tpr_key] for tracker in trackers])
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=0, ymax=_ceil_nearest(CLEARANCE * max_tpr, 0.1))
        plt.grid(color=GRID_COLOR, clip_on=False)
        _hide_spines()
        plot_dir = os.path.join('analysis', args.data, args.challenge)
        _ensure_dir_exists(plot_dir)
        _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))
        _legend_outside(**legend_kwargs)
        _save_fig(os.path.join(plot_dir, base_name + '.pdf'))


def _plot_posthoc_curve(assessments, **kwargs):
    frames = list(itertools.chain.from_iterable(
        series.values() for series in assessments.values()))
    operating_points = oxuva.posthoc_threshold(frames)
    metrics = list(map(oxuva.quality_metrics, operating_points))
    plt.plot([point['TNR'] for point in metrics],
             [point['TPR'] for point in metrics], **kwargs)


def _plot_intervals(assessments, trackers, iou_threshold, bootstrap,
                    names=None, colors=None, markers=None):
    # TODO: Add errorbars using bootstrap sampling?
    names = names or {}
    colors = colors or {}
    markers = markers or {}
    times_sec = range(0, args.max_time + 1, args.time_step)

    # Get overall stats for order in legend.
    overall_stats = {tracker: oxuva.dataset_quality(assessments[tracker][iou_threshold]['totals'],
                                                    enable_bootstrap=bootstrap,
                                                    num_trials=args.bootstrap_trials)
                     for tracker in trackers}
    order = _sort_quality(overall_stats, use_bootstrap_mean=False)

    intervals_sec = {}
    points = {}
    for mode in INTERVAL_TYPES:
        intervals_sec[mode], points[mode] = _make_intervals(times_sec, mode)

    stats = {mode: {tracker: [
        oxuva.dataset_quality_interval(
            assessments[tracker][iou_threshold]['quantized_totals'],
            min_time=None if min_time is None else FRAME_RATE * min_time,
            max_time=None if max_time is None else FRAME_RATE * max_time,
            enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)
        for min_time, max_time in intervals_sec[mode]]
        for tracker in trackers} for mode in INTERVAL_TYPES}
    tpr_key = 'TPR_mean' if bootstrap else 'TPR'
    # Find maximum TPR value over all plots (to have same axes).
    max_tpr = {mode: max(s[tpr_key] for tracker in trackers for s in stats[mode][tracker])
               for mode in INTERVAL_TYPES}

    for mode in INTERVAL_TYPES:
        plt.figure(figsize=(args.width_inches, args.height_inches))
        plt.xlabel(INTERVAL_AXIS_LABEL[mode])
        plt.ylabel('True Positive Rate')
        for tracker in order:
            tpr = [s.get(tpr_key, None) for s in stats[mode][tracker]]
            if bootstrap:
                tpr_var = [s.get('TPR_var', None) for s in stats[mode][tracker]]
                plot_func = functools.partial(
                    _errorbar,
                    yerr=args.errorbar_size * np.sqrt(tpr_var),
                    capsize=3)
            else:
                plot_func = plt.plot
            plot_func(1 / 60.0 * np.asarray(points[mode]), tpr,
                      label=names.get(tracker, tracker),
                      marker=markers.get(tracker, None),
                      color=colors.get(tracker, None),
                      markerfacecolor='none', markeredgewidth=2, clip_on=False)
        plt.xlim(xmin=0, xmax=args.max_time / 60.0)
        ymax = max(max_tpr.values()) if args.same_axes else max_tpr[mode]
        plt.ylim(ymin=0, ymax=_ceil_nearest(CLEARANCE * ymax, 0.1))
        plt.grid(color=GRID_COLOR, clip_on=False)
        _hide_spines()
        plot_dir = os.path.join('analysis', args.data, args.challenge)
        _ensure_dir_exists(plot_dir)
        base_name = ('tpr_time_iou_{}_interval_{}'.format(oxuva.float2str(iou_threshold), mode) +
                     ('_bootstrap' if bootstrap else ''))
        _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))
        _legend_outside()
        _save_fig(os.path.join(plot_dir, base_name + '.pdf'))


def _plot_present_absent(
        assessments, trackers, iou_threshold, bootstrap,
        names=None, colors=None, markers=None):
    names = names or {}
    colors = colors or {}
    markers = markers or {}

    stats_whole = {
        tracker: oxuva.dataset_quality(
            assessments[tracker][iou_threshold]['totals'],
            enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)
        for tracker in trackers}
    stats_all_present = {
        tracker: oxuva.dataset_quality_filter(
            assessments[tracker][iou_threshold]['totals'], require_none_absent=True,
            enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)
        for tracker in trackers}
    stats_any_absent = {
        tracker: oxuva.dataset_quality_filter(
            assessments[tracker][iou_threshold]['totals'], require_some_absent=True,
            enable_bootstrap=bootstrap, num_trials=args.bootstrap_trials)
        for tracker in trackers}

    order = _sort_quality(stats_whole)
    tpr_key = 'TPR_mean' if bootstrap else 'TPR'
    max_tpr = max(max([stats_all_present[tracker][tpr_key] for tracker in trackers]),
                  max([stats_any_absent[tracker][tpr_key] for tracker in trackers]))

    plt.figure(figsize=(args.width_inches, args.height_inches))
    plt.xlabel('TPR (tracks without absent labels)')
    plt.ylabel('TPR (tracks with some absent labels)')
    for tracker in order:
        if bootstrap:
            plot_func = functools.partial(
                _errorbar,
                xerr=args.errorbar_size * np.sqrt([stats_all_present[tracker]['TPR_var']]),
                yerr=args.errorbar_size * np.sqrt([stats_any_absent[tracker]['TPR_var']]),
                capsize=3)
        else:
            plot_func = plt.plot
        plot_func(
            [stats_all_present[tracker][tpr_key]], [stats_any_absent[tracker][tpr_key]],
            label=names.get(tracker, tracker),
            color=colors.get(tracker, None),
            marker=markers.get(tracker, None),
            markerfacecolor='none', markeredgewidth=2, clip_on=False)
    plt.xlim(xmin=0, xmax=_ceil_nearest(CLEARANCE * max_tpr, 0.1))
    plt.ylim(ymin=0, ymax=_ceil_nearest(CLEARANCE * max_tpr, 0.1))
    plt.grid(color=GRID_COLOR, clip_on=False)
    _hide_spines()
    # Draw a diagonal line.
    plt.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1, linestyle='dotted')
    plot_dir = os.path.join('analysis', args.data, args.challenge)
    _ensure_dir_exists(plot_dir)
    base_name = ('present_absent_iou_{}'.format(oxuva.float2str(iou_threshold)) +
                 ('_bootstrap' if bootstrap else ''))
    _save_fig(os.path.join(plot_dir, base_name + '_no_legend.pdf'))
    _legend_outside()
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
        intervals = list(zip(values, values[1:]))
        points = [0.5 * (a + b) for a, b in intervals]
    return intervals, points


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
    logger.info('write plot to %s', plot_file)
    plt.savefig(plot_file)

    if args.convert_to_png:
        name, ext = os.path.splitext(plot_file)
        if not ext.lower() == '.pdf':
            raise ValueError('plot file does not have pdf extension: {:s}'.format(plot_file))
        logger.debug('convert to png: %s', plot_file)
        png_file = name + '.png'
        try:
            subprocess.check_call(['convert', '-density', str(args.png_resolution), plot_file,
                                   '-quality', '90', png_file])
        except subprocess.CalledProcessError as ex:
            logger.warning('could not convert to png: %s', ex)


def _plot_level_sets(n=10, num_points=100):
    x = np.linspace(0, 1, num_points + 1)[1:]
    for gm in np.asfarray(range(1, n)) / n:
        # gm = sqrt(x*y); y = gm^2 / x
        y = gm**2 / x
        plt.plot(x, y, color=GRID_COLOR, linewidth=1, linestyle='dotted')


def _ceil_nearest(x, step):
    '''Rounds up to nearest multiple of step.'''
    return math.ceil(x / step) * step


def _sort_quality(quality, use_bootstrap_mean=False):
    '''
    Args:
        quality: Dict that maps tracker name to quality dict.
    '''
    def sort_key(tracker):
        return _quality_sort_key(quality[tracker],
                                 use_bootstrap_mean=use_bootstrap_mean)

    return sorted(quality.keys(), key=sort_key, reverse=True)


def _quality_sort_key(stats, use_bootstrap_mean=False):
    if use_bootstrap_mean:
        return (stats['MaxGM_mean'], stats['TPR_mean'], stats['TNR_mean'])
    else:
        return (stats['MaxGM'], stats['TPR'], stats['TNR'])


def _tracker_label(name, include_score, stats, use_bootstrap_mean):
    if not include_score:
        return name
    gm_key = 'GM_mean' if use_bootstrap_mean else 'GM'
    max_gm_key = 'MaxGM_mean' if use_bootstrap_mean else 'MaxGM'
    max_at_point = abs(stats[gm_key] - stats[max_gm_key]) <= 1e-3
    asterisk = '*' if args.asterisk and max_at_point else ''
    return '{} ({:.2f}{})'.format(name, stats[max_gm_key], asterisk)


def _errorbar(*args, **kwargs):
    container = plt.errorbar(*args, **kwargs)
    # Disable clipping for caps of errorbars.
    _, caplines, barlinecols = container
    for capline in caplines:
        capline.set_clip_on(False)
    for barlinecol in barlinecols:
        barlinecol.set_clip_on(False)
    # return container


def _hide_spines():
    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False)


def _legend_no_errorbars(**kwargs):
    '''Replaces plt.legend(). Excludes errorbars from legend.'''
    # https://swdg.io/2015/errorbar-legends/
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, matplotlib.container.ErrorbarContainer) else h
               for h in handles]
    ax.legend(handles, labels, **kwargs)


def _legend_outside(**kwargs):
    fig = plt.gcf()
    fig.set_size_inches(args.width_inches + args.legend_inches, args.height_inches)
    frac = float(args.width_inches) / (args.width_inches + args.legend_inches)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * frac, box.height])
    _legend_no_errorbars(loc='center left', bbox_to_anchor=(1.02, 0.5), **kwargs)


if __name__ == '__main__':
    main()
