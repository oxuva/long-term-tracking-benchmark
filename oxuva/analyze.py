from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import sys

from oxuva import assess
from oxuva import io
from oxuva import util

FPS = 30


def _add_arguments(parser):
    parser.add_argument('tracks_file', metavar='dataset.json|csv')
    parser.add_argument('pred_dir', metavar='predictions/',
                        help='Directory that contains tracker/vidname_objname.csv')
    parser.add_argument('--trackers', nargs='+',
                        help='Default is all sub-directories of predictions/')
    parser.add_argument('--iou_thresholds', nargs='+', type=float,
                        default=[0.3, 0.5, 0.7])
    parser.add_argument('--min_time', type=float, help='(seconds)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--permissive', action='store_true',
                        help='Silently exclude tracks which caused an error')


def main():
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    global args
    args = parser.parse_args()

    tracks = _load_annotations(args.tracks_file)
    tracks = {(video_name, obj_name): track
        for video_name, video_objs in tracks.items()
        for obj_name, track in video_objs.items()}

    trackers = args.trackers
    if not trackers:
        trackers = sorted(_list_subdirs(args.pred_dir))
        if args.verbose:
            print('found {} trackers: {}'.format(len(trackers), ', '.join(trackers)),
                  file=sys.stderr)

    # Dictionary with elements quality[tracker][(vid_name, obj_name)]
    quality = {}
    for tracker_ind, tracker in enumerate(trackers):
        tracker_pred_dir = os.path.join(args.pred_dir, tracker)
        quality[tracker] = _load_predictions_and_measure_quality(
            tracks, tracker_pred_dir=os.path.join(args.pred_dir, tracker),
            log_prefix='tracker {}/{} {}: '.format(tracker_ind+1, len(trackers), tracker))

    _print_statistics(quality)


def _load_predictions_and_measure_quality(tracks, tracker_pred_dir, log_prefix=''):
    '''Loads and assesses all tracks for a tracker.

    Calls assess.measure_quality().

    Args:
        tracker_pred_dir -- Directory that contains files video_object.csv

    Returns:
        Dictionary that maps (video_name, object_name) to track quality.
    '''
    quality = {iou: {} for iou in args.iou_thresholds}
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
            for iou in args.iou_thresholds:
                quality[iou][vid_obj] = assess.measure_quality(
                    annot['frames'], pred,
                    iou_threshold=iou,
                    min_time=None if args.min_time is None else args.min_time * FPS,
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


def _load_annotations(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.json'):
            tracks = json.load(fp)
        elif fname.endswith('.csv'):
            tracks = io.load_annotations_csv(fp)
        else:
            raise ValueError('unknown extension: {}'.format(fname))
    return tracks


def _list_subdirs(dir):
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]


if __name__ == '__main__':
    main()
