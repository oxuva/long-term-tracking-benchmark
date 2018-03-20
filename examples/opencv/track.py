from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import os
import time

import oxuva


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('predictions_dir')
    parser.add_argument('--data', default='dev')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--tracker', default='TLD')
    global args
    args = parser.parse_args()

    tracker_id = 'cv' + args.tracker
    tracker_preds_dir = os.path.join(args.predictions_dir, args.data, tracker_id)
    if not os.path.exists(tracker_preds_dir):
        os.makedirs(tracker_preds_dir, 0755)

    tracks_file = os.path.join(args.data_dir, 'annotations', args.data + '.csv')
    with open(tracks_file, 'r') as fp:
        tracks = oxuva.load_annotations_csv(fp)
    tasks = {key: oxuva.make_task_from_track(track) for key, track in tracks.items()}

    imfile = lambda vid, t: os.path.join(
        args.data_dir, 'images', args.data, vid, '{:06d}.jpeg'.format(t))

    for key, task in tasks.items():
        vid, obj = key
        if args.verbose:
            print(vid, obj)
        preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv'.format(vid, obj))
        if os.path.exists(preds_file):
            continue

        tracker = Tracker(tracker_type=args.tracker)
        tracker.init(imfile(vid, task.init_time), task.init_rect)
        preds = oxuva.SparseTimeSeries()
        start = time.time()
        for t in range(task.init_time + 1, task.last_time + 1):
            preds[t] = tracker.next(imfile(vid, t))
        dur_sec = time.time() - start
        if args.verbose:
            fps = (task.last_time - task.init_time + 1) / dur_sec
            print('fps {:.3g}'.format(fps))

        tmp_preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv.tmp'.format(vid, obj))
        with open(tmp_preds_file, 'w') as fp:
            oxuva.dump_predictions_csv(vid, obj, preds, fp)
        os.rename(tmp_preds_file, preds_file)


class Tracker:

    def __init__(self, tracker_type):
        self._tracker = create_tracker(tracker_type)

    def init(self, imfile, rect):
        im = cv2.imread(imfile, cv2.IMREAD_COLOR)
        imheight, imwidth, _ = im.shape
        if args.verbose:
            print('image size', '{}x{}'.format(imwidth, imheight))
        cvrect = rect_to_opencv(rect, imsize_hw=(imheight, imwidth))
        ok = self._tracker.init(im, cvrect)
        assert ok

    def next(self, imfile):
        im = cv2.imread(imfile, cv2.IMREAD_COLOR)
        imheight, imwidth, _ = im.shape
        ok, cvrect = self._tracker.update(im)
        if not ok:
            return oxuva.make_prediction(present=False, score=0.0)
        else:
            rect = rect_from_opencv(cvrect, imsize_hw=(imheight, imwidth))
            return oxuva.make_prediction(present=True, score=1.0, **rect)


def create_tracker(tracker_type):
    # https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
    major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker


def rect_to_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs = rect['xmin'] * imwidth
    ymin_abs = rect['ymin'] * imheight
    xmax_abs = rect['xmax'] * imwidth
    ymax_abs = rect['ymax'] * imheight
    return (xmin_abs, ymin_abs, xmax_abs - xmin_abs, ymax_abs - ymin_abs)


def rect_from_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs, ymin_abs, width_abs, height_abs = rect
    xmax_abs = xmin_abs + width_abs
    ymax_abs = ymin_abs + height_abs
    return {
        'xmin': xmin_abs / imwidth,
        'ymin': ymin_abs / imheight,
        'xmax': xmax_abs / imwidth,
        'ymax': ymax_abs / imheight,
    }


if __name__ == '__main__':
    main()
