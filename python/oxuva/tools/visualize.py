from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import os
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageColor, ImageFont

import logging
logger = logging.getLogger(__name__)

import oxuva


# <REPO_DIR>/python/oxuva/tools/analyze.py
TOOLS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.realpath(os.path.join(TOOLS_DIR, '..', '..', '..'))
DATA_DIR = os.path.join(REPO_DIR, 'dataset')


def _add_arguments(parser):
    parser.add_argument('tracker', help='Name of tracker to visualize')
    parser.add_argument('--loglevel', default='info', choices=['info', 'debug', 'warning'])
    parser.add_argument('--data', default='dev', choices=['dev', 'test'], help='{dev,test}')
    # TODO: Allow user to specify single video?
    # TODO: Plot multiple trackers together?


def main():
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    tasks_file = os.path.join(DATA_DIR, 'tasks', args.data + '.csv')
    images_dir = os.path.join(DATA_DIR, 'images', args.data)
    predictions_dir = os.path.join('predictions', args.data, args.tracker)
    output_dir = os.path.join('visualize', args.data, args.tracker)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, 0o755)

    with open(tasks_file, 'r') as fp:
        tasks = oxuva.load_dataset_tasks_csv(fp)

    for i, (key, task) in enumerate(sorted(tasks.items())):
        vid, obj = key
        logger.info('task %d/%d: (%s, %s)', i + 1, len(tasks), vid, obj)
        video_file = os.path.join(output_dir, '{}_{}.mp4'.format(vid, obj))
        if os.path.exists(video_file):
            logger.debug('skip %s %s: already exists', vid, obj)
            continue
        try:
            predictions_file = os.path.join(predictions_dir, '{}_{}.csv'.format(vid, obj))
            with open(predictions_file, 'r') as fp:
                predictions = oxuva.load_predictions_csv(fp)
            image_file_func = lambda t: os.path.join(images_dir, vid, '{:06d}.jpeg'.format(t))
            output_file = os.path.join(output_dir, '{}_{}.mp4'.format(vid, obj))
            _visualize_video(task, predictions, image_file_func, output_file)
        except (IOError, subprocess.CalledProcessError) as ex:
            logger.warning('could not visualize %s %s: %s', vid, obj, ex)


def _visualize_video(task, predictions, image_file_func, output_file):
    times = list(range(task.init_time, task.last_time + 1))
    predictions = oxuva.subset_using_previous_if_missing(predictions, times[1:])

    with _make_temp_dir(prefix='tmp-visualize-') as tmp_dir:
        logger.debug('write image files to %s', tmp_dir)
        # TODO: Delete temporary directory?
        pattern = os.path.join(tmp_dir, '%06d.jpeg')  # Messy but used by ffmpeg.

        for i, t in enumerate(times):
            im = Image.open(image_file_func(t))
            draw = ImageDraw.Draw(im)
            if t == task.init_time:
                # Draw initial rectangle.
                draw.rectangle(_pil_rect(task.init_rect, im.size), outline=_get_color('green'))
            else:
                # Draw predicted rectangle.
                draw.rectangle(_pil_rect(predictions[t], im.size), outline=_get_color('yellow'))
            del draw
            im.save(pattern % i)

        # TODO: Make video with ffmpeg.
        tmp_output_file = _tmp_name(output_file)
        command = _ffmpeg_command(['-i', pattern,
                                   '-r', '30',  # fps
                                   '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                                   tmp_output_file])
        subprocess.check_call(command)
        os.rename(tmp_output_file, output_file)


def _pil_rect(rect, size_xy):
    width, height = size_xy
    xmin = int(round(rect['xmin'] * width))
    xmax = int(round(rect['xmax'] * width))
    ymin = int(round(rect['ymin'] * height))
    ymax = int(round(rect['ymax'] * height))
    return [(xmin, ymin), (xmax, ymax)]


# https://github.com/mrmrs/colors/blob/master/js/colors.js
NICE_COLORS = {
    'aqua':    '#7fdbff',
    'blue':    '#0074d9',
    'lime':    '#01ff70',
    'navy':    '#001f3f',
    'teal':    '#39cccc',
    'olive':   '#3d9970',
    'green':   '#2ecc40',
    'red':     '#ff4136',
    'maroon':  '#85144b',
    'orange':  '#ff851b',
    'purple':  '#b10dc9',
    'yellow':  '#ffdc00',
    'fuchsia': '#f012be',
    'gray':    '#aaaaaa',
    'white':   '#ffffff',
    'black':   '#111111',
    'silver':  '#dddddd',
}


def _get_color(name):
    return ImageColor.getrgb(NICE_COLORS[name])


def _ffmpeg_command(args):
    return (
        ['ffmpeg',
         '-loglevel', 'error',  # Quiet.
         '-y',                  # Overwrite output without asking.
         '-nostdin',            # No interaction with user (q to quit).
        ] + args)


def _tmp_name(fname):
    head, tail = os.path.split(fname)
    return os.path.join(head, 'tmp_' + tail)


@contextlib.contextmanager
def _make_temp_dir(*args, **kwargs):
    temp_dir = tempfile.mkdtemp(*args, **kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
