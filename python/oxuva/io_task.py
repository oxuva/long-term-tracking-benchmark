from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from oxuva.dataset import VideoObjectDict
from oxuva.task import Task


TASK_FIELDS = [
    'video_id', 'object_id',
    'init_frame', 'last_frame', 'xmin', 'xmax', 'ymin', 'ymax',
]


def load_dataset_tasks_csv(fp):
    '''Loads the problem definitions for an entire dataset from one CSV file.'''
    reader = csv.DictReader(fp, fieldnames=TASK_FIELDS)
    rows = [row for row in reader]

    tasks = VideoObjectDict()
    for row in rows:
        key = (row['video_id'], row['object_id'])
        tasks[key] = Task(
            init_time=int(row['init_frame']),
            last_time=int(row['last_frame']),
            init_rect={
                'xmin': float(row['xmin']),
                'xmax': float(row['xmax']),
                'ymin': float(row['ymin']),
                'ymax': float(row['ymax'])})
    return tasks
