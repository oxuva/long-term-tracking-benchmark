from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from oxuva.pred import make_prediction
from oxuva import util


PREDICTION_FIELD_NAMES_V1 = [
    'video', 'object', 'imwidth', 'imheight',
    'frame', 'score', 'present', 'xmin_pix', 'ymin_pix', 'xmax_pix', 'ymax_pix',
]
PREDICTION_FIELD_NAMES_V2 = [
    'video', 'object', 'frame_num', 'present', 'score', 'xmin', 'xmax', 'ymin', 'ymax',
]
PREDICTION_FIELD_NAMES = PREDICTION_FIELD_NAMES_V2


def load_predictions_csv(fp):
    '''Loads output of tracker from CSV file.

    Args:
        fp: File-like object with read() and seek().

    Returns:
        List of (time, prediction-dict) pairs.
    '''
    # has_header = csv.Sniffer().has_header(fp.read(4<<10)) # 4 kB
    # fp.seek(0)
    reader = _dict_reader_optional_fieldnames(fp, PREDICTION_FIELD_NAMES)

    predictions = util.SparseTimeSeries()
    for row in reader:
        present = util.str2bool(row['present'])
        t = int(row['frame_num'])
        predictions[t] = make_prediction(
            present=present,
            score=float(row['score']),
            xmin=float(row['xmin']) if present else None,
            xmax=float(row['xmax']) if present else None,
            ymin=float(row['ymin']) if present else None,
            ymax=float(row['ymax']) if present else None)
    return predictions


def _dict_reader_optional_fieldnames(fp, fieldnames):
    '''Creates a csv.DictReader with the given fieldnames.

    The file may or may not contain a header row.
    If it contains a header row, it must match fieldnames.

    This function exists because csv.Sniffer() is unreliable.
    For example, it may fail if one row uses int and one row uses float.
    '''
    # Try to read headers from file.
    reader = csv.DictReader(fp, fieldnames=None)
    if set(reader.fieldnames) != set(fieldnames):
        fp.seek(0)
        reader = csv.DictReader(fp, fieldnames=fieldnames)
    return reader


def dump_predictions_csv(vid_id, obj_id, predictions, fp):
    '''Writes output of tracker for a single track to a CSV file.

    Args:
        vid_id: String.
        obj_id: String.
        predictions: SparseTimeSeries of prediction dicts.
        fp: File-like object with write().
    '''
    writer = csv.DictWriter(fp, fieldnames=PREDICTION_FIELD_NAMES)
    for t, prediction in predictions.items():
        row = {
            'video': vid_id,
            'object': obj_id,
            'frame_num': t,
            'present': util.bool2str(prediction['present']),
            'score': prediction['score'],
            'xmin': prediction['xmin'],
            'xmax': prediction['xmax'],
            'ymin': prediction['ymin'],
            'ymax': prediction['ymax'],
        }
        writer.writerow(row)
