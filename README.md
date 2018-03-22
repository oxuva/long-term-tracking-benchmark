# OxUvA long-term tracking benchmark

## How to get the data

Obtain `images_dev.tar` and `images_test.tar` as described on the website.
Extract these in `dataset/`.
The structure of `dataset/` should be:
```
dataset/images/{subset}/{video}/{frame:06d}.jpeg
dataset/tasks/{subset}.csv
dataset/annotations/{subset}.csv
```
where `{subset}` is either `dev` or `test`, `{video}` is the video ID e.g. `vid0042`, and `{frame:06d}` is the frame number starting from zero e.g. `002934`.

## How to set up your environment

To install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

You must also add the parent directory of `oxuva/` to `PYTHONPATH` to be able to import the `oxuva` package.
```bash
export PYTHONPATH="/path/to/long-term-tracking-benchmark:$PYTHONPATH"
```
This is required to run the scripts such as the OpenCV example tracker and the analysis script which generates the plots.

## How to run your tracker

Our toolkit does not execute your tracker.
Your tracker should output all predictions in the format described below.
For Python trackers, we provide the utility functions `oxuva.load_dataset_tasks_csv` and `oxuva.dump_predictions_csv` to make this easy.
See [`examples/opencv/track.py`](examples/opencv/track.py) for an example.

All rectangle co-ordinates are relative to the image: zero means top and left, one means bottom and right.
If the object extends beyond the image boundary, ground-truth rectangles are clipped to \[0, 1\].

### Task format

A tracking "task" consists of the initial and final frame numbers and the rectangle that defines the target in the initial frame.
A collection of tracking tasks are specified in a single CSV file (e.g. [`dataset/tasks/dev.csv`](dataset/tasks/dev.csv)) with the following fields.

* `video_id`: (string) Name of video.
* `object_id`: (string) Name of object within the video.
* `init_frame`: (integer) Frame in which initial rectangle is specified.
* `last_frame`: (integer) Last frame in which tracker is required to make a prediction (inclusive).
* `xmin`, `xmax`, `ymin`, `ymax`: (float) Rectangle in the initial frame. Co-ordinates are relative to the image: zero means top and left, one means bottom and right.

A tracker should output predictions for frames `init_frame` + 1, `init_frame` + 2, ..., `last_frame`.

The function `oxuva.load_dataset_tasks_csv` will return a `VideoObjectDict` of `Task`s from such a file.

### Annotation format

A track "annotation" gives the ground-truth path of an object.
This can be used for training and evaluating trackers.
The annotation includes the class, but this information is not provided for a "task", and thus will not be available at testing time.

A collection of track annotations are specified in a single CSV file (e.g. [`dataset/annotations/dev.csv`](dataset/annotations/dev.csv)) with the following fields.

* `video_id`: (string) Name of video.
* `object_id`: (string) Name of object within the video.
* `class_id`: (integer) Index of object class. Matches YTBB.
* `class_name`: (string) Name of object class. Matches YTBB.
* `contains_cuts`: (string) Either `true`, `false` or `unknown`.
* `always_visible`: (string) Either `true`, `false` or `unknown`.
* `frame_num`: (integer) Frame of current annotation.
* `object_presence`: (string) Either `present` or `absent`.
* `xmin`, `xmax`, `ymin`, `ymax`: (float) Rectangle in the current frame if present, otherwise it should be ignored.

The function `oxuva.load_dataset_annotations_csv` will return a `VideoObjectDict` of track annotation dict from such a file.
The functions `oxuva.make_track_label` and `oxuva.make_frame_label` are used to construct track annotation dicts.
The function `oxuva.make_task_from_track` converts a track annotation into a tracking task with ground-truth labels.

### Prediction format

The predictions for a tracker are specified with one file per object.
The directory structure must be:
```
predictions/{subset}/{tracker}/{video}_{object}.csv
```
The fields of this CSV file are:

* `video_id`: (string) Name of video.
* `object_id`: (string) Name of object within the video.
* `frame_num`: (integer) Frame of current annotation.
* `present`: (string) Either `present` or `absent`.
* `score`: (float) Number that represents confidence of object presence.
* `xmin`, `xmax`, `ymin`, `ymax`: (float) Rectangle in the current frame if present, otherwise it is ignored.

The score is only used for diagnostics, it does not affect the main evaluation of the tracker.
If the object is predicted `absent`, then the score and the rectangle will not be used.

## How to generate the plots

Obtain the up-to-date predictions of other trackers on the `dev` set from [this Google drive directory](https://drive.google.com/drive/folders/1dTZE4BrHIbLx0QiPkGLfGcqryYPKdveT).
These files can be downloaded on the command line using [`gdrive`](https://github.com/prasmussen/gdrive).
Extract the predictions in the repository directory.
```bash
cd long-term-tracking-benchmark/
tar -xzf predictions_dev.tgz
```

Modify the `workspace/trackers_open.json` and `workspace/trackers_constrained.json` files to include the desired trackers.

To generate the table of results:
```bash
cd workspace/
python ../scripts/analyze.py table -v --data=dev --challenge=all
```
The results table will be written to `analysis/dev/all/table.txt` in CSV format.
Use `--help` to discover the optional flags.
For example, you can use `--iou_thresholds 0.1 0.3 0.5 0.7 0.9` to generate the results with a wider range of thresholds.

Similarly, to generate the main figure, use:
```bash
python ../scripts/analyze.py plot_tpr_tnr -v --data=dev --challenge=all
```

To generate all tables and plots use the `analyze_all.sh` script in `workspace/`.
You can provide optional extra arguments:
```bash
bash analyze_all.sh -v
```

**Beware:** If you update the predictions, it may be necessary to wipe the `workspace/cache/analyze/` directory, or simply use the `--ignore_cache` flag.

## How to add your own tracker to the evaluation

Copy the output of your tracker to `predictions/dev/my-tracker/`.
There should be one file per object with the filename `{video}_{object}.csv`.

Create an entry for your tracker in one of the `trackers_{challenge}.json` files:
```json
  "my-tracker": "My Tracker",
```

If your tracker uses only the data in `annotations/ytbb_train_constrained.csv` and `annotations/dev.csv` for development, then you may add it to `trackers_constrained.json`.
Otherwise, you _must_ add your tracker to `trackers_open.json`.
**Note:** Development includes pre-training, validation and hyper-parameter search in addition to training.
For example, SINT uses pre-trained weights and SiamFC is trained from scratch on ImageNet VID.
Hence they are both in the "open" challenge.
