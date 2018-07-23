# OxUvA long-term tracking benchmark [ECCV'18]
**Beware**. This document is a work in progress.

This repository provides Python code to measure the quality of a tracker's predictions and generate all figures in the paper.
The following sections provide instructions for each stage.

1. [Obtain the data](#1-obtain-the-data)
2. [Set up the environment](#2-set-up-the-environment)
3. [Run your tracker](#3-run-your-tracker)
4. [Submit to the evaluation server](#4-submit-to-the-evaluation-server)
5. [Generate the plots for a paper](#5-generate-the-plots-for-a-paper)
6. [Add your tracker to the results repo](#6-add-your-tracker-to-the-results-repo)

The challenge is split into two tracks: "constrained" and "open".
To be eligible for the "constrained" challenge, a tracker must use *only* the data in `annotations/ytbb_train_constrained.csv` and `annotations/dev.csv` for development.
All other trackers must enter the "open" challenge.
**Note:** Development includes pre-training, validation and hyper-parameter search in addition to training.
For example, SINT uses pre-trained weights and SiamFC is trained from scratch on ImageNet VID.
Hence they are both in the "open" challenge.

The results of all *citeable* trackers are maintained in a [results repo](https://github.com/oxuva/long-term-tracking-results/).
This repo should be used for comparison against state-of-the-art.
It is updated periodically according to a [schedule](TODO).


## 1. Obtain the data

The ground-truth labels for the dev set can be found *in this repository* in [`dataset/annotations`](dataset/annotations).
The tracker initialization for the dev *and* test sets can be found in [`dataset/tasks`](dataset/tasks).

To obtain the images, fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSepA_sLCMrqnZXBPnZFNmggf-MdEGa2Um-Q7pRGQt4SxvGNeg/viewform) and then download `images_dev.tar` and `images_test.tar`.
Extract these archives in `dataset/`.

The structure of `dataset/` should be:
```
dataset/images/{subset}/{video}/{frame:06d}.jpeg
dataset/tasks/{subset}.csv
dataset/annotations/{subset}.csv
```
where `{subset}` is either `dev` or `test`, `{video}` is the video ID e.g. `vid0042`, and `{frame:06d}` is the frame number e.g. `002934`.

**Note:** Only the annotations for the *dev* set are public.
These can be useful for diagnosing failures and hyperparameter search.
For the *test* set, the annotations are secret and trackers can only be assessed via the evaluation server (explained later).

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


## 2. Set up the environment

To run the code in this repository, it is necessary to install the Python libraries listed in [`requirements.txt`](requirements.txt).
To install these dependencies using `pip` (perhaps in a virtual environment):
```bash
pip install -r requirements.txt
```

You must also add the parent directory of `oxuva/` to `PYTHONPATH` to be able to import the `oxuva` package.
```bash
export PYTHONPATH="/absolute/path/to/long-term-tracking-benchmark/python:$PYTHONPATH"
```
Alternatively, for convenience, you can `source` the script `pythonpath.sh` in `bash`:
```bash
source relative/path/to/long-term-tracking-benchmark/pythonpath.sh
```


## 3. Run your tracker

**Note:** Unlike the VOT or OTB toolkits, our toolkit does not execute your tracker.
Your tracker should output all predictions in the format described below.
For Python trackers, we provide the utility functions `oxuva.load_dataset_tasks_csv` and `oxuva.dump_predictions_csv` to make this easy.
See [`examples/opencv/track.py`](examples/opencv/track.py) for an example.

All rectangle co-ordinates are relative to the image: zero means top and left, one means bottom and right.
If the object extends beyond the image boundary, ground-truth rectangles are clipped to \[0, 1\].

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


## 4. Submit to the evaluation server

Since the annotations for the test set are secret, in order to evaluate your tracker and produce plots similar to the one in our paper you need to submit the raw prediction csv files to our [evaluation server (link todo)](https://competitions.codalab.org/competitions/19529#participate), hosted on CodaLab.

First, create a CodaLab account (if you do not already have one) and request to join the OxUvA competition.
Note that the CodaLab account is per human, not per tracker.
Do *not* create a username for your tracker.
The name of your tracker will appear when you add it to the results repo.
The organisers will manually review your request to join the competition.

To submit the results, create a zip archive containing all predictions in CSV format (as described above).
There should be one file per object with the filename `{video}_{object}.csv`.
It doesn't matter whether the CSV files are contained at the root level of the zip archive or below a single subdirectory of any name.
If a submission encounters an error (for example, a missing prediction file), you will be able to view the verbose error log, and the submission will not count towards your quota.

You can upload your predictions for the dev set to confirm that your predictions are in the correct format.

Once the submission has been successful, you can download the generated output files.
These will be used to generate the plots and submit to the results repo.


## 5. Generate the plots for a paper

First, clone the results repo.
```bash
git clone https://github.com/oxuva/long-term-tracking-results.git
cd long-term-tracking-results
```
This repo contains several snapshots of the past state-of-the-art as git tags.
The tag `eccv18` indicates the set of methods in our original paper, and successive tags are of the form `{year}-{month:02d}`, for example:
```bash
git checkout 2018-07
```
You can state in your paper which tag you are comparing against.

Add an entry for your tracker to `trackers.json`.
You must specify a human-readable name for your tracker, and whether your tracker is eligible for the constrained-data challenge.
```json
    "tracker_id": {"name": "Tracker Name", "constrained": false},
```
Use `python -m json.tool --sort-keys` to standardize the formatting and order of this file.

For the test set, copy the `iou_0dx.json` files returned by the evaluation server to the directory
```assess/test/{tracker_id}/```
The script `oxuva.tools.analyze` will try to load this summary of the tracker assessment from these files before attempting to read the complete predictions of the tracker and the ground-truth annotations.

For the dev set, you may follow the same procedure as above.
However, it is possible to evaluate your tracker's predictions locally, without using the evaluation server.
To do this, put the CSV files of your tracker's predictions (that is, the input to the evaluation server) in the directory
```predictions/dev/{tracker_id}/```
The script will generate the corresponding files in the `assess/` directory.
Note that if you update the predictions, you should erase the corresponding files in the `assess/` directory, or use `--ignore_cache`.
If desired, the predictions of other trackers on the *dev* set are available from Google Drive (TODO).
Please do not publish your predictions on the *test* set, as it may enable someone to construct an approximate ground-truth using a consensus method.

To generate all plots and tables:
```bash
bash analyze_all.sh --data=test --challenge=open --loglevel=warning
```

To just generate the table of results:
```bash
python -m oxuva.tools.analyze table --data=dev --challenge=open
```
The results table will be written to `analysis/dev/open/table.csv`.
Use `--help` to discover the optional flags.
For example, you can use `--iou_thresholds 0.1 0.3 0.5 0.7 0.9` to generate the results with a wider range of thresholds.

Similarly, to just generate the main figure, use:
```bash
python -m oxuva.tools.analyze plot_tpr_tnr --data=dev --challenge=open
```


## 6. Add your tracker to the results repo

- Need a paper with results on it (because we need a reference and description)
- Do a PR
- Does not matter if it is at the top, it allows to other people to cite your work. 
- (Please do not share results post-codalab pre-pr online.)

