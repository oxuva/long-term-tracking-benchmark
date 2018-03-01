# OxUvA long-term tracking benchmark
## How to get started


## How to run scripts

It is necessary to add the parent of the `oxuva` directory to the `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=path/to/long-term-tracking-benchmark:$PYTHONPATH
```
You can then run the scripts from anywhere, although we recommend to use `workspace` as the working directory.
For example, to run the analysis script:
```bash
cd path/to/long-term-tracking-benchmark/workspace
python -m oxuva.analyze --help
```

## How to generate the table of results

Download and extract the predictions to `workspace/predictions/dev/`:
```bash
cd path/to/workspace
tar -xzf path/to/predictions-dev.tgz
```
Then run the `oxuva.analyze` script:
```bash
python -m oxuva.analyze table --challenge=all --verbose
```
The results table will be printed to `stdout` in CSV format.

Use `oxuva.analyze table --help` to discover the optional flags.
For example, you can use `--iou_thresholds 0.1 0.3 0.5 0.7 0.9` to generate the results with a wider range of thresholds (the default is `0.3 0.5 0.7`).

## How to generate the figures

```bash
python -m oxuva.analyze plot --challenge=all --verbose
```

## How to add your own tracker

Create a directory `workspace/dev/my-tracker/`.
Add one file per object track with the filename `video_object.csv`.

Create an entry for your tracker in one of the `trackers_CHALLENGE.json` files:
```json
  "my-tracker": "Tracker Name",
```
If your tracker uses only the data in `ytbb_train_constrained.csv` and `dev.csv` for development (this includes training *as well as* pre-training, validation and hyper-parameter search), then you may add it to `trackers_constrained.json`.
Otherwise, you _must_ add your tracker to `trackers_open.json`.

To generate the table of results, follow the instructions above.
