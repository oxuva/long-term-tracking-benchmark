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
