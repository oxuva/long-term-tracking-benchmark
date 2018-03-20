#!/bin/bash

repo=..

set -x

for challenge in open constrained all
do
    for data in dev test devtest
    do
        flags="--data=$data --challenge=$challenge $@"
        python $repo/scripts/analyze.py table $flags --iou_thresholds 0.3 0.5 0.7 || exit 1
        python $repo/scripts/analyze.py plot_tpr_tnr $flags || exit 1
        python $repo/scripts/analyze.py plot_tpr_tnr_intervals $flags || exit 1
        python $repo/scripts/analyze.py plot_tpr_time $flags --max_time=300 || exit 1
        python $repo/scripts/analyze.py plot_present_absent $flags
    done
done
