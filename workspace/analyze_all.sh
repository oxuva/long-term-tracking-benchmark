#!/bin/bash

set -x

for challenge in open constrained all
do
    flags="--challenge=$challenge $@"
    python -m oxuva.tools.analyze table $flags --iou_thresholds 0.3 0.5 0.7 || exit 1
    python -m oxuva.tools.analyze plot_tpr_tnr $flags || exit 1
    python -m oxuva.tools.analyze plot_tpr_tnr_intervals $flags || exit 1
    python -m oxuva.tools.analyze plot_tpr_time $flags --max_time=300 || exit 1
    python -m oxuva.tools.analyze plot_present_absent $flags || exit 1
done
