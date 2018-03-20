#!/bin/bash

# Change into directory of script.
cd "$( dirname "${BASH_SOURCE[0]}" )"

python track.py -v ../../dataset/ ../../predictions/ --data=dev --tracker=MEDIANFLOW
