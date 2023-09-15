#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --show-dir /mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/nuscenes_vis
 