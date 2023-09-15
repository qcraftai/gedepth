#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PORT=${PORT:-29547}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/misc/visualize_point-cloud_kitti_gt_pe_pred.py $CONFIG $CHECKPOINT --output-dir point-cloud-dynamci-pe
