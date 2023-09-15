#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=-1
CONFIG=$1
CHECKPOINT=$2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval abs_rel