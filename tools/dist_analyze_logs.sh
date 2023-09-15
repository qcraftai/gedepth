#!/usr/bin/env bash

JSONLOG=$1
KEYS=$2
OUT=$3
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python /mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/tools/analyze_logs.py $JSONLOG --keys $KEYS --out $OUT
