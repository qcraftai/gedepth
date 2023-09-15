#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-38423}
cd /mnt/vepfs/ML/ml-users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/
export PATH=/root/miniconda3/envs/GE_v2/bin:$PATH
pip install IPython
pip install open3d
pip install -e .
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
