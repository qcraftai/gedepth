#! /bin/bash
set -eo pipefail

# 1. python path
PROJ_DIR="/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox"
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# . "${PROJ_DIR}/tools/aml_training/base.sh"

# echo $CURR_DIR
# cd $CURR_DIR
# cd ../../
# echo $(pwd)
cd ${PROJ_DIR}

# LINK_REPO=${PROJ_DIR}/mmdet3d
# if [ -d "${LINK_REPO}" ]; then
#   rm LINK_REPO
# fi
# rm ${LINK_REPO}
# ln -s ${PROJ_DIR}/detection3d/mmdet3d ${LINK_REPO}

# 2. check pytorch
python3 -c "import torch; print('torch.__version__ : {0}'.format(torch.__version__))"
python3 -c "import torch; print('torch.distributed.is_available() : {0}'.format(torch.distributed.is_available()))"
python3 -c "import torch; print('torch.version.cuda : {0}'.format(torch.version.cuda))"
python3 -c "import torch; print('torch.backends.cudnn.version() : {0}'.format(torch.backends.cudnn.version()))"

# sudo chmod a+w /mnt/vepfs/ML/Users/robin/qcraft/offboard/ml/pytorch/model_factory_v2/INFO

export PYTHONPATH=${PROJ_DIR} # FEN distributed training
CONFIG="/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/configs/depthformer/depthformer_swinl_22k_w7_kitti_dynamic_pe_light_att_with_ignore_smooth_pe_with_ignore_two_branch_all_k_gt_softargmax_index_100_slope_loss_01.py"

GPUS_NUM=8
PORT=${PORT:-29500}

# info "config: " ${CONFIG}
# info "GPUS_NUM: " ${GPUS_NUM}
# info "WORKDIR: " ${WORKDIR}
# info "PORT: " ${PORT}

# # clean work dir
# if [ -d $WORKDIR ]; then
#   info "${WORKDIR} exist, remove it!"
#   rm -rf $WORKDIR
# fi

# torchrun --nproc_per_node=$GPUS_NUM --master_port=$PORT \
#   tools/train.py --config $CONFIG --work-dir ${WORKDIR} --launcher pytorch # ${@:4}

# 3. launch training
OMP_NUM_THREADS=4 torchrun --nproc_per_node $MLP_WORKER_GPU \
  --master_addr $MLP_WORKER_0_HOST \
  --master_port $MLP_WORKER_0_PORT \
  --node_rank $MLP_ROLE_INDEX \
  --nnodes=$MLP_WORKER_NUM tools/train.py \
  $CONFIG \
  --launcher=pytorch
  # --work-dir=${WORKDIR} --launcher=pytorch

# popd > /dev/null
