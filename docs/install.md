## Prerequisites
- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation
We run experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.7, and Ubuntu 20.04. 

Use Anaconda to create conda environment:

```shell
conda create -n GE python=3.7
conda activate GE
```

Install PyTorch:
```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Install MMCV and our toolbox:
```shell
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/Ma-Zhuang/GEDepth.git
cd GEDepth
pip install -e .
```

For training, install TensorBoard:
```shell
pip install future tensorboard
```

More information about installation can be found in docs of MMSegmentation (see [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)).
