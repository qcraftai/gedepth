# Training and Evaluation

## KITTI
After preparing data, run the folloing script to train the model. By default, all models are trained on 8 GPUs.

### Depthformer-Vanilla
```
bash tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti_GE_Vanilla.py 8
```

### Depthformer-Adaptive
```
bash tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti_GE_Adaptive.py 8
```
