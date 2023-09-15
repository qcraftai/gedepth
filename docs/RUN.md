# Training

## KITTI

After preparing data, run the folloing script to train the model. By default, all models are trained on 8 GPUs.

### Depthformer-Vanilla
```
bash tools/dist_train.sh configs/depthformer/depthformer_v.py 8
```

### Depthformer-Adaptive
```
bash tools/dist_train.sh configs/depthformer/depthformer_a.py 8
```

# Evaluation

## KITTI

### Depthformer-Vanilla

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer-v.py  ckpt/depthformer_v.pth  8
```


### Depthformer-Adaptive

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer-a.py  ckpt/depthformer_a.pth  8
```