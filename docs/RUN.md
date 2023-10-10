# Training

After preparing the data, run the following script to train the model. By default, all models are trained on 8 GPUs.

## KITTI

### DepthFormer with GEDepth-Vanilla
```
bash tools/dist_train.sh configs/depthformer/depthformer_v.py 8
```

### DepthFormer with GEDepth-Adaptive
```
bash tools/dist_train.sh configs/depthformer/depthformer_a.py 8
```

## DDAD

### DepthFormer with GEDepth-Vanilla
```
bash tools/dist_train.sh configs/depthformer/depthformer_v_ddad.py 8
```

### DepthFormer with GEDepth-Adaptive
```
bash tools/dist_train.sh configs/depthformer/depthformer_a_ddad.py 8
```

# Evaluation

## KITTI

### DepthFormer with GEDepth-Vanilla

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer_v.py  ckpt/depthformer_v.pth  8
```

### DepthFormer with GEDepth-Adaptive

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer_a.py  ckpt/depthformer_a.pth  8
```

## DDAD

### DepthFormer with GEDepth-Vanilla

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer_v_ddad.py  ckpt/depthformer_v_ddad.pth  8
```

### DepthFormer with GEDepth-Adaptive

```bash
bash tools/dist_test.sh  configs/depthformer/depthformer_a_ddad.py  ckpt/depthformer_a_ddad.pth  8
```
