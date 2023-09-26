## Prepare datasets

It is recommended to symlink the dataset root to `$MONOCULAR-DEPTH-ESTIMATION-TOOLBOX/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

### **KITTI**

Download the offical dataset from this [link](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

Then, unzip the files into data/kitti. Remember to organizing the directory structure following instructions (Only need a few cut operations). Copy split files (whose names are started with *kitti*) in splits folder into data/kitti. Here, We utilize eigen splits following other supervised methods. The data folders structure is as follows：

```none
GEDepth
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── 2011_09_28
│   │   │   ├── ...
│   │   │   ├── 2011_10_03
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── 2011_09_26_drive_0002_sync
│   │   │   ├── ...
│   │   │   ├── 2011_10_03_drive_0047_sync
│   │   ├── kitti_eigen_train.txt
│   │   ├── kitti_eigen_test.txt

```

Finally, preprocessing the data to generate Ground-Embedding and Ground-Slope:

```shell
$ cd GEDepth
$ python tools/preprocess_data_kitti.py
```

### **DDAD**

Download the offical dataset from this [link](https://github.com/TRI-ML/DDAD).

Finally, preprocessing the data to generate Ground-Embedding and Ground-Slope:

```shell
$ cd GEDepth
$ python tools/preprocess_data_ddad.py
```