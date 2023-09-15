## GEDepth: Ground Embedding for Monocular Depth Estimation
<p align='left'>
  <img src='docs/example.gif' width='830'/>
</p>

[Xiaodong Yang](https://xiaodongyang.org/), [Zhuang Ma](), [Zhiyu Ji](https://github.com/RobinhoodKi), [Zhe Ren]() <br>
GEDepth: Ground Embedding for Monocular Depth Estimation, ICCV 2023 <br>
[[Paper]]() 

## Get Started
### Installation
Please refer to [INSTALL](docs/install.md) for the detail

### Data Preparation 

Please follow the instructions in [DATA](docs/DATA.md)


### Training and Evaluation

Please follow the instructions in [RUN](docs/RUN.md)



### Main Results

| Model |  Abs_Rel | Sq Rel |  RMSE | Checkpoint | 
| ------| -----| ------- | ------ | -------------| 
 | Depthformer-Vanilla | 0.049 | 0.144	| 2.061| [[Google Drive]](https://drive.google.com/drive/folders/1XQRl7AtSBBIPoXtZOh87M_LG0iAJPDl_?usp=sharing)
| Depthformer-Adaptive| 0.048 | 0.142| 2.044|[[Google Drive]](https://drive.google.com/drive/folders/1XQRl7AtSBBIPoXtZOh87M_LG0iAJPDl_?usp=sharing)





## Citation
 Please cite the following paper if this repo helps your research:
```
@inproceedings{yang2023gedepth,
  title={GEDepth: Ground Embedding for Monocular Depth Estimation},
  author={Yang, Xiaodong and Ma, Zhuang and Ji, Zhiyu and Ren, Zhe},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## License
Copyright (C) 2023 QCraft. All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only. For commercial use, please contact [business@qcraft.ai](business@qcraft.ai).