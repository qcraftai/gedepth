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

| Model |  Abs_Rel | Sq Rel |  RMSE | Checkpoint | Training Logs 
| ------| -----| ------- | ------ | -------------| --|
 | Depthformer-Vanilla | 0.049 | 0.145	| 2.063| [[Google Drive]](https://drive.google.com/drive/folders/1agPt7Nwecj3oX3S8WmF6wvxfDFddJSBA) | [[Google Drive]](https://drive.google.com/file/d/1faZ2_STjzlgfZB06EZ-_FpCjjd6_1FwN/view?usp=drive_link)
| Depthformer-Adaptive| xx | xx| xx|[[Google Drive]]
| BTS-Vanilla| 0.057 | 0.199| 2.476|[[Google Drive]](https://drive.google.com/drive/folders/1agPt7Nwecj3oX3S8WmF6wvxfDFddJSBA) | [[Google Drive]](https://drive.google.com/file/d/1vw_n4uAjHknEKpkXujfYpa2FYtqz-yv9/view?usp=drive_link)
| BTS-Adaptive| xx | xx| xx| [[Google Drive]]




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