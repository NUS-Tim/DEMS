# DEMS: Data-Efficient Medical Segmenter
Official Pytorch Implementation for Neural Networks Submission: “Segmenting Medical Images with Limited Data”

## Preparation

Please ensure that a "checkpoint" and "dataset" folder exists in the project root directory following
```
├── project
    ├── data
    |   ├── busi
    |   |   ├── images
    |   |   ├── masks
    |   |   ├── indexes
    |   |   └── ...
    |   ├── ddti
    |   └── ...
    ├── src
    ├── checkpoint
    └── ...
```

To generate indexes for training and validation splits, execute
```
python split.py
```

Following this, you will be able to execute our DEMS for your unique application with
```
python train.py
```

## Citation
If you find our DEMS useful for your research, please cite our paper as
```
@article{liu2024segmenting,
  title={Segmenting medical images with limited data},
  author={Liu, Zhaoshan and Lv, Qiujie and Lee, Chau Hung and Shen, Lei},
  journal={Neural Networks},
  pages={106367},
  year={2024},
  publisher={Elsevier}
}
```
