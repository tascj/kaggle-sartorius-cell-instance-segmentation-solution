# Sartorius - Cell Instance Segmentation

https://www.kaggle.com/c/sartorius-cell-instance-segmentation

## Environment setup

Build docker image

```
bash .dev_scripts/build.sh
```

Set env variables

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```

Start a docker container
```
bash .dev_scripts/start.sh all
```

## Data preparation

1. Download competition data from Kaggle
2. Download LIVECell dataset from https://github.com/sartorius-research/LIVECell (we didn't use the data provided by Kaggle)
3. Unzip the files as follows

```
├── LIVECell_dataset_2021
│   ├── images
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── train
├── train_semi_supervised
└── train.csv
```

Start a docker container and run the following commands

```
mkdir /data/checkpoints/
python tools/prepare_livecell.py
python tools/prepare_kaggle.py
```

The results should look like the 

```
├── LIVECell_dataset_2021
│   ├── images
│   ├── train_8class.json
│   ├── val_8class.json
│   ├── test_8class.json
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── train
├── train_semi_supervised
├── checkpoints
├── train.csv
├── dtrainval.json
├── dtrain_g0.json
└── dval_g0.json
```

## Training

Download COCO pretrained YOLOX-x weights from https://github.com/Megvii-BaseDetection/YOLOX

Convert the weights

```
python tools/convert_official_yolox.py /path/to/yolox_x.pth /path/to/data/checkpoints/yolox_x_coco.pth
```

Start a docker container and run the following commands for training

```
# train detector using the LIVECell dataset
python tools/det/train.py configs/det/yolox_x_livecell.py

# predict bboxes of LIVECell validataion data
python tools/det/test.py configs/det/yolox_x_livecell.py work_dirs/yolox_x_livecell/epoch_30.pth --out work_dirs/yolox_x_livecell/val_preds.pkl --eval bbox

# finetune the detector on competition data(train split)
python tools/det/train.py configs/det/yolox_x_kaggle.py --load-from work_dirs/yolox_x_livecell/epoch_15.pth

# predict bboxes of competition data(val split)
python tools/det/test.py configs/det/yolox_x_kaggle.py work_dirs/yolox_x_kaggle/epoch_30.pth --out work_dirs/yolox_x_kaggle/val_preds.pkl --eval bbox

# train segmentor using LIVECell dataset
python tools/seg/train.py configs/seg/upernet_swin-t_livecell.py

# finetune the segmentor on competition data(train split)
python tools/seg/train.py configs/seg/upernet_swin-t_kaggle.py --load-from work_dirs/upernet_swin-t_livecell/epoch_1.pth

# predict instance masks of competition data(val split)
python tools/seg/test.py configs/seg/upernet_swin-t_kaggle.py work_dirs/upernet_swin-t_kaggle/epoch_10.pth --out work_dirs/upernet_swin-t_kaggle/val_results.pkl --eval dummy
```
