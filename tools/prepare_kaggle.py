import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import mmcv
import pycocotools.mask as mask_utils
from tqdm import tqdm

CATEGORIES = ('shsy5y', 'astro', 'cort')
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IMG_HEIGHT = 520
IMG_WIDTH = 704


def init_coco():
    return {
        'info': {},
        'categories':
            [{
                'id': idx,
                'name': cat,
            } for cat, idx in CAT2IDX.items()]
    }


def krle2mask(rle, height, width):
    s = rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(height, width)  # Needed to align to RLE direction


def df2coco(df):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    for img_name, img_df in tqdm(df.groupby('id'), total=df['id'].nunique()):
        img_info = dict(
            id=img_id,
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            file_name=f'{img_name}.png',
        )
        for kaggle_rle, cell_type in zip(
            img_df['annotation'], img_df['cell_type']
        ):
            mask = krle2mask(kaggle_rle, IMG_HEIGHT, IMG_WIDTH)
            mask = np.asfortranarray(mask)
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode()
            bbox = mask_utils.toBbox(rle).tolist()
            ann_info = dict(
                id=ann_id,
                image_id=img_id,
                category_id=CAT2IDX[cell_type],
                iscrowd=0,
                segmentation=rle,
                area=bbox[2] * bbox[3],
                bbox=bbox,
            )
            ann_infos.append(ann_info)
            ann_id += 1
        img_infos.append(img_info)
        img_id += 1

    coco = init_coco()
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


def main():
    df = pd.read_csv('../data/train.csv')
    dtrainval = df2coco(df)
    mmcv.dump(dtrainval, '../data/dtrainval.json')

    all_samples = np.array(sorted(set(df['sample_id'])))
    sample2celltype = dict(zip(df['sample_id'], df['cell_type']))
    cell_types = [sample2celltype[_] for _ in all_samples]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(skf.split(all_samples, cell_types))
    for fold, (train_inds, val_inds) in enumerate(splits):
        train_samples = all_samples[train_inds]
        train_df = df[df['sample_id'].isin(train_samples)]
        train_coco = df2coco(train_df)
        mmcv.dump(train_coco, f'../data/dtrain_g{fold}.json')

        val_samples = all_samples[val_inds]
        val_df = df[df['sample_id'].isin(val_samples)]
        val_coco = df2coco(val_df)
        mmcv.dump(val_coco, f'../data/dval_g{fold}.json')

        # break


if __name__ == '__main__':
    main()
