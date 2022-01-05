import copy
import os

import mmcv
from tqdm import tqdm
from pycocotools.coco import COCO

CATEGORIES = (
    'shsy5y', 'a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'skov3', 'skbr3'
)
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


def init_coco():
    return {
        'info': {},
        'categories':
            [{
                'id': idx,
                'name': cat,
            } for cat, idx in CAT2IDX.items()]
    }


def to_multiclass(ann_file):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    coco = COCO(ann_file)
    for _id in tqdm(coco.getImgIds()):
        img_info = copy.deepcopy(coco.loadImgs(_id)[0])
        del img_info['original_filename']
        del img_info['url']
        img_info['id'] = img_id
        filename = img_info['file_name']
        subdir = filename.split('_')[0]
        img_info['file_name'] = os.path.join(subdir, filename)
        cat = subdir.lower()

        img_infos.append(img_info)
        for ann_info in coco.loadAnns(coco.getAnnIds(_id)):
            ann_info = copy.deepcopy(ann_info)
            ann_info['image_id'] = img_id
            ann_info['id'] = ann_id
            ann_info['category_id'] = CAT2IDX[cat]
            ann_infos.append(ann_info)
            ann_id += 1
        img_id += 1

    coco = init_coco()
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


if __name__ == '__main__':
    mmcv.dump(
        to_multiclass('../data/LIVECell_dataset_2021/livecell_coco_train.json'),
        '../data/LIVECell_dataset_2021/train_8class.json'
    )
    mmcv.dump(
        to_multiclass('../data/LIVECell_dataset_2021/livecell_coco_val.json'),
        '../data/LIVECell_dataset_2021/val_8class.json'
    )
    mmcv.dump(
        to_multiclass('../data/LIVECell_dataset_2021/livecell_coco_test.json'),
        '../data/LIVECell_dataset_2021/test_8class.json'
    )
