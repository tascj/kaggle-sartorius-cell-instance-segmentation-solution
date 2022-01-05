import os.path as osp
from collections import defaultdict

import numpy as np
import mmcv
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from mmseg.datasets import DATASETS, CustomDataset
from mmseg.datasets.pipelines import Compose

import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps
from mmdet.datasets import build_dataset


def to_mask(mask_ann, img_h, img_w):

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(mask_ann, img_h, img_w)
        rle = mask_util.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = mask_util.decode(rle)
    return mask


@DATASETS.register_module()
class GTBBoxDataset(CustomDataset):
    def __init__(
        self, ann_file, img_dir, pipeline, helper_dataset, test_mode=None
    ):
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        box_infos = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(img_id))
            for ann in anns:
                x1, y1, w, h = ann['bbox']
                box_infos.append(
                    dict(
                        filename=img_info['file_name'],
                        bbox=[x1, y1, x1 + w, y1 + h],
                        segmentation=ann['segmentation'],
                        score=1.0,  # GT
                        category_id=ann['category_id'],
                        img_id=img_id,
                        height=img_info['height'],
                        width=img_info['width'],
                    )
                )

        self.box_infos = box_infos
        self.coco = coco
        self.img_dir = img_dir
        self.pipeline = Compose(pipeline)

        self.helper_dataset = build_dataset(helper_dataset)

    def __len__(self):
        return len(self.box_infos)

    def __getitem__(self, idx):
        box_info = self.box_infos[idx]
        filename = box_info['filename']
        img = mmcv.imread(osp.join(self.img_dir, filename))
        mask = to_mask(
            box_info['segmentation'], box_info['height'], box_info['width']
        )

        results = dict(
            ori_filename=filename,
            filename=filename,
            img=img,
            ori_shape=img.shape,
            shape=img.shape,
            gt_semantic_seg=mask,
            seg_fields=['gt_semantic_seg'],
            bbox=box_info['bbox']
        )
        return self.pipeline(results)

    def pre_eval(self, preds, indices):
        assert len(preds) == len(indices)
        results = []
        for pred, idx in zip(preds, indices):
            mask = np.asfortranarray(pred)
            rle = mask_util.encode(mask)
            results.append(rle)
        return results

    def evaluate(self, results, metric=None, **kwargs):
        assert len(results) == len(self)
        img_results = defaultdict(list)
        for box_info, result in zip(self.box_infos, results):
            img_id = box_info['img_id']
            bbox = np.array(box_info['bbox'])
            mask = result
            label = box_info['category_id']
            img_results[img_id].append((bbox, mask, label))

        mmdet_style_results = []
        num_classes = len(self.helper_dataset.CLASSES)
        for img_id in self.coco.getImgIds():
            img_result = img_results[img_id]
            bboxes = np.array([_[0] for _ in img_result])
            masks = np.array([_[1] for _ in img_result])
            labels = np.array([_[2] for _ in img_result])
            bboxes = np.hstack([bboxes.reshape(-1, 4), np.ones((len(bboxes), 1), dtype=np.float32)])
            mmdet_bbox_result = []
            mmdet_segm_result = []
            for cat_id in range(num_classes):
                inds = labels == cat_id
                mmdet_bbox_result.append(bboxes[inds])
                mmdet_segm_result.append(masks[inds])
            mmdet_style_results.append((mmdet_bbox_result, mmdet_segm_result))

        return self.helper_dataset.evaluate(
            mmdet_style_results,
            metric=['bbox', 'segm'],
            classwise=True,
            proposal_nums=(100, 300, 2000),
            logger=kwargs.get('logger', None)
        )


def assign_gt(pr_bboxes, gt_bboxes):
    ious = bbox_overlaps(
        torch.from_numpy(pr_bboxes), torch.from_numpy(gt_bboxes)
    ).numpy()
    return ious.argmax(1)


@DATASETS.register_module()
class PredBBoxDataset(CustomDataset):
    def __init__(
        self,
        ann_file,
        pred_file,
        img_dir,
        pipeline,
        helper_dataset,
        test_mode=None,
        score_thr=-1,
        mask_rerank=False
    ):
        coco = COCO(ann_file)
        preds = mmcv.load(pred_file)
        img_ids = coco.getImgIds()
        assert len(img_ids) == len(preds)
        box_infos = []
        for img_id, pred in zip(img_ids, preds):
            img_info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(img_id))
            gt_bboxes = np.array([ann['bbox'] for ann in anns])
            gt_bboxes[:, 2:] += gt_bboxes[:, :2]  # xywh2xyxy
            for cat_id, cat_pred in enumerate(pred):
                pr_bboxes = cat_pred[:, :4]
                gt_inds = assign_gt(pr_bboxes, gt_bboxes)  # TODO: how?
                for (x1, y1, x2, y2, score), gt_ind in zip(cat_pred, gt_inds):
                    if score < score_thr:
                        continue
                    box_infos.append(
                        dict(
                            filename=img_info['file_name'],
                            bbox=[x1, y1, x2, y2],
                            segmentation=anns[gt_ind]['segmentation'],
                            score=score,
                            category_id=cat_id,
                            img_id=img_id,
                            height=img_info['height'],
                            width=img_info['width'],
                        )
                    )

        self.box_infos = box_infos
        self.coco = coco
        self.img_dir = img_dir
        self.score_thr = score_thr
        self.mask_rerank = mask_rerank
        self.pipeline = Compose(pipeline)

        self.helper_dataset = build_dataset(helper_dataset)

    def __len__(self):
        return len(self.box_infos)

    def __getitem__(self, idx):
        box_info = self.box_infos[idx]
        filename = box_info['filename']
        img = mmcv.imread(osp.join(self.img_dir, filename))
        mask = to_mask(
            box_info['segmentation'], box_info['height'], box_info['width']
        )

        results = dict(
            ori_filename=filename,
            filename=filename,
            img=img,
            ori_shape=img.shape,
            shape=img.shape,
            gt_semantic_seg=mask,
            seg_fields=['gt_semantic_seg'],
            bbox=box_info['bbox']
        )
        return self.pipeline(results)

    def pre_eval(self, preds, indices):
        assert len(preds) == len(indices)
        results = []
        for pred, idx in zip(preds, indices):
            mask = np.asfortranarray(pred['seg_pred'])
            rle = mask_util.encode(mask)
            results.append((rle, pred['confidence']))
        return results

    def evaluate(self, results, **kwargs):
        assert len(results) == len(self)
        img_results = defaultdict(list)
        for box_info, (result, confidence) in zip(self.box_infos, results):
            img_id = box_info['img_id']
            bbox = np.array(box_info['bbox'])
            mask = result
            label = box_info['category_id']
            if self.mask_rerank:
                score = box_info['score'] * confidence
            else:
                score = box_info['score']
            img_results[img_id].append((bbox, mask, label, score))

        mmdet_style_results = []
        num_classes = len(self.helper_dataset.CLASSES)
        for img_id in self.coco.getImgIds():
            img_result = img_results[img_id]
            bboxes = np.array([_[0] for _ in img_result])
            masks = np.array([_[1] for _ in img_result])
            labels = np.array([_[2] for _ in img_result])
            scores = np.array([_[3] for _ in img_result])
            bboxes = np.hstack([bboxes.reshape(-1, 4), scores.reshape(-1, 1)])
            mmdet_bbox_result = []
            mmdet_segm_result = []
            for cat_id in range(num_classes):
                inds = labels == cat_id
                mmdet_bbox_result.append(bboxes[inds])
                mmdet_segm_result.append(masks[inds])
            mmdet_style_results.append((mmdet_bbox_result, mmdet_segm_result))

        return self.helper_dataset.evaluate(
            mmdet_style_results,
            metric=['bbox', 'segm'],
            classwise=True,
            proposal_nums=(100, 300, 2000),
            logger=kwargs.get('logger', None)
        )
