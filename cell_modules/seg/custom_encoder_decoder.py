import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmseg.models.segmentors import EncoderDecoder, BaseSegmentor
from mmseg.models.builder import build_segmentor, SEGMENTORS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask


@SEGMENTORS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    def simple_test(self, img, img_meta, bbox, rescale=True):
        """Simple test with single image."""
        assert len(img_meta) == 1 and len(bbox) == 1

        seg_logit = self.inference(
            img, img_meta, rescale
        )  # softmax in inference
        seg_prob = seg_logit[:, 1]

        # map (prob, box) back to original img
        masks = seg_prob.unsqueeze(0)
        img_h, img_w, _ = img_meta[0]['ori_shape']
        boxes = torch.stack(bbox[0])
        seg_prob, _ = _do_paste_mask(masks, boxes, img_h, img_w, False)
        seg_pred = seg_prob >= 0.5
        confidence = seg_prob[seg_pred].mean()

        seg_pred = seg_pred.cpu().numpy()[0]
        confidence = confidence.item()
        return [dict(seg_pred=seg_pred, confidence=confidence)]

    def aug_test(self, imgs, img_metas, bbox, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_prob = seg_logit[:, 1]

        # map (prob, box) back to original img
        masks = seg_prob.unsqueeze(0)
        img_h, img_w, _ = img_metas[0][0]['ori_shape']
        boxes = torch.stack(bbox[0])
        seg_prob, _ = _do_paste_mask(masks, boxes, img_h, img_w, False)
        seg_pred = seg_prob >= 0.5
        confidence = seg_prob[seg_pred].mean()

        seg_pred = seg_pred.cpu().numpy()[0]
        confidence = confidence.item()
        return [dict(seg_pred=seg_pred, confidence=confidence)]


@SEGMENTORS.register_module()
class EnsembleSegmentor(BaseSegmentor):
    def __init__(
        self,
        configs,
        checkpoints,
        weights,
        train_cfg=None,
        test_cfg=None,
        **kwargs
    ):
        super(EnsembleSegmentor, self).__init__()
        self.models = nn.ModuleList()
        for config, checkpoint in zip(configs, checkpoints):
            model = build_segmentor(config)
            load_checkpoint(model, checkpoint, strict=False, map_location='cpu')
            self.models.append(model)
        self.weights = weights

        self.test_cfg = test_cfg

    def ensemble_inference(self, img, img_meta, rescale):
        seg_logits = []
        for model, weight in zip(self.models, self.weights):
            seg_logit = model.inference(img, img_meta, rescale)
            seg_logits.append(seg_logit * weight)
        seg_logit = sum(seg_logits) / sum(self.weights)
        return seg_logit

    def simple_test(self, img, img_meta, bbox, rescale=True):
        assert len(img_meta) == 1 and len(bbox) == 1
        seg_logit = self.ensemble_inference(img, img_meta, rescale)
        seg_prob = seg_logit[:, 1]

        # map (prob, box) back to original img
        masks = seg_prob.unsqueeze(0)
        img_h, img_w, _ = img_meta[0]['ori_shape']
        boxes = torch.stack(bbox[0])
        seg_prob, _ = _do_paste_mask(masks, boxes, img_h, img_w, False)
        seg_pred = seg_prob >= 0.5
        confidence = seg_prob[seg_pred].mean()

        seg_pred = seg_pred.cpu().numpy()[0]
        confidence = confidence.item()
        return [dict(seg_pred=seg_pred, confidence=confidence)]

    def aug_test(self, imgs, img_metas, bbox, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.ensemble_inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.ensemble_inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_prob = seg_logit[:, 1]

        # map (prob, box) back to original img
        masks = seg_prob.unsqueeze(0)
        img_h, img_w, _ = img_metas[0][0]['ori_shape']
        boxes = torch.stack(bbox[0])
        seg_prob, _ = _do_paste_mask(masks, boxes, img_h, img_w, False)
        seg_pred = seg_prob >= 0.5
        confidence = seg_prob[seg_pred].mean()

        seg_pred = seg_pred.cpu().numpy()[0]
        confidence = confidence.item()
        return [dict(seg_pred=seg_pred, confidence=confidence)]

    def encode_decode(self, img, img_metas):
        raise NotImplementedError()

    def extract_feat(self, imgs):
        raise NotImplementedError()

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError()
