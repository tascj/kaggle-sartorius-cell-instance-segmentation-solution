from mmdet.models.backbones import SwinTransformer
from mmseg.models import BACKBONES

BACKBONES.register_module(name='SwinTransformer', module=SwinTransformer, force=True)
