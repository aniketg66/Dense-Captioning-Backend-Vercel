from mmdet.registry import MODELS
from .custom_cascade import CustomCascadeWithMeta
from ..heads.custom_heads import FCHead, RegHead

# Register the custom modules
MODELS.register_module(module=CustomCascadeWithMeta)
MODELS.register_module(module=FCHead)
MODELS.register_module(module=RegHead) 