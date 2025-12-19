# This file makes the custom_models directory a Python package 

print("🔧 [PLUGIN] legend_match_swin.custom_models __init__ loaded")

# Import custom models to register them with MMDetection
from .custom_cascade_with_meta import CustomCascadeWithMeta
from .custom_faster_rcnn_with_meta import CustomFasterRCNNWithMeta
from .flexible_load_annotations import FlexibleLoadAnnotations
from .mask_filter import MaskFilter

# Import modules required by config files
from . import register
from . import custom_hooks
from . import progressive_loss_hook
from . import flexible_load_annotations

print("✅ [PLUGIN] legend_match_swin.custom_models package ready for import!")
print(f"🔧 [PLUGIN] Registered models: CustomCascadeWithMeta, CustomFasterRCNNWithMeta, FlexibleLoadAnnotations, MaskFilter") 