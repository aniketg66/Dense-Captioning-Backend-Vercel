from mmdet.registry import MODELS, DATASETS, TRANSFORMS, HOOKS
from .custom_heads import FCHead, RegHead, DataSeriesHead
from .custom_cascade_with_meta import CustomCascadeWithMeta
from .custom_dataset import (
    ChartDataset, RobustLoadImageFromFile, CreateDummyImg,
    ClampBBoxes, SetScaleFactor, EnsureScaleFactor, SetInputs, CustomPackDetInputs
)
from .flexible_load_annotations import FlexibleLoadAnnotations
from .custom_hooks import (
    ChartTypeDistributionHook, SkipInvalidLossHook, RuntimeErrorHook, MissingImageReportHook, SkipBadSamplesHook, CompatibleCheckpointHook
)
from .nan_recovery_hook import NanRecoveryHook
from .progressive_loss_hook import ProgressiveLossHook, AdaptiveLossHook
from .square_fcn_mask_head import SquareFCNMaskHead

def register_all_modules():
    """Register all enhanced modules for comprehensive chart detection."""
    
    # Register models and heads
    MODELS.register_module(module=FCHead, force=True)
    MODELS.register_module(module=RegHead, force=True)
    MODELS.register_module(module=DataSeriesHead, force=True)
    MODELS.register_module(module=CustomCascadeWithMeta, force=True)
    MODELS.register_module(module=SquareFCNMaskHead, force=True)

    # Register datasets
    DATASETS.register_module(module=ChartDataset, force=True)
    
    # Register transforms
    TRANSFORMS.register_module(module=RobustLoadImageFromFile, force=True)
    TRANSFORMS.register_module(module=CreateDummyImg, force=True)
    TRANSFORMS.register_module(module=ClampBBoxes, force=True)
    TRANSFORMS.register_module(module=SetScaleFactor, force=True)
    TRANSFORMS.register_module(module=EnsureScaleFactor, force=True)
    TRANSFORMS.register_module(module=SetInputs, force=True)
    TRANSFORMS.register_module(module=CustomPackDetInputs, force=True)
    TRANSFORMS.register_module(module=FlexibleLoadAnnotations, force=True)
    
    # Register hooks
    HOOKS.register_module(module=ChartTypeDistributionHook, force=True)
    HOOKS.register_module(module=SkipInvalidLossHook, force=True)
    HOOKS.register_module(module=RuntimeErrorHook, force=True)
    HOOKS.register_module(module=MissingImageReportHook, force=True)
    HOOKS.register_module(module=SkipBadSamplesHook, force=True)
    HOOKS.register_module(module=NanRecoveryHook, force=True)
    HOOKS.register_module(module=CompatibleCheckpointHook, force=True)
    HOOKS.register_module(module=ProgressiveLossHook, force=True)
    HOOKS.register_module(module=AdaptiveLossHook, force=True)

    print("✅ Enhanced chart detection modules registered:")
    print("   📊 Models: FCHead, RegHead, DataSeriesHead, CustomCascadeWithMeta")
    print("   📁 Datasets: ChartDataset (21 categories)")
    print("   🔄 Transforms: RobustLoadImageFromFile, ClampBBoxes, SetScaleFactor, FlexibleLoadAnnotations, etc.")
    print("   🎯 Hooks: SkipInvalidLossHook, RuntimeErrorHook, MissingImageReportHook, SkipBadSamplesHook, NanRecoveryHook, CompatibleCheckpointHook, ProgressiveLossHook, AdaptiveLossHook")

# Register all modules when this module is imported
register_all_modules() 