import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from collections import defaultdict
import cv2
import easyocr
import json
from typing import List, Dict, Tuple, Optional
import logging
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🚨 CRITICAL FIX: Delay MMDetection imports and registration until needed
# This prevents custom model registration from interfering with ResNet-50 loading
MMDET_AVAILABLE = False
_mmdet_registered = False

def _ensure_mmdet_available():
    """Lazy import and registration of MMDetection modules only when needed"""
    global MMDET_AVAILABLE, _mmdet_registered
    
    if _mmdet_registered:
        return MMDET_AVAILABLE
        
    try:
        # Import MMDetection modules only when needed
        from mmdet.utils import register_all_modules
        from mmengine.registry import MODELS
        from mmengine.config import Config
        from mmdet.apis import inference_detector
        
        # Register standard MMDetection modules
        register_all_modules()
        
        # Import and register custom models only when needed
        try:
            from legend_match_swin.custom_models.register import register_all_modules
            # Register all custom modules (heads, models, transforms, etc.)
            register_all_modules()
            logger.info("Successfully registered all custom models from legend_match_swin.custom_models package")
        except ImportError as e:
            logger.warning(f"Could not import legend_match_swin.custom_models package: {e}")
            # Fallback to individual custom model if available
            try:
                from legend_match_swin.custom_models.custom_cascade_with_meta import CustomCascadeWithMeta
                MODELS.register_module(module=CustomCascadeWithMeta, force=True)
                logger.info("Using standalone CustomCascadeWithMeta")
            except ImportError:
                logger.warning("Custom model not available. Using standard MMDetection.")
        
        MMDET_AVAILABLE = True
        _mmdet_registered = True
        logger.info("✅ MMDetection modules registered successfully")
        
    except ImportError:
        MMDET_AVAILABLE = False
        logger.warning("MMDetection not available, chart element detection will be limited")
        
    return MMDET_AVAILABLE

# Enhanced 21-class categories for comprehensive chart element detection
ENHANCED_CLASS_NAMES = [
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
]

# Chart type classification categories - DocFigure Dataset (28 categories)
CHART_TYPE_CLASSES = [
    'Line graph', 'Natural image', 'Table', '3D objects', 'Bar plot', 
    'Scatter plot', 'Medical image', 'Sketch', 'Geographic map', 'Flow chart',
    'Heat map', 'Mask', 'Block diagram', 'Venn diagram', 'Confusion matrix',
    'Histogram', 'Box plot', 'Vector plot', 'Pie chart', 'Surface plot',
    'Algorithm', 'Contour plot', 'Tree diagram', 'Bubble chart', 'Polar plot',
    'Area chart', 'Pareto chart', 'Radar chart'
]

# Color scheme for different element types
ELEMENT_COLORS = {
    # Text elements - blue family
    'title': '#1f77b4', 'subtitle': '#aec7e8', 'axis-title': '#2ca02c',
    'x-axis-label': '#d62728', 'y-axis-label': '#d62728',
    'x-tick-label': '#ff7f0e', 'y-tick-label': '#ff7f0e', 'tick-label': '#ff7f0e',
    'data-label': '#9467bd',
    
    # Legend elements - purple family  
    'legend': '#8c564b', 'legend-title': '#e377c2', 'legend-item': '#f7b6d3', 'legend-text': '#c5b0d5',
    
    # Data visualization elements - green/red family
    'data-point': '#2ca02c', 'data-line': '#98df8a', 'data-bar': '#d62728', 'data-area': '#ff9896',
    
    # Structure elements - gray family
    'x-axis': '#7f7f7f', 'y-axis': '#7f7f7f', 'grid-line': '#c7c7c7', 'plot-area': '#17becf'
}

class ScientificImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 🎯 Two-Stage Architecture Initialization
        # Stage 1: Chart Type Classification Model
        self.chart_type_model = None
        self.chart_type_transform = None
        self._init_chart_type_model()
        
        # Stage 2: Chart Element Detection Model  
        # Note: Model will be created fresh for each detection to avoid state issues
        self.chart_element_model = None
        
        # Initialize OCR
        self.reader = easyocr.Reader(['en'])
        self.ocr_threshold = 0.5

    def _init_chart_type_model(self):
        """Stage 1: Initialize ResNet-50 chart type classification model"""
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', 'chart_type.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Chart type checkpoint not found: {checkpoint_path}")
            return
            
        try:
            logger.info("🎯 STAGE 1: Initializing Chart Type Classification")
            logger.info("Model: ResNet-50 with custom multi-layer head")
            logger.info("Input: Raw image (224×224, ImageNet preprocessing)")
            logger.info("Output: 28 chart type predictions with confidence scores")
            
            # Create ResNet-50 model with EXACT training architecture
            model = models.resnet50(pretrained=False)
            
            # Create multi-layer head matching training code exactly
            in_features = model.fc.in_features
            dropout = nn.Dropout(0.6)  # Match training dropout rate
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),        # backbone.fc.0.*
                nn.ReLU(inplace=True),             # backbone.fc.1
                dropout,                           # backbone.fc.2  
                nn.Linear(512, len(CHART_TYPE_CLASSES))  # backbone.fc.3.*
            )
            
            logger.info(f"✅ Created model with multi-layer head (training-compatible)")
            logger.info(f"   Architecture: {in_features} → 512 → {len(CHART_TYPE_CLASSES)} classes")

            # Load checkpoint using simple robust method - MATCH WORKING SCRIPT EXACTLY
            checkpoint = self.safe_torch_load_simple(checkpoint_path)
            
            # Extract state dict with better handling - MATCH WORKING SCRIPT
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("✅ Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("✅ Found 'state_dict' in checkpoint")
            else:
                state_dict = checkpoint
                logger.info("✅ Using checkpoint directly as state_dict")

            # Enhanced key adaptation for multi-layer head - MATCH WORKING SCRIPT EXACTLY
            adapted_state_dict = {}
            for key, value in state_dict.items():
                original_key = key
                
                # Handle backbone prefix removal
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                else:
                    new_key = key
                
                adapted_state_dict[new_key] = value
                
                # Debug key mapping for classifier head
                if 'fc.' in original_key:
                    logger.info(f"   📋 Mapped: {original_key} → {new_key}")

            # Load state dict with strict=False to see what's missing - MATCH WORKING SCRIPT
            missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
            
            # CRITICAL: Check if classification head loaded properly - MATCH WORKING SCRIPT
            classifier_keys_found = [k for k in adapted_state_dict.keys() if k.startswith('fc.')]
            expected_classifier_keys = ['fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias']
            
            logger.info(f"🔍 Classification Head Analysis:")
            logger.info(f"   Expected keys: {expected_classifier_keys}")
            logger.info(f"   Found keys: {classifier_keys_found}")
            
            classifier_loaded = all(key in adapted_state_dict for key in expected_classifier_keys)
            logger.info(f"   Classification head loaded: {'✅ YES' if classifier_loaded else '❌ NO'}")
            
            if not classifier_loaded:
                logger.warning("⚠️  WARNING: Classification head not loaded properly!")
                logger.warning("This will cause random predictions!")
            
            # Validate model has correct number of output classes
            actual_output_classes = model.fc[3].out_features  # Last layer in Sequential
            if actual_output_classes != len(CHART_TYPE_CLASSES):
                logger.warning(f"⚠️  Warning: Model has {actual_output_classes} outputs, expected {len(CHART_TYPE_CLASSES)}")
            
            # Move to device and set eval mode
            model.to(self.device)
            model.eval()  # CRITICAL: Set to eval mode

            logger.info(f"🎯 Model loaded successfully on device: {self.device}")
            logger.info(f"🔧 Model in eval mode: {not model.training}")
            
            if missing_keys:
                logger.info(f"⚠️  Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    logger.info(f"   Missing: {missing_keys}")
            if unexpected_keys:
                logger.info(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    logger.info(f"   Unexpected: {unexpected_keys}")
            
            self.chart_type_model = model
            
            # Stage 1: EXACT ImageNet preprocessing matching working script
            logger.info(f"🔄 Using DocFigure ImageNet preprocessing:")
            logger.info(f"   Size: 224×224")
            logger.info(f"   Color: RGB order")
            logger.info(f"   Range: [0,1] via ToTensor()")
            logger.info(f"   Norm: ImageNet mean/std")
            
            self.chart_type_transform = transforms.Compose([
                transforms.Resize((224, 224)),              # 224×224 input
                transforms.ToTensor(),                      # scales to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],            # ImageNet mean
                    std=[0.229, 0.224, 0.225]              # ImageNet std
                )
            ])
            
            logger.info("✅ Stage 1 model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading Stage 1 model: {e}")
            import traceback
            traceback.print_exc()
            self.chart_type_model = None

    def _create_swin_cascade_config(self):
        """Create EXACT model configuration matching chart_label+.pth training setup"""
        
        # Try to import the EXACT training configuration like in working script
        try:
            from legend_match_swin.cascade_rcnn_r50_fpn_meta import model as training_model_config
            logger.info("✅ Successfully imported training model config")
            # If successfully imported, could use that config, but for now use manual config for reliability
        except Exception as e:
            logger.info(f"Could not import training config: {e}, using manual config")
        
        return dict(
            type='CustomCascadeWithMeta',  # Use custom model with coordinate handling
            coordinate_standardization=dict(
                enabled=True,
                origin='bottom_left',      # Match annotation creation coordinate system
                normalize=True,
                relative_to_plot=False,    # Keep simple for now
                scale_to_axis=False        # Keep simple for now
            ),
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_size_divisor=32),
            # ----- Swin Transformer Base (22K) Backbone + FPN -----
            backbone=dict(
                type='SwinTransformer',
                embed_dims=128,  # Swin Base embedding dimensions
                depths=[2, 2, 18, 2],  # Swin Base depths
                num_heads=[4, 8, 16, 32],  # Swin Base attention heads
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.3,  # Slightly higher for more complex model
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'
                )
            ),
            neck=dict(
                type='FPN',
                in_channels=[128, 256, 512, 1024],  # Swin Base: embed_dims * 2^(stage)
                out_channels=256,
                num_outs=6,
                start_level=0,
                add_extra_convs='on_input'
            ),
            # Enhanced RPN with smaller anchors for tiny objects + improved losses
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[1, 2, 4, 8],  # Even smaller scales for tiny objects
                    ratios=[0.5, 1.0, 2.0],  # Multiple aspect ratios
                    strides=[4, 8, 16, 32, 64, 128]),  # Extended FPN strides
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            # Progressive Loss Strategy with 10x data-point weighting for all 3 stages
            roi_head=dict(
                type='CascadeRoIHead',
                num_stages=3,
                stage_loss_weights=[1, 0.5, 0.25],
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=[
                    # Stage 1: SmoothL1Loss (coarse detection) + 10x data-point weighting
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=21,  # 21 enhanced categories
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.05, 0.05, 0.1, 0.1]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0,
                            class_weight=[1.0,  # background class (index 0)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                         10.0,  # data-point at index 12 gets 10x weight (11+1 for background)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                    # Stage 2: SmoothL1Loss (intermediate refinement) + 10x data-point weighting
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=21,  # 21 enhanced categories
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.033, 0.033, 0.067, 0.067]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0,
                            class_weight=[1.0,  # background class (index 0)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                         10.0,  # data-point at index 12 gets 10x weight (11+1 for background)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                    # Stage 3: Post-progressive loss switching (could be CIoU if trained with ProgressiveLossHook)
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=21,  # 21 enhanced categories
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.02, 0.02, 0.05, 0.05]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0,
                            class_weight=[1.0,  # background class (index 0)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                         10.0,  # data-point at index 12 gets 10x weight (11+1 for background)
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                        loss_bbox=dict(type='CIoULoss', loss_weight=1.0))  # FIXED: Stage 3 uses CIoULoss after epoch 3
                ]),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False),
                    allowed_border=0,
                    pos_weight=-1,
                    debug=False),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=2000,
                    nms=dict(type='nms', iou_threshold=0.8),
                    min_bbox_size=0),
                rcnn=[
                    # Stage 1: Lower IoU thresholds for better small object (data-point) matching
                    dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.3,  # Lower for small data-points
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True,  # Enable for small objects
                            ignore_iof_thr=-1),
                        sampler=dict(
                            type='RandomSampler',
                            num=512,
                            pos_fraction=0.25,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=True),
                        pos_weight=-1,
                        debug=False),
                    # Stage 2: Moderate IoU thresholds for data-point refinement  
                    dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,  # Lower for small data-points
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                            match_low_quality=True,  # Enable for small objects
                            ignore_iof_thr=-1),
                        sampler=dict(
                            type='RandomSampler',
                            num=512,
                            pos_fraction=0.25,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=True),
                        pos_weight=-1,
                        debug=False),
                    # Stage 3: Final refinement - still lower IoU for data-point precision
                    dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.6,  # Lower for small data-points
                            neg_iou_thr=0.6,
                            min_pos_iou=0.6,
                            match_low_quality=True,  # Enable for small objects
                            ignore_iof_thr=-1),
                        sampler=dict(
                            type='RandomSampler',
                            num=512,
                            pos_fraction=0.25,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=True),
                        pos_weight=-1,
                        debug=False)
                ]),
            # Enhanced test configuration with soft-NMS for small object detection
            test_cfg=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    score_thr=0.001,  # Very low threshold for testing
                    nms=dict(
                        type='soft_nms',  # Soft-NMS for better small object detection
                        iou_threshold=0.3,  # Lower for data-points to avoid over-suppression
                        min_score=0.001,
                        method='gaussian',
                        sigma=0.5),
                    max_per_img=500)))  # Allow more detections

    def _create_fresh_chart_element_model(self):
        """Create a fresh chart element detection model (avoids state caching issues)"""
        # 🚨 CRITICAL FIX: Ensure MMDetection is available before creating model
        if not _ensure_mmdet_available():
            logger.warning("MMDetection not available, cannot create chart element model")
            return None
            
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', 'chart_label+.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Chart element detection checkpoint not found: {checkpoint_path}")
            # Fallback to older model versions
            checkpoint_candidates = [
                os.path.join(os.path.dirname(__file__), 'models', 'chart_label.pth'),
                'latest.pth', 'epoch_10.pth', 'epoch_9.pth', 'epoch_8.pth', 
                'epoch_7.pth', 'epoch_6.pth', 'epoch_5.pth', 'epoch_4.pth', 
                'epoch_3.pth', 'epoch_2.pth', 'epoch_1.pth'
            ]
            
            checkpoint_path = None
            for candidate in checkpoint_candidates:
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
            
            if checkpoint_path is None:
                logger.warning("No chart element detection checkpoint found")
                return None
        
        try:
            # Import needed MMDetection classes now that registration is complete
            from mmengine.registry import MODELS
            from mmengine.config import Config
            
            logger.info("🔄 Creating fresh Stage 2 model (avoiding state caching)")
            logger.info("Model: Swin Transformer + Cascade R-CNN with custom metadata heads")
            logger.info("Input: Raw image (1120×672, optimized for 14x14 data points with 7x7 Swin windows)")
            logger.info("Output: 21 chart element categories with bounding boxes")
            
            # Create model configuration
            model_cfg_dict = self._create_swin_cascade_config()
            
            # Convert to Config object for MMDetection compatibility
            model_cfg = Config(model_cfg_dict)
            
            # Build model
            model = MODELS.build(model_cfg)
            
            # Stage 2: Optimized preprocessing pipeline for 1120×672 input (14x14 data points with 7x7 Swin windows)
            model_cfg.test_dataloader = dict(
                dataset=dict(
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='Resize', scale=(1120, 672), keep_ratio=True),  # Optimized for 14x14 data points with 7x7 windows
                        dict(type='ClampBBoxes'),  # Ensure bboxes stay within image bounds
                        dict(type='PackDetInputs')
                    ]
                )
            )
            
            # Add cfg attribute to model (required by MMDetection inference API)
            model.cfg = model_cfg
            
            # Load checkpoint with robust handling
            logger.info("Loading checkpoint with compatibility handling...")
            checkpoint = self.safe_torch_load_simple(checkpoint_path)
            
            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Load state dict with flexible matching
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
            
            # Move to device and set eval mode
            model.to(self.device)
            model.eval()
            
            logger.info("✅ Fresh Stage 2 model created successfully!")
            return model
            
        except Exception as e:
            logger.error(f"Error creating fresh Stage 2 model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def safe_torch_load_simple(self, checkpoint_path):
        """Simple and robust PyTorch loading based on working script approach"""
        loading_methods = [
            {'map_location': 'cuda' if torch.cuda.is_available() else 'cpu', 'weights_only': False},
            {'map_location': 'cpu', 'weights_only': False},
            {'weights_only': False},
        ]

        for i, method_kwargs in enumerate(loading_methods):
            try:
                logger.info(f"Attempting Method {i+1}: torch.load with {method_kwargs}")
                checkpoint = torch.load(checkpoint_path, **method_kwargs)
                logger.info(f"✅ Success with Method {i+1}!")
                return checkpoint
            except Exception as e:
                logger.info(f"Method {i+1} failed: {e}")
                continue

        raise Exception("All PyTorch loading methods failed. Check checkpoint path and compatibility.")

    def validate_image(self, img_path):
        """Validate image file and properties"""
        try:
            from PIL import Image
            img = Image.open(img_path)
            width, height = img.size
            
            if width < 32 or height < 32:
                logger.warning(f"Image is very small ({width}×{height}). Results may be poor.")
            
            if img.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unusual image mode '{img.mode}'. Converting to RGB.")
            
            return True
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return False

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image for processing"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def classify_chart_type(self, image_path: str) -> Optional[List[Tuple[str, float]]]:
        """Exact replication of working script logic"""
        # Debug: Check if the image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Absolute path: {os.path.abspath(image_path)}")
            return None
            
        chart_type_checkpoint = os.path.join(os.path.dirname(__file__), 'models', 'chart_type.pth')
        
        if not os.path.exists(chart_type_checkpoint):
            logger.warning(f"Chart type checkpoint not found: {chart_type_checkpoint}")
            return None
        
        try:
            # CREATE MODEL FRESH (like working script)
            model = models.resnet50(pretrained=False)
            in_features = model.fc.in_features
            dropout = nn.Dropout(0.6)
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                dropout,
                nn.Linear(512, len(CHART_TYPE_CLASSES))
            )
            
            # LOAD CHECKPOINT FRESH
            checkpoint = self.safe_torch_load_simple(chart_type_checkpoint)
            
            # EXACT STATE DICT MAPPING
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            adapted_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                else:
                    new_key = key
                adapted_state_dict[new_key] = value
            
            model.load_state_dict(adapted_state_dict, strict=False)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            # EXACT PREPROCESSING
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # EXACT INFERENCE
            img = Image.open(image_path).convert('RGB')
            img_tensor = val_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_prob, top_class = torch.topk(probabilities, k=min(5, len(CHART_TYPE_CLASSES)))

            results = []
            for i in range(top_prob.shape[1]):
                class_idx = top_class[0][i].item()
                confidence = top_prob[0][i].item()
                chart_type = CHART_TYPE_CLASSES[class_idx]
                results.append((chart_type, confidence))

            return results
            
        except Exception as e:
            logger.error(f"Error in chart type classification: {e}")
            return None

    def _load_datapoint_model(self):
        """Load the new data-point model using Swin config and chart_datapoint.pth"""
        from mmdet.apis import init_detector
        import sys
        
        # Add the parent directory to sys.path so legend_match_swin can be imported
        parent_dir = os.path.abspath('..')
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Also add current directory to make sure relative imports work
        current_dir = os.path.abspath('.')
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import and register custom modules before loading the config
        try:
            # Import the package to trigger registration
            import legend_match_swin.custom_models
            logger.info("✅ Custom modules imported successfully")
        except ImportError as e:
            logger.warning(f"Failed to import custom modules: {e}")
            return None
        
        config_path = os.path.join('legend_match_swin', 'mask_rcnn_swin_meta.py')
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', 'chart_datapoint.pth')
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            logger.warning(f"Data-point model config or checkpoint not found: {config_path}, {checkpoint_path}")
            return None
        model = init_detector(config_path, checkpoint_path, device=str(self.device))
        model.to(self.device)
        model.eval()
        return model

    def _filter_out_class(self, detections, class_names, exclude_class='data-point', score_thr=0.2):
        pred_instances = detections.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        keep = [i for i, l in enumerate(labels) if class_names[l] != exclude_class and scores[i] >= score_thr]
        filtered = {
            'bboxes': bboxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }
        return filtered

    def _filter_only_class(self, detections, class_names, target_class='data-point', score_thr=0.5):
        """Filter to keep ONLY the target class, reject everything else"""
        pred_instances = detections.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Keep ONLY detections that match the target class with sufficient confidence
        keep = [i for i, l in enumerate(labels) if class_names[l] == target_class and scores[i] >= score_thr]
        
        logger.info(f"Datapoint model: {len(bboxes)} total detections, keeping {len(keep)} {target_class} detections (score >= {score_thr})")
        
        # Log what we're rejecting
        rejected_classes = {}
        for i, l in enumerate(labels):
            class_name = class_names[l]
            if class_name != target_class:
                rejected_classes[class_name] = rejected_classes.get(class_name, 0) + 1
        
        if rejected_classes:
            rejected_summary = ', '.join([f"{cls}: {count}" for cls, count in rejected_classes.items()])
            logger.info(f"Datapoint model: Rejected non-{target_class} classes: {rejected_summary}")
        
        filtered = {
            'bboxes': bboxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]  # All remaining labels should be data-point class index
        }
        return filtered

    def _filter_out_classes(self, detections, class_names, exclude_classes=['data-point'], score_thr=0.2):
        """Filter out multiple classes from detections"""
        pred_instances = detections.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Keep detections that are NOT in exclude_classes and meet score threshold
        keep = [i for i, l in enumerate(labels) if class_names[l] not in exclude_classes and scores[i] >= score_thr]
        
        logger.info(f"Original model: {len(bboxes)} total detections, keeping {len(keep)} after excluding {exclude_classes}")
        
        filtered = {
            'bboxes': bboxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }
        return filtered

    def _filter_data_elements(self, detections, class_names, target_classes=['data-point', 'data-bar'], score_thr=0.5):
        """Filter to keep data-point and data-bar classes from datapoint model (matching main branch)"""
        pred_instances = detections.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Keep detections that match target classes with sufficient confidence
        keep = [i for i, l in enumerate(labels) if class_names[l] in target_classes and scores[i] >= score_thr]
        
        logger.info(f"Datapoint model: {len(bboxes)} total detections, keeping {len(keep)} {target_classes} detections (score >= {score_thr})")
        
        # Log what we're keeping and rejecting
        kept_classes = {}
        rejected_classes = {}
        for i, l in enumerate(labels):
            class_name = class_names[l]
            if class_name in target_classes and scores[i] >= score_thr:
                kept_classes[class_name] = kept_classes.get(class_name, 0) + 1
            else:
                rejected_classes[class_name] = rejected_classes.get(class_name, 0) + 1
        
        if kept_classes:
            kept_summary = ', '.join([f"{cls}: {count}" for cls, count in kept_classes.items()])
            logger.info(f"Datapoint model: Kept {target_classes} classes: {kept_summary}")
        
        if rejected_classes:
            rejected_summary = ', '.join([f"{cls}: {count}" for cls, count in rejected_classes.items()])
            logger.info(f"Datapoint model: Rejected other classes: {rejected_summary}")
        
        filtered = {
            'bboxes': bboxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }
        return filtered

    def detect_chart_elements(self, image_path: str) -> List[Dict]:
        """Combined chart element detection: original model for all except data-points and data-bars, new model for data-points and data-bars."""
        if not _ensure_mmdet_available():
            logger.warning("MMDetection not available, Stage 2 model cannot be used")
            return []
        try:
            from mmdet.apis import inference_detector
            logger.info("🎯 Combined Chart Element Detection (segmentation branch)")
            # --- Original model (all except data-point and data-bar) ---
            orig_model = self._create_fresh_chart_element_model()
            if orig_model is None:
                logger.warning("Failed to create original chart element model")
                return []
            orig_result = inference_detector(orig_model, image_path)
            # Filter out both data-point and data-bar from original model
            filtered_orig = self._filter_out_classes(orig_result, ENHANCED_CLASS_NAMES, exclude_classes=['data-point', 'data-bar'], score_thr=0.2)
            
            # --- New data-point model (for data-point and data-bar) ---
            logger.info("🎯 Loading datapoint model...")
            datapoint_model = self._load_datapoint_model()
            if datapoint_model is None:
                logger.warning("Failed to load data-point model")
                return []
            
            logger.info("🎯 Running datapoint model inference...")
            datapoint_result = inference_detector(datapoint_model, image_path)
            
            logger.info("🎯 Filtering datapoint model results...")
            
            # Log what the datapoint model detected BEFORE filtering
            datapoint_pred_instances = datapoint_result.pred_instances
            datapoint_bboxes = datapoint_pred_instances.bboxes.cpu().numpy()
            datapoint_scores = datapoint_pred_instances.scores.cpu().numpy()
            datapoint_labels = datapoint_pred_instances.labels.cpu().numpy()
            
            # Count detections by class before filtering
            class_counts = {}
            for i, label in enumerate(datapoint_labels):
                class_name = ENHANCED_CLASS_NAMES[label]
                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(datapoint_scores[i])
            
            logger.info("🎯 Datapoint model raw detections by class:")
            for class_name, scores in class_counts.items():
                logger.info(f"    {class_name}: {len(scores)} detections (scores: {[f'{s:.3f}' for s in sorted(scores, reverse=True)[:5]]})")
            
            # Filter to keep data-point and data-bar from datapoint model (matching main branch)
            filtered_datapoint = self._filter_data_elements(datapoint_result, ENHANCED_CLASS_NAMES, target_classes=['data-point', 'data-bar'], score_thr=0.005)
            
            logger.info(f"🎯 Datapoint model final results: {len(filtered_datapoint['bboxes'])} data elements (data-point + data-bar)")
            # --- Combine detections ---
            logger.info(f"🎯 Original model contributed: {len(filtered_orig['bboxes'])} detections")
            logger.info(f"🎯 Datapoint model contributed: {len(filtered_datapoint['bboxes'])} detections")
            
            bboxes = np.concatenate([filtered_orig['bboxes'], filtered_datapoint['bboxes']], axis=0)
            scores = np.concatenate([filtered_orig['scores'], filtered_datapoint['scores']], axis=0)
            labels = np.concatenate([filtered_orig['labels'], filtered_datapoint['labels']], axis=0)
            
            # Mark which detections are from the new datapoint model
            datapoint_indices = set(range(len(filtered_orig['bboxes']), len(bboxes)))
            logger.info(f"🎯 Datapoint indices: {datapoint_indices}")
            detections = []
            elements_by_type = defaultdict(list)
            detection_count = 0
            for idx in range(len(bboxes)):
                score = scores[idx]
                label = labels[idx]
                class_name = ENHANCED_CLASS_NAMES[label]
                x1, y1, x2, y2 = bboxes[idx].astype(int)
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detection_count += 1
                is_from_datapoint_model = (idx in datapoint_indices and class_name in ['data-point', 'data-bar'])
                detection = {
                    'id': detection_count,
                    'element_type': class_name,
                    'confidence': float(score),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area': int(area),
                    'center': [int(center_x), int(center_y)],
                    'color': ELEMENT_COLORS.get(class_name, '#bcbd22'),
                    'source_model': 'datapoint' if is_from_datapoint_model else 'original'
                }
                detections.append(detection)
                elements_by_type[class_name].append(detection)
            self._log_comprehensive_detection_summary(elements_by_type, detections)
            return detections
        except Exception as e:
            logger.error(f"Error in combined chart element detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _log_comprehensive_detection_summary(self, elements_by_type: Dict, all_detections: List[Dict]):
        """Log comprehensive detection summary matching reference script format"""
        
        # Group elements by category for organized reporting
        category_groups = {
            "TEXT ELEMENTS": ['title', 'subtitle', 'axis-title', 'data-label'],
            "AXIS LABELS": ['x-axis-label', 'y-axis-label', 'x-tick-label', 'y-tick-label', 'tick-label'],
            "DATA VISUALIZATION": ['data-point', 'data-line', 'data-bar', 'data-area'],
            "LEGEND COMPONENTS": ['legend', 'legend-title', 'legend-item', 'legend-text'],
            "STRUCTURAL ELEMENTS": ['x-axis', 'y-axis', 'grid-line', 'plot-area']
        }
        
        logger.info("="*80)
        logger.info("CHART ELEMENT DETECTION RESULTS")
        logger.info("="*80)
        
        for group_name, group_classes in category_groups.items():
            group_elements = {cls: elements_by_type[cls] for cls in group_classes if cls in elements_by_type}
            
            if group_elements:
                logger.info(f"\n{group_name}")
                logger.info("-" * 60)
                
                for class_name, elements in group_elements.items():
                    logger.info(f"\n  {class_name.upper().replace('-', ' ')} ({len(elements)} found):")
                    
                    for elem in sorted(elements, key=lambda x: x['confidence'], reverse=True):
                        x1, y1, x2, y2 = elem['bbox']
                        cx, cy = elem['center']
                        logger.info(f"    Detection #{elem['id']:2d}: conf={elem['confidence']:.3f} | "
                                  f"bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] | "
                                  f"center=({cx:4d},{cy:4d}) | area={elem['area']:6d}px²")
        
        # Log ALL detections in order of confidence
        logger.info(f"\n{'='*80}")
        logger.info("ALL DETECTIONS RANKED BY CONFIDENCE")
        logger.info(f"{'='*80}")
        
        sorted_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        
        for i, elem in enumerate(sorted_detections):
            x1, y1, x2, y2 = elem['bbox']
            cx, cy = elem['center']
            logger.info(f"  #{i+1:2d}: {elem['element_type']:<20} | conf={elem['confidence']:.3f} | "
                      f"bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] | center=({cx:4d},{cy:4d})")
        
        # Complete element type report
        logger.info(f"\n{'='*80}")
        logger.info("COMPREHENSIVE DETECTION SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info("\nComplete Element Type Report:")
        for class_name in ENHANCED_CLASS_NAMES:
            count = len(elements_by_type.get(class_name, []))
            if count > 0:
                elements = elements_by_type[class_name]
                confidences = [e['confidence'] for e in elements]
                avg_conf = np.mean(confidences)
                max_conf = np.max(confidences)
                logger.info(f"  ✅ {class_name:<20}: {count:2d} detections | "
                          f"avg_conf={avg_conf:.3f} | max_conf={max_conf:.3f}")
            else:
                logger.info(f"  ❌ {class_name:<20}: {count:2d} detections")
        
        logger.info(f"\n✅ Chart element detection completed successfully!")
        logger.info(f"🎯 Detected elements from {len(ENHANCED_CLASS_NAMES)} possible categories")

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image using OCR"""
        try:
            results = self.reader.readtext(image)
            
            text_detections = []
            for (bbox, text, conf) in results:
                if conf >= self.ocr_threshold:
                    # Convert bbox format
                    bbox_coords = np.array(bbox).astype(int)
                    x1, y1 = bbox_coords.min(axis=0)
                    x2, y2 = bbox_coords.max(axis=0)

                    text_detections.append({
                        'type': 'text',
                        'text': text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Convert to native Python int
                        'confidence': float(conf),
                        'area': int((x2 - x1) * (y2 - y1))  # Convert to native Python int
                    })
            
            return text_detections
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return []

    def filter_tiny_text(self, detections: List[Dict], min_area: int = 100) -> List[Dict]:
        """Filter out tiny text detections that are likely noise"""
        return [d for d in detections if d.get('area', 0) >= min_area]

    def generate_caption(self, chart_detections: List[Dict], text_detections: List[Dict], chart_type: Optional[str] = None) -> str:
        """
        Generate comprehensive caption from chart elements and text, including detailed element information
        """
        try:
            # Group chart elements by type for better organization
            elements_by_type = defaultdict(list)
            for detection in chart_detections:
                element_type = detection.get('element_type', 'unknown')
                elements_by_type[element_type].append(detection)
            
            # Start caption with chart type if available
            if chart_type:
                caption_parts = [f"This is a {chart_type.lower()}."]
            else:
                caption_parts = ["This is a scientific chart or graph."]
            
            # Add comprehensive element detection summary
            total_elements = len(chart_detections)
            if total_elements > 0:
                caption_parts.append(f"The chart contains {total_elements} detected elements across {len(elements_by_type)} different categories.")
                
                # Describe major structural elements
                structural_elements = []
                if 'plot-area' in elements_by_type:
                    structural_elements.append(f"{len(elements_by_type['plot-area'])} plot area(s)")
                if 'x-axis' in elements_by_type:
                    structural_elements.append(f"{len(elements_by_type['x-axis'])} x-axis elements")
                if 'y-axis' in elements_by_type:
                    structural_elements.append(f"{len(elements_by_type['y-axis'])} y-axis elements")
                
                if structural_elements:
                    caption_parts.append(f"Key structural elements include: {', '.join(structural_elements)}.")
                
                # Describe data elements
                data_elements = []
                for data_type in ['data-point', 'data-line', 'data-bar', 'data-area']:
                    if data_type in elements_by_type:
                        count = len(elements_by_type[data_type])
                        data_elements.append(f"{count} {data_type.replace('-', ' ')}{'s' if count > 1 else ''}")
                
                if data_elements:
                    caption_parts.append(f"Data visualization includes: {', '.join(data_elements)}.")
                
                # Describe text elements
                text_elements = []
                for text_type in ['title', 'subtitle', 'axis-title', 'x-tick-label', 'y-tick-label']:
                    if text_type in elements_by_type:
                        count = len(elements_by_type[text_type])
                        text_elements.append(f"{count} {text_type.replace('-', ' ')}{'s' if count > 1 else ''}")
                
                if text_elements:
                    caption_parts.append(f"Text elements include: {', '.join(text_elements)}.")
                
                # Describe legend if present
                legend_elements = []
                for legend_type in ['legend', 'legend-title', 'legend-item', 'legend-text']:
                    if legend_type in elements_by_type:
                        count = len(elements_by_type[legend_type])
                        legend_elements.append(f"{count} {legend_type.replace('-', ' ')}{'s' if count > 1 else ''}")
                
                if legend_elements:
                    caption_parts.append(f"Legend components include: {', '.join(legend_elements)}.")
            
            # Add OCR text information if available
            if text_detections:
                visible_texts = [d.get('text', '') for d in text_detections if d.get('text', '').strip()]
                if visible_texts:
                    caption_parts.append(f"Detected text includes: {', '.join(visible_texts[:5])}")  # Limit to first 5 texts
                    if len(visible_texts) > 5:
                        caption_parts.append(f"and {len(visible_texts) - 5} additional text elements.")
            
            return " ".join(caption_parts)
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "This is a scientific chart or graph with multiple detected elements."

    def _ensure_json_serializable(self, obj):
        """Ensure all values in the object are JSON serializable"""
        if isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def analyze_image(self, image_path: str) -> Dict:
        """
        Comprehensive analysis of scientific images with detailed element detection
        matching the reference inference script approach
        """
        if not self.validate_image(image_path):
            return {"error": "Invalid image file"}
        
        try:
            results = {
                'image_path': image_path,
                'timestamp': str(np.datetime64('now')),
                'chart_type': None,
                'chart_type_confidence': None,
                'chart_elements': [],
                'text_elements': [],
                'caption': '',
                'element_summary': {},
                'total_detections': 0
            }
            
            # Step 1: Chart type classification
            logger.info("="*60)
            logger.info("CHART TYPE CLASSIFICATION")
            logger.info("="*60)
            
            chart_type_results = self.classify_chart_type(image_path)
            if chart_type_results:
                logger.info("Chart Type Prediction Results:")
                for i, (chart_type, confidence) in enumerate(chart_type_results):
                    rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    logger.info(f"  {rank_emoji} {chart_type}: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Get the top prediction
                top_chart_type, top_confidence = chart_type_results[0]
                results['chart_type'] = top_chart_type
                results['chart_type_confidence'] = float(top_confidence)
                logger.info(f"\n🎯 **PREDICTED CHART TYPE: {top_chart_type}** (confidence: {top_confidence:.3f})")
            else:
                logger.warning("Chart type classification not available")
            
            logger.info("="*60)
            
            # Step 2: Chart element detection with comprehensive analysis
            chart_elements = self.detect_chart_elements(image_path)
            results['chart_elements'] = chart_elements
            results['total_detections'] = len(chart_elements)
            
            # Create element summary by type
            elements_by_type = defaultdict(int)
            for element in chart_elements:
                elements_by_type[element['element_type']] += 1
            results['element_summary'] = dict(elements_by_type)
            
            # Step 3: OCR text extraction
            image = self.load_image(image_path)
            text_elements = self.extract_text(image)
            results['text_elements'] = text_elements
            
            # Step 4: Generate comprehensive caption
            results['caption'] = self.generate_caption(
                chart_elements, 
                text_elements, 
                results['chart_type']
            )
            
            # Log final summary
            logger.info(f"\n{'='*80}")
            logger.info("ANALYSIS COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"✅ Image analysis completed successfully!")
            logger.info(f"📊 Chart Type: {results['chart_type']} (confidence: {results.get('chart_type_confidence', 0):.3f})")
            logger.info(f"🎯 Total Chart Elements: {results['total_detections']}")
            logger.info(f"📝 Text Elements: {len(text_elements)}")
            logger.info(f"{'='*80}")
            
            return self._ensure_json_serializable(results)
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    analyzer = ScientificImageAnalyzer()
    test_image_path = "static/images/science1.jpg"
    if os.path.exists(test_image_path):
        print(f"Analyzing image: {test_image_path}")
        result = analyzer.analyze_image(test_image_path)
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please update test_image_path to a valid image.") 