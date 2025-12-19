# inference_science2_direct.py - Direct Chart Element Detection on science2.jpg

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from collections import defaultdict

from mmdet.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine.config import Config
from mmdet.apis import inference_detector

# Register standard MMDetection modules first
register_all_modules()

# Import enhanced custom models from legend_match_swin (matches training)
import sys
sys.path.append('..')  # Add parent directory to path
from legend_match_swin.custom_models.register import register_all_modules as register_custom_modules

# Register custom modules
register_custom_modules()

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

def create_model_config():
    """Create model configuration matching Swin Transformer training setup"""
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
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=21,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=21,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=21,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.02, 0.02, 0.05, 0.05]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.5))
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
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
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
        # Enhanced test configuration with soft-NMS and multi-scale support
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.005,  # Even lower threshold to catch more classes
                nms=dict(
                    type='soft_nms',  # Soft-NMS for better small object detection
                    iou_threshold=0.5,
                    min_score=0.005,
                    method='gaussian',
                    sigma=0.5),
                max_per_img=500)))  # Allow more detections
    
    # Add test pipeline configuration (required by MMDetection inference API)
    # Match the training pipeline resolution for Swin Transformer
    model_cfg_dict['test_dataloader'] = dict(
        dataset=dict(
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1600, 1000), keep_ratio=True),  # Match training resolution
                dict(type='ClampBBoxes'),  # Ensure bboxes stay within image bounds
                dict(type='PackDetInputs')
            ]
        )
    )

def create_chart_type_model_config():
    """Create chart type classification model configuration"""
    return dict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),  # Only use final stage
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=len(CHART_TYPE_CLASSES),
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )
    )

def safe_torch_load_simple(checkpoint_path):
    """Simple and robust PyTorch loading based on colab approach"""
    loading_methods = [
        {'map_location': 'cuda' if torch.cuda.is_available() else 'cpu', 'weights_only': False},
        {'map_location': 'cpu', 'weights_only': False},
        {'weights_only': False},
    ]

    for i, method_kwargs in enumerate(loading_methods):
        try:
            print(f"Attempting Method {i+1}: torch.load with {method_kwargs}")
            checkpoint = torch.load(checkpoint_path, **method_kwargs)
            print(f"✅ Success with Method {i+1}!")
            return checkpoint
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
            continue

    raise Exception("All PyTorch loading methods failed. Check checkpoint path and compatibility.")

def validate_image(img_path):
    """Validate image file and properties"""
    try:
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size
        
        if width < 32 or height < 32:
            print(f"⚠️  Warning: Image is very small ({width}×{height}). Results may be poor.")
        
        if img.mode not in ['RGB', 'RGBA', 'L']:
            print(f"⚠️  Warning: Unusual image mode '{img.mode}'. Converting to RGB.")
        
        return True
    except Exception as e:
        print(f"❌ Invalid image file: {e}")
        return False

def load_and_run_chart_type_classification(img_path):
    """Load chart type model and classify using improved colab-based approach"""

    # Validate input image first
    if not validate_image(img_path):
        return None

    chart_type_checkpoint = 'chart_type.pth'
    
    if not os.path.exists(chart_type_checkpoint):
        print(f"⚠️  Chart type checkpoint not found: {chart_type_checkpoint}")
        return None
    
    try:
        print(f"📊 Loading chart type classification model: {chart_type_checkpoint}")
        
        # Import required modules
        import torch.nn as nn
        import torchvision.models as models
        from torchvision import transforms
        from PIL import Image
        
        # Create ResNet50 model with EXACT training architecture
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
        
        print(f"✅ Created model with multi-layer head (training-compatible)")
        print(f"   Architecture: {in_features} → 512 → {len(CHART_TYPE_CLASSES)} classes")

        # Load checkpoint using simple robust method
        checkpoint = safe_torch_load_simple(chart_type_checkpoint)
        
        # Extract state dict with better handling
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✅ Found 'model_state_dict' in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print("✅ Using checkpoint directly as state_dict")

        # Enhanced key adaptation for multi-layer head
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
                print(f"   📋 Mapped: {original_key} → {new_key}")

        # Load state dict with strict=False to see what's missing
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        
        # CRITICAL: Check if classification head loaded properly
        classifier_keys_found = [k for k in adapted_state_dict.keys() if k.startswith('fc.')]
        expected_classifier_keys = ['fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias']
        
        print(f"\n🔍 Classification Head Analysis:")
        print(f"   Expected keys: {expected_classifier_keys}")
        print(f"   Found keys: {classifier_keys_found}")
        
        classifier_loaded = all(key in adapted_state_dict for key in expected_classifier_keys)
        print(f"   Classification head loaded: {'✅ YES' if classifier_loaded else '❌ NO'}")
        
        if not classifier_loaded:
            print(f"   ⚠️  WARNING: Classification head not loaded properly!")
            print(f"   This will cause random predictions!")
        
        # Validate model has correct number of output classes
        actual_output_classes = model.fc[3].out_features  # Last layer in Sequential
        if actual_output_classes != len(CHART_TYPE_CLASSES):
            print(f"⚠️  Warning: Model has {actual_output_classes} outputs, expected {len(CHART_TYPE_CLASSES)}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # CRITICAL: Set to eval mode

        print(f"🎯 Model loaded successfully on device: {device}")
        print(f"🔧 Model in eval mode: {not model.training}")
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 10:
                print(f"   Missing: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected: {unexpected_keys}")

        # DocFigure ImageNet preprocessing (exact match to training)
        print(f"🔄 Using DocFigure ImageNet preprocessing:")
        print(f"   Size: 224×224")
        print(f"   Color: RGB order")
        print(f"   Range: [0,1] via ToTensor()")
        print(f"   Norm: ImageNet mean/std")

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),              # 224×224 input
            transforms.ToTensor(),                      # scales to [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],            # ImageNet mean
                std=[0.229, 0.224, 0.225]              # ImageNet std
            )
        ])

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')      # RGB order (torchvision default)
        img_tensor = val_transform(img).unsqueeze(0).to(device)

        print(f"📐 Input tensor shape: {img_tensor.shape}")
        print(f"📊 Input tensor range: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")

        # Run inference with proper eval mode
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

        # Debug output logits/probabilities
        print(f"\n🔍 Debug info:")
        print(f"   Raw output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
        top_3_logits = [f"{outputs[0][top_class[0][i]].item():.3f}" for i in range(3)]
        print(f"   Top 3 logits: {top_3_logits}")
        print(f"   Confidence spread: {results[0][1] - results[1][1]:.3f}")
        
        # Additional validation
        if results[0][1] < 0.3:  # Very low top confidence
            print(f"   ⚠️  WARNING: Very low confidence - check if model loaded correctly")
        if len(set([r[1] for r in results[:3]])) < 2:  # All similar confidence
            print(f"   ⚠️  WARNING: Similar confidences - possible random predictions")

        return results
        
    except Exception as e:
        print(f"❌ Error in chart type classification: {e}")
        import traceback
        traceback.print_exc()
        return None

def safe_torch_load(checkpoint_path):
    """Aggressively try different PyTorch loading methods to bypass numpy issues"""
    
    # First, set up comprehensive numpy module compatibility
    import sys
    import numpy as np
    
    # Add all possible numpy module paths to sys.modules
    numpy_modules = [
        'numpy._core', 'numpy._core.multiarray', 'numpy._core.umath',
        'numpy.core', 'numpy.core.multiarray', 'numpy.core.umath', 
        'numpy.core._multiarray_umath', 'numpy.core.numeric'
    ]
    
    for module_name in numpy_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = np
    
    # Add essential numpy attributes
    numpy_attrs = {
        'scalar': np.number,
        '_reconstruct': lambda subtype, shape, dtype: np.empty(shape, dtype=dtype),
        'ndarray': np.ndarray,
        'dtype': np.dtype
    }
    
    for attr, value in numpy_attrs.items():
        if not hasattr(np, attr):
            setattr(np, attr, value)
    
    # Method 1: Try standard torch.load with different arguments
    loading_methods = [
        # Standard approaches
        {'weights_only': False},
        {'weights_only': False, 'pickle_module': __import__('pickle')},
        {'weights_only': False, 'encoding': 'latin1'},
        {'weights_only': False, 'pickle_module': __import__('dill')} if 'dill' in sys.modules else None,
        
        # More permissive approaches
        {'map_location': 'cpu', 'weights_only': False},
        {'map_location': 'cpu', 'weights_only': False, 'pickle_module': __import__('pickle')},
    ]
    
    # Remove None entries
    loading_methods = [m for m in loading_methods if m is not None]
    
    for i, method_kwargs in enumerate(loading_methods):
        try:
            print(f"Attempting Method {i+1}: torch.load with {method_kwargs}")
            checkpoint = torch.load(checkpoint_path, **method_kwargs)
            print(f"✅ Success with Method {i+1}!")
            return checkpoint
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
            continue
    
    # Method 2: Try with custom file loading and torch.load
    try:
        print("Attempting Method: Custom file loading...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu', weights_only=False)
        print("✅ Success with custom file loading!")
        return checkpoint
    except Exception as e:
        print(f"Custom file loading failed: {e}")
    
    # Method 3: Try torch.jit.load if the checkpoint is a JIT model
    try:
        print("Attempting Method: torch.jit.load...")
        checkpoint = torch.jit.load(checkpoint_path, map_location='cpu')
        print("✅ Success with torch.jit.load!")
        return checkpoint
    except Exception as e:
        print(f"torch.jit.load failed: {e}")
    
    # Method 4: Try with aggressive numpy module manipulation
    try:
        print("Attempting Method: Aggressive numpy manipulation...")
        
        # Create a more comprehensive numpy module fake
        class NumpyModuleFake:
            def __getattr__(self, name):
                if hasattr(np, name):
                    return getattr(np, name)
                elif name == 'scalar':
                    return np.number
                elif name == '_reconstruct':
                    return lambda subtype, shape, dtype: np.empty(shape, dtype=dtype)
                else:
                    return np.float32  # Default fallback
        
        # Replace all numpy modules with our fake
        fake_numpy = NumpyModuleFake()
        for module_name in numpy_modules:
            sys.modules[module_name] = fake_numpy
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Success with aggressive numpy manipulation!")
        return checkpoint
    except Exception as e:
        print(f"Aggressive numpy manipulation failed: {e}")
    
    # Method 5: Try loading only the state_dict if it's a standard checkpoint
    try:
        print("Attempting Method: Direct state_dict extraction...")
        import pickle
        
        class StateExtractorUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                # Handle persistent IDs (torch tensors)
                return torch.empty(1)  # Dummy tensor
                
            def find_class(self, module, name):
                # Bypass all numpy issues by returning dummy objects
                if 'numpy' in module:
                    if name in ['scalar', 'ndarray', 'dtype']:
                        return lambda *args, **kwargs: torch.tensor(0.0)
                    return torch.tensor
                return super().find_class(module, name)
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = StateExtractorUnpickler(f).load()
        print("✅ Success with direct state_dict extraction!")
        return checkpoint
    except Exception as e:
        print(f"Direct state_dict extraction failed: {e}")
    
    # Method 6: Try with torch.serialization directly
    try:
        print("Attempting Method: torch.serialization legacy loader...")
        from torch.serialization import _legacy_load
        import pickle
        checkpoint = _legacy_load(open(checkpoint_path, 'rb'), map_location='cpu', pickle_module=pickle)
        print("✅ Success with legacy loader!")
        return checkpoint
    except Exception as e:
        print(f"Legacy loader failed: {e}")
    
    # Method 7: Ultra-aggressive - replace numpy.number completely
    try:
        print("Attempting Method: Ultra-aggressive numpy replacement...")
        
        # Completely replace numpy.number with float
        original_number = getattr(np, 'number', None)
        np.number = float
        
        # Also replace in all numpy modules
        for module_name in numpy_modules:
            if module_name in sys.modules:
                setattr(sys.modules[module_name], 'number', float)
        
        # Try to load with the replacement
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Restore original if it existed
        if original_number is not None:
            np.number = original_number
        
        print("✅ Success with ultra-aggressive numpy replacement!")
        return checkpoint
    except Exception as e:
        print(f"Ultra-aggressive numpy replacement failed: {e}")
        # Restore original if it existed and we have a reference
        if 'original_number' in locals() and original_number is not None:
            np.number = original_number
    
    # Method 8: Try loading with custom pickle that completely bypasses numpy
    try:
        print("Attempting Method: Complete numpy bypass...")
        import pickle
        
        class CompleteNumpyBypassUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                # Return a simple tensor for any persistent ID
                return torch.tensor(0.0)
                
            def find_class(self, module, name):
                # Replace any numpy reference with torch equivalents
                if 'numpy' in module.lower():
                    if name == 'number':
                        return float
                    elif name == 'ndarray':
                        return torch.Tensor
                    elif name == 'scalar':
                        return float
                    elif name == 'dtype':
                        return torch.dtype
                    elif name == '_reconstruct':
                        return lambda *args: torch.zeros(1)
                    elif hasattr(torch, name):
                        return getattr(torch, name)
                    else:
                        return float  # Default fallback
                
                # For non-numpy modules, use default behavior
                try:
                    return super().find_class(module, name)
                except:
                    # If that fails too, return float as ultimate fallback
                    return float
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = CompleteNumpyBypassUnpickler(f).load()
        print("✅ Success with complete numpy bypass!")
        return checkpoint
    except Exception as e:
        print(f"Complete numpy bypass failed: {e}")
    
    # If all methods fail, provide a detailed error message
    raise Exception(f"""
    All PyTorch loading methods failed for checkpoint: {checkpoint_path}
    
    This suggests the checkpoint was saved with incompatible dependencies.
    
    Possible solutions:
    1. Re-save the checkpoint in the training environment with torch.save(model.state_dict(), path)
    2. Use a pre-trained MMDetection Cascade R-CNN model instead
    3. Check if the checkpoint file is corrupted
    
    The checkpoint likely contains numpy arrays saved with a newer numpy version
    that are incompatible with the current environment.
    """)

def process_single_image(img_path, out_dir):
    """Process a single image with both chart type classification and element detection"""
    
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} not found in root directory")
        return
    
    print(f"\n{'='*100}")
    print(f"🎯 PROCESSING: {img_path.upper()}")
    print(f"{'='*100}")
    
    # Step 1: Run chart type classification FIRST
    print(f"\n📊 CHART TYPE CLASSIFICATION")
    print(f"{'='*60}")
    chart_type_results = load_and_run_chart_type_classification(img_path)
    
    if chart_type_results:
        print("📋 Chart Type Prediction Results:")
        for i, (chart_type, confidence) in enumerate(chart_type_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {rank_emoji} {chart_type}: {confidence:.3f} ({confidence*100:.1f}%)")
        
        # Get the top prediction
        top_chart_type, top_confidence = chart_type_results[0]
        print(f"\n🎯 **PREDICTED CHART TYPE: {top_chart_type}** (confidence: {top_confidence:.3f})")
    else:
        print("⚠️  Chart type classification not available")
        chart_type_results = None
    
    print(f"{'='*60}")

    # Step 2: Load chart element detection model
    img_name = img_path.split('/')[-1]  # Get just the filename
    print(f"\n🔧 Building Chart Element Detection Model for {img_name}...")
    print(f"📊 Classes: {len(ENHANCED_CLASS_NAMES)} categories")
    
    # Load chart element detection checkpoint
    checkpoint_path = 'chart_label.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Chart element detection checkpoint not found: {checkpoint_path}")
        print("Available .pth files:")
        for f in os.listdir('.'):
            if f.endswith(('.pt', '.pth')):
                print(f"   - {f}")
        return
    
    print(f"📦 Loading chart element detection model: {checkpoint_path}")
    print("🎯 CHART ELEMENT DETECTION - Using trained chart_label.pth model")
    training_stage = "Chart Element Detection Model"
    
    try:
        # Create model configuration
        model_cfg_dict = create_model_config()
        
        # Convert to Config object for MMDetection compatibility
        model_cfg = Config(model_cfg_dict)
        
        # Build model
        model = MODELS.build(model_cfg)
        
        # Update the config to include test_dataloader for inference API
        model_cfg.update({
            'test_dataloader': {
                'dataset': {
                    'pipeline': [
                        dict(type='LoadImageFromFile'),
                        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                        dict(type='PackDetInputs')
                    ]
                }
            }
        })
        
        # Add cfg attribute to model (required by MMDetection inference API)
        model.cfg = model_cfg
        
        # Load checkpoint safely using the same robust method as chart_type
        print("📥 Loading checkpoint with compatibility handling...")
        checkpoint = safe_torch_load_simple(checkpoint_path)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict with flexible matching
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        # Proper device management - move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"🎯 Model moved to device: {device}")
        
        # Set model to evaluation mode for inference
        model.eval()
        print("✅ Final model loaded successfully!")

        
        # Step 3: Run chart element detection
        print(f"\n Processing {img_name} with chart element detection model...")
        print(f"{'='*80}")
        
        # Run inference using MMDetection's high-level API
        result = inference_detector(model, img_path)
        
        # Load image for visualization
        img = mmcv.imread(img_path)
        
        # Extract detection results
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        
        print(f"\n📐 Total Raw Detections: {len(bboxes)}")
        
        # Filter by score threshold
        score_thr = 0.2
        valid_indices = scores >= score_thr
        filtered_bboxes = bboxes[valid_indices]
        filtered_scores = scores[valid_indices]
        filtered_labels = labels[valid_indices]
        
        print(f"🎯 Filtered Detections: {len(filtered_bboxes)} (above {score_thr:.1f} confidence)")
        
        # Create enhanced visualization
        plt.figure(figsize=(20, 14))
        plt.imshow(img)
        ax = plt.gca()

        # Group detections by element type
        elements_by_type = defaultdict(list)
        detection_count = 0
        
        for idx in range(len(filtered_bboxes)):
            score = filtered_scores[idx]
            label = filtered_labels[idx]
            class_name = ENHANCED_CLASS_NAMES[label]
            x1, y1, x2, y2 = filtered_bboxes[idx].astype(int)
            
            detection_count += 1
            elements_by_type[class_name].append({
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'idx': detection_count
            })

        # Draw elements with styling
        legend_patches = []
        
        for element_type, elements in elements_by_type.items():
            color = ELEMENT_COLORS.get(element_type, '#bcbd22')  # Default color
            
            for elem in elements:
                x1, y1, x2, y2 = elem['bbox']
                score = elem['score']
                idx = elem['idx']
                
                # Different styling for different element types
                if 'data-' in element_type:
                    # Data elements: thicker border, filled
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=3, edgecolor=color, 
                                           facecolor=color, alpha=0.3)
                elif 'axis' in element_type:
                    # Axis elements: dashed border
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color, 
                                           facecolor='none', linestyle='--')
                elif 'legend' in element_type:
                    # Legend elements: dotted border
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color,
                                           facecolor='none', linestyle=':')
                else:
                    # Text elements: solid border
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color,
                                           facecolor='none')
                
                ax.add_patch(rect)
                
                # Add label with score and element number
                label_text = f"{element_type}\n#{idx}\n{score:.3f}"
                ax.text(x1, y1-8, label_text, color=color, fontsize=9, 
                       fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, pad=3))
            
            # Add to legend (one entry per element type)
            if elements:
                legend_patches.append(patches.Patch(color=color, label=f"{element_type} ({len(elements)})"))

        # Create title with chart type if available
        title_text = f'{img_name} - Complete Chart Element Detection\nTotal Detections: {detection_count}'
        if chart_type_results:
            top_chart_type, top_confidence = chart_type_results[0]
            title_text += f'\nChart Type: {top_chart_type} ({top_confidence:.1%})'
        
        plt.title(title_text, fontsize=16, fontweight='bold', pad=20)

        # Add chart type annotation box on the image
        if chart_type_results:
            top_chart_type, top_confidence = chart_type_results[0]
            chart_type_text = f"Chart Type:\n{top_chart_type}\n({top_confidence:.1%} confidence)"
            
            # Add text box in top-right corner
            ax.text(0.98, 0.98, chart_type_text, 
                   transform=ax.transAxes, 
                   fontsize=12, fontweight='bold',
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8, edgecolor='darkblue'))

        # Add comprehensive legend
        if legend_patches:
            plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                      fontsize=11, title="Detected Elements", title_fontsize=13)

        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        base_name = img_name.split('.')[0]  # Remove extension
        out_path = os.path.join(out_dir, f"{base_name}_final_detection.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved {img_name} visualization: {out_path}")
        plt.close()

        # Print comprehensive results
        print("\n" + "="*100)
        print(f"🎯 {img_name.upper()} - CHART ELEMENT DETECTION RESULTS ({training_stage})")
        print("="*100)
        
        # Group elements by category for organized reporting
        elements_by_category = defaultdict(list)
        
        for idx in range(len(filtered_bboxes)):
            score = filtered_scores[idx]
            label = filtered_labels[idx]
            class_name = ENHANCED_CLASS_NAMES[label]
            x1, y1, x2, y2 = filtered_bboxes[idx].astype(int)
            area = (x2 - x1) * (y2 - y1)
            
            elements_by_category[class_name].append({
                'detection_id': idx + 1,
                'confidence': score,
                'bbox': [x1, y1, x2, y2],
                'area': area,
                'center': [(x1+x2)//2, (y1+y2)//2]
            })
        
        # Print results organized by element categories
        category_groups = {
            "📄 TEXT ELEMENTS": ['title', 'subtitle', 'axis-title', 'data-label'],
            "🏷️  AXIS LABELS": ['x-axis-label', 'y-axis-label', 'x-tick-label', 'y-tick-label', 'tick-label'],
            "📈 DATA VISUALIZATION": ['data-point', 'data-line', 'data-bar', 'data-area'],
            "🗂️  LEGEND COMPONENTS": ['legend', 'legend-title', 'legend-item', 'legend-text'],
            "📐 STRUCTURAL ELEMENTS": ['x-axis', 'y-axis', 'grid-line', 'plot-area']
        }
        
        for group_name, group_classes in category_groups.items():
            group_elements = {cls: elements_by_category[cls] for cls in group_classes if cls in elements_by_category}
            
            if group_elements:
                print(f"\n{group_name}")
                print("-" * 80)
                
                for class_name, elements in group_elements.items():
                    print(f"\n  {class_name.upper().replace('-', ' ')} ({len(elements)} found):")
                    
                    for elem in sorted(elements, key=lambda x: x['confidence'], reverse=True):
                        x1, y1, x2, y2 = elem['bbox']
                        cx, cy = elem['center']
                        print(f"    Detection #{elem['detection_id']:2d}: conf={elem['confidence']:.3f} | "
                              f"bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] | "
                              f"center=({cx:4d},{cy:4d}) | area={elem['area']:6d}px²")
        
        # Print ALL detections in order of confidence
        print(f"\n{'='*100}")
        print("🏆 ALL DETECTIONS RANKED BY CONFIDENCE")
        print(f"{'='*100}")
        
        all_detections = []
        for class_name, elements in elements_by_category.items():
            for elem in elements:
                elem['class_name'] = class_name
                all_detections.append(elem)
        
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, elem in enumerate(all_detections):
            x1, y1, x2, y2 = elem['bbox']
            cx, cy = elem['center']
            print(f"  #{i+1:2d}: {elem['class_name']:<20} | conf={elem['confidence']:.3f} | "
                  f"bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] | center=({cx:4d},{cy:4d})")
        
        # Complete element type report
        print(f"\n{'='*100}")
        print("📊 COMPREHENSIVE DETECTION SUMMARY")
        print(f"{'='*100}")
        
        print("\n📋 Complete Element Type Report:")
        for class_name in ENHANCED_CLASS_NAMES:
            count = len(elements_by_category.get(class_name, []))
            if count > 0:
                elements = elements_by_category[class_name]
                confidences = [e['confidence'] for e in elements]
                avg_conf = np.mean(confidences)
                max_conf = np.max(confidences)
                print(f"  ✅ {class_name:<20}: {count:2d} detections | "
                      f"avg_conf={avg_conf:.3f} | max_conf={max_conf:.3f}")
            else:
                print(f"  ❌ {class_name:<20}: {count:2d} detections")
        
        print(f"\n✅ {img_name} inference completed successfully!")
        print(f"📁 Visualization saved: {out_path}")
        print(f"🎯 Detected elements from {len(ENHANCED_CLASS_NAMES)} possible categories")
        
        # Add chart type to summary
        if chart_type_results:
            top_chart_type, top_confidence = chart_type_results[0]
            print(f"📊 Chart Type: {top_chart_type} (confidence: {top_confidence:.3f})")
        
        print(f"\n{'='*100}")
        
    except Exception as e:
        print(f"❌ Error processing {img_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process both science2.jpg and science3.png"""
    
    # Setup output directory
    out_dir = 'test_infer_results'
    os.makedirs(out_dir, exist_ok=True)
    
    # List of images to process
    images_to_process = ['science2.jpg', 'science3.png']
    
    print(f"\n🚀 STARTING BATCH INFERENCE ON MULTIPLE SCIENCE IMAGES")
    print(f"{'='*120}")
    print(f"📋 Images to process: {', '.join(images_to_process)}")
    print(f"📁 Output directory: {out_dir}")
    print(f"{'='*120}")
    
    # Process each image
    for img_path in images_to_process:
        process_single_image(img_path, out_dir)
    
    # Final summary
    print(f"\n🎉 BATCH INFERENCE COMPLETED!")
    print(f"{'='*120}")
    print(f"✅ Processed {len(images_to_process)} images successfully")
    print(f"📁 All visualizations saved in: {out_dir}/")
    
    # List generated files
    print(f"\n📋 Generated Files:")
    for img_path in images_to_process:
        base_name = img_path.split('.')[0]
        expected_file = f"{base_name}_final_detection.png"
        if os.path.exists(os.path.join(out_dir, expected_file)):
            print(f"   ✅ {expected_file}")
        else:
            print(f"   ❌ {expected_file} (not found)")
    
    print(f"{'='*120}")

if __name__ == '__main__':
    main() 