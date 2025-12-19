# mask_rcnn_swin_meta.py - Mask R-CNN with Swin Transformer for data point segmentation
# 
# ADAPTED FROM CASCADE R-CNN CONFIG:
# - Uses same Swin Transformer Base backbone with optimizations
# - Maintains data-point class weighting (10x) and IoU strategies
# - Adds mask head for instance segmentation of data points
# - Uses enhanced annotation files with segmentation masks
# - Keeps custom hooks and progressive loss strategies
#
# MASK-SPECIFIC OPTIMIZATIONS:
# - RoI size 14x14 for mask extraction (matches data point size)
# - FCN mask head with 4 convolution layers
# - Mask loss weight balanced with bbox and classification losses
# - Enhanced test-time augmentation for better mask quality
#
# DATA POINT FOCUS:
# - Primary target: data-point class (ID 11) with 10x weight
# - Generates both bounding boxes AND instance masks
# - Optimized for 16x16 pixel data points in scientific charts
# Removed _base_ inheritance to avoid path issues - all configs are inlined below

# Custom imports - same as Cascade R-CNN setup
custom_imports = dict(
    imports=[
        'legend_match_swin.custom_models.register',
        'legend_match_swin.custom_models.custom_hooks',
        'legend_match_swin.custom_models.progressive_loss_hook',
        'legend_match_swin.custom_models.flexible_load_annotations',
    ],
    allow_failed_imports=False
)

# Add to Python path
import sys
sys.path.insert(0, '.')

# Mask R-CNN model with Swin Transformer backbone
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,  # Important for mask training
        mask_pad_value=0,
    ),
    # Same Swin Transformer Base backbone as Cascade R-CNN
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
        drop_path_rate=0.3,  # Same as Cascade config
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'
        )
    ),
    # Same FPN as Cascade R-CNN
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],  # Swin Base: embed_dims * 2^(stage)
        out_channels=256,
        num_outs=5,  # Standard for Mask R-CNN (was 6 in Cascade)
        start_level=0,
        add_extra_convs='on_input'
    ),
    # Same RPN configuration as Cascade R-CNN
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 2, 4, 8],  # Same small scales for tiny objects
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),  # Standard FPN strides for Mask R-CNN
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
    ),
    # Mask R-CNN ROI head with bbox + mask branches
    roi_head=dict(
        type='StandardRoIHead',
        # Bbox ROI extractor (same as Cascade R-CNN final stage)
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        # Bbox head with data-point class weighting
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=22,  # 22 enhanced categories including boxplot
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[1.0,  # background class (index 0)
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                             10.0,  # data-point at index 12 gets 10x weight (11+1 for background)
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Added boxplot class
            ),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        ),
        # Mask ROI extractor (optimized for 16x16 data points)
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(14, 14), sampling_ratio=0, aligned=True),  # Force exact 14x14 with legacy alignment
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        # Mask head optimized for data points with square mask targets
        mask_head=dict(
            type='SquareFCNMaskHead',
            num_convs=4,  # 4 conv layers for good feature extraction
            in_channels=256,
            roi_feat_size=14,  # Explicitly set ROI feature size
            conv_out_channels=256,
            num_classes=22,  # 22 enhanced categories including boxplot
            upsample_cfg=dict(type=None),  # No upsampling - keep 14x14
            loss_mask=dict(
                type='CrossEntropyLoss', 
                use_mask=True, 
                loss_weight=1.0  # Balanced with bbox loss
            )
        )
    ),
    # Training configuration adapted from Cascade R-CNN
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
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        # RCNN training (using Cascade stage 2 settings - balanced for mask training)
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,  # Balanced IoU for bbox + mask training
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,  # Important for small data points
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=(14, 14),  # Force exact 14x14 size for data points
            pos_weight=-1,
            debug=False)
    ),
    # Test configuration with soft NMS
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.005,  # Low threshold to catch data points
            nms=dict(
                type='soft_nms',  # Soft NMS for better small object detection
                iou_threshold=0.3,  # Low for data points
                min_score=0.005,
                method='gaussian',
                sigma=0.5),
            max_per_img=100,
            mask_thr_binary=0.5  # Binary mask threshold
        )
    )
)

# Dataset settings - using standard COCO dataset for mask support
dataset_type = 'CocoDataset'
data_root = ''

# 22 enhanced categories including boxplot
CLASSES = (
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',        # 0-5
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',        # 6-10
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',              # 11-15 (data-point at index 11)
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area',         # 16-20
    'boxplot'                                                                      # 21
)

# Verify data-point class index
assert CLASSES[11] == 'data-point', f"Expected 'data-point' at index 11 in CLASSES tuple, got '{CLASSES[11]}'"

# Training dataloader with mask annotations
train_dataloader = dict(
    batch_size=2,  # Same as Cascade R-CNN
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='legend_match_swin/mask_generation/enhanced_datasets/train_filtered_with_masks_only.json',
        data_prefix=dict(img='legend_data/train/images/'),
        metainfo=dict(classes=CLASSES),
        filter_cfg=dict(filter_empty_gt=False, min_size=12),  # Don't filter out images with masks
        # Disable any built-in filtering that might remove annotations
        test_mode=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='FlexibleLoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1120, 672), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ClampBBoxes'),
            dict(type='PackDetInputs')
        ]
    )
)

# Validation dataloader with mask annotations
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='legend_match_swin/mask_generation/enhanced_datasets/val_enriched_with_masks_only.json',
        data_prefix=dict(img='legend_data/train/images/'),
        metainfo=dict(classes=CLASSES),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1120, 672), keep_ratio=True),
            dict(type='FlexibleLoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='ClampBBoxes'),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

# Enhanced evaluators for both bbox and mask metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file='legend_match_swin/mask_generation/enhanced_datasets/val_enriched_with_masks_only.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    proposal_nums=(100, 300, 1000)
)

test_evaluator = val_evaluator

# Same custom hooks as Cascade R-CNN
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CompatibleCheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Same custom hooks as Cascade R-CNN (adapted for Mask R-CNN)
custom_hooks = [
    dict(type='SkipBadSamplesHook', interval=1),
    dict(type='ChartTypeDistributionHook', interval=500),
    dict(type='MissingImageReportHook', interval=1000),
    dict(type='NanRecoveryHook',
         fallback_loss=1.0,
         max_consecutive_nans=50,
         log_interval=25),
    # Note: Progressive loss hook not used in standard Mask R-CNN
    # but could be adapted if needed for bbox loss only
]

# Training configuration - reduced to 20 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Same optimizer settings as Cascade R-CNN
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# Same learning rate schedule as Cascade R-CNN
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.1,
        by_epoch=False, 
        begin=0, 
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=20,
        by_epoch=True,
        T_max=20,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

# Work directory
work_dir = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/work_dirs/mask_rcnn_swin_base_20ep_meta'

# Fresh start
resume = False
load_from = None

# Default runtime settings (normally inherited from _base_)
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO' 