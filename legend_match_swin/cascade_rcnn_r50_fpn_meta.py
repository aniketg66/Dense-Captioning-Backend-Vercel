# cascade_rcnn_r50_fpn_meta.py - Enhanced config with Swin Transformer backbone
# 
# PROGRESSIVE LOSS STRATEGY:
# - All 3 Cascade stages start with SmoothL1Loss for stable initial training
# - At epoch 5, Stage 3 (final stage) switches to GIoULoss via ProgressiveLossHook  
# - Stage 1 & 2 remain SmoothL1Loss throughout training
# - This ensures model stability before introducing more complex IoU-based losses
_base_ = [
    '/content/mmdetection/configs/_base_/datasets/coco_detection.py', 
    '/content/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/content/mmdetection/configs/_base_/default_runtime.py'
]

# Custom imports - this registers our modules without polluting config namespace
custom_imports = dict(
    imports=[
        'legend_match_swin.custom_models.custom_dataset',
        'legend_match_swin.custom_models.register',
        'legend_match_swin.custom_models.custom_hooks',
        'legend_match_swin.custom_models.progressive_loss_hook',
    ],
    allow_failed_imports=False
)

# Add to Python path
import sys
sys.path.insert(0, '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch')

# Custom Cascade model with coordinate handling for chart data
model = dict(
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
    # Progressive Loss Strategy: Start with SmoothL1 for all 3 stages
    # Stage 3 (final stage) will switch to GIoU at epoch 5 via ProgressiveLossHook
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
            # Stage 1: Always SmoothL1Loss (coarse detection)
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
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            # Stage 2: Always SmoothL1Loss (intermediate refinement)
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
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            # Stage 3: SmoothL1 → GIoU at epoch 5 (progressive switching)
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
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
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
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
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

# Dataset settings - using cleaned annotations
dataset_type = 'ChartDataset'
data_root = ''  # Remove data_root duplication

# Define the 21 chart element classes that match the annotations
CLASSES = (
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
)

# Updated to use cleaned annotation files
train_dataloader = dict(
    batch_size=2,  # Increased back to 2
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='legend_data/annotations_JSON_cleaned/train_enriched.json',  # Full path
        data_prefix=dict(img='legend_data/train/images/'),  # Full path
        metainfo=dict(classes=CLASSES),  # Tell dataset what classes to expect
        filter_cfg=dict(filter_empty_gt=True, min_size=0, class_specific_min_sizes={
            'data-point': 16,    # Back to 16x16 from 32x32 
            'data-bar': 16,      # Back to 16x16 from 32x32
            'tick-label': 16,    # Back to 16x16 from 32x32
            'x-tick-label': 16,  # Back to 16x16 from 32x32  
            'y-tick-label': 16   # Back to 16x16 from 32x32
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1600, 1000), keep_ratio=True),  # Higher resolution for tiny objects
            dict(type='RandomFlip', prob=0.5),
            dict(type='ClampBBoxes'),  # Ensure bboxes stay within image bounds
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='legend_data/annotations_JSON_cleaned/val_enriched_with_info.json',  # Full path
        data_prefix=dict(img='legend_data/train/images/'),  # All images are in train/images
        metainfo=dict(classes=CLASSES),  # Tell dataset what classes to expect
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1600, 1000), keep_ratio=True),  # Base resolution for validation
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='ClampBBoxes'),  # Ensure bboxes stay within image bounds
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

# Enhanced evaluators with debugging
val_evaluator = dict(
    type='CocoMetric',
    ann_file='legend_data/annotations_JSON_cleaned/val_enriched_with_info.json',  # Using cleaned annotations
    metric='bbox',
    format_only=False,
    classwise=True,  # Enable detailed per-class metrics table
    proposal_nums=(100, 300, 1000))  # More detailed AR metrics

test_evaluator = val_evaluator

# Add custom hooks for debugging empty results
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CompatibleCheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Add NaN recovery hook for graceful handling like Faster R-CNN
custom_hooks = [
    dict(type='SkipBadSamplesHook', interval=1),           # Skip samples with bad GT data
    dict(type='ChartTypeDistributionHook', interval=500),  # Monitor class distribution
    dict(type='MissingImageReportHook', interval=1000),    # Track missing images
    dict(type='NanRecoveryHook',                           # For logging & monitoring
         fallback_loss=1.0,
         max_consecutive_nans=100,
         log_interval=50),
    dict(type='ProgressiveLossHook',                       # Progressive loss switching
         switch_epoch=5,                                   # Switch stage 3 to GIoU at epoch 5
         target_loss_type='GIoULoss',                      # Use GIoU for stage 3 (final stage)
         loss_weight=1.0,                                  # Keep same loss weight
         warmup_epochs=2,                                  # Monitor for 2 epochs after switch
         monitor_stage_weights=True),                      # Log stage loss details
]

# Training configuration - extended to 40 epochs for Swin Base on small objects
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer with standard stable settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35.0, norm_type=2)
)

# Extended learning rate schedule with cosine annealing for Swin Base
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.05,  # 1e-4 / 2e-2 = 0.05 (warmup from 1e-4 to 2e-2)
        by_epoch=False, 
        begin=0, 
        end=1000),  # 1k iteration warmup
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=40,  # Match max_epochs
        by_epoch=True,
        T_max=40,
        eta_min=1e-6,  # Minimum learning rate
        convert_to_iter_based=True)
]

# Work directory 
work_dir = './work_dirs/cascade_rcnn_swin_base_40ep_cosine_fpn_meta'

# Multi-scale test configuration (uncomment to enable)
# img_scales = [(800, 500), (1600, 1000), (2400, 1500)]  # 0.5x, 1.0x, 1.5x scales
# tta_model = dict(
#     type='DetTTAModel',
#     tta_cfg=dict(
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=100)
# )

# Fresh start
resume = False
load_from = None

