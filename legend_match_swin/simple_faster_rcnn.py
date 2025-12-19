# simple_faster_rcnn.py - Ultra-stable Faster R-CNN for chart detection
_base_ = [
    '/content/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '/content/mmdetection/configs/_base_/datasets/coco_detection.py', 
    '/content/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/content/mmdetection/configs/_base_/default_runtime.py'
]

# Custom imports
custom_imports = dict(
    imports=[
        'legend_match_swin.custom_models',  # Import the full package to register all models
        'legend_match_swin.custom_models.custom_dataset',
        'legend_match_swin.custom_models.custom_faster_rcnn_with_meta',  # Explicitly import the custom model
        'legend_match_swin.custom_models.register'
    ],
    allow_failed_imports=False
)

# Custom Faster R-CNN model with coordinate handling for chart data
model = dict(
    type='CustomFasterRCNNWithMeta',  # Use custom model with coordinate handling
    coordinate_standardization=dict(
        enabled=True,
        origin='bottom_left',      # Match annotation creation coordinate system
        normalize=True,
        relative_to_plot=False,    # Keep simple for now
        scale_to_axis=False        # Keep simple for now
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=21  # 21 enhanced categories instead of 80
        )
    ),
    # Ultra-conservative test config
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300))
)

# Dataset configuration
dataset_type = 'ChartDataset'
data_root = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/'

# Simple training pipeline
train_pipeline = [
    dict(type='RobustLoadImageFromFile', try_real_images=True, fallback_to_dummy=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ClampBBoxes', min_size=12),  # Increased to match larger bounding boxes (up to 16x16)
    dict(type='Resize', scale=(800, 800), keep_ratio=True),  # Smaller scale for stability
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='RobustLoadImageFromFile', try_real_images=True, fallback_to_dummy=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ClampBBoxes', min_size=12),  # Increased to match larger bounding boxes (up to 16x16)
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Conservative data loaders
train_dataloader = dict(
    batch_size=1,  # Very small batch size for stability
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_JSON/train_enriched.json',  # Use enriched annotations
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=12),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_JSON/val_enriched_with_info.json',  # Use enriched annotations
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=12),
        test_mode=True,
        pipeline=test_pipeline))

# Simple evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_JSON/val_enriched_with_info.json',  # Use enriched annotations
    metric='bbox',
    format_only=False,
    classwise=True)

# Ultra-conservative training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
val_cfg = dict(type='ValLoop')

# Rock-solid optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),  # Standard LR
    clip_grad=dict(max_norm=1.0, norm_type=2))  # Very aggressive clipping

# Simple learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),  # Short warmup
    dict(
        type='MultiStepLR',
        begin=0,
        end=3,
        by_epoch=True,
        milestones=[2],
        gamma=0.1)
]

# Minimal hooks - disable problematic ones
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# NO custom hooks that could cause issues
custom_hooks = []

# Clean work directory
work_dir = './work_dirs/simple_faster_rcnn_stable'

# Fresh start
resume = False
load_from = None 