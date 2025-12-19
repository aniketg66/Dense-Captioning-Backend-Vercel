#!/usr/bin/env python3
"""
Combined inference script:
- Uses the original chart_label+.pth model (with legend_match_swin config) for all chart elements except data-point.
- Uses the new chart_datapoint.pth model (with Swin config) for data-point detection only.
- Combines results and visualizes in the same style as inference_science2_direct.py.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from collections import defaultdict

# Add legend_match_swin to Python path FIRST
sys.path.insert(0, '.')
sys.path.insert(0, '../legend_match_swin')

from mmdet.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine.config import Config
from mmdet.apis import inference_detector, init_detector

register_all_modules()
from legend_match_swin.custom_models.register import register_all_modules as register_custom_modules
register_custom_modules()

# Enhanced class names (21 categories)
ENHANCED_CLASS_NAMES = [
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
]

ELEMENT_COLORS = {
    'title': '#1f77b4', 'subtitle': '#aec7e8', 'axis-title': '#2ca02c',
    'x-axis-label': '#d62728', 'y-axis-label': '#d62728',
    'x-tick-label': '#ff7f0e', 'y-tick-label': '#ff7f0e', 'tick-label': '#ff7f0e',
    'data-label': '#9467bd',
    'legend': '#8c564b', 'legend-title': '#e377c2', 'legend-item': '#f7b6d3', 'legend-text': '#c5b0d5',
    'data-point': '#2ca02c', 'data-line': '#98df8a', 'data-bar': '#d62728', 'data-area': '#ff9896',
    'x-axis': '#7f7f7f', 'y-axis': '#7f7f7f', 'grid-line': '#c7c7c7', 'plot-area': '#17becf'
}

def create_model_config():
    # Copied from inference_science2_direct.py
    return dict(
        type='CustomCascadeWithMeta',
        coordinate_standardization=dict(
            enabled=True,
            origin='bottom_left',
            normalize=True,
            relative_to_plot=False,
            scale_to_axis=False
        ),
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
        backbone=dict(
            type='SwinTransformer',
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
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
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=6,
            start_level=0,
            add_extra_convs='on_input'
        ),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[1, 2, 4, 8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64, 128]),
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
                        loss_weight=1.0,
                        class_weight=[1.0]*12+[10.0]+[1.0]*8),
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
                        loss_weight=1.0,
                        class_weight=[1.0]*12+[10.0]+[1.0]*8),
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
                        target_stds=[0.02, 0.02, 0.04, 0.04]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0,
                        class_weight=[1.0]*12+[10.0]+[1.0]*8),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            ],
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=21,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=300,
                mask_thr_binary=0.5
            )
        )
    )

def load_datapoint_model():
    config_path = "../legend_match_swin/mask_rcnn_swin_meta.py"
    checkpoint_path = "chart_datapoint.pth"
    model = init_detector(config_path, checkpoint_path, device='cpu')
    return model

def filter_out_class(detections, class_names, exclude_class='data-point', score_thr=0.2):
    # Remove all detections of exclude_class
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

def filter_only_class(detections, class_names, target_class='data-point', score_thr=0.5):
    # Keep only detections of target_class
    pred_instances = detections.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    keep = [i for i, l in enumerate(labels) if class_names[l] == target_class and scores[i] >= score_thr]
    filtered = {
        'bboxes': bboxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }
    return filtered

def process_single_image(img_path, out_dir, orig_model, datapoint_model):
    img_name = os.path.basename(img_path)
    print(f"\n=== PROCESSING: {img_name} ===")
    img = mmcv.imread(img_path)

    # Run original model (exclude data-point)
    orig_result = inference_detector(orig_model, img_path)
    filtered_orig = filter_out_class(orig_result, ENHANCED_CLASS_NAMES, exclude_class='data-point', score_thr=0.2)

    # Run new data-point model (only data-point)
    datapoint_result = inference_detector(datapoint_model, img_path)
    filtered_datapoint = filter_only_class(datapoint_result, ENHANCED_CLASS_NAMES, target_class='data-point', score_thr=0.5)

    # Combine detections
    bboxes = np.concatenate([filtered_orig['bboxes'], filtered_datapoint['bboxes']], axis=0)
    scores = np.concatenate([filtered_orig['scores'], filtered_datapoint['scores']], axis=0)
    labels = np.concatenate([filtered_orig['labels'], filtered_datapoint['labels']], axis=0)

    # Visualization (reuse logic from inference_science2_direct.py)
    plt.figure(figsize=(20, 14))
    plt.imshow(img)
    ax = plt.gca()
    elements_by_type = defaultdict(list)
    detection_count = 0
    # Track which detections are from the new datapoint model
    datapoint_indices = set(range(len(filtered_orig['bboxes']), len(bboxes)))
    for idx in range(len(bboxes)):
        score = scores[idx]
        label = labels[idx]
        class_name = ENHANCED_CLASS_NAMES[label]
        x1, y1, x2, y2 = bboxes[idx].astype(int)
        detection_count += 1
        # Mark if this is a datapoint from the new model
        is_new_datapoint = (idx in datapoint_indices and class_name == 'data-point')
        elements_by_type[class_name].append({
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'idx': detection_count,
            'is_new_datapoint': is_new_datapoint
        })
    legend_patches = []
    for element_type, elements in elements_by_type.items():
        for elem in elements:
            x1, y1, x2, y2 = elem['bbox']
            score = elem['score']
            idx = elem['idx']
            # Use a distinct color for data-point bboxes from the new model
            if element_type == 'data-point' and elem['is_new_datapoint']:
                color = '#e600e6'  # Magenta for new data-point detections
            else:
                color = ELEMENT_COLORS.get(element_type, '#bcbd22')
            if 'data-' in element_type:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=color, facecolor=color, alpha=0.3)
            elif 'axis' in element_type:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
            elif 'legend' in element_type:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none', linestyle=':')
            else:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            label_text = f"{element_type}\n#{idx}\n{score:.3f}"
            ax.text(x1, y1-8, label_text, color=color, fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, pad=3))
        # Add to legend (one entry per element type)
        if elements:
            if element_type == 'data-point':
                legend_patches.append(patches.Patch(color='#e600e6', label=f"data-point (new model)"))
                legend_patches.append(patches.Patch(color=ELEMENT_COLORS.get('data-point', '#2ca02c'), label=f"data-point (original model)"))
            else:
                legend_patches.append(patches.Patch(color=ELEMENT_COLORS.get(element_type, '#bcbd22'), label=f"{element_type} ({len(elements)})"))
    title_text = f'{img_name} - Combined Chart Element Detection\nTotal Detections: {detection_count}'
    plt.title(title_text, fontsize=16, fontweight='bold', pad=20)
    if legend_patches:
        plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, title="Detected Elements", title_fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    base_name = img_name.split('.')[0]
    out_path = os.path.join(out_dir, f"{base_name}_combined_detection.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved {img_name} visualization: {out_path}")
    plt.close()

def main():
    out_dir = 'test_infer_results'
    os.makedirs(out_dir, exist_ok=True)
    images_to_process = ['science2.jpg', 'science3.png']
    print(f"\n🚀 STARTING COMBINED INFERENCE ON MULTIPLE SCIENCE IMAGES")
    print(f"{'='*120}")
    print(f"📋 Images to process: {', '.join(images_to_process)}")
    print(f"📁 Output directory: {out_dir}")
    print(f"{'='*120}")
    # Load models once
    print("Loading original chart_label+.pth model...")
    orig_model_cfg = Config(create_model_config())
    orig_model = MODELS.build(orig_model_cfg)
    orig_model.cfg = orig_model_cfg
    # Add test_dataloader for inference API compatibility (after model is built)
    orig_model.cfg.test_dataloader = {
        'dataset': {
            'pipeline': [
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1120, 672), keep_ratio=True),
                dict(type='ClampBBoxes'),
                dict(type='PackDetInputs')
            ]
        }
    }
    checkpoint_path = 'chart_label+.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    orig_model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orig_model.to(device)
    orig_model.eval()
    print("Loading chart_datapoint.pth model...")
    datapoint_model = load_datapoint_model()
    datapoint_model.to(device)
    datapoint_model.eval()
    for img_path in images_to_process:
        img_full_path = os.path.join('test', img_path)
        process_single_image(img_full_path, out_dir, orig_model, datapoint_model)
    print(f"\n🎉 COMBINED INFERENCE COMPLETED!")
    print(f"{'='*120}")
    print(f"✅ Processed {len(images_to_process)} images successfully")
    print(f"📁 All visualizations saved in: {out_dir}/")
    print(f"\n📋 Generated Files:")
    for img_path in images_to_process:
        base_name = img_path.split('.')[0]
        expected_file = f"{base_name}_combined_detection.png"
        if os.path.exists(os.path.join(out_dir, expected_file)):
            print(f"   ✅ {expected_file}")
        else:
            print(f"   ❌ {expected_file} (not found)")
    print(f"{'='*120}")

if __name__ == '__main__':
    main() 