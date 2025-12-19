#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and Filter Chart Annotations"""

import json
import os
from tqdm import tqdm

def validate_annotations(ann_file_path):
    """Validate annotations and find invalid bboxes"""
    
    print(f"Validating annotations in: {ann_file_path}")
    
    with open(ann_file_path, 'r') as f:
        ann = json.load(f)
    
    # Create image ID to image info mapping for quick lookup
    images_dict = {img['id']: img for img in ann['images']}
    
    bad_boxes = []
    valid_annotations = []
    
    print(f"Checking {len(ann['annotations'])} annotations...")
    
    for idx, obj in enumerate(tqdm(ann['annotations'])):
        x, y, w, h = obj['bbox']
        image_id = obj['image_id']
        
        # Get image dimensions
        if image_id not in images_dict:
            print(f"Warning: annotation {idx} references non-existent image_id {image_id}")
            bad_boxes.append({
                'type': 'missing_image',
                'annotation_idx': idx,
                'bbox': obj['bbox'],
                'image_id': image_id
            })
            continue
            
        img_info = images_dict[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        is_valid = True
        error_reasons = []
        
        # Check for negative or zero width/height
        if w <= 0:
            is_valid = False
            error_reasons.append(f"width <= 0 ({w})")
        if h <= 0:
            is_valid = False
            error_reasons.append(f"height <= 0 ({h})")
        
        # Check for negative coordinates
        if x < 0:
            is_valid = False
            error_reasons.append(f"x < 0 ({x})")
        if y < 0:
            is_valid = False
            error_reasons.append(f"y < 0 ({y})")
        
        # Check if bbox extends outside image bounds
        # Remember: coords are from bottom-left to top-right
        if x + w > img_width:
            is_valid = False
            error_reasons.append(f"x+w > img_width ({x}+{w} > {img_width})")
        if y + h > img_height:
            is_valid = False
            error_reasons.append(f"y+h > img_height ({y}+{h} > {img_height})")
        
        if not is_valid:
            bad_boxes.append({
                'type': 'invalid_bbox',
                'annotation_idx': idx,
                'bbox': obj['bbox'],
                'image_id': image_id,
                'image_dims': (img_width, img_height),
                'errors': error_reasons
            })
        else:
            valid_annotations.append(obj)
    
    print(f"\nValidation Results:")
    print(f"  Total annotations: {len(ann['annotations'])}")
    print(f"  Valid annotations: {len(valid_annotations)}")
    print(f"  Invalid annotations: {len(bad_boxes)}")
    
    if bad_boxes:
        print(f"\nFirst 10 invalid bboxes:")
        for i, bad in enumerate(bad_boxes[:10]):
            print(f"  {i+1}. Index {bad['annotation_idx']}: {bad['errors']}")
            print(f"     bbox: {bad['bbox']}, image_dims: {bad.get('image_dims', 'unknown')}")
    
    return valid_annotations, bad_boxes, ann

def clean_annotations(ann_file_path, output_file_path=None):
    """Clean annotations by removing invalid bboxes"""
    
    valid_annotations, bad_boxes, original_ann = validate_annotations(ann_file_path)
    
    if len(bad_boxes) == 0:
        print("No invalid annotations found. No cleaning needed.")
        return
    
    # Create cleaned annotation file
    cleaned_ann = original_ann.copy()
    cleaned_ann['annotations'] = valid_annotations
    
    # Set output path
    if output_file_path is None:
        base, ext = os.path.splitext(ann_file_path)
        output_file_path = f"{base}_cleaned{ext}"
    
    # Save cleaned annotations
    with open(output_file_path, 'w') as f:
        json.dump(cleaned_ann, f, indent=2)
    
    print(f"\nCleaned annotations saved to: {output_file_path}")
    print(f"Removed {len(bad_boxes)} invalid annotations")
    
    return output_file_path

def main():
    """Main function to validate all annotation files"""
    
    data_root = 'legend_data/'
    
    # Files to validate
    files_to_check = [
        'annotations_JSON/train.json',
        'annotations_JSON/val_with_info.json'
    ]
    
    print("="*60)
    print("CHART ANNOTATION VALIDATION")
    print("="*60)
    
    for ann_file in files_to_check:
        full_path = os.path.join(data_root, ann_file)
        
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found, skipping...")
            continue
        
        print(f"\n{'-'*40}")
        print(f"Validating: {ann_file}")
        print(f"{'-'*40}")
        
        # Validate and clean
        try:
            clean_annotations(full_path)
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
    
    print(f"\n{'='*60}")
    print("Validation complete!")
    print("If any cleaned files were created, update your training config to use them.")
    print("="*60)

if __name__ == '__main__':
    main() 