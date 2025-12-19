#!/usr/bin/env python3
"""
Clean dataset by removing images with invalid bounding boxes.
This prevents NaN issues during training by filtering problematic data upfront.
"""

import json
import numpy as np
import os
from pathlib import Path

def is_valid_bbox(bbox):
    """Check if a bounding box is valid (COCO format: [x, y, width, height])."""
    try:
        # Handle different bbox formats
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, width, height = bbox
        elif hasattr(bbox, '__iter__') and len(bbox) == 4:
            x, y, width, height = bbox
        else:
            print(f"   ❌ Invalid bbox format: {bbox}")
            return False
        
        # Check for NaN or Inf
        if any(not np.isfinite(val) for val in [x, y, width, height]):
            print(f"   ❌ NaN/Inf values: {bbox}")
            return False
        
        # Check for negative coordinates or dimensions
        if x < 0 or y < 0:
            print(f"   ❌ Negative coordinates: {bbox}")
            return False
        
        # Check for valid width/height (must be positive)
        if width <= 0 or height <= 0:
            print(f"   ❌ Invalid dimensions (width={width}, height={height}): {bbox}")
            return False
        
        # Check for reasonable size (not too small)
        if width < 1 or height < 1:
            print(f"   ❌ Too small (width={width}, height={height}): {bbox}")
            return False
        
        # Check for extremely large bboxes (likely errors)
        if width > 10000 or height > 10000:
            print(f"   ❌ Too large (width={width}, height={height}): {bbox}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking bbox {bbox}: {e}")
        return False

def clean_annotation_file(input_file, output_file):
    """Clean annotation file by removing images with invalid bboxes."""
    print(f"\n🔍 Processing: {input_file}")
    
    # Load annotations
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    original_images = len(data['images'])
    original_annotations = len(data['annotations'])
    
    print(f"📊 Original: {original_images} images, {original_annotations} annotations")
    
    # Track invalid images
    invalid_image_ids = set()
    valid_annotations = []
    
    # Check all annotations for invalid bboxes
    for i, ann in enumerate(data['annotations']):
        if i % 1000 == 0:
            print(f"   Checking annotation {i}/{original_annotations}...")
        
        bbox = ann.get('bbox', [])
        if not is_valid_bbox(bbox):
            print(f"   🚫 Invalid bbox in image_id {ann['image_id']}: {bbox}")
            invalid_image_ids.add(ann['image_id'])
        else:
            valid_annotations.append(ann)
    
    # Filter out images with invalid annotations
    valid_images = []
    for img in data['images']:
        if img['id'] not in invalid_image_ids:
            valid_images.append(img)
    
    # Create cleaned dataset
    cleaned_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data.get('categories', []),
        'images': valid_images,
        'annotations': valid_annotations
    }
    
    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Report results
    cleaned_images = len(cleaned_data['images'])
    cleaned_annotations = len(cleaned_data['annotations'])
    removed_images = original_images - cleaned_images
    removed_annotations = original_annotations - cleaned_annotations
    
    print(f"✅ Cleaned: {cleaned_images} images, {cleaned_annotations} annotations")
    print(f"🗑️ Removed: {removed_images} images ({removed_images/original_images*100:.1f}%), {removed_annotations} annotations ({removed_annotations/original_annotations*100:.1f}%)")
    print(f"💾 Saved to: {output_file}")
    
    return removed_images, removed_annotations

def main():
    """Clean all annotation files."""
    base_dir = Path("legend_data/annotations_JSON")
    output_dir = Path("legend_data/annotations_JSON_cleaned")
    
    # Files to clean
    files_to_clean = [
        "train_enriched.json",
        "val_enriched_with_info.json"
    ]
    
    total_removed_images = 0
    total_removed_annotations = 0
    
    print("🧹 Starting dataset cleaning...")
    print("="*60)
    
    for filename in files_to_clean:
        input_file = base_dir / filename
        output_file = output_dir / filename
        
        if input_file.exists():
            removed_img, removed_ann = clean_annotation_file(input_file, output_file)
            total_removed_images += removed_img
            total_removed_annotations += removed_ann
        else:
            print(f"⚠️ File not found: {input_file}")
    
    print("="*60)
    print(f"🎉 CLEANING COMPLETE!")
    print(f"📊 Total removed: {total_removed_images} images, {total_removed_annotations} annotations")
    print(f"📁 Cleaned files saved to: {output_dir}")
    print("\n🔧 Next steps:")
    print("1. Update your training config to use the cleaned annotation files")
    print("2. Restart training with the clean dataset")

if __name__ == "__main__":
    main() 