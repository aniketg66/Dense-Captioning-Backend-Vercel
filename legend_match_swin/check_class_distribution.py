#!/usr/bin/env python3
# check_class_distribution.py - Diagnose class distribution issues
import json
import os
from collections import Counter

def check_class_distribution(data_root='/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/'):
    """Check class distribution in training data to understand 0.0 mAP issues."""
    
    # Load training annotations
    train_file = os.path.join(data_root, 'annotations_JSON/train.json')
    val_file = os.path.join(data_root, 'annotations_JSON/val_with_info.json')
    
    # Class names from your config
    class_names = [
        'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
        'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
        'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
        'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
    ]
    
    print("🔍 CHECKING CLASS DISTRIBUTION FOR 0.0 mAP DIAGNOSIS")
    print("=" * 60)
    
    for split_name, ann_file in [('TRAIN', train_file), ('VAL', val_file)]:
        print(f"\n📊 {split_name} SET:")
        
        if not os.path.exists(ann_file):
            print(f"❌ File not found: {ann_file}")
            continue
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Count annotations per class
        class_counts = Counter()
        for ann in data.get('annotations', []):
            class_id = ann['category_id']
            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
                class_counts[class_name] += 1
        
        print(f"Total annotations: {len(data.get('annotations', []))}")
        print(f"Total images: {len(data.get('images', []))}")
        
        # Show class distribution
        print("\nClass distribution:")
        print(f"{'Class Name':<15} {'Count':<8} {'Percentage':<10}")
        print("-" * 35)
        
        total_anns = len(data.get('annotations', []))
        classes_with_zero = []
        
        for class_name in class_names:
            count = class_counts.get(class_name, 0)
            percentage = (count / total_anns * 100) if total_anns > 0 else 0
            
            if count == 0:
                classes_with_zero.append(class_name)
                print(f"{class_name:<15} {count:<8} {percentage:<10.1f}% ❌ ZERO!")
            elif count < 100:
                print(f"{class_name:<15} {count:<8} {percentage:<10.1f}% ⚠️  LOW")
            else:
                print(f"{class_name:<15} {count:<8} {percentage:<10.1f}% ✅")
        
        print(f"\n🚨 Classes with ZERO annotations: {len(classes_with_zero)}")
        for cls in classes_with_zero:
            print(f"   • {cls}")
    
    # Check if enriched files exist (might have more annotations)
    enriched_files = {
        'train_enriched.json': 'ENRICHED TRAIN',
        'val_enriched.json': 'ENRICHED VAL'
    }
    
    print(f"\n🔍 CHECKING FOR ENRICHED ANNOTATION FILES:")
    for filename, description in enriched_files.items():
        filepath = os.path.join(data_root, 'annotations_JSON', filename)
        if os.path.exists(filepath):
            print(f"✅ Found {description}: {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            class_counts = Counter()
            for ann in data.get('annotations', []):
                class_id = ann['category_id']
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    class_counts[class_name] += 1
            
            print(f"   Total annotations: {len(data.get('annotations', []))}")
            
            # Show top classes
            print("   Top 5 classes:")
            for class_name, count in class_counts.most_common(5):
                print(f"     {class_name}: {count}")
                
        else:
            print(f"❌ Missing {description}: {filepath}")
    
    print(f"\n💡 DIAGNOSIS RECOMMENDATIONS:")
    print("=" * 60)
    print("1. If many classes have ZERO annotations:")
    print("   → Use enriched annotation files or fix data preparation")
    print("2. If classes have <100 annotations:")
    print("   → Consider class merging or more training data")
    print("3. If enriched files exist:")
    print("   → Update your config to use train_enriched.json/val_enriched.json")
    
if __name__ == "__main__":
    check_class_distribution() 