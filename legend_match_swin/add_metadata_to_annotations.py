import json
import os
from datetime import datetime

def add_metadata_to_enriched_annotations(base_dir: str = None):
    """Add complete metadata to enriched annotation files for COCO compatibility."""
    
    # ─── Flexible path detection (same as enhanced merger) ─────────
    if base_dir is None:
        # Try to detect current environment
        try:
            # For regular Python script execution
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # For Jupyter notebook execution
            BASE_DIR = os.getcwd()
            print("🔧 Running in Jupyter/Colab environment, using current working directory")
    else:
        BASE_DIR = base_dir
    
    # Look for legend_data directory in common locations
    possible_legend_dirs = [
        os.path.join(BASE_DIR, "..", "legend_data"),  # Up one level (for legend_match/)
        os.path.join(BASE_DIR, "legend_data"),        # Same level
        os.path.join(BASE_DIR, "CHART-DeMatch", "legend_data"),  # In CHART-DeMatch subfolder
        "/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data"  # Colab path
    ]
    
    data_root = None
    for path in possible_legend_dirs:
        if os.path.exists(path):
            data_root = path
            break
    
    if data_root is None:
        # Try to find it by searching
        for root, dirs, files in os.walk(BASE_DIR):
            if "legend_data" in dirs:
                data_root = os.path.join(root, "legend_data")
                break
        
        if data_root is None:
            raise OSError(f"Could not find legend_data directory. Searched in: {possible_legend_dirs}")
    
    annotations_dir = os.path.join(data_root, 'annotations_JSON')
    
    print(f"📁 Working with annotations in: {annotations_dir}")
    
    # ─── Updated categories to match enhanced merger (21 categories) ─────────
    enhanced_categories = [
        {"id": 0, "name": "title", "supercategory": "text_element"},
        {"id": 1, "name": "subtitle", "supercategory": "text_element"}, 
        {"id": 2, "name": "x-axis", "supercategory": "axis_element"},
        {"id": 3, "name": "y-axis", "supercategory": "axis_element"},
        {"id": 4, "name": "x-axis-label", "supercategory": "text_element"},
        {"id": 5, "name": "y-axis-label", "supercategory": "text_element"},
        {"id": 6, "name": "x-tick-label", "supercategory": "text_element"},
        {"id": 7, "name": "y-tick-label", "supercategory": "text_element"},
        {"id": 8, "name": "legend", "supercategory": "chart_element"},
        {"id": 9, "name": "legend-title", "supercategory": "text_element"},
        {"id": 10, "name": "legend-item", "supercategory": "chart_element"},
        {"id": 11, "name": "data-point", "supercategory": "data_element"},
        {"id": 12, "name": "data-line", "supercategory": "data_element"},
        {"id": 13, "name": "data-bar", "supercategory": "data_element"},
        {"id": 14, "name": "data-area", "supercategory": "data_element"},
        {"id": 15, "name": "grid-line", "supercategory": "chart_element"},
        {"id": 16, "name": "axis-title", "supercategory": "text_element"},
        {"id": 17, "name": "tick-label", "supercategory": "text_element"},
        {"id": 18, "name": "data-label", "supercategory": "text_element"},
        {"id": 19, "name": "legend-text", "supercategory": "text_element"},
        {"id": 20, "name": "plot-area", "supercategory": "chart_element"}
    ]
    
    # ─── Process both train and val enriched files ─────────────────────────
    splits = ['train', 'val']
    
    for split in splits:
        # Look for enriched files first, then fall back to regular files
        enriched_file = os.path.join(annotations_dir, f'{split}_enriched.json')
        regular_file = os.path.join(annotations_dir, f'{split}.json')
        
        if os.path.exists(enriched_file):
            orig_file = enriched_file
            output_file = os.path.join(annotations_dir, f'{split}_enriched_with_info.json')
            print(f"📝 Processing enriched file: {split}_enriched.json")
        elif os.path.exists(regular_file):
            orig_file = regular_file
            output_file = os.path.join(annotations_dir, f'{split}_with_info.json')
            print(f"📝 Processing regular file: {split}.json")
        else:
            print(f"⚠️ No annotation file found for {split} split, skipping...")
            continue

        # Load original JSON
        with open(orig_file, 'r') as f:
            ann = json.load(f)

        # Inject a complete "info" block
        ann['info'] = {
            "description": "Enhanced Chart Element Detection Dataset with comprehensive annotations",
            "version": "2.0",
            "year": 2025,
            "contributor": "Enhanced Chart-DeMatch Pipeline", 
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": "https://github.com/microsoft/unilm/tree/master/layoutlmft/layoutlmft/data/funsd",
            "annotation_type": "enhanced_chart_elements",
            "total_categories": len(enhanced_categories),
            "features": [
                "plot_areas", "text_elements", "visual_elements", "data_points", 
                "axis_elements", "data_series_analysis", "element_counts"
            ]
        }

        # Ensure all required top-level keys exist
        for key in ('licenses', 'images', 'annotations', 'categories'):
            ann.setdefault(key, [])

        # Add license information
        if not ann.get('licenses'):
            ann['licenses'] = [
                {
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "id": 0,
                    "name": "Attribution License"
                }
            ]

        # Update categories to match enhanced merger exactly
        ann['categories'] = enhanced_categories

        # Add metadata to each image while preserving existing enriched data
        for img in ann['images']:
            # Preserve all existing enriched fields (data_series_stats, element_counts, etc.)
            
            # Add required COCO metadata fields if they don't exist
            if 'license' not in img:
                img['license'] = 0
            if 'flickr_url' not in img:
                img['flickr_url'] = ""
            if 'coco_url' not in img:
                img['coco_url'] = ""
            if 'date_captured' not in img:
                img['date_captured'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            # Ensure enhanced metadata fields have defaults if missing
            if 'chart_type' not in img:
                img['chart_type'] = "unknown"
            if 'plot_bb' not in img:
                img['plot_bb'] = {}
            if 'axes_info' not in img:
                img['axes_info'] = {"x-axis": {}, "y-axis": {}}
            if 'data_series_stats' not in img:
                img['data_series_stats'] = {
                    "num_data_points": 0,
                    "x_data_type": "unknown", 
                    "y_data_type": "unknown"
                }
            if 'element_counts' not in img:
                img['element_counts'] = {
                    "text_elements": 0,
                    "visual_lines": 0,
                    "visual_bars": 0,
                    "scatter_points": 0,
                    "x_axis_ticks": 0,
                    "y_axis_ticks": 0
                }

        # Add validation statistics
        image_count = len(ann['images'])
        annotation_count = len(ann['annotations'])
        avg_annotations = annotation_count / image_count if image_count > 0 else 0
        
        # Calculate data point statistics
        total_data_points = sum(
            img.get("data_series_stats", {}).get("num_data_points", 0) 
            for img in ann['images']
        )
        avg_data_points = total_data_points / image_count if image_count > 0 else 0
        
        # Add dataset statistics to info
        ann['info'].update({
            "statistics": {
                "total_images": image_count,
                "total_annotations": annotation_count,
                "avg_annotations_per_image": round(avg_annotations, 2),
                "total_data_points": total_data_points,
                "avg_data_points_per_image": round(avg_data_points, 2),
                "categories_count": len(enhanced_categories)
            }
        })

        # Write to new file
        with open(output_file, 'w') as f:
            json.dump(ann, f, indent=2)

        print(f"✅ Created {os.path.basename(output_file)} with enhanced metadata")
        print(f"   • Images: {image_count}")
        print(f"   • Annotations: {annotation_count}")
        print(f"   • Avg annotations/image: {avg_annotations:.1f}")
        print(f"   • Total data points: {total_data_points}")
        print(f"   • Avg data points/image: {avg_data_points:.1f}")
    
    print(f"\n🎉 All annotation files updated with enhanced metadata!")
    return annotations_dir

if __name__ == "__main__":
    add_metadata_to_enriched_annotations() 