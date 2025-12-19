import os
import json
import random
import re
from PIL import Image
import time
import numpy as np

def fast_jpeg_size(path):
    """Quick width/height from a JPEG's SOF0 marker."""
    with open(path, "rb") as f:
        data = f.read(2048)
    pos = data.find(b"\xFF\xC0")
    if pos < 0:
        raise ValueError(f"SOF0 marker not found in {path}")
    h = int.from_bytes(data[pos+5:pos+7], "big")
    w = int.from_bytes(data[pos+7:pos+9], "big")
    return w, h

def get_image_size(path):
    """Get image dimensions efficiently."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return fast_jpeg_size(path)
    else:
        with Image.open(path) as img:
            return img.width, img.height

def create_bbox_from_polygon(polygon, padding=5):
    """Convert polygon coordinates to bounding box [x0, y0, width, height] with padding."""
    if not polygon:
        return None
    
    # Handle different polygon formats
    if isinstance(polygon, dict):
        # Format: {"x0": ..., "x1": ..., "y0": ..., "y1": ...}
        if all(k in polygon for k in ("x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3")):
            xs = [polygon[f"x{i}"] for i in range(4)]
            ys = [polygon[f"y{i}"] for i in range(4)]
        else:
            return None
    elif isinstance(polygon, list):
        # Format: [[x0, y0], [x1, y1], ...]
        xs = [p[0] for p in polygon if len(p) >= 2]
        ys = [p[1] for p in polygon if len(p) >= 2]
    else:
        return None
    
    if not xs or not ys:
        return None
        
    # Add padding to make text elements more detectable
    x0, y0 = min(xs) - padding, min(ys) - padding
    x1, y1 = max(xs) + padding, max(ys) + padding
    width, height = x1 - x0, y1 - y0
    
    return [x0, y0, width, height] if width > 0 and height > 0 else None

def create_bbox_from_points(points, padding=2):
    """Create bounding box from a list of points with padding."""
    if not points:
        return None
        
    xs = [p.get("x", 0) for p in points if "x" in p]
    ys = [p.get("y", 0) for p in points if "y" in p]
    
    if not xs or not ys:
        return None
        
    x0, y0 = min(xs) - padding, min(ys) - padding
    x1, y1 = max(xs) + padding, max(ys) + padding
    width, height = x1 - x0, y1 - y0
    
    return [x0, y0, width, height] if width > 0 and height > 0 else None

def analyze_data_series(data_series):
    """Analyze data series to extract statistics and metadata."""
    if not data_series:
        return {
            "num_data_points": 0,
            "x_data_type": "unknown",
            "y_data_type": "unknown", 
            "x_range": None,
            "y_range": None
        }
    
    num_points = len(data_series)
    x_values = [point.get("x") for point in data_series if "x" in point]
    y_values = [point.get("y") for point in data_series if "y" in point]
    
    # Analyze X data type
    x_numerical = [x for x in x_values if isinstance(x, (int, float))]
    x_categorical = [x for x in x_values if isinstance(x, str)]
    
    if len(x_numerical) > len(x_categorical):
        x_data_type = "numerical"
        x_range = [min(x_numerical), max(x_numerical)] if x_numerical else None
    else:
        x_data_type = "categorical"
        x_range = None
        
    # Analyze Y data type (usually numerical)
    y_numerical = [y for y in y_values if isinstance(y, (int, float))]
    if y_numerical:
        y_data_type = "numerical"
        y_range = [min(y_numerical), max(y_numerical)]
    else:
        y_data_type = "categorical"
        y_range = None
    
    return {
        "num_data_points": num_points,
        "x_data_type": x_data_type,
        "y_data_type": y_data_type,
        "x_range": x_range,
        "y_range": y_range,
        "x_values_sample": x_values[:5],  # First 5 for debugging
        "y_values_sample": y_values[:5]
    }

def merge_and_split_enhanced_chart_data(seed: int = 42, base_dir: str = None):
    """Enhanced merger that extracts ALL chart elements."""
    
    # ─── Configuration (Flexible paths for different environments) ─────────
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
    
    LEGEND_DATA_DIR = None
    for path in possible_legend_dirs:
        if os.path.exists(path):
            LEGEND_DATA_DIR = path
            break
    
    if LEGEND_DATA_DIR is None:
        # Try to find it by searching
        for root, dirs, files in os.walk(BASE_DIR):
            if "legend_data" in dirs:
                LEGEND_DATA_DIR = os.path.join(root, "legend_data")
                break
        
        if LEGEND_DATA_DIR is None:
            raise OSError(f"Could not find legend_data directory. Searched in: {possible_legend_dirs}")
    
    ANN_DIR = os.path.join(LEGEND_DATA_DIR, "train", "annotations")
    IMG_DIR = os.path.join(LEGEND_DATA_DIR, "train", "images") 
    OUTPUT_FOLDER = os.path.join(LEGEND_DATA_DIR, "annotations_JSON")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"📁 Working directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Legend Data: {LEGEND_DATA_DIR}")
    print(f"   Annotations: {ANN_DIR}")
    print(f"   Images: {IMG_DIR}")
    print(f"   Output: {OUTPUT_FOLDER}")

    # Verify directories
    if not os.path.exists(ANN_DIR):
        raise OSError(f"Annotations directory does not exist: {ANN_DIR}")
    if not os.path.exists(IMG_DIR):
        raise OSError(f"Images directory does not exist: {IMG_DIR}")

    # ─── Enhanced Categories (matching METAINFO) ─────────────
    categories = [
        {"id": 0, "name": "title"},
        {"id": 1, "name": "subtitle"}, 
        {"id": 2, "name": "x-axis"},
        {"id": 3, "name": "y-axis"},
        {"id": 4, "name": "x-axis-label"},
        {"id": 5, "name": "y-axis-label"},
        {"id": 6, "name": "x-tick-label"},
        {"id": 7, "name": "y-tick-label"},
        {"id": 8, "name": "legend"},
        {"id": 9, "name": "legend-title"},
        {"id": 10, "name": "legend-item"},
        {"id": 11, "name": "data-point"},
        {"id": 12, "name": "data-line"},
        {"id": 13, "name": "data-bar"},
        {"id": 14, "name": "data-area"},
        {"id": 15, "name": "grid-line"},
        {"id": 16, "name": "axis-title"},
        {"id": 17, "name": "tick-label"},
        {"id": 18, "name": "data-label"},
        {"id": 19, "name": "legend-text"},
        {"id": 20, "name": "plot-area"}  # Additional for plot bounding box
    ]
    
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    
    # ─── Role to Category Mapping ─────────────────────────────
    role_to_category = {
        "chart_title": "title",
        "title": "title",
        "subtitle": "subtitle",
        "axis_title": "axis-title",
        "x_axis_title": "x-axis-label", 
        "y_axis_title": "y-axis-label",
        "tick_label": "tick-label",
        "x_tick_label": "x-tick-label",
        "y_tick_label": "y-tick-label",
        "legend": "legend",
        "legend_title": "legend-title",
        "legend_item": "legend-item",
        "legend_text": "legend-text",
        "data_label": "data-label",
        "grid_line": "grid-line"
    }

    # ─── Build image map ─────────────────────────────────────
    print("Building image filename map...")
    image_map = {}
    for fn in os.listdir(IMG_DIR):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = re.match(r"^([0-9a-f]+)", name)
        if not m:
            continue
        hexid = m.group(1)
        image_map.setdefault(hexid, fn)
    print(f"Found {len(image_map)} valid image files")

    # ─── Gather valid JSON files ─────────────────────────────
    print("Gathering JSON files for valid images...")
    all_jsons = []
    for base in image_map.keys():
        json_file = f"{base}.json"
        json_path = os.path.join(ANN_DIR, json_file)
        if os.path.exists(json_path):
            all_jsons.append(json_file)

    print(f"Found {len(all_jsons)} valid JSON files")
    random.seed(seed)
    random.shuffle(all_jsons)
    split_idx = int(0.7 * len(all_jsons))
    splits = {
        "train": all_jsons[:split_idx],
        "val": all_jsons[split_idx:]
    }

    # ─── Process each split ─────────────────────────────────
    for split_name, json_files in splits.items():
        print(f"\n🔄 Processing {split_name} split...")
        coco = {"images": [], "annotations": [], "categories": categories}
        img_id_map = {}
        next_img_id = 1
        next_ann_id = 1
        processed_count = 0
        total_files = len(json_files)
        total_annotations = 0

        for jf in json_files:
            try:
                base = os.path.splitext(jf)[0]
                img_fn = image_map.get(base)
                if img_fn is None:
                    continue

                img_path = os.path.join(IMG_DIR, img_fn)
                w, h = get_image_size(img_path)

                # Load original annotation
                ann_path = os.path.join(ANN_DIR, jf)
                try:
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"❌ Error reading {jf}: {e}")
                    continue

                # Analyze data series for enhanced metadata
                data_series_stats = analyze_data_series(ann.get("data-series", []))

                # Register image with enhanced metadata
                if img_fn not in img_id_map:
                    img_id_map[img_fn] = next_img_id
                    image_info = {
                        "id": next_img_id,
                        "file_name": img_fn,
                        "width": w,
                        "height": h,
                        "source": ann.get("source", "generated"),
                        "chart_type": ann.get("chart-type", ""),
                        "plot_bb": ann.get("plot-bb", {}),
                        "data_series": ann.get("data-series", []),
                        "data_series_stats": data_series_stats,  # NEW: Enhanced stats with data point count
                        "axes_info": {
                            "x-axis": ann.get("axes", {}).get("x-axis", {}),
                            "y-axis": ann.get("axes", {}).get("y-axis", {})
                        },
                        # Additional element counts for analysis
                        "element_counts": {
                            "text_elements": len(ann.get("text", [])),
                            "visual_lines": len(ann.get("visual-elements", {}).get("lines", [])),
                            "visual_bars": len(ann.get("visual-elements", {}).get("bars", [])),
                            "scatter_points": len(ann.get("visual-elements", {}).get("scatter points", [])),
                            "x_axis_ticks": len(ann.get("axes", {}).get("x-axis", {}).get("ticks", [])),
                            "y_axis_ticks": len(ann.get("axes", {}).get("y-axis", {}).get("ticks", []))
                        }
                    }
                    coco["images"].append(image_info)
                    next_img_id += 1
                img_id = img_id_map[img_fn]

                def add_annotation(bbox, category_name, additional_info=None):
                    """Add annotation with validation."""
                    nonlocal next_ann_id, total_annotations
                    
                    if not bbox or len(bbox) != 4:
                        return False
                        
                    x0, y0, width, height = bbox
                    
                    # Validate bbox
                    if width <= 0 or height <= 0:
                        return False
                    if x0 < 0 or y0 < 0 or x0 + width > w or y0 + height > h:
                        # Clamp to image bounds
                        x0 = max(0, min(x0, w-1))
                        y0 = max(0, min(y0, h-1))
                        width = min(width, w - x0)
                        height = min(height, h - y0)
                        
                        if width <= 0 or height <= 0:
                            return False
                    
                    if category_name not in cat_name_to_id:
                        print(f"⚠️ Unknown category: {category_name}")
                        return False
                    
                    annotation = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_name_to_id[category_name],
                        "bbox": [x0, y0, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    
                    if additional_info:
                        annotation.update(additional_info)
                    
                    coco["annotations"].append(annotation)
                    next_ann_id += 1
                    total_annotations += 1
                    return True

                # ─── 1. Plot Area ─────────────────────────────────
                plot_bb = ann.get("plot-bb", {})
                if all(k in plot_bb for k in ("x0", "y0", "width", "height")):
                    bbox = [plot_bb["x0"], plot_bb["y0"], plot_bb["width"], plot_bb["height"]]
                    add_annotation(bbox, "plot-area", {
                        "element_type": "plot_area",
                        "plot_bb": plot_bb,
                        "data_points_inside": data_series_stats["num_data_points"]  # KEY FEATURE: Data point count
                    })

                # ─── 2. Text Elements ─────────────────────────────
                for text_elem in ann.get("text", []):
                    polygon = text_elem.get("polygon", {})
                    bbox = create_bbox_from_polygon(polygon)
                    
                    if bbox:
                        role = text_elem.get("role", "")
                        category_name = role_to_category.get(role, "tick-label")  # Default fallback
                        
                        # Handle axis-specific mappings
                        if role == "tick_label":
                            # Try to determine if it's x or y axis based on text_id and axes info
                            text_id = text_elem.get("id", -1)
                            is_x_tick = False
                            is_y_tick = False
                            
                            # Check x-axis ticks
                            for tick in ann.get("axes", {}).get("x-axis", {}).get("ticks", []):
                                if tick.get("id") == text_id:
                                    is_x_tick = True
                                    break
                            
                            # Check y-axis ticks  
                            if not is_x_tick:
                                for tick in ann.get("axes", {}).get("y-axis", {}).get("ticks", []):
                                    if tick.get("id") == text_id:
                                        is_y_tick = True
                                        break
                            
                            if is_x_tick:
                                category_name = "x-tick-label"
                            elif is_y_tick:
                                category_name = "y-tick-label"
                            else:
                                category_name = "tick-label"
                        
                        add_annotation(bbox, category_name, {
                            "text": text_elem.get("text", ""),
                            "role": role,
                            "text_id": text_elem.get("id", -1),
                            "element_type": "text"
                        })

                # ─── 3. Visual Elements ────────────────────────────
                visual_elements = ann.get("visual-elements", {})
                
                # Lines
                for i, line in enumerate(visual_elements.get("lines", [])):
                    bbox = create_bbox_from_points(line, padding=10)  # Increased from 5 to 10 for better detection
                    if bbox:
                        add_annotation(bbox, "data-line", {
                            "line_points": line,
                            "element_type": "line",
                            "line_id": i,
                            "num_points_in_line": len(line)
                        })
                
                # Bars 
                for i, bar in enumerate(visual_elements.get("bars", [])):
                    if isinstance(bar, dict) and "bbox" in bar:
                        bbox = bar["bbox"]
                    else:
                        bbox = create_bbox_from_points(bar if isinstance(bar, list) else [bar], padding=8)  # Increased from 2 to 8 for better detection
                    
                    if bbox:
                        add_annotation(bbox, "data-bar", {
                            "bar_data": bar,
                            "element_type": "bar",
                            "bar_id": i
                        })
                
                # Scatter points
                for i, point in enumerate(visual_elements.get("scatter points", [])):
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        bbox = [point["x"]-8, point["y"]-8, 16, 16]  # Increased to 16x16 box for better detection
                        add_annotation(bbox, "data-point", {
                            "point": point,
                            "element_type": "scatter_point",
                            "point_id": i
                        })
                
                # Dot points
                for i, point in enumerate(visual_elements.get("dot points", [])):
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        bbox = [point["x"]-8, point["y"]-8, 16, 16]  # Increased to 16x16 box for better detection
                        add_annotation(bbox, "data-point", {
                            "point": point,
                            "element_type": "dot_point", 
                            "point_id": i
                        })

                # ─── 4. Individual Data Points from data-series ──────────
                data_series = ann.get("data-series", [])
                if data_series:
                    # Try to map logical data coordinates to pixel coordinates
                    plot_bb = ann.get("plot-bb", {})
                    if all(k in plot_bb for k in ("x0", "y0", "width", "height")):
                        plot_x0, plot_y0 = plot_bb["x0"], plot_bb["y0"]
                        plot_w, plot_h = plot_bb["width"], plot_bb["height"]
                        
                        # Extract data ranges for scaling
                        data_x_values = []
                        data_y_values = []
                        
                        for point in data_series:
                            x_val = point.get("x")
                            y_val = point.get("y")
                            
                            # Handle different x-value types
                            if isinstance(x_val, (int, float)):
                                data_x_values.append(float(x_val))
                            elif isinstance(x_val, str):
                                # For categorical data, use index position
                                try:
                                    # Try to parse as number first
                                    data_x_values.append(float(x_val))
                                except:
                                    # Use enumeration for categorical
                                    data_x_values.append(len(data_x_values))
                            
                            # Handle y-values (usually numerical)
                            if isinstance(y_val, (int, float)):
                                data_y_values.append(float(y_val))
                        
                        if data_x_values and data_y_values:
                            x_min, x_max = min(data_x_values), max(data_x_values)
                            y_min, y_max = min(data_y_values), max(data_y_values)
                            
                            # Create bounding boxes for each data point
                            for i, (point, x_val, y_val) in enumerate(zip(data_series, data_x_values, data_y_values)):
                                if x_max > x_min and y_max > y_min:
                                    # Scale to plot coordinates
                                    x_scaled = plot_x0 + (x_val - x_min) / (x_max - x_min) * plot_w
                                    y_scaled = plot_y0 + plot_h - (y_val - y_min) / (y_max - y_min) * plot_h  # Flip Y
                                    
                                    # Create larger bounding box around the data point for better detection
                                    point_size = 16  # 16x16 pixel box (increased from 12x12)
                                    bbox = [x_scaled - point_size//2, y_scaled - point_size//2, point_size, point_size]
                                    
                                    # Determine data point type based on chart type
                                    chart_type = ann.get("chart-type", "")
                                    if chart_type in ["scatter", "dot"]:
                                        category_name = "data-point"
                                    elif chart_type in ["line"]:
                                        category_name = "data-point"  # Line chart data points
                                    elif chart_type in ["vertical_bar", "horizontal_bar", "bar"]:
                                        category_name = "data-bar"  # Bar endpoints
                                    else:
                                        category_name = "data-point"  # Default
                                    
                                    add_annotation(bbox, category_name, {
                                        "data_point": point,
                                        "element_type": "data_series_point",
                                        "point_id": i,
                                        "chart_type": chart_type,
                                        "data_type": "numerical" if isinstance(point.get("x"), (int, float)) else "categorical",
                                        "total_data_points": len(data_series)  # FEATURE: Total data points per annotation
                                    })

                # ─── 5. Axis Ticks (as separate elements) ─────────
                for axis_name in ["x-axis", "y-axis"]:
                    for tick in ann.get("axes", {}).get(axis_name, {}).get("ticks", []):
                        pt = tick.get("tick_pt", {})
                        if "x" in pt and "y" in pt:
                            # Create larger bbox around tick point for better detection
                            bbox = [pt["x"]-6, pt["y"]-6, 12, 12]  # Increased from 8x8 to 12x12
                            category = "x-axis" if axis_name == "x-axis" else "y-axis"
                            
                            add_annotation(bbox, category, {
                                "tick_point": pt,
                                "tick_id": tick.get("id", -1),
                                "axis": axis_name,
                                "element_type": "axis_tick"
                            })

                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"📊 Processed {processed_count}/{total_files} files ({(processed_count/total_files)*100:.1f}%) - {total_annotations} annotations so far")

            except Exception as e:
                print(f"❌ Error processing {jf}: {e}")
                continue

        # ─── Save output ─────────────────────────────────────
        out_path = os.path.join(OUTPUT_FOLDER, f"{split_name}_enriched.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        # ─── Validation ─────────────────────────────────────
        print(f"\n📋 {split_name.upper()} SPLIT SUMMARY:")
        print(f"   • Images: {len(coco['images'])}")
        print(f"   • Annotations: {len(coco['annotations'])}")
        print(f"   • Avg annotations per image: {len(coco['annotations'])/len(coco['images']):.1f}")
        
        # Data series statistics (KEY FEATURE)
        total_data_points = sum(img.get("data_series_stats", {}).get("num_data_points", 0) for img in coco["images"])
        avg_data_points = total_data_points / len(coco["images"]) if coco["images"] else 0
        print(f"   • Total data points in data-series: {total_data_points}")
        print(f"   • Average data points per image: {avg_data_points:.1f}")
        
        # Category breakdown
        cat_counts = {}
        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            cat_name = next(c["name"] for c in categories if c["id"] == cat_id)
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
        
        print(f"   • Category breakdown:")
        for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {cat_name}: {count}")
        
        print(f"✅ Wrote {out_path}")

    print(f"\n🎉 COMPLETE! Enhanced chart annotations extracted successfully!")
    print(f"📁 Output files saved in: {OUTPUT_FOLDER}")
    return OUTPUT_FOLDER

if __name__ == "__main__":
    merge_and_split_enhanced_chart_data() 