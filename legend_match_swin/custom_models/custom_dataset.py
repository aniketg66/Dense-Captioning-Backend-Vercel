import json
import os.path as osp
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
import warnings

# ─── Enhanced robust image loader for real images ───
@TRANSFORMS.register_module()
class RobustLoadImageFromFile(LoadImageFromFile):
    """Enhanced image loader: tries real images first, falls back to dummy if needed."""
    
    # Class variable to track missing images
    missing_count = 0
    
    def __init__(self, try_real_images=True, fallback_to_dummy=True, **kwargs):
        super().__init__(**kwargs)
        self.try_real_images = try_real_images
        self.fallback_to_dummy = fallback_to_dummy
    
    def transform(self, results):
        """Try to load real image first, fall back to dummy if not found."""
        if self.try_real_images:
            try:
                # Try standard MMDet image loading first
                results = super().transform(results)
                return results
                
            except (FileNotFoundError, OSError, Exception) as e:
                # Count missing image
                RobustLoadImageFromFile.missing_count += 1
                
                # Log warning every 10 missing images to avoid spam
                if RobustLoadImageFromFile.missing_count % 10 == 1:
                    warnings.warn(f"Missing image #{RobustLoadImageFromFile.missing_count}: {results.get('img_path', 'unknown')}. "
                                f"Total missing so far: {RobustLoadImageFromFile.missing_count}", 
                                UserWarning)
                
                if not self.fallback_to_dummy:
                    raise e
                # Fall through to create dummy image
        
        # Create dummy image (either by choice or because real image loading failed)
        if 'img_shape' in results:
            h, w = results['img_shape'][:2]
        else:
            h = results.get('height', 800)
            w = results.get('width', 600)
            
        results['img'] = np.zeros((h, w, 3), dtype=np.uint8)
        results['img_shape'] = (h, w, 3)
        results['ori_shape'] = (h, w, 3)
        return results
    
    @classmethod
    def get_missing_count(cls):
        """Get the total count of missing images."""
        return cls.missing_count
    
    @classmethod
    def reset_missing_count(cls):
        """Reset the missing image counter."""
        cls.missing_count = 0

# ─── Legacy support for old transform name ───
@TRANSFORMS.register_module()
class CreateDummyImg(RobustLoadImageFromFile):
    """Legacy alias for RobustLoadImageFromFile."""
    pass

@TRANSFORMS.register_module()
class ClampBBoxes(BaseTransform):
    """Simple bbox clamping transform - only clamps coordinates, doesn't filter."""
    def __init__(self, min_size=1):
        self.min_size = min_size
    
    def transform(self, results):
        """Clamp bboxes to image bounds without removing any boxes."""
        if 'gt_bboxes' not in results:
            return results
            
        h, w = results['img_shape'][:2]
        
        # Handle both numpy arrays and MMDet's HorizontalBoxes objects
        gt_bboxes = results['gt_bboxes']
        if hasattr(gt_bboxes, 'tensor'):
            # MMDet HorizontalBoxes object - clamp in place
            gt_bboxes.tensor[:, 0].clamp_(0, w)  # x1
            gt_bboxes.tensor[:, 1].clamp_(0, h)  # y1  
            gt_bboxes.tensor[:, 2].clamp_(0, w)  # x2
            gt_bboxes.tensor[:, 3].clamp_(0, h)  # y2
        else:
            # Regular numpy array - clamp in place
            if len(gt_bboxes) > 0:
                gt_bboxes[:, 0] = np.clip(gt_bboxes[:, 0], 0, w)  # x1
                gt_bboxes[:, 1] = np.clip(gt_bboxes[:, 1], 0, h)  # y1
                gt_bboxes[:, 2] = np.clip(gt_bboxes[:, 2], 0, w)  # x2
                gt_bboxes[:, 3] = np.clip(gt_bboxes[:, 3], 0, h)  # y2
        
        # Don't drop anything here - let filter_cfg handle empty GT filtering
        results['gt_bboxes'] = gt_bboxes
        return results

@TRANSFORMS.register_module()
class SetScaleFactor(BaseTransform):
    """Compute scale_factor from data_series & plot_bb before any Resize."""
    def __init__(self, default_scale=(1.0, 1.0)):
        self.default_scale = default_scale

    def calculate_scale_factor(self, results):
        bb = results.get('plot_bb', {})
        w, h = bb.get('width', 0), bb.get('height', 0)
        xs, ys = [], []
        for series in results.get('data_series', []):
            for pt in series.get('data', []):
                x, y = pt.get('x'), pt.get('y')
                if isinstance(x, (int, float)): xs.append(x)
                if isinstance(y, (int, float)): ys.append(y)
        if xs and max(xs) != min(xs):
            x_scale = w / (max(xs) - min(xs))
        else:
            x_scale = self.default_scale[0]
        if ys and max(ys) != min(ys):
            y_scale = -h / (max(ys) - min(ys))
        else:
            y_scale = self.default_scale[1]
        return (x_scale, y_scale)

    def transform(self, results):
        try:
            sf = self.calculate_scale_factor(results)
            results['scale_factor'] = np.array(sf, dtype=np.float32)
        except Exception:
            results['scale_factor'] = np.array(self.default_scale, dtype=np.float32)
        H, W = results.get('height', 0), results.get('width', 0)
        results['img_shape'] = (H, W, 3)
        return results

@TRANSFORMS.register_module()
class EnsureScaleFactor(BaseTransform):
    """Fallback if no scale_factor set yet."""
    def transform(self, results):
        results['scale_factor'] = np.array([1.0, 1.0], dtype=np.float32)
        return results

@TRANSFORMS.register_module()
class SetInputs(BaseTransform):
    """Copy dummy img into inputs for DetDataPreprocessor."""
    def transform(self, results):
        if 'img' in results:
            results['inputs'] = results['img'].copy()
        return results

@TRANSFORMS.register_module()
class CustomPackDetInputs(PackDetInputs):
    """Final packing into DetDataSample, ensure inputs present."""
    def transform(self, results):
        if 'img' in results:
            results['inputs'] = results['img'].copy()
        return super().transform(results)

@DATASETS.register_module()
class ChartDataset(BaseDetDataset):
    """Enhanced dataset for comprehensive chart element detection and analysis."""
    
    # Updated METAINFO with 21 enhanced categories
    METAINFO = {
        'classes': [
            'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
            'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
            'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
            'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
        ]
    }

    # Chart-type specific element filtering based on actual dataset distribution
    # Data from analyze_chart_types.py:
    # • line (41.9%): 1710 images → data-line only
    # • scatter (18.2%): 742 images → data-point only  
    # • vertical_bar (30.5%): 1246 images → data-bar only
    # • dot (9.2%): 374 images → data-point only
    # • horizontal_bar (0.2%): 9 images → data-bar only
    CHART_TYPE_ELEMENT_MAPPING = {
        # Line charts (41.9% - 1710 images): ONLY data-line
        'line': {
            'allowed_data_elements': {'data-line'},
            'forbidden_data_elements': {'data-point', 'data-bar', 'data-area'}
        },
        # Scatter charts (18.2% - 742 images): ONLY data-point
        'scatter': {
            'allowed_data_elements': {'data-point'},
            'forbidden_data_elements': {'data-line', 'data-bar', 'data-area'}
        },
        # Vertical bar charts (30.5% - 1246 images): ONLY data-bar
        'vertical_bar': {
            'allowed_data_elements': {'data-bar'},
            'forbidden_data_elements': {'data-point', 'data-line', 'data-area'}
        },
        # Dot charts (9.2% - 374 images): ONLY data-point
        'dot': {
            'allowed_data_elements': {'data-point'},
            'forbidden_data_elements': {'data-line', 'data-bar', 'data-area'}
        },
        # Horizontal bar charts (0.2% - 9 images): ONLY data-bar
        'horizontal_bar': {
            'allowed_data_elements': {'data-bar'},
            'forbidden_data_elements': {'data-point', 'data-line', 'data-area'}
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metainfo.update(self.METAINFO)
        
        # Print configuration info
        print(f"📊 ChartDataset initialized with {len(self.METAINFO['classes'])} categories:")
        for i, cls_name in enumerate(self.METAINFO['classes']):
            print(f"   {i}: {cls_name}")
        
        # Print chart-type filtering info
        print(f"🎯 Chart-type specific filtering enabled:")
        for chart_type, mapping in self.CHART_TYPE_ELEMENT_MAPPING.items():
            allowed = mapping.get('allowed_data_elements', set())
            forbidden = mapping.get('forbidden_data_elements', set())
            print(f"   • {chart_type}: ✅ {allowed} | 🚫 {forbidden}")
        
        # Debug print the data configuration
        print(f"📁 Dataset configuration:")
        print(f"   • data_root: {getattr(self, 'data_root', 'None')}")
        print(f"   • data_prefix: {getattr(self, 'data_prefix', 'None')}")
        print(f"   • ann_file: {getattr(self, 'ann_file', 'None')}")

    def load_data_list(self):
        """Load enhanced annotation files with priority order."""
        
        # Auto-detect best annotation file (same logic as config)
        def get_best_ann_file(split):
            ann_dir = osp.join(self.data_root, 'annotations_JSON')
            
            # Priority order with flexible naming
            candidates = [
                f'{split}_enriched_with_info.json',
                f'{split}_enriched.json',
                f'{split}_with_info.json',  # Added: Handles val_with_info.json
                f'{split}.json',
                f'{split}_cleaned.json'
            ]
            
            for candidate in candidates:
                full_path = osp.join(ann_dir, candidate)
                if osp.exists(full_path):
                    print(f"📁 ChartDataset using {candidate}")
                    return full_path
            
            # Fallback to ann_file if specified
            if hasattr(self, 'ann_file') and self.ann_file:
                fallback_path = osp.join(self.data_root, self.ann_file)
                if osp.exists(fallback_path):
                    print(f"📁 Using fallback annotation file: {self.ann_file}")
                    return fallback_path
            
            raise FileNotFoundError(f"No annotation files found in {ann_dir}")
        
        # Determine file path
        if hasattr(self, 'ann_file') and self.ann_file:
            ann_file_path = osp.join(self.data_root, self.ann_file)
        else:
            # Try to auto-detect based on common patterns
            for split in ['train', 'val']:
                try:
                    ann_file_path = get_best_ann_file(split)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError("Could not find any annotation files")
        
        # Load annotation file
        with open(ann_file_path, 'r') as f:
            ann = json.load(f)
        
        print(f"📊 Loading from {ann_file_path}")
        print(f"   • Images: {len(ann.get('images', []))}")
        print(f"   • Annotations: {len(ann.get('annotations', []))}")
        
        # Build image lookup
        img_id_to_info = {img['id']: img for img in ann['images']}
        
        # Group annotations by image
        img_id_to_anns = {}
        for ann_data in ann.get('annotations', []):
            img_id = ann_data['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann_data)
        
        # Create data list with enhanced metadata
        data_list = []
        for img_id, img_info in img_id_to_info.items():
            annotations = img_id_to_anns.get(img_id, [])
            
            # Skip images without annotations if filter_empty_gt is enabled
            if not annotations and self.filter_cfg.get('filter_empty_gt', False):
                    continue
            
            # Convert annotations to instances format
            instances = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format for MMDet
                bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                instance = {
                    'bbox': bbox_xyxy,
                    'bbox_label': ann['category_id'],
                    'ignore_flag': 0,
                    'annotation_id': ann.get('id', -1),
                    'area': ann.get('area', bbox[2] * bbox[3]),
                    'element_type': ann.get('element_type', 'unknown')
                }
                
                # Add additional annotation metadata if available
                for key in ['text', 'role', 'data_point', 'chart_type', 'total_data_points']:
                    if key in ann:
                        instance[key] = ann[key]
                
                instances.append(instance)
            
            # Create data info with enhanced metadata
            # Fix: Construct full image path using data_prefix (like standard MMDet datasets)
            filename = img_info['file_name']
            if self.data_prefix.get('img'):
                img_path = osp.join(self.data_prefix['img'], filename)
            else:
                img_path = filename  # Fallback to original filename
                
            data_info = {
                'img_id': img_info['id'],
                'img_path': img_path,  # Use constructed path
                'height': img_info['height'],
                'width': img_info['width'],
                'instances': instances,
                # Enhanced metadata from enriched annotations
                'chart_type': img_info.get('chart_type', ''),
                'plot_bb': img_info.get('plot_bb', {}),
                'data_series': img_info.get('data_series', []),
                'data_series_stats': img_info.get('data_series_stats', {}),
                'axes_info': img_info.get('axes_info', {}),
                'element_counts': img_info.get('element_counts', {}),
                'source': img_info.get('source', 'unknown')
            }
            
            data_list.append(data_info)
        
        print(f"✅ Loaded {len(data_list)} images with enhanced metadata")
        return data_list

    def parse_data_info(self, raw_data_info):
        """Parse data info with enhanced metadata support."""
        d = raw_data_info.copy()
        
        # Debug logging for first few images to verify path construction
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count <= 3:
            print(f"🔍 Path verification debug #{self._debug_count}:")
            print(f"   • img_path from load_data_list: {d['img_path']}")
            print(f"   • data_root: {getattr(self, 'data_root', 'None')}")
            full_path = osp.join(self.data_root, d['img_path']) if hasattr(self, 'data_root') else d['img_path']
            print(f"   • Full absolute path: {full_path}")
            print(f"   • Path exists: {osp.exists(full_path)}")

        # Create or get image information
        img_h, img_w = d['height'], d['width']
        
        # Get class names for class-specific filtering
        class_names = self.METAINFO['classes']
        
        # Get filter configuration
        min_size = self.filter_cfg.get('min_size', 1)
        class_specific_min_sizes = self.filter_cfg.get('class_specific_min_sizes', {})

        # Handle bboxes and labels from instances with enhanced filtering
        bboxes, labels = [], []
        filtered_count = 0
        enlarged_count = 0
        chart_type_filtered_count = 0
        
        # Get chart type for filtering
        chart_type = d.get('chart_type', '').lower()
        chart_mapping = self.CHART_TYPE_ELEMENT_MAPPING.get(chart_type, {})
        allowed_data_elements = chart_mapping.get('allowed_data_elements', set())
        forbidden_data_elements = chart_mapping.get('forbidden_data_elements', set())
        
        for inst in d.get('instances', []):
            bbox = inst['bbox']
            label_id = inst['bbox_label']
            
            # Get class name for this label
            class_name = class_names[label_id] if 0 <= label_id < len(class_names) else 'unknown'
            
            # Chart-type specific filtering: Skip forbidden data elements
            if chart_type and class_name in forbidden_data_elements:
                chart_type_filtered_count += 1
                if self._debug_count <= 3 and chart_type_filtered_count <= 3:
                    print(f"   🚫 Filtered {class_name} from {chart_type} chart (inappropriate data element)")
                continue
            
            # Chart-type specific validation: Log allowed data elements
            if chart_type and class_name in allowed_data_elements:
                if self._debug_count <= 3:
                    print(f"   ✅ Keeping {class_name} for {chart_type} chart (appropriate data element)")
            
            # Validate and clamp bbox
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1, min(x2, img_w))
            y2 = max(y1, min(y2, img_h))
            
            # Skip invalid bboxes
            if x2 <= x1 or y2 <= y1:
                filtered_count += 1
                continue
            
            # Calculate current bbox dimensions
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_min_dim = min(bbox_w, bbox_h)
            
            # Check class-specific minimum size
            required_min_size = class_specific_min_sizes.get(class_name, min_size)
            
            # If bbox is smaller than required, enlarge it to meet the minimum size
            if bbox_min_dim < required_min_size:
                # Calculate expansion needed
                expand_w = max(0, required_min_size - bbox_w) / 2
                expand_h = max(0, required_min_size - bbox_h) / 2
                
                # Expand bbox while keeping it within image bounds
                new_x1 = max(0, x1 - expand_w)
                new_y1 = max(0, y1 - expand_h)
                new_x2 = min(img_w, x2 + expand_w)
                new_y2 = min(img_h, y2 + expand_h)
                
                # Update bbox coordinates
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                enlarged_count += 1
                
                if self._debug_count <= 3 and enlarged_count <= 3:
                    print(f"   📏 Enlarged {class_name} bbox: {bbox_w:.1f}x{bbox_h:.1f} → {(x2-x1):.1f}x{(y2-y1):.1f}")
                
            bboxes.append([x1, y1, x2, y2])
            labels.append(label_id)

        # Log filtering and enlargement statistics for first few images
        if self._debug_count <= 3:
            print(f"   📊 Bbox processing: {len(bboxes)} kept, {filtered_count} filtered (invalid), {chart_type_filtered_count} filtered (chart-type), {enlarged_count} enlarged")
            if chart_type:
                print(f"   📈 Chart type: {chart_type} | Allowed data elements: {allowed_data_elements}")
                if forbidden_data_elements:
                    print(f"   🚫 Forbidden data elements for {chart_type}: {forbidden_data_elements}")

        # Convert to arrays
        d['gt_bboxes'] = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
        d['gt_bboxes_labels'] = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Enhanced scale factor calculation using data_series_stats
        d['scale_factor'] = np.array([1.0, 1.0], dtype=np.float32)

        # Use enhanced metadata for better scale factor calculation
        data_series_stats = d.get('data_series_stats', {})
        plot_bb = d.get('plot_bb', {})
        
        if data_series_stats and plot_bb and all(k in plot_bb for k in ['width', 'height']):
            x_range = data_series_stats.get('x_range')
            y_range = data_series_stats.get('y_range')
            
            if x_range and len(x_range) == 2 and x_range[1] != x_range[0]:
                d['scale_factor'][0] = plot_bb['width'] / (x_range[1] - x_range[0])
            if y_range and len(y_range) == 2 and y_range[1] != y_range[0]:
                d['scale_factor'][1] = -plot_bb['height'] / (y_range[1] - y_range[0])

        # Required MMDet fields
        d.update({
            'img_shape': (img_h, img_w, 3),
            'ori_shape': (img_h, img_w, 3),
            'pad_shape': (img_h, img_w, 3),
            'flip': False,
            'flip_direction': None,
            'img_fields': ['img'],
            'bbox_fields': ['bbox'],
        })
        
        # Additional metadata for training
        d['img_info'] = {
            'height': img_h,
            'width': img_w,
            'img_shape': d['img_shape'],
            'ori_shape': d['ori_shape'],
            'pad_shape': d['pad_shape'],
            'scale_factor': d['scale_factor'].copy(),
            'flip': d['flip'],
            'flip_direction': d['flip_direction'],
            # Enhanced metadata
            'chart_type': d.get('chart_type', ''),
            'num_data_points': data_series_stats.get('num_data_points', 0),
            'element_counts': d.get('element_counts', {})
        }
        
        return d

def print_missing_image_summary():
    """Print summary of missing images."""
    count = RobustLoadImageFromFile.get_missing_count()
    if count > 0:
        print(f"📊 MISSING IMAGES SUMMARY: {count} images were not found and replaced with dummy images")
    else:
        print("✅ All images loaded successfully!")

def print_dataset_summary():
    """Print summary of dataset configuration."""
    print("📊 ENHANCED CHART DATASET SUMMARY:")
    print(f"   • 21 categories supported for comprehensive chart element detection")
    print(f"   • Auto-detects best annotation files (enriched_with_info > enriched > regular)")
    print(f"   • Enhanced metadata: chart_type, data_series_stats, element_counts, axes_info")
    print(f"   • Robust image loading with fallback to dummy images")
    print(f"   • Multiple annotations per image (not just plot areas)")

print("✅ [PLUGIN] Enhanced ChartDataset + transforms registered!")
print_dataset_summary()
