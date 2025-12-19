# -*- coding: utf-8 -*-
import json
import os.path as osp
from typing import List, Dict, Any

from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class ChartTypeDataset(BaseDataset):
    """Chart Type Classification Dataset.
    
    This dataset extracts chart type information from the JSON annotations
    and creates a classification dataset for chart type prediction.
    
    Chart types mapping:
    - line: 0
    - bar: 1  
    - scatter: 2
    - pie: 3
    - area: 4
    """
    
    CHART_TYPES = ['line', 'bar', 'scatter', 'pie', 'area']
    
    def __init__(self, data_root: str, ann_file: str, **kwargs):
        # Set up chart type to label mapping
        self.chart_type_to_label = {chart_type: idx for idx, chart_type in enumerate(self.CHART_TYPES)}
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            **kwargs
        )
    
    def load_data_list(self) -> List[Dict[str, Any]]:
        """Load annotation file and return list of data info."""
        data_list = []
        
        # Load annotation file
        ann_file_path = osp.join(self.data_root, self.ann_file)
        with open(ann_file_path, 'r') as f:
            annotations = json.load(f)
        
        # Process each image
        for img_info in annotations['images']:
            # Get image path
            img_path = osp.join(self.data_prefix.get('img', ''), img_info['file_name'])
            
            # Extract chart type
            chart_type = img_info.get('chart_type', 'unknown')
            
            # Skip if chart type is not in our defined types
            if chart_type not in self.chart_type_to_label:
                print(f"Warning: Unknown chart type '{chart_type}' in image {img_info['file_name']}, skipping...")
                continue
            
            # Get label
            gt_label = self.chart_type_to_label[chart_type]
            
            # Create data info
            data_info = {
                'img_path': img_path,
                'gt_label': gt_label,
                'chart_type': chart_type,  # Keep for reference
                'img_id': img_info.get('id', ''),
                'height': img_info.get('height', 0),
                'width': img_info.get('width', 0)
            }
            
            data_list.append(data_info)
        
        print(f"Loaded {len(data_list)} samples from {ann_file_path}")
        
        # Print class distribution
        class_counts = {}
        for data_info in data_list:
            chart_type = data_info['chart_type']
            class_counts[chart_type] = class_counts.get(chart_type, 0) + 1
        
        print("Class distribution:")
        for chart_type, count in class_counts.items():
            print(f"  {chart_type}: {count}")
        
        return data_list
    
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index."""
        return [self.data_list[idx]['gt_label']]
    
    def get_gt_labels(self) -> List[int]:
        """Get ground truth labels for all samples."""
        return [data_info['gt_label'] for data_info in self.data_list]
    
    def get_cat_names(self, cat_ids: List[int]) -> List[str]:
        """Get category names by category ids."""
        return [self.CHART_TYPES[cat_id] for cat_id in cat_ids] 