import numpy as np
from typing import Dict, Optional
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.registry import TRANSFORMS
import logging
from mmdet.structures.mask import BitmapMasks

logger = logging.getLogger(__name__)

@TRANSFORMS.register_module()
class FlexibleLoadAnnotations(LoadAnnotations):
    """
    Flexible annotation loader that handles mixed mask/bbox datasets.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_mask: bool = True,
                 with_seg: bool = False,
                 poly2mask: bool = True,
                 **kwargs):
        super().__init__(
            with_bbox=with_bbox,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            **kwargs
        )
        self.mask_stats = {'total': 0, 'with_masks': 0, 'without_masks': 0}

    def _load_masks(self, results: dict) -> dict:
        """Load mask annotations from COCO format instances."""
        if not self.with_mask or not isinstance(results, dict):
            return results

        # Check for ann_info format (what COCO dataset actually provides)
        ann_info = results.get('ann_info')
        if isinstance(ann_info, dict):
            # Check if segmentation is in ann_info
            if 'segmentation' in ann_info:
                segmentation = ann_info['segmentation']
                if segmentation and isinstance(segmentation, list) and len(segmentation) > 0:
                    # Convert to mask format
                    ann_info['masks'] = segmentation
                    return super()._load_masks(results)
            
            # Check for polygon data in ann_info
            if 'polygon' in ann_info:
                polygon = ann_info['polygon']
                if polygon and isinstance(polygon, dict):
                    try:
                        # Convert polygon to COCO segmentation format
                        coords = []
                        for j in range(4):  # Assuming 4-point polygons
                            x_key = f'x{j}'
                            y_key = f'y{j}'
                            if x_key in polygon and y_key in polygon:
                                coords.extend([polygon[x_key], polygon[y_key]])
                        
                        if len(coords) >= 6:  # Need at least 3 points (6 coordinates)
                            # Convert to COCO format: [x1, y1, x2, y2, x3, y3, ...]
                            segmentation = [coords]
                            ann_info['segmentation'] = segmentation
                            ann_info['masks'] = segmentation
                            return super()._load_masks(results)
                    except Exception as e:
                        logger.debug(f"Polygon conversion failed: {e}")
        
        # Handle COCO format: instances with segmentation
        instances = results.get('instances')
        if isinstance(instances, list):
            # Process ALL instances - keep both with and without masks
            valid_instances = []
            
            for i, instance in enumerate(instances):
                self.mask_stats['total'] += 1
                
                # Check for segmentation in COCO format (COCO dataset stores it in 'mask' field)
                segmentation = instance.get('mask') or instance.get('segmentation')
                if segmentation and isinstance(segmentation, list) and len(segmentation) > 0:
                    # Handle nested list format: [[x1, y1, x2, y2, ...]]
                    if isinstance(segmentation[0], list):
                        # Nested format - check if inner list has enough coordinates
                        inner_seg = segmentation[0]
                        if len(inner_seg) >= 6:  # Need at least 3 points (6 coordinates)
                            instance['mask'] = segmentation  # Keep original nested format for parent
                            valid_instances.append(instance)
                            self.mask_stats['with_masks'] += 1
                        else:
                            # Keep instance for bbox training even without valid mask
                            instance['mask'] = []
                            valid_instances.append(instance)
                            self.mask_stats['without_masks'] += 1
                    else:
                        # Flat format - already correct
                        instance['mask'] = segmentation
                        valid_instances.append(instance)
                        self.mask_stats['with_masks'] += 1
                else:
                    # Check for polygon data and convert to segmentation
                    polygon = instance.get('polygon')
                    if polygon and isinstance(polygon, dict):
                        # Convert polygon to COCO segmentation format
                        try:
                            # Extract polygon coordinates
                            coords = []
                            for j in range(4):  # Assuming 4-point polygons
                                x_key = f'x{j}'
                                y_key = f'y{j}'
                                if x_key in polygon and y_key in polygon:
                                    coords.extend([polygon[x_key], polygon[y_key]])
                            
                            if len(coords) >= 6:  # Need at least 3 points (6 coordinates)
                                # Convert to COCO format: [x1, y1, x2, y2, x3, y3, ...]
                                segmentation = [coords]
                                instance['segmentation'] = segmentation
                                instance['mask'] = segmentation
                                valid_instances.append(instance)
                                self.mask_stats['with_masks'] += 1
                            else:
                                # Keep instance for bbox training even without mask
                                # Add empty mask field to prevent KeyError in parent class
                                instance['mask'] = []
                                valid_instances.append(instance)
                                self.mask_stats['without_masks'] += 1
                        except Exception as e:
                            # Keep instance for bbox training even if polygon conversion fails
                            # Add empty mask field to prevent KeyError in parent class
                            instance['mask'] = []
                            valid_instances.append(instance)
                            self.mask_stats['without_masks'] += 1
                    else:
                        # Keep instance for bbox training even without segmentation
                        # Add empty mask field to prevent KeyError in parent class
                        instance['mask'] = []
                        valid_instances.append(instance)
                        self.mask_stats['without_masks'] += 1
            
            # Update results with valid instances only
            results['instances'] = valid_instances
            
            # Call parent method to process the filtered instances
            if valid_instances:
                super()._load_masks(results)  # Parent modifies results in place
                return results
            else:
                # No valid masks, create empty mask structure
                h, w = results.get('img_shape', (0, 0))
                results['gt_masks'] = BitmapMasks([], h, w)
                results['gt_ignore_flags'] = np.array([], dtype=bool)
                return results

        # Check for direct segmentation in results
        if 'segmentation' in results:
            segmentation = results['segmentation']
            if segmentation and isinstance(segmentation, list) and len(segmentation) > 0:
                results['masks'] = segmentation
                return super()._load_masks(results)

        return results

    def transform(self, results: dict) -> dict:
        """Transform function to load annotations."""
        # ensure we always return a dict
        if not isinstance(results, dict):
            logger.error(f"Expected dict, got {type(results)}")
            return {}

        # Call parent transform to handle bbox loading
        results = super().transform(results)
        
        # Handle mask loading with our custom logic
        results = self._load_masks(results)
        
        # periodic logging
        if self.mask_stats['total'] % 1000 == 0:
            t = self.mask_stats['total']
            w = self.mask_stats['with_masks']
            wo = self.mask_stats['without_masks']
            logger.info(f"Mask stats - total: {t}, with_masks: {w}, without_masks: {wo}")
        
        return results

    def __repr__(self) -> str:
        """String representation."""
        return (f'{self.__class__.__name__}('
                f'with_bbox={self.with_bbox}, '
                f'with_mask={self.with_mask}, '
                f'with_seg={self.with_seg}, '
                f'poly2mask={self.poly2mask})')
