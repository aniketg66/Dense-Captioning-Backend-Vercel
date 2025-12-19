import numpy as np
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
import logging

logger = logging.getLogger(__name__)

@TRANSFORMS.register_module()
class MaskFilter(BaseTransform):
    """Filter out images with no valid masks during training.
    
    This transform checks if there are any valid masks in the image and
    returns None if no masks are found, which will cause the image to be skipped.
    """
    
    def __init__(self, min_masks=1):
        self.min_masks = min_masks
    
    def transform(self, results):
        """Filter results based on mask availability.
        
        Args:
            results (dict): Result dict from dataset.
            
        Returns:
            dict or None: Returns results if valid masks found, None otherwise.
        """
        # Check if we have valid masks
        gt_masks = results.get('gt_masks')
        
        if gt_masks is None:
            logger.warning("MaskFilter: No gt_masks found, skipping image")
            return None
        
        # Count valid masks
        if hasattr(gt_masks, 'masks'):
            num_masks = len(gt_masks.masks)
        elif hasattr(gt_masks, 'polygons'):
            num_masks = len(gt_masks.polygons)
        else:
            num_masks = 0
        
        if num_masks < self.min_masks:
            logger.info(f"MaskFilter: Only {num_masks} masks found (min: {self.min_masks}), skipping image")
            return None
        
        logger.info(f"MaskFilter: {num_masks} masks found, keeping image for training")
        return results 