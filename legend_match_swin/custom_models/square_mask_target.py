# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from mmdet.registry import MODELS
from mmdet.structures.mask.mask_target import mask_target as original_mask_target


def square_mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    """Compute square mask target for positive proposals in multiple images.
    
    This function forces all mask targets to be square regardless of the original
    aspect ratio to avoid tensor size mismatches.
    
    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        Tensor: Square mask target of each image, has shape (num_pos, size, size).
    """
    # Get the target size (should be a tuple like (14, 14))
    mask_size = _pair(cfg.mask_size)
    
    # Force square size by using the minimum dimension
    square_size = min(mask_size)
    
    # Create a proper ConfigDict object
    from mmengine.config import ConfigDict
    square_cfg = ConfigDict({'mask_size': (square_size, square_size)})
    
    # Call the original mask target function with square size
    mask_targets = original_mask_target(pos_proposals_list, pos_assigned_gt_inds_list, 
                                       gt_masks_list, square_cfg)
    
    print(f"🔍 SQUARE_MASK_TARGET: Original mask_targets shape: {mask_targets.shape}")
    print(f"🔍 SQUARE_MASK_TARGET: Expected square_size: {square_size}")
    
    # Force square shape by padding or cropping if necessary
    if mask_targets.size(1) != square_size or mask_targets.size(2) != square_size:
        print(f"🔍 SQUARE_MASK_TARGET: Forcing square shape from {mask_targets.shape} to ({mask_targets.size(0)}, {square_size}, {square_size})")
        
        # Create new tensor with square shape
        num_masks = mask_targets.size(0)
        square_targets = torch.zeros(num_masks, square_size, square_size, 
                                   device=mask_targets.device, dtype=mask_targets.dtype)
        
        # Copy the mask data, padding with zeros if necessary
        h, w = mask_targets.size(1), mask_targets.size(2)
        h_copy = min(h, square_size)
        w_copy = min(w, square_size)
        
        square_targets[:, :h_copy, :w_copy] = mask_targets[:, :h_copy, :w_copy]
        mask_targets = square_targets
        
        print(f"🔍 SQUARE_MASK_TARGET: Final mask_targets shape: {mask_targets.shape}")
    else:
        print(f"🔍 SQUARE_MASK_TARGET: Masks already square: {mask_targets.shape}")
    
    return mask_targets


# Register the custom function
MODELS.register_module(name='square_mask_target', module=square_mask_target) 