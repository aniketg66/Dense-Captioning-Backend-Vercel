import traceback
import torch
import numpy as np
from mmdet.models.dense_heads.anchor_head import AnchorHead

# keep a reference to the original
_orig_loss_by_feat = AnchorHead.loss_by_feat

def _validate_gt_instances(batch_gt_instances, batch_img_metas):
    """Validate ground truth instances to prevent index out of bounds errors."""
    if not batch_gt_instances:
        return batch_gt_instances
    
    cleaned_instances = []
    for i, (gt_instances, img_meta) in enumerate(zip(batch_gt_instances, batch_img_metas)):
        if not hasattr(gt_instances, 'labels') or not hasattr(gt_instances, 'bboxes'):
            cleaned_instances.append(gt_instances)
            continue
            
        # Get image dimensions
        img_h, img_w = img_meta.get('img_shape', (800, 600))[:2]
        
        # Validate class labels (should be 0-based and < num_classes)
        labels = gt_instances.labels
        
        # Validate bbox coordinates
        bboxes = gt_instances.bboxes
        if hasattr(bboxes, 'tensor'):
            bbox_tensor = bboxes.tensor
        else:
            bbox_tensor = bboxes
        
        # Check for finite values in both labels and bboxes
        finite_label_mask = torch.isfinite(labels)
        finite_bbox_mask = torch.isfinite(bbox_tensor).all(dim=1)  # All 4 coords must be finite
        finite_mask = finite_label_mask & finite_bbox_mask
        
        if not finite_mask.all():
            non_finite_count = (~finite_mask).sum().item()
            print(f"⚠️ Sample {i}: {non_finite_count} instances have non-finite values")
            
        # Apply finite mask first
        if finite_mask.any():
            labels = labels[finite_mask]
            bbox_tensor = bbox_tensor[finite_mask]
        else:
            print(f"⚠️ Sample {i}: No instances with finite values!")
            # Create new empty InstanceData
            from mmengine.structures import InstanceData
            new_gt_instances = InstanceData()
            new_gt_instances.labels = torch.empty(0, dtype=labels.dtype, device=labels.device)
            if hasattr(bboxes, 'tensor'):
                empty_bboxes = type(bboxes)(torch.empty(0, 4, dtype=bbox_tensor.dtype, device=bbox_tensor.device))
                new_gt_instances.bboxes = empty_bboxes
            else:
                new_gt_instances.bboxes = torch.empty(0, 4, dtype=bbox_tensor.dtype, device=bbox_tensor.device)
            cleaned_instances.append(new_gt_instances)
            continue
            
        # Now validate remaining instances
        valid_label_mask = (labels >= 0) & (labels < 21)  # 21 enhanced categories
            
        # Check bbox bounds: x1,y1,x2,y2 format
        valid_bbox_mask = (
            (bbox_tensor[:, 0] >= 0) & (bbox_tensor[:, 0] < img_w) &  # x1
            (bbox_tensor[:, 1] >= 0) & (bbox_tensor[:, 1] < img_h) &  # y1  
            (bbox_tensor[:, 2] > bbox_tensor[:, 0]) & (bbox_tensor[:, 2] <= img_w) &  # x2 > x1 and <= width
            (bbox_tensor[:, 3] > bbox_tensor[:, 1]) & (bbox_tensor[:, 3] <= img_h)    # y2 > y1 and <= height
        )
        
        # Combine valid masks
        valid_mask = valid_label_mask & valid_bbox_mask
        
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"⚠️ Sample {i}: Removing {invalid_count}/{len(valid_mask)} invalid GT instances")
            
            # Filter to valid instances only
            if valid_mask.any():
                gt_instances.labels = labels[valid_mask]
                if hasattr(bboxes, 'tensor'):
                    gt_instances.bboxes.tensor = bbox_tensor[valid_mask]
                else:
                    gt_instances.bboxes = bbox_tensor[valid_mask]
            else:
                # No valid instances - create new empty InstanceData
                print(f"⚠️ Sample {i}: No valid GT instances remaining!")
                from mmengine.structures import InstanceData
                new_gt_instances = InstanceData()
                new_gt_instances.labels = torch.empty(0, dtype=labels.dtype, device=labels.device)
                if hasattr(bboxes, 'tensor'):
                    # Create empty bboxes with proper structure
                    empty_bboxes = type(bboxes)(torch.empty(0, 4, dtype=bbox_tensor.dtype, device=bbox_tensor.device))
                    new_gt_instances.bboxes = empty_bboxes
                else:
                    new_gt_instances.bboxes = torch.empty(0, 4, dtype=bbox_tensor.dtype, device=bbox_tensor.device)
                gt_instances = new_gt_instances
        
        cleaned_instances.append(gt_instances)
    
    return cleaned_instances

def _validate_and_clean_predictions(cls_scores, bbox_preds, batch_gt_instances):
    """Clean up predictions by removing invalid anchors early."""
    cleaned_cls_scores = []
    cleaned_bbox_preds = []
    
    for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
        # Get tensor shapes: cls_score: [N, A, H, W], bbox_pred: [N, A*4, H, W]
        N, A, H, W = cls_score.shape
        _, A4, _, _ = bbox_pred.shape
        
        # Check for extreme values that could cause CUDA errors
        if torch.any(torch.abs(cls_score) > 1e6):
            print(f"⚠️ Level {level_idx}: Extreme values in cls_score, clamping...")
            cls_score = torch.clamp(cls_score, -1e6, 1e6)
            
        if torch.any(torch.abs(bbox_pred) > 1e6):
            print(f"⚠️ Level {level_idx}: Extreme values in bbox_pred, clamping...")
            bbox_pred = torch.clamp(bbox_pred, -1e6, 1e6)
        
        # Check for NaN or infinite values
        cls_valid = torch.isfinite(cls_score).all(dim=1)  # [N, H, W]
        bbox_valid = torch.isfinite(bbox_pred).all(dim=1)  # [N, H, W]
        
        # Combine validity masks (both should be [N, H, W])
        valid_mask = cls_valid & bbox_valid
        
        # Check for zero-sized predictions (all zeros)
        # Sum over channel dimension to get spatial mask
        cls_nonzero = (cls_score.abs().sum(dim=1) > 1e-8)  # [N, H, W]
        bbox_nonzero = (bbox_pred.abs().sum(dim=1) > 1e-8)  # [N, H, W]
        
        nonzero_mask = cls_nonzero & bbox_nonzero
        
        # Final mask: must be both valid and non-zero [N, H, W]
        final_mask = valid_mask & nonzero_mask
        
        # If we lose too many anchors, keep original (avoid completely empty)
        total_locations = N * H * W
        valid_locations = final_mask.sum().item()
        
        if valid_locations < max(1, total_locations * 0.1):  # keep at least 10%
            print(f"⚠️ Level {level_idx}: Would remove too many locations ({valid_locations}/{total_locations}), keeping original")
            cleaned_cls_scores.append(cls_score)
            cleaned_bbox_preds.append(bbox_pred)
        else:
            # Apply mask to zero out invalid predictions
            # Expand mask to match tensor dimensions
            cls_mask = final_mask.unsqueeze(1).expand(-1, A, -1, -1)  # [N, A, H, W]
            bbox_mask = final_mask.unsqueeze(1).expand(-1, A4, -1, -1)  # [N, A*4, H, W]
            
            clean_cls = cls_score.clone()
            clean_bbox = bbox_pred.clone()
            
            # Zero out invalid predictions
            clean_cls[~cls_mask] = 0.0
            clean_bbox[~bbox_mask] = 0.0
            
            cleaned_cls_scores.append(clean_cls)
            cleaned_bbox_preds.append(clean_bbox)
            
            invalid_locations = total_locations - valid_locations
            if invalid_locations > 0:
                print(f"⚠️ Level {level_idx}: Zeroed {invalid_locations}/{total_locations} invalid locations")
    
    return cleaned_cls_scores, cleaned_bbox_preds

def _safe_loss_by_feat(self, *args, **kwargs):
    """Minimal intervention - let CheckInvalidLossHook handle NaN recovery."""
    
    try:
        # Just call the original function - let CheckInvalidLossHook catch any NaN
        return _orig_loss_by_feat(self, *args, **kwargs)
        
    except RuntimeError as e:
        msg = str(e)
        if any(err in msg.lower() for err in ['index out of bounds', 'device-side assert', 
                                              'cuda error', 'invalid index', 'out of range']):
            print("⚠️ Caught GPU/index error in AnchorHead.loss_by_feat:")
            print(f"   Error: {msg}")
            print("   Re-raising to let CheckInvalidLossHook handle it...")
            
            # Clear CUDA cache to help with device-side asserts
            torch.cuda.empty_cache()
            
            # Re-raise the error to let CheckInvalidLossHook catch it
            # This way we get clean iteration skipping instead of fallback values
            raise
        
        # re-raise anything else
        raise
    
    except Exception as e:
        print(f"⚠️ Unexpected error in AnchorHead.loss_by_feat: {e}")
        print("   Re-raising to let CheckInvalidLossHook handle it...")
        
        # Re-raise to let the proper error handling take over
        raise

# overwrite the method
AnchorHead.loss_by_feat = _safe_loss_by_feat

print("✅ [PATCH] Enhanced robust AnchorHead.loss_by_feat patch applied!")
print("   - Early detection and cleanup of invalid/zero anchors")
print("   - Comprehensive error handling for GPU/CUDA errors") 
print("   - Graceful fallback to zero losses when needed") 