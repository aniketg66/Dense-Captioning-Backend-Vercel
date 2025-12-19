# progressive_loss_hook.py - Progressive Loss Switching Hook for Cascade R-CNN
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmdet.models.losses import SmoothL1Loss, GIoULoss, CIoULoss, DIoULoss

@HOOKS.register_module()
class ProgressiveLossHook(Hook):
    """
    Progressive Loss Switching Hook for Cascade R-CNN.
    
    Starts with SmoothL1Loss for all stages, then progressively switches
    stage 3 (final stage) to GIoU/CIoU/DIoU after the model stabilizes.
    
    Args:
        switch_epoch (int): Epoch to switch stage 3 from SmoothL1 to target loss
        target_loss_type (str): Target loss type for stage 3 ('GIoULoss', 'CIoULoss', or 'DIoULoss')
        loss_weight (float): Loss weight for the new loss function
        warmup_epochs (int): Number of epochs to gradually blend the losses
        monitor_stage_weights (bool): Whether to log stage loss weights
        nan_detection (bool): Whether to enable NaN detection and rollback
        max_nan_tolerance (int): Maximum consecutive NaN losses before rollback
    """
    
    def __init__(self, 
                 switch_epoch=5,
                 target_loss_type='GIoULoss',
                 loss_weight=1.0,
                 warmup_epochs=2,
                 monitor_stage_weights=True,
                 nan_detection=False,
                 max_nan_tolerance=5):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.target_loss_type = target_loss_type
        self.loss_weight = loss_weight
        self.warmup_epochs = warmup_epochs
        self.monitor_stage_weights = monitor_stage_weights
        self.nan_detection = nan_detection
        self.max_nan_tolerance = max_nan_tolerance
        self.switched = False
        self.original_loss = None
        self.consecutive_nans = 0
        self.rollback_performed = False
        
    def before_train_epoch(self, runner):
        """Check if we should switch the loss function."""
        current_epoch = runner.epoch
        
        # Switch at the specified epoch
        if current_epoch >= self.switch_epoch and not self.switched:
            self._switch_stage2_loss(runner)
            self.switched = True
            runner.logger.info(
                f"Epoch {current_epoch}: Switched Stage 3 loss to {self.target_loss_type}")
        
        # Monitor during warmup period
        elif current_epoch >= self.switch_epoch and current_epoch < self.switch_epoch + self.warmup_epochs:
            if self.monitor_stage_weights:
                self._log_loss_info(runner, current_epoch)
    
    def _switch_stage2_loss(self, runner):
        """Switch stage 3 bbox loss from SmoothL1 to target loss."""
        model = runner.model
        
        # Navigate to stage 3 bbox head (index 2) - final refinement stage
        try:
            # Handle DDP wrapper
            if hasattr(model, 'module'):
                bbox_head_stage2 = model.module.roi_head.bbox_head[2]
            else:
                bbox_head_stage2 = model.roi_head.bbox_head[2]
            
            # Store original loss for comparison
            self.original_loss = bbox_head_stage2.loss_bbox
            
            # Create new loss function
            if self.target_loss_type == 'GIoULoss':
                new_loss = GIoULoss(loss_weight=self.loss_weight)
                # Enable decoded bbox regression for IoU losses
                bbox_head_stage2.reg_decoded_bbox = True
            elif self.target_loss_type == 'CIoULoss':
                new_loss = CIoULoss(loss_weight=self.loss_weight)
                # Enable decoded bbox regression for IoU losses
                bbox_head_stage2.reg_decoded_bbox = True
            elif self.target_loss_type == 'DIoULoss':
                new_loss = DIoULoss(loss_weight=self.loss_weight)
                # Enable decoded bbox regression for IoU losses
                bbox_head_stage2.reg_decoded_bbox = True
            else:
                raise ValueError(f"Unsupported target loss type: {self.target_loss_type}")
            
            # Store the switch information with loss-specific benefits
            if self.target_loss_type == 'CIoULoss':
                runner.logger.info(f"🎯 CIoU Loss Benefits for Data Points:")
                runner.logger.info(f"   • Directly optimizes center point distance")
                runner.logger.info(f"   • Enforces aspect ratio consistency (square-ish data points)")
                runner.logger.info(f"   • Better convergence for small objects")
                runner.logger.info(f"   • Most complete bounding box quality metric")
            elif self.target_loss_type == 'DIoULoss':
                runner.logger.info(f"🎯 DIoU Loss Benefits for Data Points:")
                runner.logger.info(f"   • Directly optimizes center point distance")
                runner.logger.info(f"   • Better convergence for small objects")
                runner.logger.info(f"   • More precise localization for data points")
            elif self.target_loss_type == 'GIoULoss':
                runner.logger.info(f"🎯 GIoU Loss Benefits:")
                runner.logger.info(f"   • Improved IoU-based optimization")
                runner.logger.info(f"   • Better than standard IoU loss")
            
            # Replace the loss function
            bbox_head_stage2.loss_bbox = new_loss
            
            runner.logger.info(
                f"Progressive Loss Switch: Stage 3 changed from "
                f"{type(self.original_loss).__name__} to {self.target_loss_type}")
                
        except Exception as e:
            runner.logger.error(f"Failed to switch loss function: {e}")
    
    def _log_loss_info(self, runner, epoch):
        """Log information about current loss configuration."""
        try:
            model = runner.model
            if hasattr(model, 'module'):
                bbox_heads = model.module.roi_head.bbox_head
            else:
                bbox_heads = model.roi_head.bbox_head
            
            loss_info = {}
            for i, head in enumerate(bbox_heads):
                loss_type = type(head.loss_bbox).__name__
                loss_weight = head.loss_bbox.loss_weight
                loss_info[f'stage_{i+1}'] = f"{loss_type}(w={loss_weight})"
            
            runner.logger.info(f"Epoch {epoch} Loss Configuration: {loss_info}")
            
        except Exception as e:
            runner.logger.warning(f"Could not log loss info: {e}")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Monitor loss values during training and detect NaN."""
        if self.switched and outputs is not None and isinstance(outputs, dict):
            # NaN detection and rollback logic
            if self.nan_detection and not self.rollback_performed:
                total_loss = outputs.get('loss', None)
                if total_loss is not None and torch.isnan(total_loss):
                    self.consecutive_nans += 1
                    runner.logger.warning(f"🚨 NaN detected in total loss! Consecutive: {self.consecutive_nans}/{self.max_nan_tolerance}")
                    
                    if self.consecutive_nans >= self.max_nan_tolerance:
                        self._rollback_loss(runner)
                        self.consecutive_nans = 0
                        self.rollback_performed = True
                        runner.logger.error(f"🔄 EMERGENCY ROLLBACK: Switched back to SmoothL1Loss due to {self.max_nan_tolerance} consecutive NaN losses")
                        return
                elif total_loss is not None and torch.isfinite(total_loss):
                    # Reset NaN counter on successful iteration
                    self.consecutive_nans = 0
            
            # Log individual stage losses if available
            log_vars = outputs.get('log_vars', {})
            stage_losses = {}
            
            for key, value in log_vars.items():
                if 'loss_bbox' in key and isinstance(value, (int, float)):
                    stage_losses[key] = value
            
            if stage_losses and self.monitor_stage_weights:
                # Log every 100 iterations to avoid spam
                if runner.iter % 100 == 0:
                    loss_summary = ", ".join([f"{k}: {v:.4f}" for k, v in stage_losses.items()])
                    runner.logger.info(f"Stage Losses - {loss_summary}")

    def after_train_epoch(self, runner):
        """Check epoch completion and reset NaN counters."""
        if self.nan_detection and self.switched:
            # Log current status
            if self.consecutive_nans > 0:
                runner.logger.warning(f"Epoch {runner.epoch} completed with {self.consecutive_nans} NaN occurrences")
            else:
                runner.logger.info(f"Epoch {runner.epoch} completed successfully with {self.target_loss_type}")

    def _rollback_loss(self, runner):
        """Rollback stage 3 to SmoothL1Loss."""
        try:
            model = runner.model
            if hasattr(model, 'module'):
                bbox_head_stage2 = model.module.roi_head.bbox_head[2]
            else:
                bbox_head_stage2 = model.roi_head.bbox_head[2]
            
            # Create new SmoothL1Loss
            rollback_loss = SmoothL1Loss(beta=1.0, loss_weight=1.0)
            bbox_head_stage2.loss_bbox = rollback_loss
            bbox_head_stage2.reg_decoded_bbox = False  # Disable decoded bbox for SmoothL1
            
            runner.logger.info(f"✅ Successfully rolled back Stage 3 from {self.target_loss_type} to SmoothL1Loss")
            
        except Exception as e:
            runner.logger.error(f"❌ Failed to rollback loss function: {e}")


@HOOKS.register_module()
class AdaptiveLossHook(Hook):
    """
    Adaptive version that switches based on training stability metrics.
    
    Monitors IoU overlap quality and switches when model is stable.
    """
    
    def __init__(self,
                 min_epoch=3,
                 min_avg_iou=0.4,
                 target_loss_type='GIoULoss',
                 loss_weight=1.0,
                 check_interval=100):
        super().__init__()
        self.min_epoch = min_epoch
        self.min_avg_iou = min_avg_iou
        self.target_loss_type = target_loss_type
        self.loss_weight = loss_weight
        self.check_interval = check_interval
        self.switched = False
        self.iou_history = []
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Monitor training stability through IoU metrics."""
        if (not self.switched and 
            runner.epoch >= self.min_epoch and 
            runner.iter % self.check_interval == 0):
            
            # Extract IoU information from outputs if available
            if outputs and isinstance(outputs, dict):
                log_vars = outputs.get('log_vars', {})
                
                # Look for any IoU-related metrics
                iou_metrics = [v for k, v in log_vars.items() 
                              if 'iou' in k.lower() and isinstance(v, (int, float))]
                
                if iou_metrics:
                    avg_iou = sum(iou_metrics) / len(iou_metrics)
                    self.iou_history.append(avg_iou)
                    
                    # Keep only recent history
                    if len(self.iou_history) > 10:
                        self.iou_history.pop(0)
                    
                    # Check if we should switch
                    if (len(self.iou_history) >= 5 and 
                        sum(self.iou_history[-5:]) / 5 >= self.min_avg_iou):
                        
                        self._switch_stage2_loss(runner)
                        self.switched = True
                        
                        recent_iou = sum(self.iou_history[-5:]) / 5
                        runner.logger.info(
                            f"Adaptive switch at epoch {runner.epoch}, iter {runner.iter}: "
                            f"avg IoU {recent_iou:.3f} >= {self.min_avg_iou}")
    
    def _switch_stage2_loss(self, runner):
        """Same switching logic as ProgressiveLossHook."""
        model = runner.model
        try:
            if hasattr(model, 'module'):
                bbox_head_stage2 = model.module.roi_head.bbox_head[2]
            else:
                bbox_head_stage2 = model.roi_head.bbox_head[2]
            
            if self.target_loss_type == 'GIoULoss':
                new_loss = GIoULoss(loss_weight=self.loss_weight)
                bbox_head_stage2.reg_decoded_bbox = True
            elif self.target_loss_type == 'CIoULoss':
                new_loss = CIoULoss(loss_weight=self.loss_weight)
                bbox_head_stage2.reg_decoded_bbox = True
            elif self.target_loss_type == 'DIoULoss':
                new_loss = DIoULoss(loss_weight=self.loss_weight)
                bbox_head_stage2.reg_decoded_bbox = True
            else:
                raise ValueError(f"Unsupported target loss type: {self.target_loss_type}")
            
            bbox_head_stage2.loss_bbox = new_loss
            
            runner.logger.info(f"Adaptive Loss Switch: Stage 3 → {self.target_loss_type}")
                
        except Exception as e:
            runner.logger.error(f"Failed to switch loss function: {e}") 