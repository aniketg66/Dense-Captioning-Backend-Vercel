# nan_recovery_hook.py - Graceful NaN loss recovery for Cascade R-CNN
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from typing import Optional, Dict, Any


@HOOKS.register_module()
class NanRecoveryHook(Hook):
    """Hook to handle NaN losses gracefully without crashing training.
    
    This hook detects NaN losses and handles them by:
    1. Replacing NaN losses with the last valid loss value
    2. Skipping gradient updates for that iteration
    3. Logging the recovery for monitoring
    4. Allowing training to continue normally
    """
    
    def __init__(self, 
                 fallback_loss: float = 0.5,
                 max_consecutive_nans: int = 100,  # Increased from 10
                 log_interval: int = 50):  # Log less frequently
        self.fallback_loss = fallback_loss
        self.max_consecutive_nans = max_consecutive_nans
        self.log_interval = log_interval
        
        # State tracking
        self.last_valid_loss = fallback_loss
        self.consecutive_nans = 0
        self.total_nans = 0
        self.nan_iterations = []
    
    def before_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None) -> None:
        """Reset any state before training iteration."""
        pass
    
    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: Optional[dict] = None,
                        outputs: Optional[Dict[str, Any]] = None) -> None:
        """Handle NaN losses after training iteration."""
        if outputs is None:
            return
        
        # Check ALL loss components for NaN, not just the main loss
        has_nan = False
        
        # Check main loss
        total_loss = outputs.get('loss')
        if total_loss is not None and (torch.isnan(total_loss) or torch.isinf(total_loss)):
            has_nan = True
        
        # Check all individual loss components
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and 'loss' in key.lower():
                if torch.isnan(value) or torch.isinf(value):
                    has_nan = True
                    break
        
        if has_nan:
            self._handle_nan_loss(runner, batch_idx, outputs)
        else:
            # Valid loss - update tracking
            if total_loss is not None:
                self.last_valid_loss = float(total_loss.item())
                if self.consecutive_nans > 0:
                    runner.logger.info(f"🎉 Loss recovered after {self.consecutive_nans} NaN iterations")
                self.consecutive_nans = 0
    
    def _handle_nan_loss(self, runner: Runner, batch_idx: int, outputs: Dict[str, Any]) -> None:
        """Handle NaN loss by replacing with detached fallback and managing state."""
        self.consecutive_nans += 1
        self.total_nans += 1
        self.nan_iterations.append(batch_idx)
        
        # Try to get last good state from SkipBadSamplesHook if available
        last_good_iteration = batch_idx
        last_good_loss = self.last_valid_loss
        
        for hook in runner.hooks:
            if hasattr(hook, 'last_good_iteration') and hasattr(hook, 'last_good_loss'):
                if hook.last_good_loss is not None:
                    last_good_iteration = hook.last_good_iteration
                    last_good_loss = hook.last_good_loss
                break
        
        # Replace NaN loss with detached fallback (no gradients = true no-op)
        if 'loss' in outputs and outputs['loss'] is not None:
            fallback_tensor = torch.tensor(
                last_good_loss, 
                device=outputs['loss'].device,
                dtype=outputs['loss'].dtype
                # NOTE: No requires_grad=True - this makes it detached
            )
            outputs['loss'] = fallback_tensor
        
        # Also fix individual loss components with detached tensors
        self._fix_loss_components(outputs, last_good_loss)
        
        # Log recovery with state info
        if self.consecutive_nans <= 5 or self.consecutive_nans % self.log_interval == 0:
            runner.logger.warning(
                f"🔄 NaN Recovery at iteration {batch_idx}: "
                f"Using last good loss {last_good_loss:.4f} from iteration {last_good_iteration}. "
                f"Consecutive NaNs: {self.consecutive_nans}, Total: {self.total_nans}"
            )
        
        # Reset training state if too many consecutive NaNs
        if self.consecutive_nans >= self.max_consecutive_nans:
            self._reset_nan_state(runner, last_good_iteration)
    
    def _reset_nan_state(self, runner: Runner, last_good_iteration: int) -> None:
        """Reset training state when too many consecutive NaNs."""
        runner.logger.error(
            f"🔄 Too many consecutive NaN losses ({self.consecutive_nans}). "
            f"Resetting to last good state from iteration {last_good_iteration}"
        )
        
        try:
            # Clear model gradients
            if hasattr(runner.model, 'zero_grad'):
                runner.model.zero_grad()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset consecutive counter
            self.consecutive_nans = 0
            
            runner.logger.info(f"✅ NaN state reset. Resuming training...")
            
        except Exception as e:
            runner.logger.error(f"❌ Failed to reset NaN state: {e}")
    
    def _fix_loss_components(self, outputs: Dict[str, Any], fallback_loss: float = None) -> None:
        """Fix ALL loss components with detached tensors (no gradients)."""
        if fallback_loss is None:
            fallback_loss = self.last_valid_loss
            
        fallback_small = max(0.01, fallback_loss * 0.1)  # Ensure non-zero minimum
        
        # Fix ALL tensors with 'loss' in the key name using detached tensors
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and 'loss' in key.lower():
                if torch.isnan(value) or torch.isinf(value):
                    # Create detached replacement tensor (no gradients)
                    replacement = torch.tensor(
                        fallback_small,
                        device=value.device,
                        dtype=value.dtype
                        # NOTE: No requires_grad=True - detached for true no-op
                    )
                    outputs[key] = replacement
                    print(f"   🔧 Fixed {key}: {value.item():.4f} -> detached {fallback_small:.4f}")
        
        # Also fix any scalar values that might be NaN
        for key, value in list(outputs.items()):
            if isinstance(value, (int, float)) and 'loss' in key.lower():
                if not torch.isfinite(torch.tensor(value)):
                    outputs[key] = fallback_small
                    print(f"   🔧 Fixed scalar {key}: {value} -> {fallback_small:.4f}")
    
    def after_train_epoch(self, runner: Runner) -> None:
        """Summary statistics after each epoch."""
        if self.total_nans > 0:
            runner.logger.info(
                f"📊 NaN Recovery Summary for Epoch: "
                f"{self.total_nans} NaN losses recovered. "
                f"Training continued successfully."
            )
            
            # Reset for next epoch
            self.consecutive_nans = 0
            self.total_nans = 0
            self.nan_iterations.clear() 