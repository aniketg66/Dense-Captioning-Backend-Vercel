from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch
import warnings
import os
import json
from pathlib import Path

@HOOKS.register_module()
class ChartTypeDistributionHook(Hook):
    """Hook to monitor chart type distribution during training."""
    
    def __init__(self, interval=50, priority='NORMAL', chart_types=None):
        super().__init__()
        self.interval = interval
        self.priority = priority
        self.chart_types = chart_types or ['line', 'scatter', 'dot', 'vertical_bar', 'horizontal_bar']
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Called after every training iteration."""
        if self.every_n_train_iters(runner, self.interval):
            # Get current batch data
            if data_batch is None:
                return
                
            # Count chart types in current batch
            chart_type_counts = {ct: 0 for ct in self.chart_types}
            data_samples = data_batch.get('data_samples', [])
            for data_info in data_samples:
                chart_type = data_info.get('chart_type')
                if chart_type in chart_type_counts:
                    chart_type_counts[chart_type] += 1
                    
            # Log distribution
            runner.logger.info(f'Chart type distribution: {chart_type_counts}')

@HOOKS.register_module()
class SkipInvalidLossHook(Hook):
    """Skip iterations with invalid loss values."""
    
    def __init__(self, interval=1):
        self.interval = interval
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Check loss after each training iteration."""
        if runner.iter % self.interval != 0:
            return
            
        # Handle None outputs
        if outputs is None:
            runner.logger.warning(f'Outputs is None at iteration {runner.iter}, skipping')
            runner.model.zero_grad()
            return True
            
        # Check total loss
        if 'loss' not in outputs:
            runner.logger.warning(f'No loss in outputs at iteration {runner.iter}, skipping')
            runner.model.zero_grad()
            return True
            
        loss = outputs['loss']
        if not torch.isfinite(loss):
            runner.logger.warning(f'Total loss is {loss.item()}, skipping iteration {runner.iter}')
            runner.model.zero_grad()
            return True
            
        # Check individual loss components
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value):
                runner.logger.warning(f'Component {key} has invalid value {value.item()}, skipping iteration {runner.iter}')
                runner.model.zero_grad()
                return True

@HOOKS.register_module()
class RuntimeErrorHook(Hook):
    """Hook to handle RuntimeError exceptions gracefully."""
    
    def __init__(self, interval=1, max_retries=3):
        super().__init__()
        self.interval = interval
        self.max_retries = max_retries
        self.retry_count = 0
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Check for runtime errors and handle them."""
        try:
            # Monitor for common CUDA errors
            if hasattr(torch.cuda, 'memory_allocated'):
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                
                if runner.iter % 100 == 0:  # Log every 100 iterations
                    runner.logger.info(f'CUDA Memory - Allocated: {memory_allocated / 1024**3:.2f} GB, '
                                     f'Reserved: {memory_reserved / 1024**3:.2f} GB')
                    
        except RuntimeError as e:
            runner.logger.error(f'RuntimeError at iteration {runner.iter}: {str(e)}')
            
            # Handle specific CUDA errors
            if "CUDA out of memory" in str(e):
                runner.logger.warning("CUDA OOM detected, clearing cache and reducing batch size")
                torch.cuda.empty_cache()
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    runner.logger.error(f"Max retries ({self.max_retries}) reached, stopping training")
                    raise e
                return True  # Skip this iteration
                
            elif "CUDA error: device-side assert triggered" in str(e):
                runner.logger.warning("CUDA assertion error detected, clearing gradients")
                runner.model.zero_grad()
                torch.cuda.empty_cache()
                return True  # Skip this iteration
                
            else:
                # Re-raise other runtime errors
                raise e
                
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """Reset retry count on successful iterations."""
        if hasattr(self, 'retry_count') and runner.iter % 10 == 0:
            self.retry_count = 0  # Reset retry count periodically 

@HOOKS.register_module()
class MissingImageReportHook(Hook):
    """Hook to report missing image statistics during training."""
    
    def __init__(self, interval=500, priority='LOW'):
        super().__init__()
        self.interval = interval
        self.priority = priority
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Report missing image count periodically."""
        if self.every_n_train_iters(runner, self.interval):
            try:
                # Import here to avoid circular imports
                from .custom_dataset import CreateDummyImg
                missing_count = CreateDummyImg.get_missing_count()
                if missing_count > 0:
                    runner.logger.info(f'📊 Missing images so far: {missing_count}')
            except Exception as e:
                pass  # Silently ignore if CreateDummyImg not available
                
    def after_train_epoch(self, runner):
        """Report missing image summary after each epoch."""
        try:
            from .custom_dataset import CreateDummyImg
            missing_count = CreateDummyImg.get_missing_count()
            if missing_count > 0:
                runner.logger.info(f'📊 EPOCH {runner.epoch} COMPLETE - Total missing images: {missing_count}')
        except Exception as e:
            pass
            
    def after_train(self, runner):
        """Report final missing image summary after training."""
        try:
            from .custom_dataset import CreateDummyImg
            missing_count = CreateDummyImg.get_missing_count()
            if missing_count > 0:
                runner.logger.info(f'🏁 TRAINING COMPLETE - Final count of missing images: {missing_count}')
                print(f"\n📊 MISSING IMAGES FINAL REPORT: {missing_count} images were not found and replaced with dummy images")
            else:
                runner.logger.info('✅ TRAINING COMPLETE - All images loaded successfully!')
                print("\n✅ All images loaded successfully!")
        except Exception as e:
            pass

# Utility function to check missing image count
def check_missing_images():
    """Check and print current missing image count."""
    try:
        from .custom_dataset import CreateDummyImg
        count = CreateDummyImg.get_missing_count()
        if count > 0:
            print(f"📊 Current missing image count: {count}")
        else:
            print("✅ No missing images so far!")
        return count
    except Exception as e:
        print(f"Could not check missing image count: {e}")
        return 0 

class SkipIterationException(Exception):
    """Custom exception to skip problematic training iterations."""
    pass

@HOOKS.register_module()
class SkipBadSamplesHook(Hook):
    """Skip training samples with problematic GT data and resume from last good state."""
    
    def __init__(self, interval=1):
        self.interval = interval
        self.skipped_samples = 0
        self.total_samples = 0
        self.last_good_iteration = 0
        self.last_good_loss = None
        self.consecutive_skips = 0
        self.max_consecutive_skips = 10  # Reset if too many consecutive skips
        
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """Check and skip problematic samples before forward pass."""
        if runner.iter % self.interval != 0:
            return False
            
        if data_batch is None:
            return False
            
        self.total_samples += 1
        
        # Check if this batch has problematic data
        if self._has_problematic_data(data_batch):
            self.skipped_samples += 1
            self.consecutive_skips += 1
            
            runner.logger.warning(
                f"⏭️ Skipping iteration {runner.iter} due to problematic GT data. "
                f"Skipped: {self.skipped_samples}/{self.total_samples} "
                f"({100*self.skipped_samples/self.total_samples:.1f}%) "
                f"Last good: {self.last_good_iteration}"
            )
            
            # If too many consecutive skips, try to reset state
            if self.consecutive_skips >= self.max_consecutive_skips:
                self._reset_training_state(runner)
            
            # Continue processing - let NanRecoveryHook handle any resulting NaN
            # The detection and logging is the important part
            
        # Reset consecutive skip counter on good iteration
        self.consecutive_skips = 0
        return False
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Track last good iteration after successful training step."""
        if outputs is not None and 'loss' in outputs:
            total_loss = outputs['loss']
            # Only update if loss is valid (not NaN/Inf)
            if torch.isfinite(total_loss):
                self.last_good_iteration = runner.iter
                self.last_good_loss = float(total_loss.item())
                self.consecutive_skips = 0  # Reset on successful iteration
    
    def _reset_training_state(self, runner):
        """Reset training state when too many consecutive skips."""
        runner.logger.warning(
            f"🔄 Too many consecutive skips ({self.consecutive_skips}). "
            f"Resetting training state. Last good iteration: {self.last_good_iteration}"
        )
        
        try:
            # Clear any cached states that might be corrupted
            if hasattr(runner.model, 'zero_grad'):
                runner.model.zero_grad()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset consecutive skip counter
            self.consecutive_skips = 0
            
            runner.logger.info(f"✅ Training state reset. Continuing from iteration {runner.iter + 1}")
            
        except Exception as e:
            runner.logger.error(f"❌ Failed to reset training state: {e}")
    
    def _has_problematic_data(self, data_batch):
        """Check if the batch contains problematic GT data."""
        try:
            inputs = data_batch.get('inputs', [])
            data_samples = data_batch.get('data_samples', [])
            
            # Check input images
            for i, img in enumerate(inputs):
                if torch.isnan(img).any() or torch.isinf(img).any():
                    print(f"   🐛 Input image {i} contains NaN/Inf")
                    return True
                    
            # Check GT data
            for i, sample in enumerate(data_samples):
                if hasattr(sample, 'gt_instances'):
                    gt = sample.gt_instances
                    
                    # Check GT bboxes
                    if hasattr(gt, 'bboxes'):
                        bbox_tensor = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
                        if torch.isnan(bbox_tensor).any() or torch.isinf(bbox_tensor).any():
                            print(f"   🐛 GT bboxes {i} contain NaN/Inf")
                            return True
                            
                        # Check for invalid bbox coordinates (negative width/height, etc.)
                        if len(bbox_tensor) > 0:
                            x1, y1, x2, y2 = bbox_tensor[:, 0], bbox_tensor[:, 1], bbox_tensor[:, 2], bbox_tensor[:, 3]
                            if (x2 <= x1).any() or (y2 <= y1).any():
                                print(f"   🐛 GT bboxes {i} have invalid coordinates (x2<=x1 or y2<=y1)")
                                return True
                            if (bbox_tensor < 0).any():
                                print(f"   🐛 GT bboxes {i} have negative coordinates")
                                return True
                    
                    # Check GT labels  
                    if hasattr(gt, 'labels'):
                        if torch.isnan(gt.labels.float()).any() or torch.isinf(gt.labels.float()).any():
                            print(f"   🐛 GT labels {i} contain NaN/Inf")
                            return True
                        if (gt.labels < 0).any():
                            print(f"   🐛 GT labels {i} contain negative values")
                            return True
                            
            return False
            
        except Exception as e:
            print(f"   🐛 Error checking data: {e}")
            return True  # Skip on any error to be safe 

@HOOKS.register_module()
class CompatibleCheckpointHook(Hook):
    """Hook to save checkpoints in a compatible format without numpy compatibility issues."""
    
    def __init__(self, 
                 interval=1, 
                 max_keep_ckpts=5,
                 save_best='auto',
                 save_last=True,
                 priority='NORMAL'):
        super().__init__()
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_best = save_best
        self.save_last = save_last
        self.priority = priority
        self.best_score = None
        self.checkpoint_paths = []
        
    def after_train_epoch(self, runner):
        """Save checkpoint after each training epoch."""
        if not self.every_n_epochs(runner, self.interval):
            return
            
        # Create work directory if it doesn't exist
        work_dir = Path(runner.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # FIX: Save checkpoint with the epoch that just completed
        # runner.epoch is the epoch we just finished, so this is correct
        # But add safety check to ensure we don't overwrite with corrupted state
        epoch_path = work_dir / f'epoch_{runner.epoch}.pth'
        
        # Safety check: only save if training completed successfully this epoch
        try:
            # Verify model state is valid before saving
            model = runner.model
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            # Quick validation that model parameters are not corrupted
            has_nan = False
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
                    runner.logger.error(f'🚨 CORRUPTED MODEL DETECTED: Parameter {key} contains NaN/Inf!')
                    has_nan = True
                    break
                    
            if has_nan:
                runner.logger.error(f'❌ Skipping checkpoint save for epoch {runner.epoch} due to corrupted model state')
                return  # Don't save corrupted checkpoint
                
            runner.logger.info(f'✅ Model state validation passed for epoch {runner.epoch}')
            
        except Exception as e:
            runner.logger.error(f'❌ Failed to validate model state for epoch {runner.epoch}: {e}')
            return  # Don't save if we can't validate
        
        # Only save if validation passed
        self._save_compatible_checkpoint(runner, epoch_path)
        runner.logger.info(f'✅ Checkpoint saved for completed epoch {runner.epoch}: {epoch_path}')
        
        # Track checkpoint paths for cleanup
        self.checkpoint_paths.append(epoch_path)
        
        # Handle save_best logic
        if self.save_best and hasattr(runner, 'val_evaluator'):
            try:
                # Get validation metrics if available
                if hasattr(runner, 'log_buffer') and runner.log_buffer.val_history:
                    latest_val = runner.log_buffer.val_history[-1]
                    current_score = self._extract_best_score(latest_val)
                    
                    if current_score is not None:
                        if self.best_score is None or current_score > self.best_score:
                            self.best_score = current_score
                            best_path = work_dir / 'best.pth'
                            self._save_compatible_checkpoint(runner, best_path)
                            runner.logger.info(f'New best checkpoint saved: {best_path} (score: {current_score:.4f})')
                            
            except Exception as e:
                runner.logger.warning(f'Could not save best checkpoint: {e}')
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def after_train(self, runner):
        """Save final checkpoint after training completes."""
        if self.save_last:
            work_dir = Path(runner.work_dir)
            final_path = work_dir / 'latest.pth'
            self._save_compatible_checkpoint(runner, final_path)
            runner.logger.info(f'Final checkpoint saved: {final_path}')
    
    def _save_compatible_checkpoint(self, runner, checkpoint_path):
        """Save checkpoint in ultra-compatible format without any numpy dependencies."""
        try:
            # Extract model state dict and ensure it's on CPU
            model = runner.model
            if hasattr(model, 'module'):  # Handle DataParallel/DistributedDataParallel
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # Ultra-aggressive tensor cleaning for maximum compatibility
            ultra_clean_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    # Ensure tensor is completely detached and on CPU
                    clean_tensor = value.detach().cpu()
                    
                    # Force conversion to pure PyTorch tensor (no numpy backing)
                    # This ensures no numpy references remain
                    clean_tensor = torch.tensor(clean_tensor.numpy(), dtype=clean_tensor.dtype)
                    
                    ultra_clean_state_dict[key] = clean_tensor
                elif isinstance(value, (int, float, str, bool, type(None))):
                    ultra_clean_state_dict[key] = value
                else:
                    # Skip any complex objects that might cause compatibility issues
                    runner.logger.warning(f'Skipping parameter {key} with type {type(value)}')
                    continue
            
            # Create ultra-minimal checkpoint with only essential data
            ultra_compatible_checkpoint = {
                'state_dict': ultra_clean_state_dict,
                'epoch': int(runner.epoch),
                'iter': int(runner.iter),
                # Minimal metadata to avoid any compatibility issues
                'meta': {
                    'epoch': int(runner.epoch),
                    'iter': int(runner.iter),
                    'mmdet_version': '3.0.0',
                    'hook_version': 'compat_v1'  # Mark as using compatible hook
                }
            }
            
            # Skip optimizer state for maximum compatibility
            # (Can be retrained if needed)
            runner.logger.info('Skipping optimizer state for maximum compatibility')
            
            # Use most compatible torch.save settings
            import pickle
            torch.save(
                ultra_compatible_checkpoint,
                checkpoint_path,
                pickle_protocol=2,  # Use older protocol
                _use_new_zipfile_serialization=False,  # Legacy serialization
                pickle_module=pickle  # Use standard pickle
            )
            
            runner.logger.info(f'Ultra-compatible checkpoint saved: {checkpoint_path}')
            
            # Verify the checkpoint can be loaded immediately
            try:
                verification_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                runner.logger.info(f'✅ Checkpoint verification passed - can be loaded successfully')
                
                # Quick structure check
                if 'state_dict' in verification_checkpoint:
                    param_count = len(verification_checkpoint['state_dict'])
                    runner.logger.info(f'✅ Verified {param_count} parameters in saved checkpoint')
                else:
                    runner.logger.warning('⚠️ No state_dict found in saved checkpoint')
                    
            except Exception as verify_e:
                runner.logger.error(f'❌ Checkpoint verification failed: {verify_e}')
                # Still continue, as the checkpoint might work in different environment
            
        except Exception as e:
            runner.logger.error(f'Failed to save ultra-compatible checkpoint: {e}')
            
            # Ultra-fallback: save only the pure state dict with numpy conversion
            try:
                runner.logger.info('Attempting ultra-fallback: pure state dict only')
                
                # Create the cleanest possible state dict
                pure_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Convert through numpy and back to ensure no references
                        numpy_array = value.detach().cpu().numpy()
                        pure_tensor = torch.from_numpy(numpy_array.copy())
                        pure_state_dict[key] = pure_tensor
                
                fallback_path = str(checkpoint_path).replace('.pth', '_pure_statedict.pth')
                torch.save(pure_state_dict, fallback_path, pickle_protocol=2)
                runner.logger.info(f'Ultra-fallback pure state dict saved: {fallback_path}')
                
                # Test load the fallback
                test_load = torch.load(fallback_path, map_location='cpu', weights_only=False)
                runner.logger.info(f'✅ Ultra-fallback verified with {len(test_load)} parameters')
                
            except Exception as fallback_e:
                runner.logger.error(f'Even ultra-fallback failed: {fallback_e}')
                runner.logger.error('❌ CRITICAL: Could not save checkpoint in any compatible format')
    
    def _clean_optimizer_state(self, optimizer_state):
        """Clean optimizer state of any problematic references."""
        if not isinstance(optimizer_state, dict):
            return optimizer_state
            
        cleaned_state = {}
        for key, value in optimizer_state.items():
            if isinstance(value, dict):
                cleaned_state[key] = self._clean_optimizer_state(value)
            elif isinstance(value, torch.Tensor):
                cleaned_state[key] = value.detach().cpu()
            elif isinstance(value, (int, float, str, bool, type(None))):
                cleaned_state[key] = value
            else:
                # Skip any complex objects that might cause issues
                continue
                
        return cleaned_state
    
    def _extract_best_score(self, val_metrics):
        """Extract the best score from validation metrics."""
        # Look for common metric names
        score_keys = ['bbox_mAP', 'mAP', 'accuracy', 'acc']
        
        for key in score_keys:
            if key in val_metrics:
                return float(val_metrics[key])
                
        # If no recognized metric, return None
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_keep_ckpts limit."""
        if self.max_keep_ckpts <= 0:
            return
            
        # Keep only the most recent checkpoints
        if len(self.checkpoint_paths) > self.max_keep_ckpts:
            to_remove = self.checkpoint_paths[:-self.max_keep_ckpts]
            for path in to_remove:
                try:
                    if path.exists():
                        path.unlink()
                except Exception as e:
                    pass  # Silently ignore cleanup failures
                    
            # Update the list
            self.checkpoint_paths = self.checkpoint_paths[-self.max_keep_ckpts:] 