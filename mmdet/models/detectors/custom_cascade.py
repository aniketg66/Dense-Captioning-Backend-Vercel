import os
print('LOADED FILE:', __file__)
with open(__file__, 'r') as f:
    for i in range(20):
        print(f.readline().rstrip())

from mmdet.models.detectors import CascadeRCNN
from mmdet.registry import MODELS
import torch
from mmdet.structures import DetDataSample
import torch.nn as nn

@MODELS.register_module(name='CustomCascadeWithMeta')
class CustomCascadeWithMeta(CascadeRCNN):
    """Custom Cascade R-CNN with metadata prediction heads."""
    
    def __init__(self,
                 *args,
                 chart_cls_head=None,
                 plot_reg_head=None,
                 axes_info_head=None,
                 data_series_head=None,
                 coordinate_standardization=None,
                 data_series_config=None,
                 axis_aware_feature=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store configuration parameters
        self.coord_std = coordinate_standardization or {}
        self.data_series_cfg = data_series_config or {}
        self.axis_feature_cfg = axis_aware_feature or {}
        
        # Initialize metadata prediction heads
        if chart_cls_head is not None:
            self.chart_cls_head = MODELS.build(chart_cls_head)
        if plot_reg_head is not None:
            self.plot_reg_head = MODELS.build(plot_reg_head)
        if axes_info_head is not None:
            self.axes_info_head = MODELS.build(axes_info_head)
        if data_series_head is not None:
            self.data_series_head = MODELS.build(data_series_head)
            
        # Initialize axis-aware feature extraction if enabled
        if self.axis_feature_cfg.get('enabled', False):
            self.axis_feature_extractor = nn.Sequential(
                nn.Conv2d(self.axis_feature_cfg['input_channels'], 
                         self.axis_feature_cfg['feature_channels'], 1),
                nn.BatchNorm2d(self.axis_feature_cfg['feature_channels']),
                nn.ReLU(inplace=True)
            )
            
            if self.axis_feature_cfg.get('attention_type') == 'spatial':
                self.axis_attention = nn.Sequential(
                    nn.Conv2d(self.axis_feature_cfg['feature_channels'], 1, 1),
                    nn.Sigmoid()
                )
    
    def transform_coordinates(self, coords, img_shape, plot_bb=None, axes_info=None):
        """Transform coordinates based on standardization settings."""
        if not self.coord_std.get('enabled', False):
            return coords
            
        # Get image dimensions
        img_height, img_width = img_shape[-2:]
        
        # Convert to tensor if not already
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, device=img_shape.device)
            
        # Ensure coords is 2D
        if coords.dim() == 1:
            coords = coords.view(-1, 2)
            
        # Normalize coordinates if needed
        if self.coord_std.get('normalize', True):
            coords = coords / torch.tensor([img_width, img_height], device=coords.device)
            
        # Ensure bottom-left to top-right orientation
        if self.coord_std.get('origin', 'bottom_left') == 'bottom_left':
            # Flip y-coordinates to ensure bottom-left origin
            coords[:, 1] = 1.0 - coords[:, 1]
            
        # Transform relative to plot area if available
        if plot_bb is not None and self.coord_std.get('relative_to_plot', True):
            if not isinstance(plot_bb, torch.Tensor):
                plot_bb = torch.tensor(plot_bb, device=coords.device)
            plot_bb = plot_bb.view(-1, 4)  # x1, y1, x2, y2
            
            # Normalize plot coordinates
            plot_bb = plot_bb / torch.tensor([img_width, img_height, img_width, img_height], 
                                           device=plot_bb.device)
            
            # Transform coordinates relative to plot area
            coords[:, 0] = (coords[:, 0] - plot_bb[0, 0]) / (plot_bb[0, 2] - plot_bb[0, 0])
            coords[:, 1] = (coords[:, 1] - plot_bb[0, 1]) / (plot_bb[0, 3] - plot_bb[0, 1])
            
            # Scale coordinates according to axis ranges if available
            if axes_info is not None and self.coord_std.get('scale_to_axis', True):
                if not isinstance(axes_info, torch.Tensor):
                    axes_info = torch.tensor(axes_info, device=coords.device)
                axes_info = axes_info.view(-1, 8)  # x_min, x_max, y_min, y_max, x_type, y_type, x_tick_type, y_tick_type
                
                # Scale x coordinates
                if axes_info[0, 4] < 0.5:  # Numerical x-axis
                    x_range = axes_info[0, 1] - axes_info[0, 0]
                    if x_range > 0:  # Avoid division by zero
                        coords[:, 0] = coords[:, 0] * x_range + axes_info[0, 0]
                
                # Scale y coordinates
                if axes_info[0, 5] < 0.5:  # Numerical y-axis
                    y_range = axes_info[0, 3] - axes_info[0, 2]
                    if y_range > 0:  # Avoid division by zero
                        coords[:, 1] = coords[:, 1] * y_range + axes_info[0, 2]
            
        return coords
    
    def inverse_transform_coordinates(self, coords, img_shape, plot_bb=None, axes_info=None):
        """Inverse transform coordinates back to image space."""
        if not self.coord_std.get('enabled', False):
            return coords
            
        # Get image dimensions
        img_height, img_width = img_shape[-2:]
        
        # Convert to tensor if not already
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, device=img_shape.device)
            
        # Ensure coords is 2D
        if coords.dim() == 1:
            coords = coords.view(-1, 2)
            
        # Transform from plot-relative coordinates if needed
        if plot_bb is not None and self.coord_std.get('relative_to_plot', True):
            if not isinstance(plot_bb, torch.Tensor):
                plot_bb = torch.tensor(plot_bb, device=coords.device)
            plot_bb = plot_bb.view(-1, 4)  # x1, y1, x2, y2
            
            # Scale coordinates according to axis ranges if available
            if axes_info is not None and self.coord_std.get('scale_to_axis', True):
                if not isinstance(axes_info, torch.Tensor):
                    axes_info = torch.tensor(axes_info, device=coords.device)
                axes_info = axes_info.view(-1, 8)  # x_min, x_max, y_min, y_max, x_type, y_type, x_tick_type, y_tick_type
                
                # Scale x coordinates
                if axes_info[0, 4] < 0.5:  # Numerical x-axis
                    x_range = axes_info[0, 1] - axes_info[0, 0]
                    if x_range > 0:  # Avoid division by zero
                        coords[:, 0] = (coords[:, 0] - axes_info[0, 0]) / x_range
                
                # Scale y coordinates
                if axes_info[0, 5] < 0.5:  # Numerical y-axis
                    y_range = axes_info[0, 3] - axes_info[0, 2]
                    if y_range > 0:  # Avoid division by zero
                        coords[:, 1] = (coords[:, 1] - axes_info[0, 2]) / y_range
            
            # Transform coordinates from plot-relative to image space
            coords[:, 0] = coords[:, 0] * (plot_bb[0, 2] - plot_bb[0, 0]) + plot_bb[0, 0]
            coords[:, 1] = coords[:, 1] * (plot_bb[0, 3] - plot_bb[0, 1]) + plot_bb[0, 1]
            
        # Denormalize coordinates if needed
        if self.coord_std.get('normalize', True):
            coords = coords * torch.tensor([img_width, img_height], device=coords.device)
            
        # Ensure bottom-left to top-right orientation
        if self.coord_std.get('origin', 'bottom_left') == 'bottom_left':
            # Flip y-coordinates back to image space
            coords[:, 1] = img_height - coords[:, 1]
            
        return coords
    
    def extract_feat(self, img):
        """Extract features from images."""
        # Get the normal FPN features (backbone + neck, all out_channels=256)
        feats = super().extract_feat(img)
        
        # Apply axis-aware feature extraction if enabled
        if self.axis_feature_cfg.get('enabled', False):
            # Run on feats[-1] which has 256 channels from FPN
            axis_feat = self.axis_feature_extractor(feats[-1])
            if self.axis_feature_cfg.get('attention_type') == 'spatial':
                att = self.axis_attention(axis_feat)
                axis_feat = axis_feat * att
            
            # Sum fusion keeps the channel count at 256
            feats = list(feats)
            feats[-1] = feats[-1] + axis_feat
            feats = tuple(feats)
        
        return feats
    
    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
    
    def forward_train(self,
                     img,
                     img_metas,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     gt_masks=None,
                     proposals=None,
                     **kwargs):
        """Forward function during training."""
        x = self.extract_feat(img)
        
        # Get metadata from kwargs
        chart_type = kwargs.get('chart_type', None)
        plot_bb = kwargs.get('plot_bb', None)
        axes_info = kwargs.get('axes_info', None)
        data_series = kwargs.get('data_series', None)
        
        # Transform coordinates if needed
        if plot_bb is not None:
            plot_bb = self.transform_coordinates(plot_bb, img.shape)
        if data_series is not None:
            # Ensure data series is properly formatted
            if isinstance(data_series, list):
                # Convert list of points to tensor
                data_series = torch.tensor(data_series, device=img.device)
            elif isinstance(data_series, dict):
                # Convert dict with x,y keys to tensor
                data_series = torch.stack([
                    torch.tensor(data_series['x'], device=img.device),
                    torch.tensor(data_series['y'], device=img.device)
                ], dim=1)
            
            # Transform data series coordinates
            data_series = self.transform_coordinates(data_series, img.shape, plot_bb, axes_info)
            
            # Validate data series coordinates
            if self.data_series_cfg.get('validate_coordinates', True):
                # Check if coordinates are within plot area
                if plot_bb is not None:
                    plot_bb = plot_bb.view(-1, 4)
                    valid_x = (data_series[:, 0] >= plot_bb[0, 0]) & (data_series[:, 0] <= plot_bb[0, 2])
                    valid_y = (data_series[:, 1] >= plot_bb[0, 1]) & (data_series[:, 1] <= plot_bb[0, 3])
                    valid_coords = valid_x & valid_y
                    if not valid_coords.all():
                        print(f"Warning: Some data series points are outside plot area")
                
                # Check if coordinates follow expected pattern
                if self.data_series_cfg.get('check_pattern', True):
                    # Sort by x-coordinate
                    sorted_indices = torch.argsort(data_series[:, 0])
                    data_series = data_series[sorted_indices]
                    
                    # Check for duplicate x-coordinates
                    x_diff = torch.diff(data_series[:, 0])
                    if (x_diff == 0).any():
                        print(f"Warning: Duplicate x-coordinates found in data series")
                    
                    # Check for monotonicity if specified
                    if self.data_series_cfg.get('check_monotonicity', False):
                        y_diff = torch.diff(data_series[:, 1])
                        if not (y_diff >= 0).all() and not (y_diff <= 0).all():
                            print(f"Warning: Data series is not monotonic")
        
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        
        # Cascade forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                               gt_bboxes, gt_labels,
                                               gt_bboxes_ignore, gt_masks,
                                               **kwargs)
        losses.update(roi_losses)
        
        # Metadata prediction losses
        if hasattr(self, 'chart_cls_head') and chart_type is not None:
            # Chart type classification loss
            pred_chart_type = self.chart_cls_head(x)
            if isinstance(chart_type, torch.Tensor):
                chart_type = chart_type.long()
            chart_cls_loss = torch.nn.functional.cross_entropy(
                pred_chart_type, chart_type, reduction='mean')
            losses['loss_chart_type'] = chart_cls_loss
            
        if hasattr(self, 'plot_reg_head') and plot_bb is not None:
            # Plot area regression loss
            pred_plot_bb = self.plot_reg_head(x)
            if isinstance(plot_bb, torch.Tensor):
                plot_bb = plot_bb.view(-1, 4)  # x1, y1, x2, y2
            
            # Calculate L1 loss for plot area coordinates
            plot_loss = torch.abs(pred_plot_bb - plot_bb)
            
            # Apply weights if specified
            if hasattr(self.plot_reg_head, 'loss_weight'):
                plot_loss = plot_loss * self.plot_reg_head.loss_weight
            
            losses['loss_plot_bb'] = plot_loss.mean()
            
        if hasattr(self, 'axes_info_head') and axes_info is not None:
            # Axes information regression loss
            pred_axes_info = self.axes_info_head(x)
            if isinstance(axes_info, torch.Tensor):
                axes_info = axes_info.view(-1, 8)  # x_min, x_max, y_min, y_max, x_type, y_type, x_tick_type, y_tick_type
            
            # Calculate losses for different components
            # Range losses (L1)
            range_loss = torch.abs(pred_axes_info[:, :4] - axes_info[:, :4])
            
            # Type classification losses (Cross Entropy)
            x_type_loss = torch.nn.functional.cross_entropy(
                pred_axes_info[:, 4:6], axes_info[:, 4].long(), reduction='none')
            y_type_loss = torch.nn.functional.cross_entropy(
                pred_axes_info[:, 6:8], axes_info[:, 5].long(), reduction='none')
            
            # Combine losses
            axes_loss = range_loss.mean() + (x_type_loss + y_type_loss).mean()
            
            # Apply weights if specified
            if hasattr(self.axes_info_head, 'loss_weight'):
                axes_loss = axes_loss * self.axes_info_head.loss_weight
            
            losses['loss_axes_info'] = axes_loss
            
        if hasattr(self, 'data_series_head') and data_series is not None:
            # Get predicted data series coordinates
            pred_series = self.data_series_head(x)
            
            # Ensure predictions are properly formatted
            if isinstance(pred_series, torch.Tensor):
                if pred_series.dim() == 1:
                    pred_series = pred_series.view(-1, 2)
                elif pred_series.dim() > 2:
                    pred_series = pred_series.view(-1, 2)
            
            # Calculate coordinate-wise losses
            if self.data_series_cfg.get('loss_type', 'l1') == 'l1':
                # L1 loss for each coordinate
                coord_loss = torch.abs(pred_series - data_series)
            elif self.data_series_cfg.get('loss_type', 'l1') == 'l2':
                # L2 loss for each coordinate
                coord_loss = (pred_series - data_series) ** 2
            else:
                # Default to L1 loss
                coord_loss = torch.abs(pred_series - data_series)
            
            # Apply coordinate weights if specified
            if self.data_series_cfg.get('coordinate_weights', None) is not None:
                weights = torch.tensor(self.data_series_cfg['coordinate_weights'], 
                                     device=coord_loss.device)
                coord_loss = coord_loss * weights
            
            # Calculate final loss
            if self.data_series_cfg.get('loss_reduction', 'mean') == 'mean':
                data_series_loss = {'loss_data_series': coord_loss.mean()}
            else:
                data_series_loss = {'loss_data_series': coord_loss.sum()}
            
            # Add individual coordinate losses for monitoring
            data_series_loss.update({
                'loss_data_series_x': coord_loss[:, 0].mean(),
                'loss_data_series_y': coord_loss[:, 1].mean()
            })
            
            losses.update(data_series_loss)
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        
        # Get RPN proposals
        if proposals is None:
            proposal_list = self.rpn_head.predict(x, img_metas)
        else:
            proposal_list = proposals
        
        # Get detection results using the standard MMDetection pipeline
        results = self.roi_head.predict(
            x, proposal_list, img_metas, rescale=rescale)
        
        # Add metadata predictions to the results
        for i, result in enumerate(results):
            # Add metadata predictions
            if hasattr(self, 'chart_cls_head'):
                chart_type = self.chart_cls_head(x[i:i+1])  # Use forward instead of simple_test
                chart_type = torch.argmax(chart_type, dim=1)  # Convert logits to class index
                result.set_metainfo({'chart_type': chart_type})
                
            if hasattr(self, 'plot_reg_head'):
                plot_bb = self.plot_reg_head(x[i:i+1])  # Use forward instead of simple_test
                # Transform plot coordinates back to image space
                plot_bb = self.inverse_transform_coordinates(plot_bb, img.shape)
                result.set_metainfo({'plot_bb': plot_bb})
                
            if hasattr(self, 'axes_info_head'):
                axes_info = self.axes_info_head(x[i:i+1])  # Use forward instead of simple_test
                result.set_metainfo({'axes_info': axes_info})
                
            if hasattr(self, 'data_series_head'):
                data_series = self.data_series_head(x[i:i+1])  # Use forward instead of simple_test
                # Transform data series coordinates back to image space
                data_series = self.inverse_transform_coordinates(data_series, img.shape, plot_bb, axes_info)
                result.set_metainfo({'data_series': data_series})
        
        return results 