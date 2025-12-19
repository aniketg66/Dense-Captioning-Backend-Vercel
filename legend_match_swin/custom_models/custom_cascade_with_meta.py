from mmdet.models.detectors import CascadeRCNN
from mmdet.registry import MODELS
import torch
from mmdet.structures import DetDataSample
import torch.nn as nn

@MODELS.register_module()
class CustomCascadeWithMeta(CascadeRCNN):
    """Custom Cascade R-CNN with metadata prediction heads."""
    
    def __init__(self,
                 *args,
                 chart_cls_head=None,
                 plot_reg_head=None,
                 axes_info_head=None,
                 data_series_head=None,
                 data_points_count_head=None,
                 coordinate_standardization=None,
                 data_series_config=None,
                 axis_aware_feature=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize metadata prediction heads
        if chart_cls_head is not None:
            self.chart_cls_head = MODELS.build(chart_cls_head)
        if plot_reg_head is not None:
            self.plot_reg_head = MODELS.build(plot_reg_head)
        if axes_info_head is not None:
            self.axes_info_head = MODELS.build(axes_info_head)
        if data_series_head is not None:
            self.data_series_head = MODELS.build(data_series_head)
        if data_points_count_head is not None:
            self.data_points_count_head = MODELS.build(data_points_count_head)
        else:
            # Default simple regression head for data point count
            self.data_points_count_head = nn.Sequential(
                nn.Linear(2048, 512),  # Assuming ResNet-50 backbone features
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1)  # Single output for count
            )
            
        # Store configurations
        self.coordinate_standardization = coordinate_standardization
        self.data_series_config = data_series_config
        self.axis_aware_feature = axis_aware_feature
        
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        """Forward function during training."""
        # Get base detector predictions
        x = self.extract_feat(img)
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
                ann_weight=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = kwargs.get('proposals', None)
            
        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                               gt_bboxes, gt_labels, **kwargs)
        losses.update(roi_losses)
        
        # Get global features for metadata prediction
        global_feat = x[-1].mean(dim=[2, 3])  # Global average pooling
        
        # Extract ground truth data point counts from img_metas
        gt_data_point_counts = []
        for img_meta in img_metas:
            count = img_meta.get('img_info', {}).get('num_data_points', 0)
            gt_data_point_counts.append(count)
        gt_data_point_counts = torch.tensor(gt_data_point_counts, dtype=torch.float32, device=global_feat.device)
        
        # Predict data point counts and compute loss
        pred_data_point_counts = self.data_points_count_head(global_feat).squeeze(-1)
        data_points_count_loss = nn.MSELoss()(pred_data_point_counts, gt_data_point_counts)
        losses['data_points_count_loss'] = data_points_count_loss
        
        # Use predicted data point count as additional feature for ROI head
        # Expand the global feature with data point count information
        normalized_counts = torch.sigmoid(pred_data_point_counts / 100.0)  # Normalize to 0-1 range
        enhanced_global_feat = torch.cat([global_feat, normalized_counts.unsqueeze(-1)], dim=-1)
        
        # Metadata prediction losses
        if hasattr(self, 'chart_cls_head'):
            chart_cls_loss = self.chart_cls_head(enhanced_global_feat)
            losses['chart_cls_loss'] = chart_cls_loss
            
        if hasattr(self, 'plot_reg_head'):
            plot_reg_loss = self.plot_reg_head(enhanced_global_feat)
            losses['plot_reg_loss'] = plot_reg_loss
            
        if hasattr(self, 'axes_info_head'):
            axes_info_loss = self.axes_info_head(enhanced_global_feat)
            losses['axes_info_loss'] = axes_info_loss
            
        if hasattr(self, 'data_series_head'):
            data_series_loss = self.data_series_head(enhanced_global_feat)
            losses['data_series_loss'] = data_series_loss
            
        return losses
        
    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg.rcnn, **kwargs)
        
        # Get global features for metadata prediction
        global_feat = x[-1].mean(dim=[2, 3])  # Global average pooling
        
        # Predict data point counts
        pred_data_point_counts = self.data_points_count_head(global_feat).squeeze(-1)
        
        # Use predicted data point count as additional feature
        normalized_counts = torch.sigmoid(pred_data_point_counts / 100.0)  # Normalize to 0-1 range
        enhanced_global_feat = torch.cat([global_feat, normalized_counts.unsqueeze(-1)], dim=-1)
        
        # Get metadata predictions
        results = []
        for i, (bboxes, labels) in enumerate(zip(det_bboxes, det_labels)):
            result = DetDataSample()
            result.bboxes = bboxes
            result.labels = labels
            
            # Add data point count prediction
            result.predicted_data_points = pred_data_point_counts[i].item()
            
            # Add metadata predictions using enhanced features
            if hasattr(self, 'chart_cls_head'):
                result.chart_type = self.chart_cls_head(enhanced_global_feat[i:i+1])
            if hasattr(self, 'plot_reg_head'):
                result.plot_bb = self.plot_reg_head(enhanced_global_feat[i:i+1])
            if hasattr(self, 'axes_info_head'):
                result.axes_info = self.axes_info_head(enhanced_global_feat[i:i+1])
            if hasattr(self, 'data_series_head'):
                result.data_series = self.data_series_head(enhanced_global_feat[i:i+1])
                
            results.append(result)
            
        return results 