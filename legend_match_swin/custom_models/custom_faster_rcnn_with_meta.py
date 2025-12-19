# custom_faster_rcnn_with_meta.py - Faster R-CNN with coordinate handling for chart data
import torch
import torch.nn as nn
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.registry import MODELS


@MODELS.register_module()
class CustomFasterRCNNWithMeta(FasterRCNN):
    """Faster R-CNN with coordinate standardization for chart detection."""
    
    def __init__(self,
                 *args,
                 coordinate_standardization=None,
                 data_points_count_head=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store coordinate standardization settings
        self.coord_std = coordinate_standardization or {}
        
        # Initialize data points count head
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
        
        print(f"🎯 CustomFasterRCNNWithMeta initialized with coordinate handling:")
        print(f"   • Enabled: {self.coord_std.get('enabled', False)}")
        print(f"   • Origin: {self.coord_std.get('origin', 'top_left')}")
        print(f"   • Normalize: {self.coord_std.get('normalize', False)}")
        print(f"   • Data points count prediction: Enabled")
    
    def transform_coordinates(self, coords, img_shape, plot_bb=None, axes_info=None):
        """Transform coordinates based on standardization settings."""
        if not self.coord_std.get('enabled', False):
            return coords
            
        # Get image dimensions
        img_height, img_width = img_shape[-2:]
        
        # Convert to tensor if not already
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, device=img_shape.device if hasattr(img_shape, 'device') else 'cpu')
            
        # Ensure coords is 2D
        if coords.dim() == 1:
            coords = coords.view(-1, 2)
            
        # Normalize coordinates if needed
        if self.coord_std.get('normalize', True):
            coords = coords / torch.tensor([img_width, img_height], device=coords.device)
            
        # Handle bottom-left to top-left origin conversion
        if self.coord_std.get('origin', 'bottom_left') == 'bottom_left':
            # Flip y-coordinates to convert from bottom-left to top-left origin
            coords[:, 1] = 1.0 - coords[:, 1]
            
        # Convert back to pixel coordinates
        if self.coord_std.get('normalize', True):
            coords = coords * torch.tensor([img_width, img_height], device=coords.device)
            
        return coords
    
    def forward_train(self,
                     img,
                     img_metas,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     **kwargs):
        """Forward function during training with coordinate transformation."""
        
        # Transform ground truth bboxes if coordinate standardization is enabled
        if self.coord_std.get('enabled', False) and gt_bboxes is not None:
            transformed_gt_bboxes = []
            for i, bboxes in enumerate(gt_bboxes):
                if len(bboxes) > 0:
                    # Convert bbox format for transformation
                    # MMDet uses [x1, y1, x2, y2] format
                    centers = torch.stack([
                        (bboxes[:, 0] + bboxes[:, 2]) / 2,  # center_x
                        (bboxes[:, 1] + bboxes[:, 3]) / 2   # center_y
                    ], dim=1)
                    
                    # Transform centers
                    img_shape = img.shape if hasattr(img, 'shape') else (img_metas[i]['img_shape'][0], img_metas[i]['img_shape'][1])
                    transformed_centers = self.transform_coordinates(
                        centers, img_shape,
                        plot_bb=img_metas[i].get('plot_bb'),
                        axes_info=img_metas[i].get('axes_info')
                    )
                    
                    # Reconstruct bboxes with transformed centers
                    widths = bboxes[:, 2] - bboxes[:, 0]
                    heights = bboxes[:, 3] - bboxes[:, 1]
                    
                    transformed_bboxes = torch.stack([
                        transformed_centers[:, 0] - widths / 2,   # x1
                        transformed_centers[:, 1] - heights / 2,  # y1
                        transformed_centers[:, 0] + widths / 2,   # x2
                        transformed_centers[:, 1] + heights / 2   # y2
                    ], dim=1)
                    
                    transformed_gt_bboxes.append(transformed_bboxes)
                else:
                    transformed_gt_bboxes.append(bboxes)
            
            gt_bboxes = transformed_gt_bboxes
        
        # Call parent forward_train with transformed coordinates to get losses
        losses = super().forward_train(
            img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs)
        
        # Extract features for data point count prediction
        x = self.extract_feat(img)
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
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Simple test function with coordinate inverse transformation."""
        # Get predictions from parent
        results = super().simple_test(img, img_metas, proposals, rescale)
        
        # Extract features for data point count prediction
        x = self.extract_feat(img)
        global_feat = x[-1].mean(dim=[2, 3])  # Global average pooling
        
        # Predict data point counts
        pred_data_point_counts = self.data_points_count_head(global_feat).squeeze(-1)
        
        # Add data point count predictions to results
        if results is not None:
            for i, result in enumerate(results):
                if hasattr(result, 'pred_instances'):
                    result.pred_instances.predicted_data_points = pred_data_point_counts[i].item()
                elif hasattr(result, 'bboxes'):
                    # For older MMDet versions, add as additional attribute
                    result.predicted_data_points = pred_data_point_counts[i].item()
        
        # Inverse transform predictions if coordinate standardization is enabled
        if self.coord_std.get('enabled', False) and results is not None:
            # Note: For simplicity, we're not doing inverse transform in test
            # The coordinate system should be consistent during training
            pass
            
        return results 