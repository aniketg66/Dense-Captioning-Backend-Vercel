import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class FCHead(nn.Module):
    """Enhanced fully connected head for classification tasks with attention."""
    
    def __init__(self, in_channels, num_classes, loss=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads=8)
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, num_classes)
        self.loss = loss
        
    def forward(self, x):
        # Apply self-attention
        x = self.attention(x, x, x)[0]
        # Apply MLP
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@MODELS.register_module()
class RegHead(nn.Module):
    """Enhanced regression head for coordinate prediction with distance-based loss."""
    
    def __init__(self, in_channels, out_dims, max_points=None, loss=None, attention=False, use_axis_info=False):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_dims)
        self.max_points = max_points
        self.loss = loss
        self.attention = attention
        self.use_axis_info = use_axis_info
        
        if attention:
            self.attention_layer = nn.MultiheadAttention(in_channels, num_heads=8)
            
        # Add axis orientation detection
        if use_axis_info:
            self.axis_orientation = nn.Linear(in_channels, 2)  # 2 for x/y axis orientation
            
    def compute_distance_loss(self, pred_points, gt_points):
        """Compute distance-based loss between predicted and ground truth points."""
        # Ensure points are in the same format
        if pred_points.dim() == 2:
            pred_points = pred_points.unsqueeze(0)
        if gt_points.dim() == 2:
            gt_points = gt_points.unsqueeze(0)
            
        # Compute pairwise distances
        dist = torch.cdist(pred_points, gt_points)
        
        # Get minimum distance for each point
        min_dist, _ = torch.min(dist, dim=2)
        
        # Compute loss (using smooth L1 loss for robustness)
        return F.smooth_l1_loss(min_dist, torch.zeros_like(min_dist))
            
    def forward(self, x):
        if self.attention:
            x = self.attention_layer(x, x, x)[0]
            
        # Get base predictions
        pred = self.fc(x)
        
        # If using axis info, also predict axis orientation
        if self.use_axis_info:
            axis_orientation = self.axis_orientation(x)
            return pred, axis_orientation
            
        return pred

class CoordinateTransformer:
    """Helper class to transform coordinates between different spaces."""
    
    @staticmethod
    def to_axis_relative(points, axis_info):
        """Transform points to be relative to axis coordinates.
        
        Args:
            points (torch.Tensor): Points in image coordinates (N, 2)
            axis_info (torch.Tensor): Axis information [x_min, x_max, y_min, y_max, x_origin, y_origin, x_scale, y_scale]
        """
        # Extract axis information
        x_min, x_max, y_min, y_max, x_origin, y_origin, x_scale, y_scale = axis_info.unbind(1)
        
        # Normalize to [0, 1] range
        x_norm = (points[..., 0] - x_min) / (x_max - x_min)
        y_norm = (points[..., 1] - y_min) / (y_max - y_min)
        
        # Scale to axis units
        x_axis = x_norm * x_scale + x_origin
        y_axis = y_norm * y_scale + y_origin
        
        return torch.stack([x_axis, y_axis], dim=-1)
    
    @staticmethod
    def to_image_coordinates(points, axis_info):
        """Transform points from axis coordinates to image coordinates."""
        # Extract axis information
        x_min, x_max, y_min, y_max, x_origin, y_origin, x_scale, y_scale = axis_info.unbind(1)
        
        # Convert from axis units to normalized coordinates
        x_norm = (points[..., 0] - x_origin) / x_scale
        y_norm = (points[..., 1] - y_origin) / y_scale
        
        # Convert to image coordinates
        x_img = x_norm * (x_max - x_min) + x_min
        y_img = y_norm * (y_max - y_min) + y_min
        
        return torch.stack([x_img, y_img], dim=-1)

@MODELS.register_module()
class DataSeriesHead(nn.Module):
    """Specialized head for data series prediction with dual attention to coordinates and axis-relative positions."""
    
    def __init__(self, in_channels, max_points=50, loss=None):
        super().__init__()
        self.max_points = max_points
        self.loss = loss
        
        # Feature extraction
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        
        # Separate branches for absolute and relative coordinates
        self.absolute_branch = nn.Sequential(
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, max_points * 2)  # 2 coordinates per point
        )
        
        self.relative_branch = nn.Sequential(
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, max_points * 2)  # 2 coordinates per point
        )
        
        # Attention mechanisms
        self.coord_attention = nn.MultiheadAttention(in_channels, num_heads=8)
        self.axis_attention = nn.MultiheadAttention(in_channels, num_heads=8)
        self.sequence_attention = nn.MultiheadAttention(in_channels, num_heads=8)
        
        # Sequence-aware processing
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=8,
                dim_feedforward=in_channels * 4,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Pattern recognition
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 5)  # 5 for different chart patterns
        )
        
        # Coordinate transformer
        self.coord_transformer = CoordinateTransformer()
        
    def check_monotonicity(self, points, chart_type):
        """Check if points follow expected monotonicity based on chart type."""
        if chart_type in ['line', 'scatter']:
            # For line/scatter, check if points are generally increasing or decreasing
            diffs = points[..., 1].diff()
            return torch.all(diffs >= 0) or torch.all(diffs <= 0)
        return True
        
    def forward(self, x, axis_info=None, chart_type=None):
        # Apply coordinate attention
        coord_feat = self.coord_attention(x, x, x)[0]
        
        # Apply axis attention if axis info is available
        if axis_info is not None:
            axis_feat = self.axis_attention(x, x, x)[0]
            # Combine features
            x = coord_feat + axis_feat
        else:
            x = coord_feat
            
        # Apply sequence attention
        seq_feat = self.sequence_attention(x, x, x)[0]
        x = x + seq_feat
        
        # Process through sequence encoder
        x = self.sequence_encoder(x.unsqueeze(0)).squeeze(0)
        
        # Extract base features
        x = F.relu(self.fc1(x))
        
        # Get predictions from both branches
        absolute_points = self.absolute_branch(x)
        relative_points = self.relative_branch(x)
        
        # Reshape to (batch_size, max_points, 2)
        absolute_points = absolute_points.view(-1, self.max_points, 2)
        relative_points = relative_points.view(-1, self.max_points, 2)
        
        # If axis information is provided, transform relative points
        if axis_info is not None:
            relative_points = self.coord_transformer.to_axis_relative(relative_points, axis_info)
            
        # Get pattern prediction
        pattern_logits = self.pattern_recognizer(x)
        
        # Check monotonicity if chart type is provided
        if chart_type is not None:
            monotonicity = self.check_monotonicity(absolute_points, chart_type)
        else:
            monotonicity = None
        
        return absolute_points, relative_points, pattern_logits, monotonicity
        
    def compute_loss(self, pred_absolute, pred_relative, gt_absolute, gt_relative, 
                    pattern_logits, gt_pattern, axis_info=None, chart_type=None):
        """Compute combined loss for both absolute and relative coordinates."""
        # Ensure points are in the same format
        if pred_absolute.dim() == 2:
            pred_absolute = pred_absolute.unsqueeze(0)
        if pred_relative.dim() == 2:
            pred_relative = pred_relative.unsqueeze(0)
        if gt_absolute.dim() == 2:
            gt_absolute = gt_absolute.unsqueeze(0)
        if gt_relative.dim() == 2:
            gt_relative = gt_relative.unsqueeze(0)
            
        # Compute absolute coordinate loss
        absolute_loss = self.compute_distance_loss(pred_absolute, gt_absolute)
        
        # Compute relative coordinate loss
        if axis_info is not None:
            # Transform predicted absolute points to relative coordinates
            pred_absolute_relative = self.coord_transformer.to_axis_relative(pred_absolute, axis_info)
            relative_loss = self.compute_distance_loss(pred_absolute_relative, gt_relative)
        else:
            relative_loss = torch.tensor(0.0, device=pred_absolute.device)
            
        # Compute pattern recognition loss
        pattern_loss = F.cross_entropy(pattern_logits, gt_pattern)
        
        # Add monotonicity penalty if applicable
        if chart_type is not None:
            monotonicity = self.check_monotonicity(pred_absolute, chart_type)
            monotonicity_loss = F.binary_cross_entropy(monotonicity.float(), torch.ones_like(monotonicity.float()))
        else:
            monotonicity_loss = torch.tensor(0.0, device=pred_absolute.device)
            
        # Combine losses with weights
        total_loss = (absolute_loss + relative_loss + 
                     0.5 * pattern_loss + 0.3 * monotonicity_loss)
        
        return total_loss
        
    def compute_distance_loss(self, pred_points, gt_points):
        """Compute distance-based loss between predicted and ground truth points."""
        # Compute pairwise distances
        dist = torch.cdist(pred_points, gt_points)
        
        # Get minimum distance for each point
        min_dist, _ = torch.min(dist, dim=2)
        
        # Compute loss (using smooth L1 loss for robustness)
        return F.smooth_l1_loss(min_dist, torch.zeros_like(min_dist)) 