from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
import torch

@MODELS.register_module(name='FCHead')
class FCHead(nn.Module):
    """Simple fully connected head for classification."""
    
    def __init__(self, in_channels, num_classes, loss=None, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.loss_cfg = loss
        
    def forward(self, x, target=None):
        # Global average pooling
        if isinstance(x, (list, tuple)):
            x = x[-1]  # Use the last feature map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.fc(x)
        
        if target is not None:
            if self.loss_cfg is not None:
                loss = F.cross_entropy(logits, target, 
                                     weight=torch.tensor(self.loss_cfg.get('class_weight', None), 
                                                       device=logits.device))
            else:
                loss = F.cross_entropy(logits, target)
            return loss
        return logits

@MODELS.register_module(name='RegHead')
class RegHead(nn.Module):
    """Enhanced regression head with attention and axis-aware features."""
    
    def __init__(self, in_channels, out_dims, loss=None, max_points=None, 
                 attention=False, use_axis_info=False, **kwargs):
        super().__init__()
        self.attention = attention
        self.use_axis_info = use_axis_info
        self.max_points = max_points
        self.loss_cfg = loss
        
        # Main regression layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, out_dims)
        )
        
        # Attention mechanism if enabled
        if self.attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        # Axis-aware feature processing if enabled
        if self.use_axis_info:
            self.axis_processor = nn.Sequential(
                nn.Linear(in_channels + 8, 512),  # +8 for axis info
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, out_dims)
            )
        
    def forward(self, x, target=None, axis_info=None):
        # Global average pooling
        if isinstance(x, (list, tuple)):
            x = x[-1]  # Use the last feature map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Apply attention if enabled
        if self.attention:
            att_weights = self.attention_layer(x)
            x = x * att_weights
        
        # Process with axis info if enabled
        if self.use_axis_info and axis_info is not None:
            x = torch.cat([x, axis_info], dim=1)
            pred = self.axis_processor(x)
        else:
            pred = self.fc_layers(x)
        
        # Handle max points if specified
        if self.max_points is not None and pred.size(1) > self.max_points * 2:
            pred = pred[:, :self.max_points * 2]
        
        if target is not None:
            if self.loss_cfg is not None:
                loss = F.smooth_l1_loss(pred, target, 
                                      beta=self.loss_cfg.get('beta', 1.0),
                                      reduction=self.loss_cfg.get('reduction', 'mean'))
            else:
                loss = F.smooth_l1_loss(pred, target)
            return loss
        return pred 