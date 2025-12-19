from mmdet.models.losses import GIoULoss
 
class SafeGIoULoss(GIoULoss):
    def forward(self, pred, target, weight=None, avg_factor=None, reduction='mean', **kwargs):
        raw = super().forward(pred, target, weight, avg_factor, reduction, **kwargs)
        return raw.clamp(min=0.0) 