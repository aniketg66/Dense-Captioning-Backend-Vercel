# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.registry import MODELS
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from .square_mask_target import square_mask_target


@MODELS.register_module()
class SquareFCNMaskHead(FCNMaskHead):
    """FCN mask head that forces square mask targets.
    
    This head ensures that all mask targets are square regardless of the original
    aspect ratio to avoid tensor size mismatches during training.
    """

    def __init__(self, *args, **kwargs):
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Initializing SquareFCNMaskHead")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: args: {args}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: kwargs: {kwargs}")
        super().__init__(*args, **kwargs)
        print(f"🔍 SQUARE_FCN_MASK_HEAD: SquareFCNMaskHead initialized successfully")

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.

        Returns:
            Tensor: Predicted foreground masks.
        """
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Input shape: {x.shape}")
        
        for i, conv in enumerate(self.convs):
            x = conv(x)
            print(f"🔍 SQUARE_FCN_MASK_HEAD: After conv {i} shape: {x.shape}")
            
        if self.upsample is not None:
            print(f"🔍 SQUARE_FCN_MASK_HEAD: Upsampling from {x.shape}")
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            print(f"🔍 SQUARE_FCN_MASK_HEAD: After upsample shape: {x.shape}")
        else:
            print(f"🔍 SQUARE_FCN_MASK_HEAD: No upsampling, shape: {x.shape}")
            
        mask_preds = self.conv_logits(x)
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Final mask_preds shape: {mask_preds.shape}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds device: {mask_preds.device}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds dtype: {mask_preds.dtype}")
        
        return mask_preds

    def loss_and_target(self,
                       mask_preds: Tensor,
                       sampling_results: InstanceList,
                       batch_gt_instances: InstanceList,
                       rcnn_train_cfg: ConfigType) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, mask_h, mask_w).
            sampling_results (List[:obj:`SamplingResult`]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (List[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss components.
        """
        print(f"🔍 SQUARE_FCN_MASK_HEAD: loss_and_target called")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds shape: {mask_preds.shape}")
        
        # Get mask targets
        mask_targets = self.get_targets(sampling_results, batch_gt_instances,
                                       rcnn_train_cfg)
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_targets shape: {mask_targets.shape}")
        
        # Get labels for positive proposals
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels shape: {pos_labels.shape}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels: {pos_labels}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels min: {pos_labels.min()}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels max: {pos_labels.max()}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: num_classes: {self.num_classes}")
        
        # Check for out-of-bounds labels
        if pos_labels.max() >= self.num_classes:
            print(f"🔍 SQUARE_FCN_MASK_HEAD: ERROR! Found label {pos_labels.max()} >= num_classes {self.num_classes}")
            # Clamp labels to valid range
            pos_labels = torch.clamp(pos_labels, 0, self.num_classes - 1)
            print(f"🔍 SQUARE_FCN_MASK_HEAD: Clamped pos_labels max: {pos_labels.max()}")
        
        # Check for size mismatch between predictions and targets
        if mask_preds.shape[-2:] != mask_targets.shape[-2:]:
            print(f"🔍 SQUARE_FCN_MASK_HEAD: SIZE MISMATCH!")
            print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds shape: {mask_preds.shape}")
            print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_targets shape: {mask_targets.shape}")
        
        # Calculate loss - use the original approach like FCNMaskHead
        print(f"🔍 SQUARE_FCN_MASK_HEAD: About to call loss_mask")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds shape: {mask_preds.shape}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_targets shape: {mask_targets.shape}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels shape: {pos_labels.shape}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_preds device: {mask_preds.device}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: mask_targets device: {mask_targets.device}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: pos_labels device: {pos_labels.device}")
        
        # Call loss function with full mask_preds and pos_labels like the original FCN mask head
        loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Loss calculated successfully: {loss_mask}")
        
        # only return the *nested* loss dict that StandardRoIHead.update() expects
        return dict(
            loss_mask={'loss_mask': loss_mask},
            # if you really need mask_targets downstream you can still return it under a
            # different key, but it will be ignored by the standard loss updater
            mask_targets=mask_targets
        )

    def get_targets(self,
                   sampling_results: InstanceList,
                   batch_gt_instances: InstanceList,
                   rcnn_train_cfg: ConfigType) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[:obj:`SamplingResult`]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (List[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            Tensor: Mask targets of each positive proposals in the image,
                has shape (num_pos, mask_h, mask_w).
        """
        print(f"🔍 SQUARE_FCN_MASK_HEAD: get_targets called")
        
        pos_proposals_list = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks_list = [res.masks for res in batch_gt_instances]
        
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Number of sampling results: {len(sampling_results)}")
        print(f"🔍 SQUARE_FCN_MASK_HEAD: rcnn_train_cfg: {rcnn_train_cfg}")
        
        # Use our custom square mask target function
        mask_targets = square_mask_target(pos_proposals_list, pos_assigned_gt_inds_list,
                                         gt_masks_list, rcnn_train_cfg)
        
        print(f"🔍 SQUARE_FCN_MASK_HEAD: Final mask_targets shape: {mask_targets.shape}")
        return mask_targets