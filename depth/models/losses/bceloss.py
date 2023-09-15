import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

@LOSSES.register_module()
class BinaryCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss wrapper.
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
    def forward(self,
                input,
                target):
        """Forward function."""

        loss_ce = F.binary_cross_entropy(input, target)
        loss_cls = self.loss_weight * loss_ce
        return loss_cls