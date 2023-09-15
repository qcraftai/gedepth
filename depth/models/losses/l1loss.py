import torch.nn as nn
import torch.nn.functional as F
import torch
from depth.models.builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def l1_loss(pred, target):
    """Warpper of mse loss."""
    # return nn.L1Loss(pred, target)
    return F.smooth_l1_loss(pred, target)

@LOSSES.register_module()
class L1Loss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """

        valid_mask = target < 255
        loss = self.loss_weight * l1_loss(
            pred if False in torch.unique(valid_mask) and len(torch.unique(valid_mask)) == 1 else pred[valid_mask],
            pred if False in torch.unique(valid_mask) and len(torch.unique(valid_mask)) == 1 else target[valid_mask],
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        return loss