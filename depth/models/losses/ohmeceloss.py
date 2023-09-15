import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES
from IPython import embed


@LOSSES.register_module()
class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, threshold=0.3,
                 min_kept=1, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels):
        predictions = F.softmax(logits, dim=1)
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        # Get the score for gt class at each pixel location.
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = predictions.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)

        pixel_losses = pixel_losses[mask][indices]
        pixel_losses = pixel_losses[predictions < threshold]
        return pixel_losses.mean()
