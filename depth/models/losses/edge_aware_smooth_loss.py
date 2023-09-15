import torch
import torch.nn as nn

from depth.models.builder import LOSSES

def calc_smoothness(inv_depths, images, num_scales):
    """
    Calculate smoothness values for inverse depths

    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    images : list of torch.Tensor [B,3,H,W]
        Inverse depth maps
    num_scales : int
        Number of scales considered

    Returns
    -------
        res: list of torch.Tensor[1]
        smooth loss in multiscales
    """
    #normalize inv_depth map
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    inv_depths_norm = [inv_depth / (mean_inv_depth+1e-7)#.clamp(min=1e-7)
            for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]
    #inv_depth gradient
    inv_depth_gradients_x = [torch.abs(d[:, :, :, :-1] - d[:, :, :, 1:]) for d in inv_depths_norm]
    inv_depth_gradients_y = [torch.abs(d[:, :, :-1, :] - d[:, :, 1:, :]) for d in inv_depths_norm]
    #image gradient
    image_gradients_x = [torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(1,True) for image in images]
    image_gradients_y = [torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(1,True) for image in images]
    #edge aware
    weights_x = [torch.exp(-g) for g in image_gradients_x]
    weights_y = [torch.exp(-g) for g in image_gradients_y]

    # Note: Fix gradient addition
    smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in range(num_scales)]#list of torch.Tensor [B,1,H,W]
    smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in range(num_scales)]#list of torch.Tensor [B,1,H,W]

    res = [smoothness_x[i].mean()+smoothness_y[i].mean() for i in range(num_scales)]

    #calculate smoothness loss 
    smoothness_loss = sum([res[i]/(2**i)  for i in range(num_scales)])/ num_scales
    return res

@LOSSES.register_module
class Edge_Aware_Smooth_Loss(nn.Module):

    def __init__(self, num_scales,loss_weight):
        super(Edge_Aware_Smooth_Loss, self).__init__()
        self.num_scales = num_scales
        self.loss_weight = loss_weight

    def forward(self,inv_depths, rgb_original):
        if self.loss_weight == 0:
            return torch.Tensor([0.0]).cuda()
        from .loss_utils import match_scales
        rgb = match_scales(rgb_original, inv_depths, self.num_scales)
        loss = calc_smoothness(inv_depths, rgb, self.num_scales)
        return loss