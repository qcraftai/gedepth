# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# import torch.nn as nn
# import torch.nn.functional as F
# from depth.models.builder import LOSSES
# import torch
# from IPython import embed
# @LOSSES.register_module()
# class CosineSimilarityLoss(nn.Module):
#     """This Loss is usually used in vector direction similarity 
#     """

#     def __init__(
#         self,
#         is_abs=False,
#     ):
#         super(CosineSimilarityLoss, self).__init__()
#         self.metric = torch.nn.CosineSimilarity(dim=1)
#         self.is_abs = is_abs

#     def forward(self, pred, gt, mask, **kwargs):
#         '''
#         Args,
#             pred, torch.Tensor, (B,C,H,W)
#             gt ,  torch.Tensor, (B,C,H,W)
#             mask, torch.Tensor, (B,1,H,W)
#         '''


#         assert pred.shape == gt.shape
#         mask = mask.squeeze()
#         # pemask = pemask.squeeze()
#         similarity = self.metric(pred, gt)
#         selected_similarity = similarity[mask]
#         # selected_pemask = pemask[mask] * 0

#         if self.is_abs:
#             selected_similarity = torch.abs(selected_similarity)
#         # loss = torch.mean(selected_pemask*(1 - selected_similarity))
#         loss = torch.mean(1 - selected_similarity)
    
#         return loss
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from depth.models.builder import LOSSES
import torch
from IPython import embed


class CosineSimilarity(nn.Module):

    def __init__(self):
        super(CosineSimilarity, self).__init__()    
    def forward(self, pred, gt):
        return torch.cos(pred)*torch.cos(gt)+torch.sin(pred)*torch.sin(gt)





@LOSSES.register_module()
class CosineSimilarityLoss(nn.Module):
    """This Loss is usually used in vector direction similarity 
    """

    def __init__(
        self,
        loss_weight=1,
    ):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.metric = CosineSimilarity()

    def forward(self, pred, gt):
        '''
        Args,
            pred, torch.Tensor, (B,C,H,W)
            gt ,  torch.Tensor, (B,C,H,W)
            mask, torch.Tensor, (B,1,H,W)
        '''


        assert pred.shape == gt.shape
        selected_similarity = self.metric(pred,gt)
        loss = self.loss_weight * torch.mean(1 - selected_similarity)
        return loss
