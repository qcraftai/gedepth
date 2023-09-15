from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn as nn

def calculate(gt, pred):
    if gt.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)

    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0
        
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def metrics(gt, pred, min_depth=1e-3, max_depth=80):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def eval_metrics(gt, pred, min_depth=1e-3, max_depth=80):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def pre_eval_to_metrics(pre_eval_results,eval_chamfer):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
    ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
    ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
    ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
    ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
    ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
    ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
    ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
    ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])
    if eval_chamfer:
        ret_metrics['chamfer'] = np.nanmean(pre_eval_results[9])
        ret_metrics['precision'] = np.nanmean(pre_eval_results[10])
        ret_metrics['recall'] = np.nanmean(pre_eval_results[11])
        ret_metrics['F_score'] = np.nanmean(pre_eval_results[12])

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }

    return ret_metrics



class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        )

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False
        )

    def forward(self, depth, inv_K):
        """
        inv_k [12,4,4]
        depth [12,1,384,640]
        """
        device = depth.device
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.to(device))
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points  # [B,3,l]
        cam_points = torch.cat([cam_points, self.ones.to(device)], 1)  # [B,4,l]
        return cam_points

