
import mmcv
import copy
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from .pac import packernel2d
from depth.ops import resize
from depth.models.builder import build_loss
from IPython import embed


class AdaptiveDepth2normal(nn.Module):
    def __init__(self, k_size=5, dilation=1, stride=1, depth_max=10., sample_num=40, area_type=1, area_thred=0.0):
        """
        convert depth map to point cloud first, and then calculate normal map
        :param k_size: the kernel size for neighbor points
        not use fixed pattern to calculate normal map; randomly select a set of three points to calculate the normal vectors
        """
        super(AdaptiveDepth2normal, self).__init__()
        self.k_size = k_size


        self.stride = stride
        self.dilation = dilation
        self.padding = (self.dilation * (self.k_size - 1) + 1 - self.stride + 1) // 2

        self.area_thred = (k_size ** 2 * 0.5) * area_thred
        self.area_type = area_type

        self.depth_max = depth_max

        self.sample_num = sample_num

        self.unford = torch.nn.Unfold(kernel_size=(self.k_size, self.k_size), padding=self.padding, stride=self.stride,
                                      dilation=self.dilation)

        self.unford_stride = torch.nn.Unfold(kernel_size=1, padding=0, stride=self.stride)

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth).to(depth.device)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth).to(depth.device)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth).to(depth.device)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1).type_as(depth).to(
            depth.device)  # [1, 3, H, W]

        return pixel_coords

    def pixel2cam(self, depth, intrinsics_inv, pixel_coords):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, h, w = depth.size()
        current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
        cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def select_index(self):
        """
        select the indexes of three points in the kernel
        """
        num = self.k_size ** 2
        p1 = np.random.choice(num, int(self.sample_num), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(self.sample_num), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(self.sample_num), replace=True)
        np.random.shuffle(p3)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        index_list = np.stack([p1, p2, p3], axis=1)

        valid_list = []
        valid_index_set = set()
        valid_areas = []  # store the area of the triangles
        # remove invalid index: 1. adjacent points; 2. points in the same line
        for i in range(np.shape(index_list)[0]):
            [p1, p2, p3] = np.sort(index_list[i])

            if (p1, p2, p3) in valid_index_set:
                continue

            p1_x = p1 % self.k_size
            p1_y = p1 // self.k_size

            p2_x = p2 % self.k_size
            p2_y = p2 // self.k_size

            p3_x = p3 % self.k_size
            p3_y = p3 // self.k_size

            # use cross product to calculate triangles' area, dot product should use two vectors (<90 degree)
            area = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x) 

            if area > self.area_thred:
                valid_list.append([p1, p2, p3])
                valid_index_set.add((p1, p2, p3))
                valid_areas.append(area)
            elif area < -self.area_thred:
                valid_list.append([p1, p3, p2])
                valid_index_set.add((p1, p3, p2))
                valid_areas.append(-1 * area)

        valid_list = np.stack(valid_list, axis=0)  # [n,3]
        valid_areas = np.array(valid_areas)

        valid_areas = valid_areas ** self.area_type

        valid_areas = valid_areas / np.sum(valid_areas)  # [n]
        # print("valid triplets: ", np.shape(valid_list))
        return valid_list, valid_areas

    def forward(self, depth, intrinsic, guide_weight=None, if_area=True, if_pa=True):
        """
        :param depth: [B, 1, H, W]
        :param intrinsic: [B, 3, 3]
        :param weight: [B,H,W,K*K] if not None, that is weighted least square modules
        :return:
        """
        device = depth.get_device()
        if device < 0:
            device = 'cpu'
        depth = depth.squeeze(1)
        b, h, w = depth.shape
        intrinsic_inv = torch.inverse(intrinsic)

        if guide_weight is None:
            guide_weight = torch.ones([b, h // self.stride, w // self.stride, self.k_size * self.k_size]).type_as(
                depth).to(device)

        pixel_coords = self.set_id_grid(depth)
        points = self.pixel2cam(depth, intrinsic_inv, pixel_coords)  # (b, c, h, w)

        valid_condition = ((depth > 0) & (depth < self.depth_max)).type(torch.FloatTensor).to(device)
        valid_condition = valid_condition.unsqueeze(1)  # (b, 1, h, w)

        points_patches = self.unford(points)  # (N,C×∏(kernel_size),L)
        points_patches = points_patches.view(-1, 3, self.k_size * self.k_size, h // self.stride, w // self.stride)
        points_patches = points_patches.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 3)

        ## extract three points based triangle indexes
        triangle_index, triangle_area_weights = self.select_index()
        triangle_index = torch.from_numpy(triangle_index).type(torch.LongTensor).to(device)  # [n,3]
        triangle_area_weights = torch.from_numpy(triangle_area_weights).type_as(depth).to(device)  # [n]
        triangle_area_weights = triangle_area_weights.view(1, 1, 1, -1)  # [1,1,1,n]

        nn = triangle_index.shape[0]
        triangle_index_copy = triangle_index.view(-1)  # [n*3]
        triangles_patches = torch.index_select(points_patches, 3, triangle_index_copy)
        triangles_patches = triangles_patches.view(b, h, w, nn, 3, 3)

        vector01_patches = triangles_patches[:, :, :, :, 1, :] - triangles_patches[:, :, :, :, 0, :]  # (b, h, w, n, 3)
        vector02_patches = triangles_patches[:, :, :, :, 2, :] - triangles_patches[:, :, :, :, 0, :]  # (b, h, w, n, 3)

        normals_patches = torch.cross(vector01_patches, vector02_patches, dim=-1)  # (b, h, w, n, 3)
        # normalize normal vectors
        normals_patches = normals_patches / (torch.norm(normals_patches, dim=-1, keepdim=True) + 1e-5)

        # extract valid_condition_patches
        valid_condition_patches = self.unford(valid_condition)
        valid_condition_patches = valid_condition_patches.view(-1, self.k_size * self.k_size, h // self.stride,
                                                               w // self.stride)
        valid_condition_patches = valid_condition_patches.permute(0, 2, 3, 1)  # (b, h, w, self.k_size*self.k_size)

        valid_condition_triangles = torch.stack(
            [torch.index_select(valid_condition_patches, 3, triangle_index[i]) for i in
             range(triangle_index.shape[0])], dim=3)  # (b, h, w, n, 3p)

        valid_condition_triangles = valid_condition_triangles[:, :, :, :, 0] * \
                                    valid_condition_triangles[:, :, :, :, 1] * \
                                    valid_condition_triangles[:, :, :, :, 2]  # (b, h, w, n)

        # extract guide_weight triangles
        guide_weight_triangles = torch.stack(
            [torch.index_select(guide_weight, 3, triangle_index[i]) for i in
             range(triangle_index.shape[0])], dim=3)  # (b, h, w, n, 3p)

        guide_weight_triangles = guide_weight_triangles[:, :, :, :, 0] * \
                                 guide_weight_triangles[:, :, :, :, 1] * \
                                 guide_weight_triangles[:, :, :, :, 2]  # (b, h, w, n)

        final_weight_triangles = valid_condition_triangles

        if if_area:
            final_weight_triangles = final_weight_triangles * triangle_area_weights  # (b, h, w, n)

        if if_pa:
            final_weight_triangles = final_weight_triangles * guide_weight_triangles

        final_weight_triangles = torch.softmax(final_weight_triangles, dim=-1)  # (b, h, w, n)

        normals_weighted = torch.sum(normals_patches * final_weight_triangles.unsqueeze(-1), dim=3,
                                     keepdim=False)  # (b, h, w, 3)

        generated_norm_normalize = normals_weighted / (torch.norm(normals_weighted, dim=-1, keepdim=True) + 1e-5)

        valid_condition = valid_condition.squeeze(1).unsqueeze(-1).repeat(1, 1, 1, 3) > 0
        generated_norm_normalize[~valid_condition] = 0

        return generated_norm_normalize.permute(0, 3, 1, 2), points


def compute_kernel(input_for_kernel, input_mask=None, kernel_size=3, stride=1, padding=1, dilation=1,
                   kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=True):
    return packernel2d(input_for_kernel, input_mask,
                       kernel_size=kernel_size, stride=stride, padding=padding,
                       dilation=dilation, kernel_type=kernel_type,
                       smooth_kernel_type=smooth_kernel_type,
                       smooth_kernel=None,
                       inv_alpha=None,
                       inv_lambda=None,
                       channel_wise=False,
                       normalize_kernel=normalize_kernel,
                       transposed=False,
                       native_impl=None)



class DepthNormalConversion(nn.Module):
    def __init__(self, k_size, dilation, sample_num=40):
        super(DepthNormalConversion, self).__init__()
        self.k_size = k_size
        self.dilation = dilation

        self.depth2norm = AdaptiveDepth2normal(k_size=k_size, dilation=dilation, sample_num=sample_num)

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return compute_kernel(input_for_kernel, input_mask,
                              kernel_size=self.k_size,
                              dilation=self.dilation,
                              padding=self.dilation * (self.k_size - 1) // 2)[0]

    def forward(self, init_depth, intrinsic, guidance=None, if_area=True, if_pa=True):
        if guidance is not None:
            guide_weight = self.compute_kernel(guidance)  # [B, 1, K, K, H, W]
            B, C, K1, K2, H, W = guide_weight.shape

            # smooth the kernel; otherwise, the distribution is too sharp
            ones_constant = torch.ones_like(guide_weight).type_as(guide_weight).to(guide_weight.device) / (K1 * K2)
            guide_weight = guide_weight + ones_constant
            norm = guide_weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
            guide_weight = guide_weight / norm * (K1 * K2)  # scale to larger values

            guide_weight = guide_weight.squeeze(1).permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, K, K]
            B, H, W, K, K = guide_weight.shape
            guide_weight = guide_weight.view(B, H, W, K * K)
        else:
            guide_weight = None

        estimate_normal, _ = self.depth2norm(init_depth, intrinsic[:,:3,:3], guide_weight, if_area=if_area, if_pa=if_pa)

        return estimate_normal



class DepthBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (List): Input channels.
        channels (int): Channels after modules, before conv_depth.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        loss_decode (dict): Config of decode loss.
            Default: dict(type='SigLoss').
        sampler (dict|None): The config of depth map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        min_depth (int): Min depth in dataset setting.
            Default: 1e-3.
        max_depth (int): Max depth in dataset setting.
            Default: None.
        norm_cfg (dict|None): Config of norm layers.
            Default: None.
        classify (bool): Whether predict depth in a cls.-reg. manner.
            Default: False.
        n_bins (int): The number of bins used in cls. step.
            Default: 256.
        bins_strategy (str): The discrete strategy used in cls. step.
            Default: 'UD'.
        norm_strategy (str): The norm strategy on cls. probability 
            distribution. Default: 'linear'
        scale_up (str): Whether predict depth in a scale-up manner.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels=96,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='SigLoss',
                     valid_mask=True,
                     loss_weight=10),
                 loss_pe=dict(
                     type='BinaryCrossEntropyLoss',
                     loss_weight=10),
                 loss_dynamic_pe=dict(
                     type='CrossEntropyLoss',
                     loss_weight=0.08
                    ),
                #  loss_attn_pe=dict(
                #      type='L1Loss',
                #      loss_weight=0.1
                #     ),
                # loss_dynamic_pe=dict(
                #      type='SigLoss',
                #      valid_mask=True,
                #      loss_weight=1
                #     ),
                #  loss_dynamic_pe=dict(
                #      type='RMILoss'
                #     ),
                # loss_dynamic_pe=dict(
                #      type='L1Loss',
                #      loss_weight=10
                #     ),
                #  loss_dynamic_pe=dict(
                #      type='OhemCE',
                #      weight = torch.tensor(
                #         [
                #             1.0, # -0.5
                #             0.5, # 0.0
                #             1.0, # 0.5
                #         ])
                #      ),
                 loss_surface_norm = None,#dict(type='CosineSimilarityLoss',is_abs=True),
                 sampler=None,
                 align_corners=False,
                 min_depth=1e-3,
                 max_depth=None,
                 norm_cfg=None,
                 classify=False,
                 n_bins=256,
                 bins_strategy='UD',
                 norm_strategy='linear',
                 scale_up=False,
                 depth2norm=False,
                 ):
        super(DepthBaseDecodeHead, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.loss_pe = build_loss(loss_pe)
        # self.loss_attn_pe = build_loss(loss_attn_pe)
        self.loss_dynamic_pe = build_loss(loss_dynamic_pe)

        self.loss_surface_norm = loss_surface_norm
        self.depth2norm_flags = depth2norm

        if not loss_surface_norm is None and self.depth2norm_flags:
            self.loss_surface_norm = build_loss(loss_surface_norm)
            self.depth2norm = DepthNormalConversion(k_size=3, dilation=1,sample_num=40)

        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up
        self.cos = torch.nn.CosineSimilarity(dim=1)
        
        if self.classify:
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.softmax = nn.Softmax(dim=1)
            self.conv_depth = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)
        

        self.fp16_enabled = False
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def extra_repr(self):
        """Extra repr."""
        s = f'align_corners={self.align_corners}'
        return s

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass
    
    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg,pe_mask,y,guidance,pe_offset, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred,depth_y = self.forward(inputs, img_metas,pe_mask,y)
        if not pe_offset is None:
            pe_k_gt = kwargs['pe_k_gt']
            # attn_gt = kwargs['pe_k_gt']
            # attn_gt[attn_gt!=255] = 1
            losses = self.losses_dynamic_pe(depth_pred, depth_gt, pe_offset, pe_k_gt,None,None)
        elif self.depth2norm_flags:
            intrinsic = kwargs['cam_intrinsic_for_normal'].to(torch.float32)
            normals_depth = self.depth2norm(depth_pred, intrinsic, guidance)
            losses = self.losses_bebug(depth_pred, depth_gt,normals_depth,kwargs['normals_gt'])
        else:
            losses = self.losses(depth_pred, depth_gt)
        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, img, inputs, img_metas, test_cfg,pe_mask,y,**kwargs):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output depth map.
        """
        depth_pred,_ = self.forward(inputs, img_metas,pe_mask,y)
        return depth_pred

    def depth_pred(self, feat,pe,depth_y):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == 'UD':
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)
            elif self.bins_strategy == 'SID':
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == 'linear':
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == 'softmax':
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == 'sigmoid':
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum('ikmn,k->imn', [logit, bins]).unsqueeze(dim=1)

        else:
            depth_temp = None
            if self.scale_up:
                output = self.sigmoid(self.conv_depth(feat)) * self.max_depth
            else:
                depth_temp = self.relu(self.conv_depth(feat))
                if not pe is None:
                    pe = resize(
                        input=pe,
                        size=depth_temp.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
                    depth_y = resize(
                        input=depth_y,
                        size=depth_temp.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
                    # embed()
                    # exit()
                    # depth_y[pe==0] = 0
                    output = ((depth_temp*(1-depth_y))+(pe))+ self.min_depth
                else:
                    output = depth_temp + self.min_depth
        return output,depth_y

    @force_fp32(apply_to=('depth_pred', ))
    def losses_dynamic_pe(self, depth_pred, depth_gt,dynamic_pe, pe_k_gt,attn_pred,attn_gt):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        # print(pe_k_gt)
        # print('------------------------------')
        loss['loss_dynamic_pe'] = self.loss_dynamic_pe(
            dynamic_pe,
            pe_k_gt.long())

        # loss['loss_pe_attn'] = self.loss_attn_pe(
        #     attn_pred,
        #     attn_gt.unsqueeze(1))

        loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt)
        # if torch.isnan(loss['loss_dynamic_pe']):
        #     print('loss_dynamic_pe is Nan')
        #     embed()
        #     exit()
        # if torch.isnan(loss['loss_depth']):
        #     print('loss_depth is Nan')
        #     embed()
        #     exit()
        return loss



    @force_fp32(apply_to=('depth_pred', ))
    def losses_bebug(self, depth_pred, depth_gt,normals_depth=None,normals_depth_gt=None,pe_attn=False):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        if pe_attn:
            loss['loss_pe_attn'] = self.loss_pe(
            depth_pred,
            depth_gt)
        else:
            loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt)
        if not normals_depth is None:
            normals_depth = resize(
                input=normals_depth,
                size=normals_depth_gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            mask_min = 1e-3<=depth_gt
            mask_max = depth_gt<=80
            mask = torch.logical_and(mask_min,mask_max)
            loss['loss_surface_norm'] = self.loss_surface_norm(
                normals_depth,
                normals_depth_gt,
                mask,
            )
        return loss


    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, depth_pred, depth_gt,normals_depth=None,normals_depth_gt=None,normals_pe=None,normals_pe_gt=None,pe_attn=False):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        if pe_attn:
            loss['loss_pe_attn'] = self.loss_pe(
            depth_pred,
            depth_gt)
        else:
            loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt)
        if not normals_depth is None:
            normals_depth = resize(
                input=normals_depth,
                size=normals_depth_gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            normals_pe = resize(
                input=normals_pe,
                size=normals_pe_gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            mask_min = 1e-3<=depth_gt
            mask_max = depth_gt<=80
            mask = torch.logical_and(mask_min,mask_max)
            loss['loss_depth_surface_norm'] = self.loss_surface_norm(
                normals_depth,
                normals_depth_gt,
                mask,
            )
            loss['loss_pe_surface_norm'] = self.loss_surface_norm(
                normals_pe,
                normals_pe_gt,
                mask,
            )
        return loss

    def log_images(self, img_path, depth_pred, depth_gt, img_meta):
        show_img = copy.deepcopy(img_path.detach().cpu().permute(1, 2, 0))
        show_img = show_img.numpy().astype(np.float32)
        show_img = show_img[:,:,0:3]
        show_img = mmcv.imdenormalize(show_img, 
                                      img_meta['img_norm_cfg']['mean'],
                                      img_meta['img_norm_cfg']['std'], 
                                      img_meta['img_norm_cfg']['to_rgb'])
        show_img = np.clip(show_img, 0, 255)
        show_img = show_img.astype(np.uint8)
        show_img = show_img[:, :, ::-1]
        show_img = show_img.transpose(0, 2, 1)
        show_img = show_img.transpose(1, 0, 2)

        depth_pred = depth_pred / torch.max(depth_pred)
        depth_gt = depth_gt / torch.max(depth_gt)

        depth_pred_color = copy.deepcopy(depth_pred.detach().cpu())
        depth_gt_color = copy.deepcopy(depth_gt.detach().cpu())

        return {"img_rgb": show_img, "img_depth_pred": depth_pred_color, "img_depth_gt": depth_gt_color}