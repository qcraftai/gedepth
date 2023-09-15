from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decode_head import DepthBaseDecodeHead
from depth.models.builder import HEADS
from .pac import packernel2d
from IPython import embed
import numpy as np




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














class SegHead(nn.Module):
    'This class defines an naive depth estimation head, output a value between 0-1'
    def __init__(
        self, in_channels, mid_channels, output_channels, class_key, scale_factor=8.0
    ):
        # TODO(Zhiyu)
        # Add a scale factor checker in the future.
        # This code is borrowed from Panoptic Segmentation and modifed by adding simoid function to the end.
        super(SegHead, self).__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.num_head = len(output_channels)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Conv2d(mid_channels, output_channels[i], 1)

        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key
        self.scale_factor = scale_factor

    def forward(self, x):
        pred = OrderedDict()
        x = self.fuse_conv(x)
        for key in self.class_key:
            mid = self.classifier[key](x)
            mid = torch.sigmoid(mid)  # project this value to [0,1]
            out = F.interpolate(
                mid, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )
            pred[key] = out
        return out











@HEADS.register_module()
class ASNDepthHeadV2(DepthBaseDecodeHead):
    def __init__(
        self,
        input_features_dim=128,
        scale_factor=8.0,
        k_size = 3,
        sample_num = 40,
        post_processor=None,
        supvised_loss = None,
        smooth_loss = None,
        surface_norm_loss = None,
        **kwargs
    ):
        super(ASNDepthHeadV2, self).__init__()
        self.depth_head = SegHead(
            input_features_dim, input_features_dim, [1], ["disparity"], scale_factor
        )
        self.guidance_head = SegHead(
            input_features_dim, input_features_dim, [1], ["guidance"], scale_factor
        )
        self.depth2norm = DepthNormalConversion(k_size=k_size, dilation=1,
                                                sample_num=sample_num)  # Depth2NormalLight

        self.post_processor = build_neck(post_processor)       
        self.supvised_loss = build_loss(supvised_loss) if not supvised_loss==None else None
        self.smooth_loss = build_loss(smooth_loss) if not smooth_loss==None else None
        self.surface_norm_loss = build_loss(surface_norm_loss) if not surface_norm_loss==None else None

        self.fp16_enabled = False

    # @force_fp32(apply_to=("network_pred"))
    def loss(self, network_pred, **kwargs):
        loss = dict()
        mask = kwargs["depth_gt"] > 0.1
        if not self.supvised_loss==None:
            mask = kwargs['depth_gt'] > 0.1
            loss['supvised_loss'] = self.supvised_loss(network_pred['depth'], kwargs['depth_gt'], mask) 
        if not self.smooth_loss==None:
            loss['smooth_loss']=self.smooth_loss(inv_depths=[network_pred['disp']], rgb_original=kwargs['rgb'])
        if not self.surface_norm_loss == None:
            loss['surface_norm_loss'] = self.surface_norm_loss(pred=network_pred['normals'],gt=kwargs['surface_normal'],mask=mask)
        return loss

    def forward_train(self, feature_embeddings, **kwargs):

        intrinsic = kwargs['K']
        network_pred = dict()
        disp = self.depth_head(feature_embeddings)
        guidance = self.guidance_head(feature_embeddings)

        network_pred['disp'] = disp
        depth = self.post_processor(disp)
        network_pred['depth'] = depth

        normals = self.depth2norm( depth, intrinsic, guidance)
        network_pred['normals'] = normals

        # For demo usage.
        # intrinsic = kwargs['K']
        # pred_depth = depth.detach().cpu().numpy()
        # pred_normal = normals.detach().cpu().numpy()
        # gt_depth = kwargs['depth_gt'].detach().cpu().numpy()
        # gt_normal = kwargs['surface_normal'].detach().cpu().numpy()
        # K = kwargs['K'].detach().cpu().numpy()
        # rgb = kwargs['rgb'].detach().cpu().numpy()
        # rgb[:,0,...] = rgb[:,0,...] * 0.229 + 0.485
        # rgb[:,1,...] = rgb[:,1,...] * 0.224 + 0.456
        # rgb[:,2,...] = rgb[:,2,...] * 0.225 + 0.406
        # rgb = rgb * 256 
        # np.save("pred_depth.npy",pred_depth)
        # np.save("pred_normal.npy",pred_normal)
        # np.save("gt_depth.npy",gt_depth)
        # np.save("gt_normal.npy",gt_normal)
        # np.save("K.npy",K)
        # np.save("img.npy",rgb)

        
        
        depth_losses = self.loss(network_pred,**kwargs)
        return depth_losses

    def forward_test(self, feature_embeddings, **kwargs):
        # TODO(Zhiyu) Do we need this level? Maybe deprecate this level in the future if there's no other good reason to keep.
        return self.simple_test(feature_embeddings, **kwargs)

    # @force_fp32(apply_to=("feature_embeddings"))
    def simple_test(self, feature_embeddings, **kwargs):
        network_pred = dict()
        disp = self.depth_head(feature_embeddings)
        network_pred['disp'] = disp
        depth = self.post_processor(disp)
        network_pred['depth'] = depth
        return network_pred

    def onnx_export(self, feature_embeddings):
        network_pred = dict()
        network_pred.update(self.depth_head(feature_embeddings))
        network_pred["depth"] = self.post_processor(network_pred["disp"])
        return network_pred
