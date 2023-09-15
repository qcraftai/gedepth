from depth.models import depther
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.core import add_prefix
from depth.ops import resize
from depth.models import builder
from depth.models.builder import DEPTHER
from .base import BaseDepther

# for model size
import os
import cv2
import numpy as np
from IPython import embed




@DEPTHER.register_module()
class DepthEncoderDecoder(BaseDepther):
    """Encoder Decoder depther.

    EncoderDecoder typically consists of backbone, (neck) and decode_head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 pe_mask_neck=None,
                 dynamic_pe_neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(DepthEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and depther set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)
        self.pe_mask_neck_FLAGS = False
        self.dynamic_pe_neck_FLAGS = False


        if neck is not None:
            self.neck = builder.build_neck(neck)
        if pe_mask_neck is not None:
            self.pe_mask_neck = builder.build_neck(pe_mask_neck)
            self.pe_mask_neck_FLAGS = True
        
        if dynamic_pe_neck is not None:
            self.dynamic_pe_neck = builder.build_neck(dynamic_pe_neck)
            self.dynamic_pe_neck_FLAGS = True

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head


        self.indices = torch.linspace(-5, 5, 11,device=torch.cuda.current_device()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.index = 0



    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners


    def dynamic_pe(self,x,y,img,img_metas, **kwargs):
        pe_img_comput = img[:,4,:,:].clone()
        pe_img_comput = pe_img_comput.unsqueeze(1)
        pe_slope_k_for_loss = self.dynamic_pe_neck(x)
        pe_slope_k_for_loss = F.interpolate(pe_slope_k_for_loss,size=[pe_img_comput.shape[2], pe_img_comput.shape[3]],mode='bilinear')
        pe_slope_k = F.softmax(pe_slope_k_for_loss,dim=1)
        pe_slope_k = torch.sum(pe_slope_k * self.indices, dim=1)
        pe_slope_k = pe_slope_k.unsqueeze(1)
        pe_slope_k = torch.tan(torch.deg2rad(pe_slope_k))
        a = -1.65/(pe_img_comput+1e-8)
        pe_offset = -1.65/((a-pe_slope_k)+1e-8)
        pe_offset_mask = pe_offset.clone()
        pe_offset_mask[pe_offset_mask<0] = 0
        pe_offset_mask[pe_offset_mask>200] = 0
        pe_offset_mask[pe_offset_mask>0] = 1
        pe_mask = ((pe_offset*pe_offset_mask))*y
        return pe_mask,pe_slope_k_for_loss


    def extract_feat(self, img, img_metas, **kwargs):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            if self.pe_mask_neck_FLAGS:
                y,dynamic_y = self.pe_mask_neck(x)
                height_output = img.shape[2]
                width_output = img.shape[3]
                y = F.interpolate(y,size=[height_output, width_output],mode='bilinear')

                if self.dynamic_pe_neck_FLAGS:
                    pe_mask,pe_slope_k_ori = self.dynamic_pe(x,y,img,img_metas, **kwargs)
                    return x,y,pe_mask,pe_slope_k_ori
                else:                    
                    x_pe = img[:,3,:,:].clone()
                    x_pe = x_pe.unsqueeze(1)
                    pe_mask = x_pe*y * 200
                    return x,y,pe_mask,None
        return x,None,None,None

    def encode_decode(self, img, img_metas, rescale=True, **kwargs):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        x,y,pe_mask,_ = self.extract_feat(img,img_metas, **kwargs)
        out = self._decode_head_forward_test(img,x, img_metas,pe_mask,y,**kwargs)
        # crop the pred depth to the certain range.
        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        if rescale:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, img, x, img_metas, depth_gt,pe_mask,y,pe_offset, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, depth_gt, self.train_cfg,pe_mask,y,pe_offset, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_train_pe(self, img, x, img_metas, depth_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, depth_gt, self.train_cfg,True, **kwargs)
        # loss_decode['loss_depth'] = loss_decode['loss_depth']*0.1
        losses.update(add_prefix(loss_decode, 'decode.pe'))
        return losses

    def _decode_head_forward_test(self,img,x, img_metas,pe_mask,y,**kwargs):
        """Run forward function and calculate loss for decode head in
        inference."""
        depth_pred = self.decode_head.forward_test(img,x, img_metas, self.test_cfg,pe_mask,y,**kwargs)
        return depth_pred

    def forward_dummy(self, img):
        """Dummy forward function."""
        depth = self.encode_decode(img, None)

        return depth

    def forward_train(self, img, img_metas, depth_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): Depth gt
                used if the architecture supports depth estimation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x,y,pe_mask,pe_offset = self.extract_feat(img,img_metas, **kwargs)

        losses = dict()
        # the last of x saves the info from neck
        loss_decode = self._decode_head_forward_train(img, x, img_metas, depth_gt,pe_mask,y,pe_offset, **kwargs)
        # loss_decode_pe = self._decode_head_forward_train_pe(img, y, img_metas, depth_gt, **kwargs)
        losses.update(loss_decode)
        # losses.update(loss_decode_pe)

        return losses

    def whole_inference(self, img, img_meta, rescale, **kwargs):
        """Inference with full image."""
        depth_pred = self.encode_decode(img, img_meta, rescale, **kwargs)

        return depth_pred

    def inference(self, img, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output depth map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            raise NotImplementedError
        else:
            depth_pred = self.whole_inference(img, img_meta, rescale, **kwargs)
        output = depth_pred
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        depth_pred = self.inference(img, img_meta, rescale, **kwargs)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            depth_pred = depth_pred.unsqueeze(0)
            return depth_pred
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented depth logit inplace
        # depth_pred = self.inference(imgs[0], img_metas[0], rescale, **kwargs)
        if True:
            kwargs['pe_ori_point_test'] = kwargs['pe_ori_point'][0]
        if 'pe_k_gt' in kwargs:
            kwargs['pe_k_gt_test'] = kwargs['pe_k_gt'][0]
        depth_pred = self.inference(imgs[0], img_metas[0], rescale, **kwargs)
        for i in range(1, len(imgs)):
            if True:
                kwargs.update({'pe_ori_point_test':kwargs['pe_ori_point'][i]})
            if 'pe_k_gt' in kwargs:
                kwargs.update({'pe_k_gt_test':kwargs['pe_k_gt'][i]})
            cur_depth_pred = self.inference(imgs[i], img_metas[i], rescale, **kwargs)
            depth_pred += cur_depth_pred
        depth_pred /= len(imgs)
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred
