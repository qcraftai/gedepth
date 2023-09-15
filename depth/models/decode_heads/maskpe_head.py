#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from depth.models.builder import HEADS
from depth.models.builder import build_loss
from mmcv.runner import BaseModule
import os
from IPython import embed

class SimpleArgMaxInfernce(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, prob):
        '''
        TODO(Zhiyu) Only support batch size equals 1 now.
        Args:
            prob (Tensor), (1, Num_Clasees, H, W)
        Return:
            label (Tensor), (1, H, W)
        '''
        prob['mask_pe'] = torch.argmax(prob['mask_pe'],dim=1)
        return prob['mask_pe']


class PEHead(nn.Module):

    def __init__(self, in_channels, mid_channels, output_channels,class_key, scale_factor=8.0):
        #TODO(Zhiyu)
        #Add a scale factor checker in the future.
        super(PEHead, self).__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.num_head = len(output_channels)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Conv2d(mid_channels, output_channels[i], 1)

        self.classifier = nn.ModuleDict(classifier)
        self.scale_factor = scale_factor
        self.class_key = class_key
        

    def forward(self, x):
        pred = OrderedDict()
        x = self.fuse_conv(x)
        for key in self.class_key:
            mid = self.classifier[key](x)
            out = F.interpolate(mid,
                        scale_factor=self.scale_factor,
                        mode='bilinear',
                        align_corners=True)
            pred[key] = out
        return pred





@HEADS.register_module()
class MaskedPE(BaseModule):


    def __init__(self, input_features_dim=128, num_classes=2, ignore_label=255,pretrain_path='', scale_factor=8.0 ,mask_loss=None, **kwargs ):
        super(MaskedPE, self).__init__()
        self.masked_pe = PEHead(input_features_dim, input_features_dim, [num_classes],['mask_pe'],scale_factor)

        self.masked_pe_post_processor = SimpleArgMaxInfernce()
        self.maskpe_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.2,1]))
        # self.maskpe_loss = build_loss(mask_loss)
        self.ignore_label = ignore_label
        self.pretrain_path = pretrain_path



    def init_weights(self):
        """Init each componets of panoptic-deeplab"""
        if os.path.exists(self.pretrain_path):
            pretrain_weights = torch.load(self.pretrain_path,map_location='cpu')
            self.load_state_dict(pretrain_weights, strict = False)


    def loss(self, network_pred, **kwargs):
        loss = dict()
        loss['mask_pe_loss'] = self.maskpe_loss(network_pred['mask_pe'],kwargs['pe_gt'].long())
        return loss



    def forward_train(self, feature_embeddings, **kwargs):
        network_pred = dict()
        network_pred.update(self.masked_pe(feature_embeddings))
        losses = self.loss(network_pred, **kwargs)
        return losses


    def forward_test(self, feature_embeddings, **kwargs):
        return self.simple_test(feature_embeddings, **kwargs)


    def simple_test(self, feature_embeddings, post_process=True, **kwargs):
        output = self.masked_pe_post_processor(self.masked_pe(feature_embeddings))
        return output

