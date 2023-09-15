# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F
from depth.ops import resize
from depth.models.builder import NECKS

import torch
from mmcv.runner import BaseModule
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from IPython import embed

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


@NECKS.register_module()
class LightPEMASKNeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(LightPEMASKNeck, self).__init__()
        self.convfinal = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(1536, 64, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(768, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(384, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        x0,x1,x2,x3,x4 = inputs[::-1]
        x0 = self.conv0(x0)
        x0 = F.interpolate(x0, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)
        x3 = F.interpolate(x3, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x4 = self.conv4(x4)
        x = x0+x1+x2+x3+x4
        return self.sigmoid(self.convfinal(x)),x


@NECKS.register_module()
class PEMASKNeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self,
                 up_sample_channels,
                 in_channels,
                 norm_cfg,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(PEMASKNeck, self).__init__(**kwargs)
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = in_channels[::-1]
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(zip(self.in_channels, self.up_sample_channels)):
            if index == 0:
                self.conv_list.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=up_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        act_cfg=None
                    ))
            else:
                self.conv_list.append(
                    UpSample(skip_input=in_channel + up_channel_temp,
                             output_features=up_channel,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg))
            up_channel_temp = up_channel

        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        temp_feat_list = []
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)
        out = self.sigmoid(self.conv1(temp_feat_list[-1]))
        return out,None


@NECKS.register_module()
class LightDYNAMICPEMASKNeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(LightDYNAMICPEMASKNeck, self).__init__()
        self.convfinal_attn = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.convfinal_dynamic = nn.Conv2d(64, 11, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(1536, 64, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(768, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(384, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        x0,x1,x2,x3,x4 = inputs[::-1]
        x0 = self.conv0(x0)
        x0 = F.interpolate(x0, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)
        x3 = F.interpolate(x3, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        x4 = self.conv4(x4)
        x = x0+x1+x2+x3+x4
        return self.sigmoid(self.convfinal_attn(x)),self.convfinal_dynamic(x)




@NECKS.register_module()
class DYNAMICPEMASKNeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self,
                 up_sample_channels,
                 in_channels,
                 norm_cfg,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(DYNAMICPEMASKNeck, self).__init__(**kwargs)
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = in_channels[::-1]
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 11, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(zip(self.in_channels, self.up_sample_channels)):
            if index == 0:
                self.conv_list.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=up_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        act_cfg=None
                    ))
            else:
                self.conv_list.append(
                    UpSample(skip_input=in_channel + up_channel_temp,
                             output_features=up_channel,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg))
            up_channel_temp = up_channel

        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        temp_feat_list = []
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)
        out = self.sigmoid(self.conv1(temp_feat_list[-1]))
        return out,self.conv2(temp_feat_list[-1])