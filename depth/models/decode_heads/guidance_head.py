from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding
from torch.nn.modules import conv
from depth.ops import resize
from depth.models.builder import HEADS
import torch.nn.functional as F

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

@HEADS.register_module()
class GuidanceHead(nn.Module):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
        fpn (bool): Whether apply FPN head.
            Default: False
        conv_dim (int): Default channel of features in FPN head.
            Default: 256.
    """

    def __init__(self,
                 up_sample_channels,
                 in_channels,
                 act_cfg,
                 min_depth,
                 fpn=False,
                 conv_dim=256,
                 sigmoid=False,
                 **kwargs):
        super(GuidanceHead, self).__init__(**kwargs)
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = in_channels[::-1]
        self.act_cfg = act_cfg
        self.norm_cfg = None
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        self.min_depth = min_depth

        self.conv_depth = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid_flags = sigmoid
        if self.sigmoid_flags:
            self.sigmoid = nn.Sigmoid()
            self.act_cfg = None


        for index, (in_channel, up_channel) in enumerate(
                zip(self.in_channels, self.up_sample_channels)):
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

            # save earlier fusion target
            up_channel_temp = up_channel

    def forward(self, inputs,pe_mask,depth_mask_y):
        """Forward function."""        
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

        depth_temp = self.relu(self.conv_depth(temp_feat_list[-1]))
        if not pe_mask is None:
            pe = resize(
                input=pe_mask,
                size=depth_temp.shape[2:],
                mode='bilinear',
                align_corners=True,
                warning=False)
            depth_y = resize(
                input=depth_mask_y,
                size=depth_temp.shape[2:],
                mode='bilinear',
                align_corners=True,
                warning=False)
            depth_y[pe==0] = 0
            # output = depth_temp
            if self.sigmoid_flags:
                output = self.sigmoid(depth_temp)
            else:
                output = ((depth_temp*(1-depth_y))+(pe*85))+ self.min_depth
        else:
            output = depth_temp + self.min_depth
        return output
