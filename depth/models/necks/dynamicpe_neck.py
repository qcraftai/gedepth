# Copyright (c) OpenMMLab. All rights reserved.
from atexit import register
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F
from depth.ops import resize
from depth.models.builder import NECKS
from depth.models.backbones.hrnet import Bottleneck, ResLayer
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
# @NECKS.register_module()
# class DynamicPENeck(BaseModule):
#     """PEMASKNeck.
#     """

#     def __init__(self):
#         super(DynamicPENeck, self).__init__()
#         self.convfinal = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
#         self.conv0 = nn.Conv2d(1536, 64, kernel_size=3, padding=1, stride=1)
#         self.conv1 = nn.Conv2d(768, 64, kernel_size=3, padding=1, stride=1)
#         self.conv2 = nn.Conv2d(384, 64, kernel_size=3, padding=1, stride=1)
#         self.conv3 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

#         self.mlp = nn.Sequential(
#             nn.Linear(61952, 3872),
#             nn.ReLU(),
#             nn.Linear(3872, 242),
#             nn.ReLU(),
#             nn.Linear(242, 36)
#         )        
#         self.softmax = nn.Softmax(dim=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((176, 352))

#     # init weight
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')
            
#     def forward(self, inputs):
#         # torch.autograd.set_detect_anomaly(True)
#         x0,x1,x2,x3,x4 = inputs[::-1]
#         embed()
#         exit()
#         x0 = self.conv0(x0)
#         x0 = F.interpolate(x0, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
#         x1 = self.conv1(x1)
#         x1 = F.interpolate(x1, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
#         x2 = self.conv2(x2)
#         x2 = F.interpolate(x2, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
#         x3 = self.conv3(x3)
#         x3 = F.interpolate(x3, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
#         x4 = self.conv4(x4)
#         x = self.convfinal(x0+x1+x2+x3+x4)
#         x = self.avgpool(x)
#         # x = F.interpolate(x,size=[176, 352],mode='bilinear')
#         x = x.view(-1,61952)
#         x=self.mlp(x)
#         return x

@NECKS.register_module()
class DynamicATTNPENeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self,
                 out_channels=5,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(DynamicATTNPENeck, self).__init__(init_cfg=init_cfg)
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels
        out_channels = [64,128,64]
        downsample_layers = []
        for i in range(2):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                )
            )
        self.downsample_layers = nn.ModuleList(downsample_layers)

        self.final_layer = ConvModule(
            in_channels=64,
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # self.final_layer2 = ConvModule(
        #     in_channels=5,
        #     out_channels=5,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     bias=False,
        # )

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            
    def forward(self, inputs):
        # inputs = inputs.detach()
        for i in range(len(self.downsample_layers)):
            if i == 0:
                feat = self.downsample_layers[i](inputs)
            else:
                feat = self.downsample_layers[i](feat)
        feat = self.final_layer(feat)
        feat = self.maxpool(feat)
        # feat = self.final_layer2(feat)
        return feat.view(feat.size(0),-1)



@NECKS.register_module()
class DynamicPENeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(DynamicPENeck, self).__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [96, 192, 384, 768, 1536]

        self.increase_layers = ConvModule(
            in_channels=64,
            out_channels=out_channels[0],
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

        downsample_layers = []
        for i in range(len(out_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                )
            )
        self.downsample_layers = nn.ModuleList(downsample_layers)

        self.final_layer = ConvModule(
            in_channels=out_channels[4],
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )
        self.relu=nn.Tanh()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(2048, 1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(2048, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 9)
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(2048, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 9)
        # )      
          

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            
    # def forward(self, inputs_ori):
    #     inputs = [x.detach() for x in inputs_ori]
    def forward(self, inputs):
        feat = self.increase_layers(inputs[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat) + inputs[i + 1]
        feat = self.final_layer(feat)
        x = self.avgpool(feat)
        x = x.view(feat.size(0),-1)
        return self.relu(self.mlp(x))


@NECKS.register_module()
class DynamicPENeckBACKBONE(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(DynamicPENeckBACKBONE, self).__init__(init_cfg=init_cfg)
        in_channels = [18, 36, 72, 144]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [128, 256, 512, 1024]

        increase_layers = []
        for i in range(len(in_channels)):
            increase_layers.append(
                ResLayer(
                    block_type,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    num_blocks=1,
                    stride=1,
                ))
        self.increase_layers = nn.ModuleList(increase_layers)

        # Downsample feature maps in each scale.
        downsample_layers = []
        for i in range(len(in_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                ))
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # The final conv block before final classifier linear layer.
        self.final_layer = ConvModule(
            in_channels=out_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 9)
            
    def forward(self, inputs):
        feat = self.increase_layers[0](inputs[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat) + \
                self.increase_layers[i + 1](inputs[i + 1])
        feat = self.final_layer(feat)
        x = self.avgpool(feat)
        x = x.view(feat.size(0),-1)
        return self.fc(x)

@NECKS.register_module()
class DynamicPENeckSOFT2(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(DynamicPENeckSOFT2, self).__init__()
        self.convfinal = nn.Conv2d(64, 11, kernel_size=3, padding=1, stride=1)
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
        
            
    # def forward(self, inputs):
    def forward(self, inputs_ori):
        inputs = [x.detach() for x in inputs_ori]
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
        return self.convfinal(x)


@NECKS.register_module()
class DynamicPENeckSOFTHRNET(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(DynamicPENeckSOFTHRNET, self).__init__()
        self.convfinal = nn.Conv2d(18, 11, kernel_size=3, padding=1, stride=1)
        self.conv0 = nn.Conv2d(144, 18, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(72, 18, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(36, 18, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(18, 18, kernel_size=3, padding=1, stride=1)
        
        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        # torch.autograd.set_detect_anomaly(True)
        x0,x1,x2,x3 = inputs[::-1]
        x0 = self.conv0(x0)
        x0 = F.interpolate(x0, size=[x3.shape[2], x3.shape[3]], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=[x3.shape[2], x3.shape[3]], mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=[x3.shape[2], x3.shape[3]], mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)
        x = x0+x1+x2+x3
        return self.convfinal(x)




@NECKS.register_module()
class DynamicPENeckSOFTDDR(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(DynamicPENeckSOFTDDR, self).__init__()
        # self.convfinal = nn.Conv2d(18, 11, kernel_size=3, padding=1, stride=1)
        # self.conv0 = nn.Conv2d(144, 18, kernel_size=3, padding=1, stride=1)
        # self.conv1 = nn.Conv2d(72, 18, kernel_size=3, padding=1, stride=1)
        # self.conv2 = nn.Conv2d(36, 18, kernel_size=3, padding=1, stride=1)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 11, kernel_size=1),
        )

        
        

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, x):
        return self.fuse_conv(x)

@NECKS.register_module()
class DYNAMICPENeckHeavy(BaseModule):
    """
    """

    def __init__(self,
                 up_sample_channels,
                 in_channels,
                 norm_cfg,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(DYNAMICPENeckHeavy, self).__init__(**kwargs)
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = in_channels[::-1]
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1 = nn.Conv2d(64, 11, kernel_size=3, padding=1, stride=1)
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
        out = self.conv1(temp_feat_list[-1])
        return out

@NECKS.register_module()
class DynamicPENeckSOFT(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(DynamicPENeckSOFT, self).__init__()
        self.convfinal = nn.Conv2d(64, 11, kernel_size=3, padding=1, stride=1)
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
    # def forward(self, inputs_ori):
        # inputs = [x.detach() for x in inputs_ori]
        # attn_mask[attn_mask<0.1] = 0
        # attn_mask[attn_mask>0.3] = 1
        # inputs = [x.detach()*F.interpolate(attn_mask,size=[x.shape[2], x.shape[3]],mode='bilinear') for x in inputs_ori]
        x0,x1,x2,x3,x4 = inputs[::-1]
        x0 = self.conv0(x0)
        x0 = F.interpolate(x0, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        # attn_mask_x0 = F.interpolate(attn_mask,size=[x0.shape[2], x0.shape[3]],mode='bilinear')
        # x0 = x0*attn_mask_x0
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        # attn_mask_x1 = F.interpolate(attn_mask,size=[x1.shape[2], x1.shape[3]],mode='bilinear')
        # x1 = x1*attn_mask_x1
        x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        # attn_mask_x2 = F.interpolate(attn_mask,size=[x2.shape[2], x2.shape[3]],mode='bilinear')
        # x2 = x2*attn_mask_x2
        x3 = self.conv3(x3)
        x3 = F.interpolate(x3, size=[x4.shape[2], x4.shape[3]], mode='bilinear', align_corners=True)
        # attn_mask_x3 = F.interpolate(attn_mask,size=[x3.shape[2], x3.shape[3]],mode='bilinear')
        # x3 = x3*attn_mask_x3
        x4 = self.conv4(x4)
        # attn_mask_x4 = F.interpolate(attn_mask,size=[x4.shape[2], x4.shape[3]],mode='bilinear')
        # x4 = x4*attn_mask_x4
        x = x0+x1+x2+x3+x4
        return self.convfinal(x)