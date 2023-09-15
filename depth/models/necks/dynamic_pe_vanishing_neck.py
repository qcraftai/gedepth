# Copyright (c) OpenMMLab. All rights reserved.
from atexit import register
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F
from depth.ops import resize
from depth.models.builder import NECKS
import torch
from mmcv.runner import BaseModule
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from IPython import embed
import numpy as np
import math
import cv2
# from skimage.measure import label, regionprops
# from depth.models._cdht.dht_func import C_dht

def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None
    
    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H-1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W-1, y)
    else:
        k = np.tan(angle)
        if y-k*x >=0 and y-k*x < H:  #left
            if point1 == None:
                point1 = (0, int(y-k*x))
            elif point2 == None:
                point2 = (0, int(y-k*x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k*(W-1)+y-k*x >= 0 and k*(W-1)+y-k*x < H: #right
            if point1 == None:
                point1 = (W-1, int(k*(W-1)+y-k*x))
            elif point2 == None:
                point2 = (W-1, int(k*(W-1)+y-k*x)) 
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k >= 0 and x-y/k < W: #top
            if point1 == None:
                point1 = (int(x-y/k), 0)
            elif point2 == None:
                point2 = (int(x-y/k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k+(H-1)/k >= 0 and x-y/k+(H-1)/k < W: #bottom
            if point1 == None:
                point1 = (int(x-y/k+(H-1)/k), H-1)
            elif point2 == None:
                point2 = (int(x-y/k+(H-1)/k), H-1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None : point2 = point1
    return point1, 


def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []
    embed()
    exit()
    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points



class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.dht = DHT(numAngle=numAngle, numRho=numRho)
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.fist_conv(x)
        x = self.dht(x)
        x = self.convs(x)
        return x

class DHT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT, self).__init__()       
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        accum = self.line_agg(x)
        return accum



@NECKS.register_module()
class DynamicPEHoughNeck(BaseModule):
    """PEMASKNeck.
    """

    def __init__(self):
        super(DynamicPEHoughNeck, self).__init__()
        self.dht_detector1 = DHT_Layer(1536, 64, numAngle=11, numRho=22)
        self.dht_detector2 = DHT_Layer(768, 64, numAngle=22, numRho=44)
        self.dht_detector3 = DHT_Layer(384, 64, numAngle=44, numRho=88)
        self.dht_detector4 = DHT_Layer(192, 64, numAngle=88, numRho=176)
        self.dht_detector5 = DHT_Layer(64, 64, numAngle=176, numRho=352)


        self.last_conv = nn.Sequential(
                nn.Conv2d(320, 1, 1)
            )
    
    def upsample_cat(self, x0,x1,x2,x3,x4):
        x0 = nn.functional.interpolate(x0, size=(176, 352), mode='bilinear', align_corners=True)
        x1 = nn.functional.interpolate(x1, size=(176, 352), mode='bilinear', align_corners=True)
        x2 = nn.functional.interpolate(x2, size=(176, 352), mode='bilinear', align_corners=True)
        x3 = nn.functional.interpolate(x3, size=(176, 352), mode='bilinear', align_corners=True)
        # x4 = nn.functional.interpolate(x4, size=(100, 100), mode='bilinear', align_corners=True)
        return torch.cat([x0,x1,x2,x3,x4], dim=1)

    # init weight
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        
            
    def forward(self, inputs):
        x0,x1,x2,x3,x4 = inputs[::-1]
        # embed()
        # exit()
        x0 = self.dht_detector1(x0)
        x1 = self.dht_detector2(x1)
        x2 = self.dht_detector3(x2)
        x3 = self.dht_detector4(x3)
        x4 = self.dht_detector5(x4)
        cat = self.upsample_cat(x0,x1,x2,x3,x4)
        embed()
        exit()
        logist = self.last_conv(cat)

        key_points = torch.sigmoid(logist)
        binary_kmap = key_points[0].detach().squeeze().cpu().numpy() > 0.01
        kmap_label = label(binary_kmap, connectivity=1)
        props = regionprops(kmap_label)
        plist = []
        for prop in props:
            plist.append(prop.centroid)
        b_points = reverse_mapping(plist, numAngle=176, numRho=352, size=(176, 352))
        return logist