from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import spconv.pytorch as spconv

from pm.utils.misc import offset2batch,batch2offset

# from timm.models.layers import trunc_normal_
# from timm.models.layers import DropPath

# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

# from knn_cuda import KNN

from pm.utils.point_cloud import PointCloud
from pm.pointmamba import PCModule
"""
用Mamba来处理点云,目前看到的, 有下面的几项工作:
1) PointMamba:这哥们(好像还是Baidu的!!!)有点灌水,到第四版,又参考了PTV3的结构化思路.
2) Point Cloud Mamba:从PointMLP出发的Mamba
3) Mamba3D: 说他的Local Norm Pooling(LNP)是相较PointMamba的优点!

另外一项工作 Voxel Mamba:


另外Point Transformer V3的工作值得注意(尽管他是在Transformer上的工作.)!尺度和结构化,是他想解决的问题! 
1)各种点云的尺度,跨度很大!
2)点云是不规则数据集
3)放弃点集和空间逼近的思路(KNN), 用SFC方法,来结构化点云!

我这里的缩写,来之传说"Space Is a latent Sequence".
"""

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointSIS(PCModule):
    def __init__(
        self,
        in_channels=6,
        #Spatial Filling Curve!
        order=["z", "z-trans"],
        shuffle_orders=True,
        #Fatures encoding!
        enc_channels=(32, 64, 128, 256, 512),
        cls_mode=False,
        # Prompting
    ):
        super().__init__()
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
    def forward(self, data_dict):
        point = PointCloud(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.grouping()

        return point


