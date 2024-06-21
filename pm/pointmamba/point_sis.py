from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
# import torch
import torch.nn as nn
from torch import Tensor
import spconv.pytorch as spconv

from torch_geometric.nn import fps
from torch_geometric.nn import knn
from pm.utils.misc import offset2batch,batch2offset

# from timm.models.layers import trunc_normal_
# from timm.models.layers import DropPath

# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

# from knn_cuda import KNN

from pm.pointmamba import PointCloud,PCModule,PCSequential
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

class Embedding(PCModule):  # 留着,看看有没有用!
    def __init__(
        self,
        in_channels,
        embed_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PCSequential(
            # conv=spconv.SubMConv3d(
            #     in_channels,
            #     embed_channels,
            #     kernel_size=5,
            #     padding=1,
            #     bias=False,
            #     indice_key="stem",
            # ),
            proj = nn.Linear(self.in_channels, self.embed_channels),
            bn_layer = nn.BatchNorm1d(self.embed_channels), #ln_layer = nn.LayerNorm
            act_layer = nn.GELU()
        )

    def forward(self, point: PointCloud):
        point = self.stem(point)
        return point
    

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

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0]
        )

    def forward(self, data_dict):
        point = PointCloud(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        return point


