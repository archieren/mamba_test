from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from pm.utils.misc import offset2batch,batch2offset

# from timm.models.layers import trunc_normal_
# from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba

# 直接使用mamba推荐的Block, 不像Point Mamba抄过来! 
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block 
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

# from knn_cuda import KNN

from pm.utils.point_cloud import PointCloud
from pm.pointmamba import PCModule
"""
用Mamba来处理点云,目前看到的, 有下面的几项工作:
1) PointMamba:这哥们(好像还是Baidu的!!!).到第四版,参考了PTV3的结构化思路后,按他自己的说法,又跑到PCM,Mamba3D的前头.
2) Point Cloud Mamba:从PointMLP出发的Mamba
3) Mamba3D: 说他的Local Norm Pooling(LNP)是相较PointMamba的优点!

另外Point Transformer V3的工作值得注意(尽管他是在Transformer上的工作.)!尺度和结构化,是他想解决的问题! 
1)各种点云的尺度,跨度很大!
2)点云是不规则数据集
3)放弃点集和空间逼近的思路(KNN), 用SFC方法,来结构化点云!(可能这个会影响一批模型)

在Voxel影像上,还看到以下几项工作:
1)SegMamba
2)Voxel Mamba


我这里的缩写,来之传说"Space Is a latent Sequence".
"""

# @torch.inference_mode()
def group_by_fps_knn(xyz_pc:PointCloud, 
                   num_group:int,  # 分多少个组
                   group_size:int, # 组内多少个元素
                   ): 
    # 不得不面临将点云都搞成传统的Batch模式!
    # 序列分段的方式,是另一思路.先不考虑!
    # 假设xyx_pc已经serialized!
    # 这个地方很花了点时间.基本功不牢呀! index broadcasting 和 scatter_, gather的关系!
    from pointops import knn_query as knn
    from pointops import farthest_point_sampling as fps
    
    # pointops方式 :按量,返回结果还包括距离
    # s_ 解读为 samples, n_ 解读为neighbors, o_解读为ordered.
    # batch_size = xyz_pc.batch[-1] + 1
    s_offset = (torch.ones_like( xyz_pc.batch_bin)* num_group).cumsum(0).int() #[batch_size]
    s_idx = fps(xyz_pc.coord, xyz_pc.offset, s_offset)  # [batch_size*num_group ] 

    s_xyz  = xyz_pc.coord[s_idx]  # [batch_size*num_group, coord's dim]
    s_n_idx, _dist = knn(group_size, xyz_pc.coord, xyz_pc.offset, s_xyz,s_offset)    ## [batch_size*num_group, group_size ], _
    s_n = xyz_pc.coord[s_n_idx]  # [batch_size*num_group , group_size, coord's dim]
    s_n = s_n - s_xyz.unsqueeze(1)  # [batch_size*num_group , group_size, vector's dim]
    s_n = s_n[:,1:, :]        # 需不需要,去掉组内第一个vector?
    
    #排序,根据原有的排序信息,获得排序!
    s_order = torch.argsort(xyz_pc.serialized_code[:, s_idx])    # 获得样本的各种序列吗, 种类排序! [order_s, batch_size * num_group]
    src=torch.arange(0, s_order.shape[1], device=s_order.device).repeat(s_order.shape[0], 1)
    s_inverse = torch.zeros_like(s_order, device=s_order.device).scatter_(dim=1,index=s_order,src=src,) # [order_s, batch_size * num_group]
    # assert s_inverse[0, s_order[0, i]] == i
    # s_idx[s_order].gather(1, s_inverse)- s_idx 等于零矩阵!!! 注意这个关系!!!

    # s_o_xyz = s_xyz[s_order]      # [order_s, batch_size * num_group , coord's dim]
    # s_o_n = s_n[s_order]          # [order_s, batch_size * num_group , group_size, coord's dim]
    # return s_o_n, s_o_xyz, s_idx, s_xyz, s_order, s_inverse

    # s_idx是样本和数据之间的对应桥梁!!!
    return s_idx, s_n, s_xyz, s_order, s_inverse

class Grouper(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    
    def forward(self, pc:PointCloud):
        return group_by_fps_knn(pc, self.num_group, self.group_size)

class Feature_Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        print(encoder_channel)
        self.e_c = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, self.e_c, 1),
            nn.BatchNorm1d(self.e_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.e_c, self.e_c *2 , 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.e_c *4, self.e_c *4, 1),
            nn.BatchNorm1d(self.e_c *4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.e_c *4, self.e_c, 1)
        )

    def forward(self, feature):
        '''
            point_groups : BG N 3
            -----------------
            feature_global : BG C
        '''
        bg, n, _ = feature.shape
        # encoder
        feature = feature.transpose(-1, -2)  # 调整形式!
        #print("1",feature.shape)
        feature = self.first_conv(feature)
        #print("2",feature.shape)
        feature_global = torch.max(feature, dim=-1, keepdim=True)[0]
        #print("3",feature_global.shape)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=-2)
        #print("4",feature.shape)
        feature = self.second_conv(feature)
        #print("5",feature.shape)
        feature_global = torch.max(feature, dim=-1, keepdim=False)[0]
        return feature_global

class Pos_Encoder(nn.Module):  # 位置也编码!!
    def __init__(self, encoder_channel):
        super().__init__()
        self.e_c = encoder_channel
        self.encoder = nn.Sequential(
            nn.Linear(3, self.e_c *2),
            nn.GELU(),
            nn.Linear(self.e_c * 2, self.e_c)            
        )
    
    def forward(self, pos):
        """
        BG 3 -> BG C
        """
        return self.encoder(pos)

# class Block(nn.Module):
#     def __init__(self, dim, config,layer_idx):
#         super().__init__()
#         self.stem = Mamba(d_model=dim, **config.mamba_config, layer_idx=layer_idx)
    
#     def forward(self, s):
#         """
#         B G C -> B G C
#         """
#         s = self.stem(s)
#         return s

def create_block(d_model, ssm_cfg, layer_idx):  # 参考Point_Mamba里的,去掉所有的缺省值设置!
    norm_epsilon=1e-5 
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(nn.LayerNorm , eps=norm_epsilon)
    block = Block(d_model, mixer_cls, mlp_cls=nn.Identity , norm_cls=norm_cls,)
    # block.layer_idx = layer_idx
    return block


class PointSIS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config
        dim = config.trans_dim

        self.grouper = Grouper(config.num_group, config.group_size)
        self.feature_encoder = Feature_Encoder(config.feature_dims)  # 其实config.feature_dims == config.pos_dim
        self.pos_encoder = Pos_Encoder(config.pos_dims)

        self.layers = nn.ModuleList([create_block(dim, config.mamba_config, layer_idx) for layer_idx in range(config.depth)])


    def forward(self, data_dict):
        point = PointCloud(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        b_s = point.batch[-1]+1
        s_idx, s_n, s_xyz, s_order, s_inverse = self.grouper(point)
        s_n = self.feature_encoder(s_n)
        s_xyz = self.pos_encoder(s_xyz)
        s = s_n + s_xyz
        s = s[s_order]
        print(s.shape)
        #s = rearrange(s, "o (b g) d -> (o b) g d", b=b_s)
        s = rearrange(s, "o (b g) d -> b (o g) d", b=b_s)  # 将两个排序拼接,参看PointMamba的第四版!!
        print(s.shape)
        hidden_states =s
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer( hidden_states, residual)
            #hidden_states = self.drop_out_in_block(hidden_states)

        print(hidden_states.shape)
        return hidden_states


