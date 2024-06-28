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
#from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

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
        self.e_o = encoder_channel
        self.e_i = 128
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, self.e_i, 1),
            nn.BatchNorm1d(self.e_i),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.e_i, self.e_i *2 , 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.e_i *4, self.e_i *4, 1),
            nn.BatchNorm1d(self.e_i *4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.e_i *4, self.e_o, 1)
        )

    def forward(self, feature):
        '''
            point_groups : BG N 3  ( N 邻居的数量)
            -----------------
            feature_global : BG C
        '''
        bg, n, _ = feature.shape
        # encoder
        feature = feature.transpose(-1, -2)  # 调整形式! BG N 3 -> BG 3 N
        feature = self.first_conv(feature)   # BG 3 N -> BG e_i N
        feature_global = torch.max(feature, dim=-1, keepdim=True)[0] # BG e_i N -> BG e_i 1
        feature_global = feature_global.expand(-1, -1, n)   # BG e_i 1 -> BG e_i N
        feature = torch.cat([feature_global, feature], dim=-2) # BG e_i N , BG e_i N -> BG e_i*2 N 
        feature = self.second_conv(feature)  # BG e_i*2 N -> BG C N
        feature_global = torch.max(feature, dim=-1, keepdim=False)[0] # BG C N -> BG C
        return feature_global

class Pos_Encoder(nn.Module):  # 位置也编码!!
    def __init__(self, encoder_channel):
        super().__init__()
        self.e_o = encoder_channel
        self.e_i = 128
        self.encoder = nn.Sequential(
            nn.Linear(3, self.e_i *2),
            nn.GELU(),
            nn.Linear(self.e_i * 2, self.e_o)            
        )
    
    def forward(self, pos):
        """
        BG 3 -> BG C
        """
        return self.encoder(pos)

class MixerLayers(nn.Module):
    """
    残差式板块栈。直接借用Mamba官方实现里面的Block。
    这个类应当对应...mixer_seq_simple...里的MixerModel
    我看很多有关Mamba的网络，基本都是抄改这一块！！！没必要。直接将对应缺省值的分支留下就可以了！
    """
    def __init__(self, config):
        super().__init__()
        self.out_indices = config.out_indices
        self.blocks = nn.ModuleList([self.create_block(config.d_model, config.mamba_config, layer_idx) 
                                     for layer_idx in range(config.depth)])
        self.norm_f = nn.LayerNorm(config.d_model)

    @staticmethod
    def create_block(d_model, mamba_cfg, layer_idx):  
        # 直接用Mamba里实现的Block，基本用缺省值!
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **mamba_cfg)
        norm_cls = partial(nn.LayerNorm )
        mlp_cls = nn.Identity
        block = Block(d_model, mixer_cls, mlp_cls , norm_cls=norm_cls,)
        # block.layer_idx = layer_idx
        return block
    
    def forward(self, hidden_states):
        residual = None
        feature_list = []
        for idx, block in enumerate(self.blocks):
            hidden_states, residual = block( hidden_states, residual)
            if idx in self.out_indices:
                r_o = (hidden_states + residual) if residual is not None else hidden_states
                h_o = self.norm_f(r_o.to(dtype=self.norm_f.weight.dtype))
                feature_list.append(h_o)
        return feature_list            

class PointSIS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config
        d_model = config.trans_dim

        self.grouper = Grouper(config.num_group, config.group_size)
        
        self.feature_encoder = Feature_Encoder(config.feature_dims)  # 其实config.feature_dims == config.pos_dim
        self.pos_encoder = Pos_Encoder(config.pos_dims)

        self.mixers = MixerLayers(config)

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
        s = rearrange(s, "o (b g) d -> b (o g) d", b=b_s)  # 将各个排序拼接,参看PointMamba的第四版!!
        s = self.mixers(s)

        print(s[-1].shape)
        return s[-1]


