from typing import Union, Optional
from functools import partial

import torch
import torch_scatter
import torch.nn as nn

from addict import Dict
from torch import Tensor
from einops import rearrange
from pm.utils.misc import offset2batch,batch2offset
from mamba_ssm.modules.mamba_simple import Mamba

# 直接使用mamba推荐的Block, 不像Point Mamba抄过来! 
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block 
from pm.utils.point_cloud import PointCloud, group_by_fps_knn_

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Grouper(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    
    def forward(self, pc:PointCloud):
        return group_by_fps_knn_(pc, self.num_group, self.group_size)

class Feature_Encoder(nn.Module):        # 改自Point Mamba！
    def __init__(self, encoder_channel):
        super().__init__()
        print(encoder_channel)
        self.e_o = encoder_channel       # 特征编码输出的通道数！
        self.e_i = 128                   # 特征编码内部使用的通道数！
        self.first_conv = nn.Sequential(
            nn.Linear(3,self.e_i), 
            nn.LayerNorm(self.e_i),
            nn.GELU(),                       
            nn.Linear(self.e_i, self.e_i *2)
        )
        self.second_conv = nn.Sequential(
            nn.Linear(self.e_i *4, self.e_i *4),
            nn.LayerNorm(self.e_i *4 ),
            nn.GELU(),
            nn.Linear(self.e_i *4, self.e_o)
        )

    def forward(self, feature):
        '''
            point_groups : BG N 3  ( N 邻居的数量)
            -----------------
            feature_global : BG C
        '''
        BG, N, C = feature.shape
        # encoder                                                     # 
        feature = self.first_conv(feature)                            # BG N 3  -> BG N e_i*2 
        feature_global = torch.max(feature, dim=1, keepdim=True)[0]   # BG N e_i*2 -> BG 1 e_i*2  # 为什么是max?
        feature_global = feature_global.expand(-1, N, -1)             # BG 1 e_i*2 -> BG N e_i*2
        feature = torch.cat([feature, feature_global], dim=-1)        # BG N e_i*2 BG N e_i*2  -> BG N e_i*4
        feature = self.second_conv(feature)                           # BG N e_i*4 -> BG N C
        feature_global = torch.max(feature, dim=1, keepdim=False)[0]  # BG N C -> BG C
        return feature_global

class Pos_Encoder(nn.Module):  # 位置也编码!! 先放到这，肯定要修改的！
    def __init__(self, encoder_channel):
        super().__init__()
        self.e_o = encoder_channel
        self.e_i = 128
        self.encoder = nn.Sequential(
            nn.Linear(3, self.e_i *2),   # 如果换成SparseConv的化， 就是所谓的xCPE！ See PVT3
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
    我看很多有关Mamba的网络，基本都是抄改这一块！！！没必要全文摘抄。直接将对应缺省值的分支留下就可以了！
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
        block = Block(d_model, mixer_cls, mlp_cls , norm_cls=norm_cls,)  # 可以了解，Block里的缺省路径 Add -> LN -> Mixer
        # block.layer_idx = layer_idx
        return block
    
    def forward(self, hidden_states):
        residual = None
        feature_list = []
        for idx, block in enumerate(self.blocks):
            hidden_states, residual = block( hidden_states, residual)
            if idx in self.out_indices:  # 此时就需要补一个 Add -> LN 过程！！！ 才能得到合适的hidden_state output!
                r_o = (hidden_states + residual) if residual is not None else hidden_states
                h_o = self.norm_f(r_o.to(dtype=self.norm_f.weight.dtype))
                feature_list.append(h_o)
        return feature_list            

class Decoder(nn.Module):
    """
    解码，不能采用PointMlp的上采样的方式。
    """
    def __init__(self, config):
        super().__init__()

    def forward(self, ):
        pass

class PointSIS_FollowMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config

        self.grouper = Grouper(config.num_group, config.group_size)        
        self.feature_encoder = Feature_Encoder(config.feature_dims)  # 其实config.feature_dims == config.pos_dim
        self.pos_encoder = Pos_Encoder(config.pos_dims)
        self.mixers = MixerLayers(config)
        self.fuse_e = nn.Sequential(                                 # 将两个编码合并！！！
            nn.Linear(config.feature_dims+config.pos_dims, config.feature_dims),
            nn.LayerNorm(config.feature_dims),
            nn.GELU(),
            nn.Linear(config.feature_dims, config.d_model)
        )

        self.fuse_f = nn.Sequential(                                 #合并各层的特征。
            nn.Linear(config.d_model * len(config.out_indices), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        self.fuse_o = nn.Sequential(                                 #合并各排序的特征。
            nn.Linear(config.d_model * len(config.order), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

    def forward(self, data_dict):
        parent_point = PointCloud(data_dict)
        # parent_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)  # 父点集，就没必要序列化了！
        #s_idx, s_n, s_xyz, s_order, s_inverse = self.grouper(parent_point)
        s_point = self.grouper(parent_point)
        s_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        s_feat  = s_point.feat 
        s_coord = s_point.coord 
        s_order = s_point.serialized_order 
        s_inverse = s_point.serialized_inverse        
        
        b_s = s_point.batch[-1]+1
        o_s = len(self.order)        
        # s_order, s_inverse 均为 order_size batch_size*num_group
        s_feat = self.feature_encoder(s_feat)                   # => batch_size*num_group d_model
        s_coord = self.pos_encoder(s_coord)                   # => batch_size*num_group d_model  
        s = torch.cat([s_feat, s_coord], dim= -1)
        s = self.fuse_e(s)                                 # 融合各编码 => batch_size*num_group d
        s = s.unsqueeze(0).repeat(o_s,1,1)                 # 为每种排序准备排序的数据 => order_size batch_size*num_group d
        s = torch_scatter.scatter(s,index=s_order, dim=1)  # 排序 => order_size batch_size*num_group d        
        s = rearrange(s, "o (b g) d -> b (o g) d", b=b_s)  # 将各个排序拼接,参看PointMamba的第四版!!
        s = self.mixers(s)                                 # 返回的是抽取的几个层的返回结果的列表 
        s = torch.cat(s, dim= -1)                          # 将各层mamba的结果，拼接！！
        s = self.fuse_f(s)                                 # 融合各结果 
        s = rearrange(s, "b (o g) d -> o (b g) d", o=o_s)  # 拆分各排序
        s = torch_scatter.scatter(s,index=s_inverse, dim=1)# 逆排序 => order_size batch_size*num_group d
        s = rearrange(s, "o (b g) d -> b g (o d)", b= b_s) # 调整，将同一点，在各排序情况下的特征拼接到一起！
        s = self.fuse_o(s)                                 # 后面咋办？
        return s

