from typing import Union, Optional
from functools import partial

import torch
import torch_scatter
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from pm.utils.misc import offset2batch,batch2offset

from mamba_ssm.modules.mamba_simple import Mamba

# 直接使用mamba推荐的Block, 不像Point Mamba抄过来! 
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block 

from pm.utils.point_cloud import PointCloud
"""
用Mamba来处理点云,目前看到的, 有下面的几项工作:
1) PointMamba:这哥们(好像还是Baidu的!!!).到第四版,参考了PTV3的结构化思路后,按他自己的说法,又跑到PCM,Mamba3D的前头.
2) Point Cloud Mamba:从PointMLP出发的Mamba
3) Mamba3D: 说他的Local Norm Pooling(LNP)是相较PointMamba的优点!
4) Serialized Point Mamba: 感觉在灌水！是个组合体。PTv3+PointMamba
5) PoinTramba:
6) Point Mamba:(上海交大的) 走的是OctTree的序列化路线！

另外Point Transformer V3的工作值得注意(尽管他是在Transformer上的工作.)!
尺度和结构化,是他想解决的问题! 
1)各种点云的尺度,跨度很大!
2)点云是不规则数据集
3)放弃点集和空间逼近的思路(KNN), 用SFC方法,来结构化点云!(可能这个会影响一批模型)
  再结合Swin的做法，逐步扩大RF，可以处理很长的序列！！
理解：(寻求一个合理的点云遍历方法，反而放到重要的位置！)

另外的一条路线：TSegFormer

在Voxel影像上,还看到以下几项工作:
1)SegMamba
2)Voxel Mamba
3)nnMamba


我这里的缩写,来之传说"Space Is a latent Sequence".
"""

@torch.no_grad()
def group_by_fps_knn(xyz_pc:PointCloud, 
                   num_group:int,  # 分多少个组                  # 其实取多少点！
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
    s_idx = fps(xyz_pc.coord, xyz_pc.offset, s_offset)  # [batch_size*num_group ]  # 幸亏这个fps

    # 此时 还不用考虑mesh提供的norm作为特征的提取基础，直接用坐标和临域关系来构造特征！
    s_xyz  = xyz_pc.coord[s_idx]                                                     # [batch_size*num_group, coord's dim]
    s_n_idx, _dist = knn(group_size, xyz_pc.coord, xyz_pc.offset, s_xyz,s_offset)    ## [batch_size*num_group, group_size ], _
    s_n = xyz_pc.coord[s_n_idx]                                                      # [batch_size*num_group , group_size, coord's dim]
    s_n = s_n - s_xyz.unsqueeze(1)                                                   # [batch_size*num_group , group_size, vector's dim]
    s_n = s_n[:,1:, :]                                                               # 需不需要,去掉组内第一个vector? 
    
    #排序,根据原有的SFC遍历序好,获得采样点的各总次序!
    s_order = torch.argsort(xyz_pc.serialized_code[:, s_idx])                                           # 获得样本的各种序列吗, 种类排序! [order_s, batch_size * num_group]
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
        point = PointCloud(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        b_s = point.batch[-1]+1
        o_s = len(self.order)
        s_idx, s_n, s_xyz, s_order, s_inverse = self.grouper(point)        
        # s_order, s_inverse 均为 order_size batch_size*num_group
        s_n = self.feature_encoder(s_n)                   # => batch_size*num_group feature_size
        s_xyz = self.pos_encoder(s_xyz)                   # => batch_size*num_group feature_size  
        s = torch.cat([s_n, s_xyz], dim= -1)
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

