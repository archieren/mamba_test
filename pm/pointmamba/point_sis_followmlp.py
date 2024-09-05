from functools import partial

import torch
import torch_scatter
import torch.nn as nn

from addict import Dict
from torch import Tensor
from einops import rearrange,einsum
from pm.utils.misc import offset2batch,batch2offset
from mamba_ssm.modules.mamba_simple import Mamba

# 直接使用mamba推荐的Block, 不像Point Mamba抄过来! 
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block 
from pm.pointmamba.conifuguration_point_sis import Mamba1Config, PointSISConfig
from pm.pointmamba.losses import PMLoss,tooth_lables
from pm.pointmamba.pointmask import MaskDecoder
from pm.utils.point_cloud import PointCloud, group_by_group_number
from pointops import interpolation2


class Grouper_By_NumGroup(nn.Module):   # TODO：这个应当改名。采样的时候，还生成了Feature！
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    
    def forward(self, pc:PointCloud):
        return group_by_group_number(pc, self.num_group, self.group_size)


class Feature_Encoder(nn.Module):        # 改自Point Mamba！
    def __init__(self, encoder_channel):
        super().__init__()
        self.e_o = encoder_channel       # 特征编码输出的通道数！
        self.e_i = 128                   # 特征编码内部使用的通道数！
        self.first_conv = nn.Sequential(
            nn.Linear(3,self.e_i),       # TODO:3
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
            nn.Linear(3, self.e_i *2),   # 如果换成SparseConv的化,就是所谓的xCPE！ See PVT3
            nn.GELU(),
            nn.Linear(self.e_i * 2, self.e_o)            
        )
    
    def forward(self, pos):
        """
        BG 3 -> BG C
        """
        return self.encoder(pos)

class FeatPropagation(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.k = group_size
        self.interpolation = interpolation2

    def forward(self, parent_pc:PointCloud, s_pc:PointCloud): 
        xyz = s_pc.coord
        offset = s_pc.offset
        new_xyz = parent_pc.coord        # 为什么这样， new_xyz是parent_pc.coord! 想明白这个，就明白底层算法了！
        new_offset = parent_pc.offset

        input = s_pc.feat
        output = self.interpolation(xyz, new_xyz, input, offset, new_offset, self.k)
        return output
    
class MixerLayers(nn.Module):
    """
    残差式板块栈。直接借用Mamba官方实现里面的Block。
    这个类应当对应...mixer_seq_simple...里的MixerModel
    我看很多有关Mamba的网络，基本都是抄改这一块！！！没必要全文摘抄。直接将对应缺省值的分支留下就可以了！
    """
    def __init__(self, d_model, depth, mamba_config):
        super().__init__()
        self.blocks = nn.ModuleList([self.create_block(d_model, mamba_config, layer_idx) 
                                     for layer_idx in range(depth)])
        self.norm_f = nn.LayerNorm(d_model)

    @staticmethod
    def create_block(d_model:int, mamba_cfg:Mamba1Config, layer_idx):  
        # 直接用Mamba里实现的Block，基本用缺省值!
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **mamba_cfg)
        norm_cls = partial(nn.LayerNorm )
        mlp_cls = nn.Identity
        block = Block(d_model, mixer_cls, mlp_cls , norm_cls=norm_cls,)  # 可以了解，Block里的缺省路径 Add -> LN -> Mixer
        # block.layer_idx = layer_idx
        return block
    
    def forward(self, hidden_states):
        residual = None
        for idx, block in enumerate(self.blocks):
            hidden_states, residual = block( hidden_states, residual)
         # 此时就需要补一个 Add -> LN 过程！！！ 才能得到合适的hidden_state output!
        hidden_states = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states           



class PointSIS_FollowMLP(nn.Module):
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config
        
        self.feature_encoder = Feature_Encoder(config.feature_dims)  # 其实config.feature_dims == config.pos_dim
        self.pos_encoder = Pos_Encoder(config.pos_dims)
        self.mixers = nn.ModuleList([MixerLayers(d_model=config.d_model, depth= d, mamba_config=config.mamba_config)
                                     for d in config.depth])
        
        self.fuse_e = nn.Sequential(                                 # 将两个嵌入编码融合！
            nn.Linear(config.feature_dims+config.pos_dims, config.feature_dims),
            nn.LayerNorm(config.feature_dims),
            nn.GELU(),                                               # TODO：看看融合后，要不要激活函数                           
            nn.Linear(config.feature_dims, config.d_model)
        )

        self.fuse_o = nn.Sequential(                                 #合并各排序的特征。
            nn.Linear(config.d_model * len(config.order), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),                                               # TODO：看看融合后，要不要激活函数
            nn.Linear(config.d_model, config.d_model)
        )        


    def forward(self, s_pc:PointCloud):                     
        # s_意味着初始下采样，已经做了
        # TODO：后面的多尺度下采样，看情况再做！
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)        
        s_feat  = s_pc.feat   # (b g) d
        s_coord = s_pc.coord   # (b g) 3
        s_order = s_pc.serialized_order   # o (b g)
        s_inverse = s_pc.serialized_inverse  # o (b g)      
        
        b_s = s_pc.batch[-1]+1
        o_s = len(self.order)        
        # s_order, s_inverse 均为 order_size batch_size*num_group
        s_coord = self.pos_encoder(s_coord)                # => (b g) d    # 每层共用这个位置编码！
        hidden_state = self.feature_encoder(s_feat)              # => (b g) d 
        hidden_states =[]
        for _ , mixlayer in enumerate(self.mixers):        
            hidden_state = torch.cat([hidden_state, s_coord], dim= -1)               # => (b g) (d*2)  # 坐标的嵌入，应当不变！
            hidden_state = self.fuse_e(hidden_state)                                 # 融合各编码 => (b g) d
            hidden_state = hidden_state.unsqueeze(0).repeat(o_s,1,1)                 # 为每种排序准备排序的数据 => o (b g) d
            hidden_state = torch_scatter.scatter(hidden_state,index=s_order, dim=1)  # 排序 => o (b g) d       
            hidden_state = rearrange(hidden_state, "o (b g) d -> b (o g) d", b=b_s)  # 将各个排序拼接,参看PointMamba的第四版!!
            hidden_state = mixlayer(hidden_state)                                    #  
            hidden_state = rearrange(hidden_state, "b (o g) d -> o (b g) d", o=o_s)  # 拆分各排序
            hidden_state = torch_scatter.scatter(hidden_state,index=s_inverse, dim=1)# 逆排序 => order_size batch_size*num_group d
            hidden_state = rearrange(hidden_state, "o (b g) d -> b g (o d)", b= b_s) # 调整，将同一点，在各排序情况下的特征拼接到一起！
            hidden_state = self.fuse_o(hidden_state)
            hidden_state = rearrange(hidden_state, "b g d -> (b g) d")               # (b g) d 
            hidden_states.append(hidden_state)                                       # [(b g) d, ]           
        s_pc.feat = hidden_states                                  # b (l g) d  TODO: 感觉PointCloud.feat这个域被重用了太多！   
        return s_pc
    
class PointSIS_Encoder(nn.Module):         
    """
    这一模块，是尝试性的。看看有无必要，将各层的结果联合起来做个处理！
    目前来看，只是作个简单的层间的相关性操作！还不能当作尺度间的attention！
    """
    def __init__(self, config:PointSISConfig) -> None:
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.num_group = config.num_group
        self.num_feature_levels = config.num_feature_levels
        self.mixers = nn.ModuleList([MixerLayers(d_model=config.d_model, depth=config.enc_layer_depth, mamba_config=config.mamba_config)
                                     for _ in self.order])
        
                                    
    def forward(self, s_pc:PointCloud):
        s_feat  = s_pc.feat[::-1]            # [b g d, ...] 列表长l # TODO:为什么逆序,思考!
        # s_coord = s_pc.coord                 # (b g) 3      # TODO:可以考虑再次作坐标嵌入
        s_order = s_pc.serialized_order      # o (b g)
        s_inverse = s_pc.serialized_inverse  # o (b g)

        b_s = s_pc.batch[-1]+1
        o_s = len(self.order)
        l = self.num_feature_levels
        g = self.num_group

        level_feat=s_feat
        for o in range(o_s):   # TODO：order间，串行算了, 简单些！.
            # Order 
            order = s_order[o]
            o_level_feat = []
            for feat in level_feat: 
                feat = torch_scatter.scatter(feat, index=order, dim=0)        # (b g) d, (b g) -> b g d 
                feat = rearrange(feat, "(b g) d -> b g d ", b = b_s)
                o_level_feat.append(feat)
            #
            o_level_feat = torch.cat(o_level_feat, dim=1)                    # => b (l g) d
            o_level_feat = self.mixers[o](o_level_feat)                      # b (l g) d => b (l g) d
            o_level_feat = list(o_level_feat.split(g,dim=1))                 # b (l g) d => [b g d,...]
            # Inverse
            inverse = s_inverse[o]
            level_feat = []
            for feat in o_level_feat:
                feat = rearrange(feat, " b g d -> (b g) d")
                level_feat.append(torch_scatter.scatter(feat, index= inverse, dim=0))
        
        for idx, feat in enumerate(level_feat):     # 为了MaskDecoder的输入需要！！！
            feat = rearrange(feat, " (b g) d -> b g d", b=b_s)
            level_feat[idx] = feat
        s_pc.feat = level_feat
        return s_pc


class PointSISFollowmlp_SEG(nn.Module):
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.grouper = Grouper_By_NumGroup(config.num_group, config.group_size)
        self.pointsis_followmlp = PointSIS_FollowMLP(config)
        self.point_encoder = PointSIS_Encoder(config)
        self.mask_decoder = MaskDecoder(config)
        # self.seg = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.GELU(),
        #     nn.Linear(config.d_model, 2),
        #     nn.Softmax(dim=-1)           
        # )
        self.class_predict = nn.Linear(config.d_model, config.num_labels+1)
        self.feat_propagation = FeatPropagation(config.group_size)

        self.num_queries = config.num_queries
        self.query_embedder = nn.Embedding(config.num_queries, config.d_model)        # 可学习的查询！
        self.query_position_embedder = nn.Embedding(config.num_queries, config.d_model)   # TODO：位置也是可学习的？？？ Mask2Former就是如此！！！
        
        self.loss = PMLoss(config)

    def forward(self, parent_pc:PointCloud):
        #
        s_pc = self.grouper(parent_pc)
        #
        s_pc = self.pointsis_followmlp(s_pc)
        #
        s_pc = self.point_encoder(s_pc)
        b_s = s_pc.batch[-1]+1
        #
        query_embeddings = self.query_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1)
        query_position_embeddings = self.query_position_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1)
        point_embedding = s_pc.feat[-1]
        encoder_hidden_states = s_pc.feat[0:-1]
        # TODO:有个问题,mask_decoder的参数point_embedding,encoder_hidden_states是否需要序列化?在Transformer机制下，可以先不考虑?作也容易！
        pred_mask, q = self.mask_decoder(                               # -> b q g , b q d
                            query_embeddings = query_embeddings,
                            query_position_embeddings= query_position_embeddings,
                            point_embeddings = point_embedding,
                            encoder_hidden_states= encoder_hidden_states)
        # feat = self.feat_propagation(parent_pc, s_pc)
        # seg = self.seg(feat)
        pred_probs = self.class_predict(q)                             # b q d -> b q l      # l代表num_labels+1
        if "labels" in s_pc.keys():    # 如果有标签，就计算loss！！！
            labels = rearrange(s_pc.labels, "(b g) -> b g", b=b_s)
            m_i = self.loss(pred_mask,pred_probs,labels)  # 
            print(m_i) 
        return s_pc