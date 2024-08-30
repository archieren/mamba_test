from typing import Union, Optional
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
from pm.utils.point_cloud import PointCloud, group_by_group_number
from pointops import interpolation2

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

class MaskPredictor(nn.Module):
    """
    实质上,预测一个attention mask！参看 Mask2Former！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.nhead = config.nhead
        self.query_mlp = MLP(in_channels=config.d_model,
                            out_channels=config.d_model, 
                            hidden_channels=int(config.d_model *2))

    def forward(self, query:torch.Tensor, memory:torch.Tensor):
        query_emb = self.query_mlp(query)                                            # b q d -> b q d
        # TODO: 据说jit不友好,...
        out_mask = einsum(query_emb, memory, "b q d, b g d -> b q g ")                # b q d , b g d -> b q g
        attension_mask = out_mask.sigmoid().squeeze(1).repeat(1, self.nhead, 1, 1)   # b q g -> b h q g
        attension_mask = (attension_mask < 0.5).bool()
        attension_mask = attension_mask.detach()                                     # no_grad! why?
        return attension_mask                                                      

class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # Assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor:torch.Tensor):
        tensor = rearrange(tensor, "b l (h d1) -> b l h d1", d1=self.head_dim)
        tensor = rearrange(tensor," b l h d1 -> b h l d1")
        return tensor

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
    
    def forward(
        self,
        #
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Input shape: Batch x Time/Length/num_group/... x Channel"""

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj 
        # 自注意，没有key_value_states
        key_states = self._shape(self.k_proj(hidden_states))
        value_states = self._shape(self.v_proj(hidden_states_original))      # 注意 value_states是不能加位置嵌入信息的！

        #proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = rearrange(self._shape(query_states), "b h q d1 -> (b h) q d1")
        key_states   = rearrange(key_states, "b h g d1 -> (b h) g d1 ")
        value_states = rearrange(value_states, "b h g d1 -> (b h) g d1 ")

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (b h) q d1, (b h) g d1 -> (b h) q g

        if attention_mask is not None:           # 这就是 Attention_mask的含义！
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)      # (b h) q g, (b h) g d1 -> (b h) q d1
        attn_output = rearrange(attn_output, "(b h) q d1 -> b h q d1", h=self.num_heads)
        attn_output = rearrange(attn_output, "b h q d1 -> b q (h d1)")

        attn_output = self.out_proj(attn_output)               # b q d -> b q d

        return attn_output

class MaskedAttentionDecoderLayer(nn.Module):
    def __init__(self, config:PointSISConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MaskedSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.nhead,
            dropout=config.dropout,
            is_decoder=True,         # TODO：好像没用，写多了！
        )

        self.dropout = config.dropout
        self.activation_dropout = config.dropout
        self.activation_fn = nn.ReLU()                   # Mask2Former里的缺省设置，可以，TODO：改成可配置的

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, config.nhead, config.dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.dim_feedforward)
        self.fc2 = nn.Linear(config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(  # 简化,我只考虑pre_norm==False的情况.Mask2Former里的缺省设置. TODO:思考有什么区别！
        self,
        # Q
        hidden_states: torch.Tensor,
        query_position_embeddings: Optional[torch.Tensor] = None,
        # K,V
        encoder_hidden_states: Optional[torch.Tensor] = None,    # To k,v
        position_embeddings:   Optional[torch.Tensor] = None,
        # predicated_mask
        encoder_attention_mask: Optional[torch.Tensor] = None,    # TODO: 这个要理解！
    ):
        # Masked(Cross)-Attention Block
        residual = hidden_states
        hidden_states, _ = self.cross_attn(                       # 这是 torch自带的！
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key  =self.with_pos_embed(encoder_hidden_states, position_embeddings),
            value=encoder_hidden_states,
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states
        hidden_states = self.self_attn(                            # 
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class MaskDecoder(nn.Module):
    """
    可以采用MaskFormer，Mask2Former里的方法，来使用TransformerDeocoder！

    """
    def __init__(self,config:PointSISConfig):
        super().__init__()
        self.mask_predictor = MaskPredictor(config)

        self.num_feature_levels = config.num_feature_levels  # TODO: 是否考虑真的下采样？
        self.num_decoder_layers = config.num_decode_layers
        self.layers = nn.ModuleList(
            [MaskedAttentionDecoderLayer(self.config) for _ in range(self.num_decoder_layers)]
        )

        self.layernorm = nn.LayerNorm(config.hidden_dim)

    def forward(self,
        # Q         
        query_embeddings: torch.Tensor,
        query_position_embeddings: torch.Tensor = None,
        #
        point_embeddings: torch.Tensor = None,
        #
        encoder_hidden_states: torch.Tensor = None,
        ):
        hidden_states = query_embeddings  # b q d 

        intermediate_hidden_states = self.layernorm(query_embeddings)
        attention_mask = self.mask_predictor(intermediate_hidden_states, point_embeddings)
        for idx, decoder_layer in enumerate(self.layers):
            level_index = idx % self.num_feature_levels
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False  # 避免什么？
            layer_outputs = decoder_layer(
                # Q
                hidden_states,
                query_position_embeddings=query_position_embeddings,
                #level_index=level_index,
                # k,v
                encoder_hidden_states=encoder_hidden_states[level_index],
                #position_embeddings = ??,                   # TODO: 要考虑
                encoder_attention_mask=attention_mask,
            )
            intermediate_hidden_states = self.layernorm(layer_outputs)
            attention_mask = self.mask_predictor(intermediate_hidden_states,point_embeddings)
            hidden_states = layer_outputs

        return hidden_states
        # # [(b g) d, ...] 应当是四个！
        # b_s  = s_pc.batch[-1]+1                      # b

        # memory = rearrange(feat,"(b g) d - > b g d", b = b_s)
        # output = torch.arange(self.num_queries, device=memory.device).unsqueeze(0).repeat(b_s,1)
        # output = self.query_embedding(output)                    # b q d
        # print("require grad",output.shape)
        # output = self.decoder(output, memory)                    # b q d , b g d -> b q d 

        # return output

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
            nn.GELU(),
            nn.Linear(config.feature_dims, config.d_model)
        )

        self.fuse_o = nn.Sequential(                                 #合并各排序的特征。
            nn.Linear(config.d_model * len(config.order), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )        


    def forward(self, s_pc:PointCloud):                     
        # s_意味着初始下采样，已经做了
        # TODO：后面的多尺度下采样，看情况再做！
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)        
        s_feat  = s_pc.feat 
        s_coord = s_pc.coord 
        s_order = s_pc.serialized_order 
        s_inverse = s_pc.serialized_inverse        
        
        b_s = s_pc.batch[-1]+1
        o_s = len(self.order)        
        # s_order, s_inverse 均为 order_size batch_size*num_group
        s_coord = self.pos_encoder(s_coord)                # => (b g) d 
        hidden_state = self.feature_encoder(s_feat)              # => (b g) d 
        hidden_states =[]
        for _ , mixlayer in enumerate(self.mixers):        
            hidden_state = torch.cat([hidden_state, s_coord], dim= -1)               # => (b g) (d*2)  # 坐标的嵌入，应当不变！
            hidden_state = self.fuse_e(hidden_state)                                 # 融合各编码 => (b g) d
            hidden_state = hidden_state.unsqueeze(0).repeat(o_s,1,1)                 # 为每种排序准备排序的数据 => o (b g) d
            hidden_state = torch_scatter.scatter(hidden_state,index=s_order, dim=1)  # 排序 => o (b g) d       
            hidden_state = rearrange(hidden_state, "o (b g) d -> b (o g) d", b=b_s)  # 将各个排序拼接,参看PointMamba的第四版!!
            hidden_state = mixlayer(hidden_state)                                       # 返回的是抽取的几个层的返回结果的列表 
            #s = torch.cat(s, dim= -1)                          # 将各层mamba的结果，拼接！！
            #s = self.fuse_f(s)                                 # 融合各结果 
            hidden_state = rearrange(hidden_state, "b (o g) d -> o (b g) d", o=o_s)  # 拆分各排序
            hidden_state = torch_scatter.scatter(hidden_state,index=s_inverse, dim=1)# 逆排序 => order_size batch_size*num_group d
            hidden_state = rearrange(hidden_state, "o (b g) d -> b g (o d)", b= b_s) # 调整，将同一点，在各排序情况下的特征拼接到一起！
            hidden_state = self.fuse_o(hidden_state)                                 # b g d
            hidden_state = rearrange(hidden_state, "b g d -> (b g) d")               # (b g) d
            hidden_states.append(hidden_state)
        s_pc.feat = hidden_states                                                    # TODO: 感觉PointCloud.feat这个域被重用了太多！   
        return s_pc

class PointSISFollowmlp_SEG(nn.Module):
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.grouper = Grouper_By_NumGroup(config.num_group, config.group_size)
        self.pointsis_followmlp = PointSIS_FollowMLP(config)
        self.mask_decoder = MaskDecoder(config)
        self.seg = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
            nn.Softmax(dim=-1)           
        )
        self.feat_propagation = FeatPropagation(config.group_size)

        # self.num_queries = config.num_queries
        # self.query_embedding = nn.Embedding(config.num_queries, config.d_model)        # 可学习的查询！

    def forward(self, parent_pc:PointCloud):
        s_pc = self.grouper(parent_pc)
        s_pc = self.pointsis_followmlp(s_pc)
        # feat = self.feat_propagation(parent_pc, s_pc)
        # seg = self.seg(feat)        
        return s_pc