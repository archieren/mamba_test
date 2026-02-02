import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange,einsum
from typing import Optional, Tuple
from .mlp import MLP

from pm.pointmamba.configuration_point_sis import  PointSISConfig

class MaskPredictor(nn.Module):
    """
    实质上,预测一个attention mask！参看 Mask2Former和 Mask3D的论文！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.nhead = config.nhead
        self.num_classes = config.num_labels
        self.predict_class = nn.Linear(self.embed_dim, self.num_classes+1)
        # TODO: 要还是不要?好像无差别！
        self.predict_mask = MLP(in_channels=self.embed_dim,
                            out_channels=self.embed_dim, 
                            hidden_channels=int(self.embed_dim *2))
        self.query_norm = nn.LayerNorm(self.embed_dim)           

    def forward(self, query:torch.Tensor, memory:torch.Tensor) -> Tuple[Tensor]:
        query_output = self.query_norm(query)
        predicted_classes = self.predict_class(query_output)    # b q d -> b q (num_classes+1)
        
        # TODO: 据说einsum对jit不友好,...
        # 注意predict_mask和predicted_mask的区别！
        # predict_mask = self.predict_mask(query_output)        # b q d -> b q d
        predicated_mask = einsum(self.predict_mask(query_output), memory, "b q d, b g d -> b q g ")                # b q d , b g d -> b q g
        attension_mask = predicated_mask.sigmoid().squeeze(1).repeat(1, self.nhead, 1, 1)   # b q g -> b h q g
        attension_mask = rearrange(attension_mask, " b h q g -> (b h) q g")          # 注意： torch.nn.MultiheadAttention的要求！！！ 原文实现用的是flatten(0,1) 
        attension_mask = (attension_mask < 0.5).bool()
        attension_mask = attension_mask.detach()                                     # no_grad! why?
        return predicted_classes,predicated_mask, attension_mask                                                      

class MaskedSelfAttention(nn.Module):     # 这个MaskedSelfAttention和一般的SelfAttention没有任何区别！
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
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
    ) -> Tensor:
        """Input shape: Batch x Time/Length/num_group/... x Channel"""

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        else:
            hidden_states_original = hidden_states

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
    def __init__(self, config:PointSISConfig, only_cross_attn=False) -> None:
        super().__init__()
        self.only_cross_attn = only_cross_attn
        self.embed_dim = config.d_model
        self.dropout = config.dropout
        #
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, config.nhead, config.dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        #
        if not self.only_cross_attn:
            self.self_attn = MaskedSelfAttention(  # TODO: 实质上 attention_mask 没有用！
                embed_dim=self.embed_dim,
                num_heads=config.nhead,
                dropout=config.dropout,
            )
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

            #ffn
            self.activation_dropout = config.dropout
            self.activation_fn = nn.ReLU()                   # Mask2Former里的缺省设置，可以，TODO：改成可配置的
            self.fc1 = nn.Linear(self.embed_dim, config.dim_feedforward)
            self.fc2 = nn.Linear(config.dim_feedforward, self.embed_dim)
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
    
    def forward(  # 简化,我只考虑pre_norm==False的情况.Mask2Former里的缺省设置. TODO:思考有什么区别！
        self,
        # Q
        query: torch.Tensor,
        query_position_embeddings: Optional[torch.Tensor] = None,
        # K,V
        encoder_output: Optional[torch.Tensor] = None,    # To k,v
        encoder_output_position_embeddings:   Optional[torch.Tensor] = None,
        # predicated_mask
        attention_mask: Optional[torch.Tensor] = None,    # TODO: 这个要理解！
    ) -> Tensor:
        # Masked(Cross)-Attention Block
        residual = query
        query, _ = self.cross_attn(                       # 这是 torch自带的！
            query=self.with_pos_embed(query, query_position_embeddings),
            key  =self.with_pos_embed(encoder_output, encoder_output_position_embeddings),
            value=encoder_output,
            attn_mask=attention_mask,
            key_padding_mask=None,
        )
        query = nn.functional.dropout(query, p=self.dropout, training=self.training)
        query = residual + query
        query = self.cross_attn_layer_norm(query)
        if not self.only_cross_attn:
            # Self Attention Block
            residual = query
            query = self.self_attn(                            # 
                hidden_states=query,
                position_embeddings=query_position_embeddings,
                attention_mask=None,
            )
            query = nn.functional.dropout(query, p=self.dropout, training=self.training)
            query = residual + query
            query = self.self_attn_layer_norm(query)
            # Fully Connected
            residual = query
            query = self.activation_fn(self.fc1(query))
            query = nn.functional.dropout(query, p=self.activation_dropout, training=self.training)
            query = self.fc2(query)
            query = nn.functional.dropout(query, p=self.dropout, training=self.training)
            query = residual + query
            query = self.final_layer_norm(query)

        return query

class MaskDecoder(nn.Module):
    """
    可以采用MaskFormer，Mask2Former里的方法，来使用TransformerDeocoder！

    """
    def __init__(self,config:PointSISConfig):
        super().__init__()
        self.mask_predictor = MaskPredictor(config)

        self.num_feature_levels = config.num_feature_levels
        self.num_decoder_layers = config.num_decode_layers
        self.layers = nn.ModuleList(
            [MaskedAttentionDecoderLayer(config) for _ in range(self.num_decoder_layers)]
        )

        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self,
        # Q         
        query_embeddings: torch.Tensor,
        # 实质上是编码其的最后一层输出！
        mask_features: torch.Tensor,
        # 编码器各层的输出！
        encoder_hidden_states: torch.Tensor,
        #
        query_position_embeddings: torch.Tensor = None,
        ) -> Tuple[Tensor]:
        query_hidden_states = query_embeddings  # b q d                            # 作为decode_layer的输入！ 直接级联

        query_hidden_states = self.layernorm(query_embeddings)        # 作为MaskPredictor的第一次输入！须先normalize！
        predicated_classes,predicated_mask, attention_mask = self.mask_predictor(query_hidden_states, mask_features)
        for idx, decoder_layer in enumerate(self.layers):
            level_index = idx % self.num_feature_levels
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False  # 避免什么？
            query_hidden_states = decoder_layer(                            # 注: layer_output 已经被 normalized了！
                # Q
                query = query_hidden_states,
                query_position_embeddings = query_position_embeddings,
                #level_index=level_index,
                # k,v
                encoder_output = encoder_hidden_states[level_index],
                #position_embeddings = ??,                   # TODO: 要考虑
                attention_mask = attention_mask,
            )
            predicated_classes, predicated_mask, attention_mask = self.mask_predictor(query_hidden_states,mask_features)
            

        return predicated_mask, predicated_classes