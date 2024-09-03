import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange,einsum
from typing import List, Optional, Tuple
from .mlp import MLP

from pm.pointmamba.conifuguration_point_sis import  PointSISConfig

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

    def forward(self, query:torch.Tensor, memory:torch.Tensor) -> Tuple[Tensor]:
        query_emb = self.query_mlp(query)                                            # b q d -> b q d
        # TODO: 据说jit不友好,...
        predicated_mask = einsum(query_emb, memory, "b q d, b g d -> b q g ")                # b q d , b g d -> b q g
        attension_mask = predicated_mask.sigmoid().squeeze(1).repeat(1, self.nhead, 1, 1)   # b q g -> b h q g
        attension_mask = rearrange(attension_mask, " b h q g -> (b h) q g")          # 注意： torch.nn.MultiheadAttention的要求！！！ 原文实现用的是flatten(0,1) 
        attension_mask = (attension_mask < 0.5).bool()
        attension_mask = attension_mask.detach()                                     # no_grad! why?
        return predicated_mask, attension_mask                                                      

class MaskedSelfAttention(nn.Module):
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
    def __init__(self, config:PointSISConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        #
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, config.nhead, config.dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        #
        self.self_attn = MaskedSelfAttention(  # TODO: 实质上 attebtion_mask 没有用！
            embed_dim=self.embed_dim,
            num_heads=config.nhead,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = config.dropout
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
        hidden_states: torch.Tensor,
        query_position_embeddings: Optional[torch.Tensor] = None,
        # K,V
        encoder_hidden_states: Optional[torch.Tensor] = None,    # To k,v
        position_embeddings:   Optional[torch.Tensor] = None,
        # predicated_mask
        encoder_attention_mask: Optional[torch.Tensor] = None,    # TODO: 这个要理解！
    ) -> Tensor:
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
            [MaskedAttentionDecoderLayer(config) for _ in range(self.num_decoder_layers)]
        )

        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self,
        # Q         
        query_embeddings: torch.Tensor,
        # 实质上是编码其的最后一层输出！
        point_embeddings: torch.Tensor,
        # 编码器各层的输出！
        encoder_hidden_states: torch.Tensor,
        #
        query_position_embeddings: torch.Tensor = None,
        ) -> Tuple[Tensor]:
        hidden_states = query_embeddings  # b q d                            # 作为decode_layer的输入！ 直接级联

        normalized_hidden_states = self.layernorm(query_embeddings)        # 作为MaskPredictor的输入！须先normalize！
        predicated_mask, attention_mask = self.mask_predictor(normalized_hidden_states, point_embeddings)
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
            normalized_hidden_states = self.layernorm(layer_outputs)
            predicated_mask, attention_mask = self.mask_predictor(normalized_hidden_states,point_embeddings)
            hidden_states = layer_outputs

        return predicated_mask, hidden_states


#各种辅助！！
class PMHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, 
        cost_mask: float = 1.0, 
        cost_dice: float = 1.0
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
    ) -> List[Tuple[Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_group` with the predicted masks.
                (num_group 就是fps采样后的点数)
            class_labels (`torch.Tensor`):
                A tensor of dim `num_target` containing the class labels
                (num_target 就是在target点云中标注的对象个数！每个对象有个类别).
            mask_labels (`torch.Tensor`):
                A tensor of dim `num_target, num_group` containing the target masks.
                (每个对象,一个掩码.)

        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices