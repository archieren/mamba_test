from functools import partial

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from einops import rearrange, repeat

# 直接使用mamba推荐的Block, 不像Point Mamba抄过来!
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2 as Mamba

from pm.pointmamba.mlp import MLP
from pm.pointmamba._configuration_point_sis_vp import Mamba1Config, PointSISConfig
from pm.pointmamba.losses import PMLoss
from pm.pointmamba.pointmask import MaskDecoder, MaskedAttentionDecoderLayer, MaskPredictor

from pm.utils.point_cloud import FeatPropagation, Grouper_By_NumGroup, PointCloud
from pointops import farthest_point_sampling as fps

"""_这里准备尝试Dynamic Perceiver思路
"""


class Feature_Encoder(nn.Module):
    """
    特征编码，直接用MLP来处理！
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.embeding = MLP(
            in_channels=in_channel,
            out_channels=out_channel,
            hidden_channels=out_channel * 2,
        )

    def forward(self, s_pc: PointCloud):
        """
        BG i -> BG o
        """
        s_pc.feat = self.embeding(s_pc.feat)
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc


class CPE(nn.Module):
    """
    Maybe, this is the conditional position encode
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.e_o = out_channels
        #self.pos_proj = nn.Linear(3, in_channels)
        self.spconv_f = spconv.SubMConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=5, bias=True
        )
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.norm_f = nn.LayerNorm(out_channels)

    def forward(self, s_pc: PointCloud):
        # assert s_pc.feat和s_pc.sparse_conv_feat.features 应当是一致的！
        short_cut = s_pc.feat.clone()  # TODO: 需不需要运用残差结构能？
        # #掺入坐标
        # pos_embed = self.pos_proj(s_pc.coord)  # BG C
        # s_pc.feat = s_pc.feat + pos_embed
        # s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        # {- Begin 原始的CPE内容
        s_pc.sparse_conv_feat = self.spconv_f(s_pc.sparse_conv_feat)
        s_pc.feat = s_pc.sparse_conv_feat.features
        s_pc.feat = self.fc1(s_pc.feat)
        s_pc.feat = self.norm_f(s_pc.feat)  # 1
        # - End}
        s_pc.feat = short_cut + s_pc.feat  # 2
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc


class MixerLayers(nn.Module):
    """
    残差式板块栈。直接借用Mamba官方实现里面的Block。
    这个类应当对应...mixer_seq_simple...里的MixerModel
    我看很多有关Mamba的网络，基本都是抄改这一块！！！没必要全文摘抄。直接将对应缺省值的分支留下就可以了！
    """

    def __init__(self, d_model, depth, mamba_config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                self.create_block(d_model, mamba_config, layer_idx)
                for layer_idx in range(depth)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)

    @staticmethod
    def create_block(d_model: int, mamba_cfg: Mamba1Config, layer_idx):
        # 直接用Mamba里实现的Block，基本用缺省值!
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **mamba_cfg)
        norm_cls = nn.LayerNorm
        mlp_cls = nn.Identity
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
        )  # 可以了解，Block里的缺省路径 Add -> LN -> Mixer
        # block.layer_idx = layer_idx
        return block

    def forward(self, hidden_states, seq_idx=None):
        residual = None
        for idx, block in enumerate(self.blocks):
            hidden_states, residual = block(hidden_states, residual, seq_idx=seq_idx)
        # 此时就需要补一个 Add -> LN 过程！！！ 才能得到合适的hidden_state output!
        hidden_states = (
            (hidden_states + residual) if residual is not None else hidden_states
        )
        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


class Stage(nn.Module):
    """ 
    这里采取了Dyn-Perceiver的思路！
    """
    def __init__(self, config:PointSISConfig, stage_num: int) :#feat_dim, depth, order_num, mamba_config):
        super().__init__()
        #
        feat_dim=config.enc_channels[stage_num]
        depth=config.enc_depths[stage_num]
        order_num=len(config.order)
        
        self.cpe = CPE(feat_dim, feat_dim)
        self.mixer_layers = MixerLayers(
            d_model=feat_dim, depth=depth, mamba_config=config.mamba_config
        )
        self.fuse_o = nn.Sequential(  # 合并各排序的特征。
            nn.Linear(
                feat_dim * (order_num + 0), feat_dim
            ),  # TODO: 相当于一个加法平均？ +1 相当于搞了个残差？
            # nn.LayerNorm(feat_dim),                                # TODO: 前面的Mixerlayer里已经有LayerNorm了，这里还要不要？  # noqa: E501
            nn.GELU(),  # TODO: 看看融合后，要不要激活函数
            # nn.Linear(feat_dim, feat_dim),                         # TODO:
        )
        
        self.cloud_cross_query = MaskedAttentionDecoderLayer(config)
        self.query_cross_cloud = MaskedAttentionDecoderLayer(config,only_cross_attn=True)
    def scan(self, s_pc: PointCloud):
        s_order = s_pc.serialized_order  # o (b g)
        s_inverse = s_pc.serialized_inverse  # o (b g)
        # b_s = s_pc.batch[-1]+1
        seq_idx = s_pc.batch.unsqueeze(0).int()  # (b g) -> 1 (b g)
        o_s = s_order.shape[0]
        # s_order, s_inverse 均为 order_size batch_size*num_group
        hidden_state = s_pc.feat  # (b g) d
        # 将各排序拼接,走Mamba流程！

        output_gathered = []
        # output_gathered.append(hidden_state)
        for i in range(o_s):  # 对每个排序， 走一趟mixer_layers
            seq_input = torch_scatter.scatter(
                hidden_state, index=s_order[i], dim=0
            )  # 排序:    (b g) d, (b g) => (b g) d
            seq_input = seq_input.unsqueeze(0)  # "(b g) d -> 1 (b g) d"
            # 一个次序，一个分支的话，就这样写！
            # seq_output  = self.mixer_layers[i](seq_input, seq_idx = seq_idx)
            seq_output = self.mixer_layers(seq_input, seq_idx=seq_idx)
            seq_output = seq_output.squeeze(0)  # "1 (b g) d -> (b g) d"
            seq_output = torch_scatter.scatter(
                seq_output, index=s_inverse[i], dim=0
            )  # 逆排序:   (b g) d, (b g) => (b g) d
            output_gathered.append(seq_output)  #

        hidden_state = torch.stack(
            output_gathered, dim=-1
        )  # [(b g) d, ...] => [(b g) d o_s]
        hidden_state = (
            torch.sum(hidden_state, dim=-1) / o_s
        )  # TODO: 这里相当于平均？ +1 是因为多了个残差？
        # hidden_state = self.fuse_o(hidden_state)

        s_pc.feat = hidden_state
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc

    #   运用mamba的变长能力
    def forward(self, s_pc: PointCloud):
        """s_pc带着两个输入，feat及query！
        """
        b_s = s_pc.batch[-1]+1
                
        if s_pc.query is not None:
            s_pc.query = self.cloud_cross_query(
                query=s_pc.query,
                query_pos_emb=s_pc.query_pos,
                encoder_output=rearrange(s_pc.feat, "(b g) d -> b g d", b=b_s),
            )
        
        s_pc = self.scan(self.cpe(s_pc))

        if s_pc.query is not None:
            s_pc.feat = self.query_cross_cloud(
                query=rearrange(s_pc.feat, "(b g) d -> b g d", b=b_s),
                encoder_output=s_pc.query,
            )
            s_pc.feat = rearrange(s_pc.feat, "b g d -> (b g) d")
            s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc

class Chain(nn.Module):     # 构造可以回溯的输出结果链！
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
    def forward(self, point: PointCloud):       
        point_dict = Dict(
            feat = self.proj(point.feat),
            coord = point.coord,
            grid_coord=point.grid_coord,
            batch=point.batch,
            order = point.order,           # TODO: 这个地方有点隐蔽,必须穿进来的,必须有这个field!
            query = point.query.clone() if "query" in point.keys() else None,
            query_pos = point.query_pos.clone() if "query_pos" in point.keys() else None,
        )
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size
        if "s_o_i" in point.keys():
            point_dict["s_o_i"] = point.s_o_i
        if "labels" in point.keys(): 
            point_dict["labels"] = point.labels
            point_dict["shape_weight"] = point.shape_weight
        point_dict["ancestor"] = point
        point = PointCloud(point_dict)
        #TODO: 这里还有优化空间！
        point.serialization(order=point.order)
        point.sparsify()
        return point


class PointSIS_Feature_Extractor(nn.Module):
    def __init__(self, config: PointSISConfig):
        super().__init__()
        self.num_group = config.num_group
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config
        self.feature_encoder = Feature_Encoder(
            config.in_channels, config.enc_channels[0]
        )
        self.num_stages = len(config.enc_depths)
        self.enc = nn.Sequential()
        for s in range(self.num_stages):
            enc = nn.Sequential()
            if s > 0:
                enc.add_module(
                    name=f"down_{s}",
                    module=Chain(
                        in_channels=config.enc_channels[s - 1],
                        out_channels=config.enc_channels[s],
                    ),
                )
            enc.add_module(
                name=f"enc_stage_{s}",
                module=Stage(
                    config=config,
                    stage_num=s,
                ),
            )
            self.enc.add_module(f"enc_{s}", enc)


    def transform(self, s_pc: PointCloud):  
        # TODO: 这个地方才开始用到grid_size！
        # 准备好什么样的输入呢？
        assert s_pc.s_o_i is not None, "s_pc must have s_o_i field!"
        s_o_i = s_pc.s_o_i.unsqueeze(-1).repeat(1, self.num_group)
        s_o_i = rearrange(s_o_i, "b g -> (b g) 1")
        
        s_pc.feat = torch.cat([s_pc.coord, s_pc.feat, s_o_i], dim=-1)  # BG 7 #此处融入坐标！
        s_pc.order = self.order
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        s_pc.sparsify()
        return s_pc
    
    def forward(self, s_pc: PointCloud):
        s_pc = self.transform(s_pc)
        s_pc = self.feature_encoder(s_pc)  # => (b g) d
        s_pc = self.enc(s_pc)
        
        return s_pc

class Latent_Query_Generator(nn.Module):
    """
    从编码器的最后一层输出生成latent query！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.num_queries = config.num_queries
        self.query_emb = nn.Embedding(config.num_queries, config.d_model)
        self.query_pos_emb = nn.Embedding(config.num_queries, config.d_model)

    def forward(self, b_s):
        query = self.query_emb.weight.unsqueeze(0).repeat(b_s, 1, 1) # q l => b q l
        query_pos = self.query_pos_emb.weight.unsqueeze(0).repeat(b_s, 1, 1) # q l => b q l

        # 直接从点云中采样出query!
        # v0: 好像不行！
        # q_offset = (torch.ones_like( s_pc.batch_bin)* self.num_queries).cumsum(0).int()
        # q_idx = fps(s_pc.coord, s_pc.offset, q_offset)  # n_1+n_2+...+n_b  # 幸亏这个fps
        # query = gathered_feats[0][q_idx]  # (b q) d
        # query = rearrange(query, "(b q) d -> b q d", b=b_s)
        
        return query, query_pos
    
class PointSIS_Seg(nn.Module):
    """
    这部分是拿来训练的,预处理、後处理都没有学习的内容！
    """

    def __init__(self, config: PointSISConfig):
        super().__init__()
        self.num_queries = config.num_queries
        self.pointsis_feature_extractor = PointSIS_Feature_Extractor(config)
        # self.mask_decoder = MaskDecoder(config)
        self.mask_predictor = MaskPredictor(config)
        # {Query:
        # latent_query应当是个技术创新，直接从编码器的最后输出构造出query!
        self.latent_query_generator = Latent_Query_Generator(config)
        # }
        #
        self.loss = PMLoss(config)

    def gen_prompt(self, s_o_i: torch.Tensor, b_s: int):
        # 上下颌分类！后面要考虑搞成promt embedding!
        s_o_i = repeat(s_o_i, "b -> (b q) 1", q=self.num_queries)
        s_o_i = rearrange(s_o_i, "(b q) 1 -> b q 1", b=b_s)
        return s_o_i 
    
    def forward(self, s_pc: PointCloud):
        """_summary_
        Args:
            s_pc (PointCloud): The Grouped PointCloud, with "coord,feat,offset,grid_size,s_o_i" as inputs. 
                                "labels,shape_weight" are optional, depending on whether it's training or inference!
        Returns:
            PointCloud: s_pc with updated "feat" as predicted mask, and "pred_probs" as predicted class probabilities. 
                        If in training, also with "loss".
        """
        # s_pc: "coord,feat,offset,grid_size,s_o_i"可用，"labels,shape_weight"看情况!
        b_s = int(s_pc.batch[-1]) + 1
        s_pc.query, s_pc.query_pos = self.latent_query_generator(b_s)  # b q d
        #
        s_pc = self.pointsis_feature_extractor(s_pc)
        query = s_pc.query  # b q d  # TODO: 这里的query是经过Stage处理过的！
        mask_features = rearrange(s_pc.feat, "(b g) d -> b g d", b=b_s)  # b g d  
        pred_probs,pred_mask, _ = self.mask_predictor(query, mask_features)
        
        # TODO: pred_probs和 pred_mask 都无须activation！loss里面有！
        if "labels" in s_pc.keys():  # 如果有标签，就计算loss！！！
            labels = rearrange(s_pc.labels, "(b g) -> b g", b=b_s)
            shape_weight = (
                rearrange(s_pc.shape_weight, "(b g) -> b g", b=b_s)
                if s_pc.shape_weight is not None
                else None
            )
            m_i = self.loss(pred_mask, pred_probs, labels, shape_weight)  #
            s_pc.loss = m_i
        pred_mask = rearrange(pred_mask, "b q g -> b g q")
        pred_mask = rearrange(pred_mask, "b g q -> (b g) q")
        s_pc.feat = pred_mask.contiguous() # FIXME:老问题 s_pc的feat过载太多，看怎么清晰一下！！！
        s_pc.pred_probs = pred_probs
        return s_pc


class PointSIS_Seg_Model(nn.Module):
    """
    注意,训练的是PointSIS_Seg,用的是PointSIS_Seg_Model!
    """

    def __init__(self, config: PointSISConfig):
        super().__init__()
        self.grouper = Grouper_By_NumGroup(config.num_group, config.group_size)
        self.feat_propagation = FeatPropagation(config.group_size)
        self.model = PointSIS_Seg(config)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(
            state_dict, strict=strict, assign=assign
        )  # It means, this class of self has no parameters to be trained!

    def forward(self, parent_pc: PointCloud):
        s_pc = self.model(self.grouper(parent_pc))
        parent_pc.feat = self.feat_propagation(parent_pc, s_pc)
        parent_pc.pred_probs = s_pc.pred_probs

        if s_pc.loss is not None:
            parent_pc.loss = s_pc.loss
        del s_pc

        return parent_pc
