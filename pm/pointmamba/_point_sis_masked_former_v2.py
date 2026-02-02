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


from pm.pointmamba.configuration_point_sis import Mamba1Config, PointSISConfig
from pm.pointmamba.losses import PMLoss
from pm.pointmamba.pointmask import MaskDecoder, MaskedAttentionDecoderLayer

from pm.utils.point_cloud import FeatPropagation, Grouper_By_NumGroup, PointCloud
from pointops import farthest_point_sampling as fps

"""_这里准备尝试从点云中直接采样出Query_
前面的尝试中，发现几个问题:
1) 序列化方式对下、上采样，其实是不友好的。尤其在这种稀疏数据的情况下。本质上，Toplogy和Geometry是两码事了！
"""


class Feature_Encoder(nn.Module):
    """
    特征编码，直接用MLP来处理！
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.embeding = nn.Sequential(
            nn.Linear(in_channel, out_channel * 2),
            nn.GELU(),
            nn.Linear(out_channel * 2, out_channel),
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
        self.pos_proj = nn.Linear(3, in_channels)
        self.spconv_f = spconv.SubMConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=True
        )
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.norm_f = nn.LayerNorm(out_channels)

    def forward(self, s_pc: PointCloud):
        # assert s_pc.feat和s_pc.sparse_conv_feat.features 应当是一致的！
        short_cut = s_pc.feat  # TODO: 需不需要运用残差结构能？
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
    """ """

    def __init__(self, feat_dim, depth, order_num, mamba_config):
        super().__init__()
        self.cpe = CPE(feat_dim, feat_dim)
        # 如果一个次序，一个分支的话，就这样写！
        # self.mixer_layers = nn.ModuleList([MixerLayers(d_model=feat_dim, depth= depth, mamba_config=mamba_config)  # noqa: E501
        #                                   for _ in range(order_num)
        #                                   ]
        #                                 )
        self.mixer_layers = MixerLayers(
            d_model=feat_dim, depth=depth, mamba_config=mamba_config
        )
        self.fuse_o = nn.Sequential(  # 合并各排序的特征。
            nn.Linear(
                feat_dim * (order_num + 0), feat_dim
            ),  # TODO: 相当于一个加法平均？ +1 相当于搞了个残差？
            # nn.LayerNorm(feat_dim),                                # TODO: 前面的Mixerlayer里已经有LayerNorm了，这里还要不要？  # noqa: E501
            nn.GELU(),  # TODO: 看看融合后，要不要激活函数
            # nn.Linear(feat_dim, feat_dim),                         # TODO:
        )

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
        s_pc = self.scan(self.cpe(s_pc))
        return s_pc

class Chain(nn.Module):     #下
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
            coord = point.coord,  # 直接用head point的坐标！
            grid_coord=point.grid_coord,
            batch=point.batch,
            order = point.order           # TODO: 这个地方有点隐蔽,必须穿进来的,必须有这个field!
        )
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size 
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
        self.num_feature_levels = config.num_feature_levels  # len(enc_depths) - 1
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
                    feat_dim=config.enc_channels[s],
                    depth=config.enc_depths[s],
                    order_num=len(config.order),
                    mamba_config=config.mamba_config,
                ),
            )
            self.enc.add_module(f"enc_{s}", enc)


    def transform(self, s_pc: PointCloud):  # TODO: 这个地方才开始用到grid_size！
        s_pc.feat = torch.cat([s_pc.coord, s_pc.feat], dim=-1)  # BG 7 #此处融入坐标！
        s_pc.order = self.order
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        s_pc.sparsify()
        return s_pc

    def gather_enc(self, s_pc: PointCloud,b_s:int):
        feat_list = []
        # fn -> [fn, fn-1, ..., f0]
        while "ancestor" in s_pc.keys():
            feat_list.append(s_pc.feat)
            s_pc = s_pc.ancestor
        feat_list.append(s_pc.feat)
        s_pc.gathered_feat = feat_list
        return s_pc     
    
    def forward(self, s_pc: PointCloud):
        b_s = s_pc.batch[-1]+1
        s_pc = self.transform(s_pc)
        s_pc = self.feature_encoder(s_pc)  # => (b g) d
        s_pc = self.enc(s_pc)
        s_pc = self.gather_enc(s_pc,b_s)
        
        return s_pc

class Latent_Query_Generator(nn.Module):
    """
    从编码器的最后一层输出生成latent query！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.num_queries = config.num_queries
        self.query_embedding_proj_pre = nn.Sequential(
            nn.Linear(config.num_group, config.num_queries //2),
            nn.Linear(config.num_queries //2, config.num_queries),
        )
        self.query_embedding_proj_post = nn.Sequential(
            nn.Linear(config.d_model, config.d_model*2),
            nn.Linear(config.d_model*2, config.d_model),
        )
        self.layers = nn.ModuleList(
            [MaskedAttentionDecoderLayer(config) for _ in range(config.num_feature_levels+1)]
        )

    def forward(self, gathered_feats:list[torch.Tensor]):
        query = self.query_embedding_proj_pre(gathered_feats[0].transpose(1, 2))  # b d q
        query = self.query_embedding_proj_post(query.transpose(1, 2))  # b q d
        
        for idx, layer in enumerate(self.layers):
            query = layer(
                query=query,
                encoder_output=gathered_feats[idx],
            )
        return query
    
class PointSIS_Seg(nn.Module):
    """
    这部分是拿来训练的,预处理、後处理都没有学习的内容！
    """

    def __init__(self, config: PointSISConfig):
        super().__init__()
        self.num_queries = config.num_queries
        self.pointsis_feature_extractor = PointSIS_Feature_Extractor(config)
        self.mask_decoder = MaskDecoder(config)
        # {Query:
        # latent_query应当是个技术创新，直接从编码器的最后输出构造出query!
        self.latent_query_generator = Latent_Query_Generator(config)
        self.query_position_gen = nn.Embedding(config.num_queries, config.d_model)
        # }
        # {融合prompt:
        self.merge_prompt = nn.Linear(config.d_model + 1, config.d_model)
        self.merge_prompt_pos = nn.Linear(config.d_model+1, config.d_model)
        # }
        #
        self.loss = PMLoss(config)

    def forward(self, s_pc: PointCloud):
        # s_pc: "coord,feat,offset,grid_size,s_o_i"可用，"labels,shape_weight"看情况!  # noqa: E501
        s_pc = self.pointsis_feature_extractor(s_pc)
        b_s = int(s_pc.batch[-1]) + 1
        gathered_feats = s_pc.gathered_feat  # [(b g) d,...]
        # 直接从点云中采样出query!
        # v0: 好像不行！
        # q_offset = (torch.ones_like( s_pc.batch_bin)* self.num_queries).cumsum(0).int()
        # q_idx = fps(s_pc.coord, s_pc.offset, q_offset)  # n_1+n_2+...+n_b  # 幸亏这个fps
        # query = gathered_feats[0][q_idx]  # (b q) d
        # query = rearrange(query, "(b q) d -> b q d", b=b_s)
        for i in range(len(gathered_feats)):
            gathered_feats[i] = rearrange(gathered_feats[i], "(b g) d -> b g d", b=b_s)

        query = self.latent_query_generator(gathered_feats)  # b q d
        query_position = self.query_position_gen.weight.unsqueeze(0).repeat(b_s, 1, 1) # q l => b q l
        # 上下颌分类！后面要考虑搞成promt embedding!
        s_o_i = s_pc.s_o_i  # b
        s_o_i = repeat(s_o_i, "b -> (b q) 1", q=self.num_queries)
        s_o_i = rearrange(s_o_i, "(b q) 1 -> b q 1", b=b_s)
        # Merge_Prompt!
        query_embeddings = self.merge_prompt(
            torch.cat([query, s_o_i], dim=-1)
        )
        query_position_embeddings = self.merge_prompt_pos(
            torch.cat([query_position, s_o_i], dim=-1)
        )

        mask_features = gathered_feats[-1]  # s_pc.feat 此时是收集起来的一个feat list!
        encoder_hidden_states = gathered_feats[
            0:-1
        ]  # config.num_feature_levels控制！ 其实就是主干网的那几层输出!
        pred_mask, pred_probs = self.mask_decoder(  # -> b q g , b q d
            query_embeddings=query_embeddings,
            query_position_embeddings=query_position_embeddings,
            mask_features=mask_features,  # 编码主干网的最后一层输出！
            encoder_hidden_states=encoder_hidden_states,  # 编码主干网的下面几层的输出！
        )

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
        s_pc.feat = (
            pred_mask.contiguous()
        )  # FIXME:老问题 s_pc的feat过载太多，看怎么清晰一下！！！
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
