from functools import partial


import torch
import torch_scatter
import torch.nn as nn
import spconv.pytorch as spconv

from addict import Dict
from torch import Tensor
from einops import rearrange,repeat
from pm.utils.misc import offset2batch,batch2offset
from mamba_ssm.modules.mamba2 import Mamba2 as Mamba
# 直接使用mamba推荐的Block, 不像Point Mamba抄过来! 
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block 
from pm.pointmamba.conifuguration_point_sis import Mamba1Config, PointSISConfig
from pm.pointmamba.losses import PMLoss
from pm.pointmamba.pointmask import MaskDecoder
from pm.utils.point_cloud import PointCloud, Grouper_By_NumGroup,FeatPropagation


#很拙劣的东西。如果能好的话,这些代码优化!
INITIAL_FEATURE_DIMS = 4      # 在目前的构造中 是 3维的norms+1维的那个特殊的曲率！

class Feature_Encoder(nn.Module):
    """
    特征编码，直接用MLP来处理！
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.e_i = 128           #内部用的channel数！
        self.embeding = nn.Sequential(
            nn.Linear(in_channel, self.e_i *2),
            nn.GELU(),
            nn.Linear(self.e_i * 2, out_channel)            
        )
    
    def forward(self, s_pc:PointCloud):
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
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.e_o = out_channels
        # self.pos_proj = nn.Linear(3, in_channels)
        self.spconv_f = spconv.SubMConv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,bias=True)
        self.fc1      = nn.Linear(out_channels, out_channels)
        self.norm_f   = nn.LayerNorm(out_channels)
    
    def forward(self, s_pc:PointCloud):
        # assert s_pc.feat和s_pc.sparse_conv_feat.features 应当是一致的！
        short_cut = s_pc.feat  # TODO: 需不需要运用残差结构能？
        # #掺入坐标
        # pos_embed = self.pos_proj(s_pc.coord)  # BG C
        # s_pc.feat = s_pc.feat + pos_embed
        # s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        # #
        s_pc.sparse_conv_feat = self.spconv_f(s_pc.sparse_conv_feat)
        s_pc.feat = s_pc.sparse_conv_feat.features
        s_pc.feat = self.fc1(s_pc.feat)
        s_pc.feat = self.norm_f(s_pc.feat) # 1
        #
        s_pc.feat = short_cut + s_pc.feat  # 2
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc

class GridPooling(nn.Module):     #下
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="mean",
        shuffle_orders=False,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        
        self.norm = None
        self.act = None
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        if act_layer is not None:
            self.act = act_layer()

    def forward(self, point: PointCloud):
        grid_coord = point.grid_coord       
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        #Start (TODO: 要理解这儿干了什么？)
        # print(f"start- {grid_coord.shape}")
        # if "pooling_parent" in point.keys():
        #     print("has pooling_parent")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48    # 加上批号！
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)                   # 去掉批号！
        # print(f"end- {grid_coord.shape}")
        #End
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(  # 特征有个reduce！
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr( # 坐标有个reduce！
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
            order = point.order           # TODO: 这个地方有点隐蔽,必须穿进来的,必须有这个field!
        )
        if "name" in point.keys():
            point_dict["name"] = point.name

        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = PointCloud(point_dict)
        if self.norm is not None:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
        point.serialization(order=point.order, shuffle_orders=self.shuffle_orders)  
        point.sparsify()
        return point


class GridUnpooling(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        use_interpolation=True,
        k=5,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add_module("norm_l", norm_layer(out_channels))
            self.proj_skip.add_module("norm_l", norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add_module("act_l",act_layer())
            self.proj_skip.add_module("act_l",act_layer())

        if use_interpolation:
            self.use_interpolation = use_interpolation
            self.feature_propagation = FeatPropagation(group_size=k)
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pooling_parent   # pop("pooling_parent")
        inverse = point.pooling_inverse
        #
        parent.feat = self.proj_skip(parent.feat) 
        if not self.use_interpolation:                      
            parent.feat = parent.feat + self.proj(point.feat)[inverse]    # 注意这个 inverse！ 实现上采样！
        else:
            interpolated_feat = self.feature_propagation(parent, point) 
            parent.feat = parent.feat + self.proj(interpolated_feat)
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent

    
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
    
    def forward(self, hidden_states, seq_idx=None):
        residual = None
        for idx, block in enumerate(self.blocks):
            hidden_states, residual = block( hidden_states, residual, seq_idx=seq_idx)
        # 此时就需要补一个 Add -> LN 过程！！！ 才能得到合适的hidden_state output!
        hidden_states = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states           

class Stage(nn.Module):
    """
    """
    def __init__(self, feat_dim, depth, order_num, mamba_config):
        super().__init__()
        self.cpe = CPE(feat_dim, feat_dim)
        # 如果一个次序，一个分支的话，就这样写！
        # self.mixer_layers = nn.ModuleList([MixerLayers(d_model=feat_dim, depth= depth, mamba_config=mamba_config)
        #                                   for _ in range(order_num)
        #                                   ]
        #                                 ) 
        self.mixer_layers = MixerLayers(d_model=feat_dim, depth= depth, mamba_config=mamba_config)                          
        self.fuse_o = nn.Sequential(                                 #合并各排序的特征。
            nn.Linear(feat_dim * (order_num + 0), feat_dim),         # TODO: 相当于一个加法平均？ +1 相当于搞了个残差？
            # nn.LayerNorm(feat_dim),                                # TODO: 前面的Mixerlayer里已经有LayerNorm了，这里还要不要？
            nn.GELU(),                                               # TODO: 看看融合后，要不要激活函数
            # nn.Linear(feat_dim, feat_dim),                         # TODO:
        )

    def scan(self, s_pc:PointCloud):
        s_order = s_pc.serialized_order   # o (b g)
        s_inverse = s_pc.serialized_inverse  # o (b g)
        # b_s = s_pc.batch[-1]+1
        seq_idx = s_pc.batch.unsqueeze(0).int()
        o_s = s_order.shape[0]
        # s_order, s_inverse 均为 order_size batch_size*num_group
        hidden_state  = s_pc.feat   # (b g) d 
        #将各排序拼接,走Mamba流程！
        
        output_gathered = []
        #output_gathered.append(hidden_state)
        for i in range(o_s):                     # 对每个排序， 走一趟mixer_layers
            seq_input   = torch_scatter.scatter(hidden_state,index=s_order[i], dim=0)  # 排序:    (b g) d, (b g) => (b g) d 
            seq_input = seq_input.unsqueeze(0)         # "(b g) d -> 1 (b g) d"
            # 一个次序，一个分支的话，就这样写！
            #seq_output  = self.mixer_layers[i](seq_input, seq_idx = seq_idx)
            seq_output  = self.mixer_layers(seq_input, seq_idx = seq_idx)
            seq_output = seq_output.squeeze(0)     #  "1 (b g) d -> (b g) d"
            seq_output  = torch_scatter.scatter(seq_output,index=s_inverse[i], dim=0)# 逆排序:   (b g) d, (b g) => (b g) d
            output_gathered.append(seq_output) # 
        
        hidden_state = torch.stack(output_gathered, dim=-1) # [(b g) d, ...] => [(b g) d o_s]
        hidden_state = torch.sum(hidden_state, dim=-1) / o_s  # TODO: 这里相当于平均？ +1 是因为多了个残差？
        #hidden_state = self.fuse_o(hidden_state)

        s_pc.feat = hidden_state 
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc                       
    #   运用mamba的变长能力
    def forward(self, s_pc:PointCloud):
        s_pc = self.scan(self.cpe(s_pc))        
        return s_pc                 
        

    
class PointSIS_Feature_Extractor(nn.Module):
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.num_group = config.num_group
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.num_feature_levels = config.num_feature_levels  # len(enc_depths) - 1
        self.config = config
        self.feature_encoder = Feature_Encoder(config.in_channels, config.enc_channels[0])
        self.num_stages = len(config.enc_depths)
        self.has_decoder = config.has_decoder
        self.enc = nn.Sequential()
        for s in range(self.num_stages):
            enc = nn.Sequential()
            if s > 0 :
                enc.add_module(
                    name= f"down_{s}",
                    module= GridPooling(
                        in_channels =   config.enc_channels[s-1],
                        out_channels=   config.enc_channels[s],
                        stride      =   config.stride[s-1],
                        norm_layer  =   nn.LayerNorm,
                        act_layer   =   nn.GELU,                    
                    )
                )
            enc.add_module(
                name = f"enc_stage_{s}",
                module = Stage(
                    feat_dim=config.enc_channels[s], #config.d_model,
                    depth= config.enc_depths[s],
                    order_num=len(config.order), 
                    mamba_config=config.mamba_config                        
                )
            )
            self.enc.add_module(f"enc_{s}",enc)        

        # 解码器通道数列表，要加上最后一个编码器输出！ TODO: 即使没有解码器，gather也要用到这个列表！
        dec_channels = config.dec_channels + (config.enc_channels[-1],)          
        if self.has_decoder:
            self.dec = nn.Sequential()
            for s in reversed(range(self.num_feature_levels)):
                dec = nn.Sequential()
                dec.add_module(
                    name= f"up_{s}",
                    module = GridUnpooling(
                        in_channels = dec_channels[s+1] ,
                        skip_channels = config.enc_channels[s],
                        out_channels = dec_channels[s],
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                        use_interpolation=config.use_interpolation,
                        k=config.k,
                        traceable=True,  # record parent and cluster
                    )
                )
                dec.add_module(
                    name = f"dec_stage_{s}",
                    module = Stage(
                        feat_dim=dec_channels[s], #config.d_model,
                        depth= config.dec_depths[s],
                        order_num=len(config.order), 
                        mamba_config=config.mamba_config                        
                    )                
                )
                self.dec.add_module(f"dec_{s}", dec)
            
        # gather 操作用！
        self.gather_projs = nn.ModuleList()
        for s in reversed(range(self.num_feature_levels)): # 
            self.gather_projs.add_module(
                name=f"gather_proj_{s}",
                module=nn.Linear(dec_channels[s+1], config.d_model))
        # 
        self.use_interpolation = config.use_interpolation
        self.feature_propagation = FeatPropagation(group_size=config.k)      

    def transform(self, s_pc:PointCloud): #TODO: 这个地方才开始用到grid_size！
        s_pc.feat = torch.cat([s_pc.coord, s_pc.feat], dim=-1)  # BG 7 #此处融入坐标！        
        s_pc.order = self.order
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        s_pc.sparsify()
        return s_pc
    
    def gather_enc(self, s_pc:PointCloud, b_s:int):
        """如果骨干网只有编码器的部分，收集编码器的多层输出！
        Args:
            s_pc (PointCloud): 应当是解码器输出最后一层的特征输出！
            b_s (int): _description_

        Returns:
            s_pc (PointCloud): s_pc的feat被修改为编码器的多层输出！
        """
        r_pc = s_pc
        while "pooling_parent" in r_pc.keys():  # 找到encode_stage_0的输出!没有pooling_parent的就是!
            r_pc = r_pc["pooling_parent"]
        
        t_pc = s_pc  # 
        feats = []  
        while "pooling_parent" in t_pc.keys():
            if not self.use_interpolation:
                feat = t_pc.feat
                inverse = t_pc.pooling_inverse
                feats.append(feat)
                for i in range(len(feats)):
                    feats[i] =  feats[i][inverse]
            else:
                feat = self.feature_propagation(r_pc, t_pc)  # 这里要理解s_pc是祖先
                feats.append(feat)
            t_pc = t_pc["pooling_parent"]    
                
        for idx, module  in enumerate(self.gather_projs) :
            feats[idx] = rearrange(module(feats[idx]), "(b g) d -> b g d", b=b_s)
        
        feats.append(rearrange(r_pc.feat, "(b g) d -> b g d", b=b_s))  # 最后的主干网输出！
        r_pc.feat = feats        
        return r_pc
    
    def gather_dec(self, s_pc:PointCloud, b_s:int):
        """如果骨干网有解码器的部分，收集解码器的多层输出！
        Args:
            s_pc (PointCloud): #TODO: TOSEE 要搞明白 the data structure of s_pc
            b_s (int): _description_

        Returns:
            s_pc (PointCloud): s_pc的feat被修改为解码器的多层输出！
        """
        pc = s_pc
        #定位,找到需要的后代！
        for _ in range(self.num_feature_levels): 
            pc = pc["unpooling_parent"]
        #收集特征，并统一纬度！
        feats = []
        while "pooling_parent" in pc.keys():
            if not self.use_interpolation:
                feat = pc.feat
                inverse = pc.pooling_inverse
                feats.append(feat)
                for i in range(len(feats)):
                    feats[i] =  feats[i][inverse]
            else:
                feat = self.feature_propagation(s_pc, pc)  # 这里要理解s_pc是祖先
                feats.append(feat)
            pc = pc["pooling_parent"]     
                
        for idx, module  in enumerate(self.gather_projs) :
            feats[idx] = rearrange(module(feats[idx]), "(b g) d -> b g d", b=b_s)
        
        feats.append(rearrange(s_pc.feat, "(b g) d -> b g d", b=b_s))  # 最后的主干网输出！
        s_pc.feat = feats        
        return s_pc
            
    def forward(self, s_pc:PointCloud):
        b_s = s_pc.batch[-1]+1
        s_pc = self.transform(s_pc)                   
        s_pc = self.feature_encoder(s_pc)              # => (b g) d 

        s_pc = self.enc(s_pc)
        if self.has_decoder :
            s_pc = self.dec(s_pc)
            #TODO: 收集
            s_pc = self.gather_dec(s_pc, b_s)
        else:
            s_pc = self.gather_enc(s_pc, b_s)
        return s_pc
    
class PointSIS_Seg(nn.Module):
    """
    这部分是拿来训练的,预处理、後处理都没有学习的内容！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.pointsis_feature_extractor = PointSIS_Feature_Extractor(config)
        self.mask_decoder = MaskDecoder(config)
        #{Query:
        self.num_queries = config.num_queries
        # 可学习的查询！
        self.query_embedder = nn.Embedding(config.num_queries, config.d_model)
        # TODO：位置也是可学习的？？？ Mask2Former就是如此！！！ 
        self.query_position_embedder = nn.Embedding(config.num_queries, config.d_model)
        #}
        
        #{融合prompt:
        self.merge_prompt = nn.Linear(config.d_model+1, config.d_model)
        self.merge_prompt_pos = nn.Linear(config.d_model+1, config.d_model)
        #}
        #        
        self.loss = PMLoss(config)

    def forward(self, s_pc:PointCloud):
        # s_pc: "coord,feat,offset,grid_size,index_back_to_parent,s_o_i"可用，"labels,shape_weight"看情况!
        s_pc = self.pointsis_feature_extractor(s_pc)
        b_s = s_pc.batch[-1]+1
        # 上下颌分类！后面要考虑搞成promt embedding!
        s_o_i = s_pc.s_o_i                                              # b
        s_o_i = repeat(s_o_i,   "b -> (b q) 1", q = self.num_queries)
        s_o_i = rearrange(s_o_i,   "(b q) 1 -> b q 1", b = b_s)
        #
        query_embeddings = self.query_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1)      # q l => b q l
        query_position_embeddings = self.query_position_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1) # q l => b q l
        #Merge_Prompt!
        query_embeddings = self.merge_prompt(torch.cat([query_embeddings, s_o_i], dim=-1))
        query_position_embeddings = self.merge_prompt_pos(torch.cat([query_position_embeddings, s_o_i], dim=-1))
        
        # TODO:取名mask_features是为了和mask2former对应！但没有想通的问题是，s_pc.feat到底应该是什么结构！
        mask_features =  s_pc.feat[-1]    # s_pc.feat 此时是收集起来的一个feat list!
        encoder_hidden_states =  s_pc.feat[0:-1]  #config.num_feature_levels控制！ 其实就是主干网的那几层输出!
        # TODO:有个问题,mask_features,encoder_hidden_states是否需要序列化?
        # 在Transformer机制下，可以先不考虑?作也容易！
        # 是否需要为他们添加位置编码?
        pred_mask, pred_probs = self.mask_decoder(                                   #       -> b q g , b q d
                            query_embeddings = query_embeddings,
                            query_position_embeddings = query_position_embeddings,
                            mask_features = mask_features,             #       编码主干网的最后一层输出！
                            encoder_hidden_states= encoder_hidden_states    #       编码主干网的下面几层的输出！
                        )

        # TODO: pred_probs和 pred_mask 都无须activation！loss里面有！
        if "labels" in s_pc.keys():    # 如果有标签，就计算loss！！！
            labels = rearrange(s_pc.labels, "(b g) -> b g", b=b_s)
            shape_weight = rearrange(s_pc.shape_weight, "(b g) -> b g", b=b_s) if s_pc.shape_weight is not None else None
            m_i = self.loss(pred_mask,pred_probs,labels, shape_weight)  # 
            s_pc.loss = m_i
        pred_mask = rearrange(pred_mask,"b q g -> b g q")
        pred_mask = rearrange(pred_mask, "b g q -> (b g) q")
        s_pc.feat = pred_mask.contiguous()       # FIXME:老问题 s_pc的feat过载太多，看怎么清晰一下！！！
        s_pc.pred_probs = pred_probs
        return s_pc
    
class PointSIS_Seg_Model(nn.Module):
    """
    注意,训练的是PointSIS_Seg,用的是PointSIS_Seg_Model!
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.grouper = Grouper_By_NumGroup(config.num_group, config.group_size)
        self.feat_propagation = FeatPropagation(config.group_size)
        self.model = PointSIS_Seg(config)
    def load_state_dict(self, state_dict, strict = True, assign = False):
        self.model.load_state_dict(state_dict,strict = strict, assign = assign)    # It means, this class of self has no parameters to be trained!
        return self
    
    def forward(self, parent_pc:PointCloud):
        s_pc = self.model(self.grouper(parent_pc))
        parent_pc.feat = self.feat_propagation(parent_pc, s_pc)
        parent_pc.pred_probs = s_pc.pred_probs
                
        if s_pc.loss is not None:
            parent_pc.loss = s_pc.loss
        del s_pc
        
        return parent_pc

