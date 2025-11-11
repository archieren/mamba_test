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
        self.encoder = nn.Sequential(
            nn.Linear(in_channel, self.e_i *2),
            nn.GELU(),
            nn.Linear(self.e_i * 2, out_channel)            
        )
    
    def forward(self, s_pc:PointCloud):
        """
        BG 4 -> BG C
        """
        feat = self.encoder(s_pc.feat)
        s_pc.feat = feat
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc

class CPE(nn.Module):
    """
    Maybe, this is the conditional position encode
    """
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.e_o = out_channels
        self.spconv_f = spconv.SubMConv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,bias=True)
        self.fc1      = nn.Linear(out_channels, out_channels)
        self.norm_f   = nn.LayerNorm(out_channels)
    
    def forward(self, s_pc:PointCloud):
        # s_pc.feat和s_pc.sparse_conv_feat.features 应当是一致的！
        short_cut = s_pc.feat
        s_pc.sparse_conv_feat = self.spconv_f(s_pc.sparse_conv_feat)
        s_pc.feat = s_pc.sparse_conv_feat.features
        s_pc.feat = self.fc1(s_pc.feat)
        s_pc.feat = self.norm_f(s_pc.feat)
        s_pc.feat = short_cut + s_pc.feat
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

class Stage(nn.Module):
    """
    """
    def __init__(self, d_model, depth, order_num, mamba_config):
        super().__init__()
        self.cpe = CPE(d_model, d_model)
        self.mixer_layer = MixerLayers(d_model=d_model, depth= depth, mamba_config=mamba_config)
        self.fuse_o = nn.Sequential(                                 #合并各排序的特征。
            nn.Linear(d_model * order_num, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),                                               # TODO：看看融合后，要不要激活函数
            nn.Linear(d_model, d_model)
        )        
    def forward(self, s_pc:PointCloud):
        s_pc = self.cpe(s_pc)
        s_order = s_pc.serialized_order   # o (b g)
        s_inverse = s_pc.serialized_inverse  # o (b g)
        b_s = s_pc.batch[-1]+1
        o_s = s_order.shape[0]       
        # s_order, s_inverse 均为 order_size batch_size*num_group   
             
        hidden_state  = s_pc.feat   # (b g) d  # coord 有三个分量！ normals 有三个分量, point_curvature占一个分量！
        #将各排序拼接,走Mamba流程！
        hidden_state = hidden_state.unsqueeze(0).repeat(o_s,1,1)                 # 为每种排序准备排序的数据,(b g) d => o (b g) d
        hidden_state = torch_scatter.scatter(hidden_state,index=s_order, dim=1)  # 排序: o (b g) d, o (b g) => o (b g) d       
        hidden_state = rearrange(hidden_state, "o (b g) d -> b (o g) d", b=b_s)  # 将各个排序拼接,参看PointMamba的第四版!!      
        hidden_state = self.mixer_layer(hidden_state)
        #将同一点，在各排序情况下的特征拼接到一起！
        # 将阶段性结果输出！                                    #  
        hidden_state = rearrange(hidden_state, "b (o g) d -> o (b g) d", o=o_s).contiguous()  # 拆分各排序
        hidden_state = torch_scatter.scatter(hidden_state,index=s_inverse, dim=1)# 逆排序 => order_size batch_size*num_group d
        hidden_state = rearrange(hidden_state, "o (b g) d -> b g (o d)", b= b_s) # 调整，将同一点，在各排序情况下的特征拼接到一起！
        hidden_state = self.fuse_o(hidden_state)         # FIXME:注意,hidden_state没有融合,这样似乎好点！！
        hidden_state = rearrange(hidden_state, "b g d -> (b g) d")               # (b g) d 
        s_pc.feat = hidden_state
        s_pc.sparse_conv_feat = s_pc.sparse_conv_feat.replace_feature(s_pc.feat)
        return s_pc         
        

class PointSIS_Feature_Extractor(nn.Module):
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.num_group = config.num_group
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.shuffle_orders = config.shuffle_orders
        self.config = config
        self.feature_encoder = Feature_Encoder(config.in_channels, config.d_model)
        self.stages = nn.ModuleList([Stage(d_model=config.d_model, 
                                           depth= d,
                                           order_num=len(config.order), 
                                           mamba_config=config.mamba_config)
                                     for d in config.depth])
        
        self.category = nn.Linear(config.d_cat, config.feature_dims)  # 输入的数据范畴,目前只有有上下之分！      

    def transform(self, s_pc:PointCloud):
        # TODO:: 这一段代码比较臭:原因在于什么样的输入特征及位置编码用作网络的输入！
        # s_意味着初始下采样，已经做了
        # TODO:后面的多尺度下采样，看情况再做！
        # TODO:下面这段的改法，可能影响最小！
        #上下颌指示！
        s_s_o_i = s_pc.s_o_i                 # b     # s_o_i        
        #s_s_o_i = self.category(repeat(s_s_o_i, "b -> (b g) 1", g = self.num_group)) # b -> (b g) d  # TODO!
        s_s_o_i = repeat(s_s_o_i, "b -> (b g) 1", g = self.num_group)
        s_pc.feat = torch.cat([s_pc.coord, s_pc.feat, s_s_o_i], dim= -1)
        
        s_pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        s_pc.sparsify()
        return s_pc        
    def forward(self, s_pc:PointCloud):
        s_pc = self.transform(s_pc)                   
        s_pc = self.feature_encoder(s_pc)              # => (b g) d 
        
        #TODO: 怎么作下采样呢？                    # 需不需要?
        for _ , stage in enumerate(self.stages):
            s_pc = stage(s_pc)  
        return s_pc
    

    
class PointSIS_Seg(nn.Module):
    """
    这部分是拿来训练的,预处理、後处理都没有学习的内容！
    """
    def __init__(self, config:PointSISConfig):
        super().__init__()
        self.pointsis_feature_extractor = PointSIS_Feature_Extractor(config)
        self.mask_decoder = MaskDecoder(config)
        #
        self.num_queries = config.num_queries
        self.query_embedder = nn.Embedding(config.num_queries, config.d_model)        # 可学习的查询！
        self.query_position_embedder = nn.Embedding(config.num_queries, config.d_model)   # TODO：位置也是可学习的？？？ Mask2Former就是如此！！！
        #
        self.class_predict = nn.Linear(config.d_model, config.num_labels+1)
        #        
        self.loss = PMLoss(config)

    def forward(self, s_pc:PointCloud):
        # s_pc: "coord,feat,offset,grid_size,index_back_to_parent,s_o_i"可用，"labels,shape_weight"看情况!
        # 关于s_o_i,必须注意,将输入调整到统一的姿势,最终看来,还是必要的!
        s_pc = self.pointsis_feature_extractor(s_pc)
        b_s = s_pc.batch[-1]+1
        #
        s_pc.feat = rearrange(s_pc.feat, "(b g) d -> b g d", b=b_s)
        #
        query_embeddings = self.query_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1)
        query_position_embeddings = self.query_position_embedder.weight.unsqueeze(0).repeat(b_s, 1, 1)
        # TODO:To be refactored！
        point_embedding = s_pc.feat #s_pc.feat[-1]
        encoder_hidden_states = [s_pc.feat, s_pc.feat]     #s_pc.feat[0:-1]  #config.num_feature_levels控制！
        # TODO:有个问题,mask_decoder的参数point_embedding,encoder_hidden_states是否需要序列化?在Transformer机制下，可以先不考虑?作也容易！
        pred_mask, q = self.mask_decoder(                               # -> b q g , b q d
                            query_embeddings = query_embeddings,
                            query_position_embeddings= query_position_embeddings,
                            point_embeddings = point_embedding,
                            encoder_hidden_states= encoder_hidden_states)

        pred_probs = self.class_predict(q)                             # b q d -> b q l      # l代表num_labels+1
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

