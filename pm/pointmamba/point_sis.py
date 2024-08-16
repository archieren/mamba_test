from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
# class MLP(nn.Module): # 这个先留着，看看能不能用！
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, bias=False):  #
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

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



class SwinMixers(nn.Module):        # 这儿，对Patch，进行飘移操作，应当和Swin的思路一样！！          
    """
    """
    def __init__(self, config):
        super().__init__()
        self.cascade = config.cascade
        self.block_0 = self.create_block(config.d_model, config.mamba_config, 0)
        self.block_1 = self.create_block(config.d_model, config.mamba_config, 0)  # shift
        self.norm_f_0 = nn.LayerNorm(config.d_model)
        self.norm_f_1 = nn.LayerNorm(config.d_model)
        self.merge = nn.Linear(config.d_model * 2, config.d_model)
        self.norm_f = nn.LayerNorm(config.d_model)   # TODO: 待评估！！！


    @staticmethod
    def create_block(d_model, mamba_cfg, layer_idx):  
        # 直接用Mamba里实现的Block，基本用缺省值!
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **mamba_cfg)
        norm_cls = partial(nn.LayerNorm )
        mlp_cls = nn.Identity
        block = Block(d_model, mixer_cls, mlp_cls , norm_cls=norm_cls,)  # 可以了解，Block里的缺省路径 Add -> LN -> Mixer
        # block.layer_idx = layer_idx
        return block
    
    def forward(self, hidden_states,shift,shift_back, patch_size):  
        # 0
        residual = None
        hidden_states_0 = hidden_states

        hidden_states_0 = rearrange(hidden_states_0, " (n p) d -> n p d", p = patch_size)
        hidden_states_0, residual = self.block_0(hidden_states_0, residual)
        hidden_states_0 = (hidden_states_0 + residual) if residual is not None else hidden_states_0
        hidden_states_0 = self.norm_f_0(hidden_states_0.to(dtype=self.norm_f_0.weight.dtype)) #TODO
        hidden_states_0 = rearrange(hidden_states_0, " n p d -> (n p) d")
        # 1
        residual = None
        if not self.cascade: # 并联
            hidden_states_1 = hidden_states[shift]
        else:  # 串联
            hidden_states_1 = hidden_states_0[shift]    # 

        hidden_states_1 = rearrange(hidden_states_1, " (n p) d -> n p d", p = patch_size)
        hidden_states_1, residual = self.block_1(hidden_states_1,residual)
        hidden_states_1 = (hidden_states_1 + residual) if residual is not None else hidden_states_1
        hidden_states_1 = self.norm_f_1(hidden_states_1.to(dtype=self.norm_f_1.weight.dtype)) #TODO
        hidden_states_1 = rearrange(hidden_states_1, " n p d -> (n p) d")
        hidden_states_1 = hidden_states_1[shift_back]
        if not self.cascade: # 并联
            hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=-1)
            hidden_states = self.merge(hidden_states)
            hidden_states = self.norm_f(hidden_states.to(self.norm_f.weight.dtype))
        else: # 串联
            hidden_states = hidden_states_1

        return hidden_states    


class PointSIS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.order = [config.order] if isinstance(config.order, str) else config.order
        self.repeats = config.repeats
        self.shuffle_orders = config.shuffle_orders
        self.patch_size = config.patch_size
        self.config = config
        # TODO:
        self.feature_embedding =  MLP(3, config.d_model, config.d_model)
        self.pos_embedding =  MLP(3, config.d_model, config.d_model)
        self.tokening = nn.Linear(config.d_model*2, config.d_model)
        self.swin_layers = nn.ModuleList([SwinMixers(config) for i in range(len(self.order)*self.repeats)])  # 2*4*3,


    def forward(self, pc:PointCloud):
        pc.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        pad, unpad, shift, shift_back = pc.get_padding_and_inverse(self.patch_size)
        #对于Mesh数据，是否暂时可以假设数据的密度比较一致，并且normals可以充当feature呢？
        feat = self.feature_embedding(pc.feat)
        pos  = self.pos_embedding(pc.coord)
        f = feat
        for i , block in enumerate(self.swin_layers):
            the_order = i % (len(self.order))
            if the_order==0:
                f = torch.cat([f, pos], dim=-1)           # TODO: 这也是一种策略，每轮重复前都将POS加上，而不是只在开始的时候！
                f = self.tokening(f)
            f = f[pc.serialized_order[the_order]]
            f = f[pad]
            f = block(f, shift, shift_back, self.patch_size)
            f = f[unpad]
            f = f[pc.serialized_inverse[the_order]]
        return f

    
class PointSIS_SEG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pointsis = PointSIS(config)
        self.seg = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
            nn.Softmax(dim=-1)           
        )

    def forward(self, pc:PointCloud):
        f = self.pointsis(pc)
        seg = self.seg(f)
        return seg

