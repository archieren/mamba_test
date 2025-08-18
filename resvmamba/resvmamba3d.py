
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import partial
from einops import rearrange, repeat
from dataclasses import dataclass, asdict
#from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2 as Mamba
# 直接使用mamba推荐的Block.
# 这需要看片文章"On Layer Normalization in the Transformer Architecture"
from mamba_ssm.modules.block import Block

#按照Res-Vmamba的思路,写一个简单的3D Res-Vmamba.
#网络结构不复杂!讨厌的是,很多人的总是不愿意直接调用Mamba的模块！

@dataclass
class MambaConfig():     # 抄自 Mamba的初始化参数!!!
    # 要注意确保 d_model * expand / headdim = multiple of 8
    # d_model: int        # D in paper/comments
    d_state: int = 64   # N in paper/comments
    d_conv:  int = 4
    expand:  int = 2     # E in paper/comments
    headdim: int = 24    # Default is 64
    # dt_rank:    Union[int, str] = 'auto'
    # dt_min:     float = 0.001
    # dt_max:     float = 0.1
    # dt_init:    str = "random" # "random" or "constant"
    # dt_scale:   float = 1.0
    # dt_init_floor: float = 1e-4

    # conv_bias: bool = True
    # bias:      bool = False
    # use_fast_path: bool = True  # Fused kernel options

    #layer_idx = None
    # device    = None
    # dtype     = None

@dataclass
class ResVMamba3dConfig():
    cube_size = 4                          # 4 for 128; 2 for 64, 3 for 96
    in_chans = 2                           # 
    mamba_config = asdict(MambaConfig()) 
    num_classes=2 
    depths=[2, 2, 9, 2] 
    dims=[96, 96 , 192, 384, 768]  # 注意为什么哈！
    drop: float = 0.4
    drop_path: float = 0.1

@dataclass
class ResVMUnet3dConfig():
    cube_size = 8 
    in_chans =1
    mamba_config = asdict(MambaConfig()) 
    depths=[2, 2, 2, 2] 
    dims=[96, 96 , 192, 384, 768]  # 注意为什么哈！
    drop: float = 0.0
    drop_path: float = 0.1

         
def make_default_config(net_type='resvmamba'):
    if net_type == 'resvmamba':
        config = ResVMamba3dConfig()
    else :
        config =ResVMUnet3dConfig()
    return config
       
class Mlp(nn.Module):
    def __init__(self, 
                 c_in,
                #  为了需要，不用下面这种方式？ 
                #  in_features, 
                #  hidden_features, 
                #  out_features, 
                mlp_ratio = 2,
                act_layer=nn.GELU, 
                drop=0.4,
                ):
        super().__init__()
        inner_hidden_dim = c_in * mlp_ratio
        self.fc1 = nn.Linear(c_in, inner_hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(inner_hidden_dim, c_in)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    ### Squeeze and Excitation Class definition
class SE3D(nn.Module):
    def __init__(self, channel, reduction_ratio =16):
        super().__init__()
        ### Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.LeakyReLU(),     #nn.GELU(),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ , _ = x.size()   # b c i j k
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class CrossScan3DToSequences(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_dim = 1
        
    def scan(self, x: torch.Tensor):           #  -3维为主纬度! 
        dim0,dim1 = -2 , -1              # 后两维介入转置！
        start_dim, end_dim = -3, -1
        # I 
        x0 = x.flatten(start_dim, end_dim).unsqueeze(self.scan_dim) # -> (B,1,C,I*J*K)
        x1 = x.transpose(dim0=dim0, dim1=dim1).flatten(start_dim, end_dim).unsqueeze(self.scan_dim)
        x2 = torch.flip(x0, dims = [end_dim])
        x3 = torch.flip(x1, dims = [end_dim])
        return torch.cat([x0,x1,x2,x3], dim=1)                       # -> (B,4,C,I*J*K)
    
    def forward(self, x: torch.Tensor): #(B,C,I,J,K) -> (B,S,C,I*J*K)=(B,S,C,L)        
        #B, C, D, H, W = x.shape      # Batch Channel Depth Hight Width 
        #B, C, I, J, K = x.shape       # Or: Batch Channel I J K
        #S = 4*3                        # 扫描的次序有12中,正向6,逆向6!
        # assert I == J and I == K, "嘿嘿,改吧!"
 
        
        xs_i = rearrange(x, "b c i j k -> b c i j k")        # 凑个形式！
        xs_i = self.scan(x).contiguous()                     # -> (B, 4, C, L)
        
        xs_j = rearrange(x, "b c i j k -> b c j k i")          
        xs_j = self.scan(xs_j).contiguous()                  # -> (B, 4, C, L)
        
        xs_k = rearrange(x, "b c i j k -> b c k i j")
        xs_k = self.scan(xs_k).contiguous()                  # -> (B, 4, C, L) 
        
        return torch.cat([xs_i, xs_j, xs_k], dim=1)          # -> (B, 12,C, L)    一个Image，得到12个Sequence！

class MergeCrossScanSequencesTo3D(nn.Module):
    # 仔细读了VMamba的代码，它有太多的将就之处，而且，有些搞法和怪异！不好说！
    def __init__(self, c_out, **kwargs):
        super().__init__( **kwargs)
        self.scan_dim = 1
        self.reduct_f = nn.Conv3d(12*c_out,c_out,kernel_size=1,stride=1)  # 等效nn.linear
        self.norm_f = nn.BatchNorm3d(c_out)
            
    def forward(self, ys:torch.tensor,i,j,k):  #  b s c l -> b c i j k
        B, _, _, _ = ys.shape  
        scan_dim = self.scan_dim
        end_dim = -1 

        
        def merge(x:torch.Tensor,i,j,k):     # 
            x0,x1,x2,x3 = x.split([1,1,1,1], dim=scan_dim)
            x_0_2 = torch.cat([x0.squeeze(scan_dim),
                               x2.squeeze(scan_dim).flip(dims=[end_dim])],
                              dim=1)
            x_1_3 = torch.cat([x1.squeeze(scan_dim),
                               x3.squeeze(scan_dim).flip(dims=[end_dim])],
                              dim=1)
            y = torch.cat([x_0_2.view(B,-1,i,j,k),
                           x_1_3.view(B,-1,i,k,j).transpose(dim0=-2, dim1=-1)],
                          dim=1)
            return y

        xs_i, xs_j, xs_k = ys.split([4,4,4], dim=1)
        xs_i = merge(xs_i,i,j,k)                                     # 注意
        xs_i = rearrange(xs_i, "b c i j k ->b c i j k")        # 凑个形式
        xs_j = merge(xs_j,j,k,i) # 恢复原序!
        xs_j = rearrange(xs_j, "b c j k i ->b c i j k")
        xs_k = merge(xs_k,k,i,j) # 恢复原序!
        xs_k = rearrange(xs_k, "b c k i j ->b c i j k")
        
        xs = torch.cat([xs_i,
                        xs_j,
                        xs_k],
                       dim=1)   # b 12*c j j k
        xs = self.reduct_f(xs)
        xs = self.norm_f(xs)
        return  xs

class CubeDownSample(nn.Module): #用卷积实现所谓的Space_To_Depth 
    def __init__(self, c_in:int, c_out:int, s:int=2, is_s = True):
        super().__init__()
        self.conv_f = nn.Conv3d(c_in, c_out,kernel_size=s,stride=s)
        #最后一个不作conv_f_s
        #self.conv_f_s = nn.Conv3d(c_out, c_out,kernel_size=3,padding='same',stride=1) if is_s else nn.Identity()
        self.norm_f = nn.BatchNorm3d(c_out)

    def forward(self, x): 
        #设定输入的Tensor:(B,C,D,H,W)或(B,C,I,J,K)
        x = self.conv_f(x)
        #x = self.conv_f_s(x)
        x = self.norm_f(x)
        return x

class CubeEmbedding(nn.Module):
    def __init__(self, c_in, c_out, cube_size):
        super().__init__()
        self.conv_f = nn.Conv3d(c_in, c_out,kernel_size=cube_size,stride=cube_size)
        self.conv_f_s = nn.Conv3d(c_out, c_out,kernel_size=3,padding='same',stride=1)
        self.norm_f = nn.BatchNorm3d(c_out)
        
    def forward(self, x:torch.Tensor):
        # x.shape : "b c i j k"
        x = self.conv_f(x)
        res = self.conv_f_s(x)   # 弄个简单的残差！
        x = x + res
        x = self.norm_f(x)
        return x
    
# class CubeEmbedding(nn.Module):
#     def __init__(self, c_in, c_out, cube_size):
#         super().__init__()
#         self.embed_f = nn.Linear(c_in*(cube_size**3), c_out)
#         self.norm_f = nn.LayerNorm(c_out)
#         self.cube_size = cube_size
        
#     def forward(self, x:torch.Tensor):
#         # x.shape : "b c i j k"
#         s = self.cube_size
#         y = rearrange(x, "b c i j k -> b i j k c")
#         y = rearrange(y, 
#                       "b (i s1) (j s2) (k s3) c -> b i j k (c s1 s2 s3)",
#                       s1=s, s2=s, s3=s,
#                       )
#         y = self.embed_f(y)
#         y = self.norm_f(y)
#         y = rearrange(y, "b i j k c -> b c i j k")
#         return y

# class CubeDownSample(nn.Module): #所谓的Space_To_Depth #和那个CubeEmbedding一模一样！
#     def __init__(self, c_in:int, c_out:int, s:int=2):
#         super().__init__()
#         self.s :int = s
#         self.s_to_d :int = s**3
#         self.reduction = nn.Linear( self.s_to_d * c_in, c_out, bias=False)
#         self.norm = nn.LayerNorm(c_out)

#     def forward(self, x): 
#         #设定输入的Tensor:(B,C,D,H,W)或(B,C,I,J,K)
#         x = rearrange(x, " b c i j k -> b i j k c")
#         x = rearrange(x, "b (i s1) (j s2) (k s3) c -> b i j k (c s1 s2 s3)",s1=self.s,s2=self.s,s3=self.s)  # 
#         x = self.reduction(x)
#         x = self.norm(x)
#         x = rearrange(x, "b i j k c -> b c i j k")
#         return x

class CubeUpExpand(nn.Module): # 流着备用! 后面可能搞个MM-Unet之类的！
    def __init__(self, c_in, up_scale):
        super().__init__()
        self.up_scale = up_scale
        self.expand = nn.Linear(c_in, (up_scale**3)*c_in, bias=False)
        self.norm_f = nn.LayerNorm(c_in)

    def forward(self, x):
        # x.shape : "b c i j k"
        y = rearrange(x, "b c i j k -> b i j k c")
        y = self.expand(y)   # "b c i j k -> b i j k (up_scale up_scale up_scale c)"

        s = self.up_scale
        y = rearrange(y, 
                      'b i j k ( c s1 s2 s3)-> b (i s1) (j s2) (k s3) c', 
                      s1=s, s2=s, s3=s, 
                      )
        y = self.norm_f(y)
        y = rearrange(y, "b i j k c -> b c i j k")
        return y            

                    
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
    def create_block(d_model:int, mamba_cfg:MambaConfig, layer_idx):  
        # 直接用Mamba里实现的Block，基本用缺省值!
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **mamba_cfg)
        norm_cls = partial(nn.LayerNorm)
        mlp_cls =  partial(Mlp) # nn.Identity     #
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
    def __init__(self, c_in, c_out, depth, mamba_config, is_res=False):
        super().__init__()
        self.is_res = is_res
        self.down_sampler = CubeDownSample(c_in=c_in, c_out=c_out)
        self.cross_scan = CrossScan3DToSequences()
        self.mixlayers = MixerLayers(c_out, depth, mamba_config)  # c_out as d_model
        self.cross_merge = MergeCrossScanSequencesTo3D(c_out)
        self.se3d = SE3D(c_out)
    
    def forward(self, x:torch.Tensor):
        x_in = self.down_sampler(x)    # 下采样！
        B,_,I,J,K = x_in.shape       
        #
        res = self.cross_scan(x_in)  # b c i j k ->  b s c (i*j*k)
        _,S,_,_ = res.shape
        #
        res = rearrange(res, "b s c l -> b (s l) c")
        res = self.mixlayers(res)   # b (s l) c ->  b (s l) c
        #
        res = rearrange(res,"b (s l) c -> b s c l", b=B, s=S)
        res = self.cross_merge(res, i=I, j=J, k=K)   # b s c l -> b (s c) i j k

        res = self.se3d(res)                        # Squeeze and Excition ? 标准SE！
        y =  res + x_in if self.is_res else res    # 就这一个加号,一篇文章！
        
        return y

class Emb(nn.Module):
    def __init__(self, num_features): #, num_classes):
        super().__init__()
        self.norm_f = nn.BatchNorm3d(num_features) #nn.LayerNorm(num_features)
        #self.head_f = nn.Linear(num_features, num_classes)
        self.avgpool_f = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x:torch.Tensor):
        #x.shape : "b c i j k"
        x = self.norm_f(x)      # -> b c i j k
        x = self.avgpool_f(x)   # -> b c 1 1 1
        x = x.flatten(1)        # -> b c
        #y = self.head_f(x)      # -> b c'
        return x
        
               
class ResVMamba3D(nn.Module):
    def __init__(self, config:ResVMamba3dConfig):
        super().__init__()
        self.num_stages = len(config.depths)
        self.depths = config.depths
        self.dims = config.dims
        self.drop_f = nn.Dropout3d(p=config.drop)
        
        embed_dims = config.dims[0]
        self.cube_embeder = CubeEmbedding(c_in=config.in_chans, c_out=embed_dims, cube_size=config.cube_size)
        
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            c_in = self.dims[i_stage]
            c_out = self.dims[i_stage+1]
            depth = self.depths[i_stage]
            mamba_config = config.mamba_config
            is_res = True
            self.stages.append(Stage(c_in, c_out, depth, mamba_config, is_res=is_res))
        
    def forward(self, x:torch.Tensor):
        #x.shape : "b c i j k"
        y = self.cube_embeder(x)
        y = self.drop_f(y)
        #y.shape : "b c i j k"
        for i , stage in enumerate(self.stages):
            y = stage(y)

            
        return y

class ArcFace(nn.Module):
    def __init__(self, feature_size, num_classes, K:int=4, scale=64, margin=0.5, easy_margin=False, **kwargs):
        super().__init__()
        self.K = K
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes*K, feature_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor=None):
        if labels==None :
            return embeddings, None, None
        
        _,indices_as_cls = torch.max(labels, dim=1)
        # 计算余弦相似度！ K个子类中心！
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        cos_theta, _ = torch.max(rearrange(cos_theta, 'b (c k) -> b c k', k=self.K), dim=-1)
        # 提取目标类别的余弦值、正弦值!
        cos_theta_y = torch.gather(cos_theta, 1, indices_as_cls.view(-1,1))
        sin_theta_y = torch.sqrt((1.0 - torch.pow(cos_theta_y, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        # 处理Margin!
        cos_theta_y_m = cos_theta_y * self.cos_m - sin_theta_y * self.sin_m   # 两角和余弦公式!
        if self.easy_margin:
            cos_theta_y_m = torch.where(cos_theta_y > 0, cos_theta_y_m, cos_theta_y)
        else:
            cos_theta_y_m = torch.where(cos_theta_y > self.th, cos_theta_y_m, cos_theta_y - self.mm)
        # 修改目标类别的余弦相似度
        output = torch.scatter(cos_theta, 1, indices_as_cls.view(-1,1), cos_theta_y_m)
        # output = cos_theta + one_hot
        output *= self.scale
        
        loss = F.cross_entropy(output, labels) #+ F.cross_entropy(cos_theta, labels)

        _, indices_as_preds = torch.max(output, dim=1)
        correct_mask = (indices_as_preds == indices_as_cls)
        
        return embeddings, loss, correct_mask
        
class ResVMamba3dClassifierModel(nn.Module):
    def __init__(self, config:ResVMamba3dConfig):
        super().__init__()
        self.resvmamva3d = ResVMamba3D(config)
        
        num_features = config.dims[-1]
        num_classes = config.num_classes
        self.emb = Emb(num_features=num_features) # ,num_classes=num_classes)
        self.loss = ArcFace(feature_size=num_features, num_classes=num_classes)
    def forward(self, x:torch.Tensor, labels:torch.Tensor=None):
        feat = self.resvmamva3d(x)
        emb = self.emb(feat)
        #return cls, feat
        emb, loss, correct_mask = self.loss(emb, labels)
        return emb , loss , correct_mask      
        




    
class ResVMUnet3d(nn.Module):
    def __init__(self, config:ResVMUnet3dConfig):
        pass