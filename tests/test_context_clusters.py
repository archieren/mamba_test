"阅度Image as point set 的"
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple
from einops import rearrange
import torch.nn.functional as F

class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size, stride, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :控制区域里的锚点数,值点数
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :控制区域数!
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h

    def forward(self, p):  # [b,c,w,h]
        p_v = self.v(p)  # 所谓的值空间. 值空间讲的是中心.
        p_s = self.f(p)  # 所谓的相似空间. 相似空间讲的是锚点.
        p_s = rearrange(p_s, "b (e c) w h -> (b e) c w h", e=self.heads)
        p_v = rearrange(p_v, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:  
            # region 区域划分,看了下源代码的内容,最终保证每个区域,w=h=7,共49个点
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = p_s.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            p_s = rearrange(p_s, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            p_v = rearrange(p_v, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)

        b, c, w, h = p_s.shape
        sim_centers   = rearrange(self.centers_proposal(p_s), 'b c w h -> b (w h) c')  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        
        # sim的计算
        p_s = rearrange(p_s, 'b c w h -> b (w h) c')
        sim = torch.sigmoid(self.sim_beta +self.sim_alpha * pairwise_cos_sim(sim_centers,p_s))  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N] 注意 mask 起得作用!!!分cluster了!
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask

        value_centers = rearrange(self.centers_proposal(p_v), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        p_v = rearrange(p_v, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        p_g = ((p_v.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        # dispatch step, return to each point in a cluster
        p_g = (p_g.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,M,1,C] * [B,M,N,1] => [B,M,N,C].sum(dim=1) => [B,N,D] @注意
        p_out = rearrange(p_g, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            p_out = rearrange(p_out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        p_out = rearrange(p_out, "(b e) c w h -> b (e c) w h", e=self.heads)
        p_out = self.proj(p_out)
        return p_out

class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x
    
class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))  # 这...
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x))) # 或这,才完成Feature Dispatching
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def test_Cluster():
    from torchinfo import summary
    embed_dim = 64
    batch, channel, width, height = 2, embed_dim, 56, 56
    x = torch.randn(batch, channel, width, height).to("cuda")
    model_layer = Cluster( 
            dim=embed_dim,
            out_dim=embed_dim, 
            proposal_w=2, 
            proposal_h=2, 
            fold_w=8, 
            fold_h=8, 
            heads=4, 
            head_dim=24
            ).cuda()
    # input_shape = (batch, channel, width, height)
    # summary(model_layer, input_shape)

    y = model_layer(x)
    print(y.shape)

def test_ClusterBlock():
    from torchinfo import summary
    embed_dim = 64
    batch, channel, width, height = 2, embed_dim, 56, 56
    x = torch.randn(batch, channel, width, height).to("cuda")
    model_layer = ClusterBlock( 
            dim=embed_dim,
            proposal_w=2, 
            proposal_h=2, 
            fold_w=8, 
            fold_h=8, 
            heads=4, 
            head_dim=24
            ).cuda()
    input_shape = (batch, channel, width, height)
    summary(model_layer, input_shape)

    y = model_layer(x)
    print(y.shape)

def test_broadcast_mechanism():
    """
    经常脑袋转不开的东西.
    """
    B, M, N, D = 1, 2, 4, 2
    p_g = torch.randn(B, M, D)
    sim = torch.randn(B, M, N)
    
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)
    sim = sim * mask

    print(sim)
    print(sim.shape)
    print(p_g.unsqueeze(dim=2).shape)
    print(sim.unsqueeze(dim=-1).shape)
    p_g = (p_g.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1) 
    print(p_g.shape)

    x= torch.randn(1,3)
    y= torch.randn(3,1)
    print(x)
    print(y)
    print(x*y)

test_Cluster()
# test_ClusterBlock()
# test_broadcast_mechanism()