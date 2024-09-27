import sys
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from pm.utils.point_cloud import PointCloud


# class Feature_Encoder(nn.Module):        # 改自Point Mamba！ 但当特征为normals and point curvature时，就不需要了
#     def __init__(self, encoder_channel):
#         super().__init__()
#         self.e_o = encoder_channel       # 特征编码输出的通道数！
#         self.e_i = 128                   # 特征编码内部使用的通道数！
#         self.first_conv = nn.Sequential(
#             nn.Linear(3,self.e_i), 
#             nn.LayerNorm(self.e_i),
#             nn.GELU(),                       
#             nn.Linear(self.e_i, self.e_i *2)
#         )
#         self.second_conv = nn.Sequential(
#             nn.Linear(self.e_i *4, self.e_i *4),
#             nn.LayerNorm(self.e_i *4 ),
#             nn.GELU(),
#             nn.Linear(self.e_i *4, self.e_o)
#         )

#     def forward(self, feature):
#         '''
#             point_groups : BG N 3  ( N 邻居的数量)
#             -----------------
#             feature_global : BG C
#         '''
#         BG, N, C = feature.shape
#         # encoder                                                     # 
#         feature = self.first_conv(feature)                            # BG N 3  -> BG N e_i*2 
#         feature_global = torch.max(feature, dim=1, keepdim=True)[0]   # BG N e_i*2 -> BG 1 e_i*2  # 为什么是max?
#         feature_global = feature_global.expand(-1, N, -1)             # BG 1 e_i*2 -> BG N e_i*2
#         feature = torch.cat([feature, feature_global], dim=-1)        # BG N e_i*2 BG N e_i*2  -> BG N e_i*4
#         feature = self.second_conv(feature)                           # BG N e_i*4 -> BG N C
#         feature_global = torch.max(feature, dim=1, keepdim=False)[0]  # BG N C -> BG C
#         return feature_global

class PCModule(nn.Module):
    r"""
    此模块的子类都从PointSparseSequential中取参数!.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PCSequential(PCModule):
    r"""
    A sequential container.
    (类似mmcv中的SparseSequential.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PCModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, PointCloud):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, PointCloud):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input
