import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange,einsum,repeat
from typing import Optional, Tuple

from pm.pointmamba.conifuguration_point_sis import  PointSISConfig

"""
在point_sis_masked_former里,出现训练结果很好,但即使用训练数据做测试,结果也极差的情况!不知为何！
也许在输入的时候,以某种方法得到输入的shape encoding，然后再作Feature Encoding！
思考
1）Perceiver或Perceive IO结构,情况会发生变化!
2）VecSet一系列的工作(3DShape2VecSet,Dora-Vae,Crafts3DMan-Vae,COD-Vae等).
3) StruMamba3D里面提供的"处理形状结构"方法!
期望什么呢?
"""