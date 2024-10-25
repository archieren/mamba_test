import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange,einsum,repeat
from typing import Optional, Tuple

from pm.pointmamba.conifuguration_point_sis import  PointSISConfig

"""
在point_sis里,对数据的泛化适应能力,不强!
也许在输入的时候,加入Perceiver或Perceive IO结构,情况会发生变化!
"""