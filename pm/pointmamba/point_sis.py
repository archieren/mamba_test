from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import fps
from torch_geometric.nn import knn

from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

from knn_cuda import KNN

"""
用Mamba来处理点云,有下面的几项工作:
PointMamba:这哥们有点灌水,到第四版,又惨考了PTV3的结构化思路.
Mamba3D:
Point Cloud Mamba:
另外Point Transformer V3的工作值得注意!尺度规模和结构化,是他想解决的问题!
"""

