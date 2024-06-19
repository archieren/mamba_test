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
用Mamba来处理点云,有下面的几样工作
"""

