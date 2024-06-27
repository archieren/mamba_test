import math

from dataclasses import dataclass, asdict
from typing import Union,Any

@dataclass
class Mamba1Config:     # 抄自 Mamba1的初始化参数!!!

    # d_model: int        # D in paper/comments
    d_state: int = 16   # N in paper/comments
    d_conv:  int = 4
    expand:  int = 2     # E in paper/comments

    dt_rank:    Union[int, str] = 'auto'
    dt_min:     float = 0.001
    dt_max:     float = 0.1
    dt_init:    str = "random" # "random" or "constant"
    dt_scale:   float = 1.0
    dt_init_floor: float = 1e-4

    conv_bias: bool = True
    bias:      bool = False
    use_fast_path: bool = True  # Fused kernel options

    #layer_idx = None
    device    = None
    dtype     = None

@dataclass
class PointSISConfig():
        in_channels:  int = 3
        # About SFC
        #Spatial Filling Curve!
        #{"z", "z-trans", "hilbert", "hilbert-trans"}
        order=["z", "z-trans", "hilbert", "hilbert-trans"]
        shuffle_orders=False
        # About group
        num_group:    int = 16384
        group_size:   int = 17
        trans_dim:    int = 64       # trans_dim feature_dim pos_dim d_model 是一样的!
        feature_dims: int = trans_dim
        pos_dims:     int = trans_dim
        # mamba
        depth:        int = 12       # 控制层数！！！
        mamba_config = asdict(Mamba1Config())
        #Fatures encoding!
        cls_mode=False
        # Prompting

def make_default_config():
    config = PointSISConfig()
    return config