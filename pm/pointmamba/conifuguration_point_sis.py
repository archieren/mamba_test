import math

from dataclasses import dataclass, asdict
from typing import Union,List

@dataclass
class Mamba1Config:     # 抄自 Mamba1的初始化参数!!!

    # d_model: int        # D in paper/comments
    d_state: int = 32   # N in paper/comments
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
        order             = ["hilbert"] #["hilbert", "hilbert-trans"]  # ["z", "z-trans", "hilbert", "hilbert-trans"]# 
        shuffle_orders    =False
        # AboutGroup
        num_group:    int = 1024 # 8096 # 16384
        group_size:   int = 17
        # AboutSwin
        patch_size:   int = 1024

        d_model:      int = 128       # feature_dim pos_dim d_model 是一样的!, 未将d_model放到mamba_config里！
        feature_dims: int = d_model
        pos_dims:     int = d_model
        # mamba
        #d_model:      int =  
        depth:        int = 12 # 控制层数！！！
        out_indices       = [3,7,11]   # []
        mamba_config = asdict(Mamba1Config())
        #Fatures encoding!
        cls_mode=False
        # Prompting

def make_default_config():
    config = PointSISConfig()
    return config