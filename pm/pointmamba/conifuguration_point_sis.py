import math

from dataclasses import dataclass, asdict
from typing import Union,List

TEETH_num = {18,17,16,15,14,13,12,11,
             28,27,26,25,24,23,22,21,
             38,37,36,35,34,33,32,31,
             48,47,46,45,44,43,42,41}

TEETH_num_cls = {18: 8,17: 7,16: 6,15: 6,14: 4,13: 3,12: 2,11: 1,
                 28:16,27:15,26:14,25:13,24:12,23:11,22:10,21: 9,
                 38:24,37:23,36:22,35:21,34:20,33:19,32:18,31:17,
                 48:32,47:31,46:30,45:29,44:28,43:27,42:26,41:25}

kp_name_cls =   {'buccal':1, 
                'buccal-cusp':2, 
                'contact':3, 
                'dental-cusp':4,
                'edge':5,
                'fossa':6,
                'lingual':7,
                'lingual-cusp':8,
                'groove':9}

kp_cls_name =   {1:'buccal', 
                2:'buccal-cusp', 
                3:'contact', 
                4:'dental-cusp',
                5:'edge',
                6:'fossa',
                7:'lingual',
                8:'lingual-cusp',
                9:'groove'}

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
        order              = ["hilbert", "hilbert-trans"] # ["z", "z-trans", "hilbert", "hilbert-trans"]# 
        shuffle_orders:bool=False
        mamba_config = asdict(Mamba1Config())
        d_model:      int = 128       # feature_dim pos_dim d_model 是一样的!, 未将d_model放到mamba_config里！
        feature_dims: int = d_model
        pos_dims:     int = d_model
        
        #Follow SWIN
        patch_size:   int = 4096
        #mamba in swin
        cascade:      bool=False
        repeats:      int = 3
        
        #Follow MLP 
        # AboutGroup
        group_ratio:  float = 0.09 # 按ratio方式下采样！
        num_group:    int = 8096 # 4096 # 8172 # 16384
        group_size:   int = 11  # 邻居个数       
        depth             = [3, 2, 3, 2] # 每层的mamba堆叠深度！！！
        #out_indices       = [3, 7, 11]   # 弃用！

        #MaskEncoder
        enc_layer_depth: int = 3
        #MaskDecoder!
        nhead:               int = 4
        dim_feedforward:     int = 2048
        num_feature_levels:  int = 3
        num_decode_layers:   int = int(num_feature_levels*3)
        num_labels:          int = 34           # 目前，只有34类 单个teeth，所有tooth，牙龈
        num_queries:         int = 64           # 35 #  >  7*4 + 1  
        dropout:             float = 0.1
        ## About loss
        class_weight:        float = 2.0
        mask_weight:         float = 5.0
        dice_weight:         float = 5.0
        no_object_weight:    float = 0.1            
        # Prompting

def make_default_config():
    config = PointSISConfig()
    return config