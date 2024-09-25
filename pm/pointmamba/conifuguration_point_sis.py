import math
import torch

from dataclasses import dataclass, asdict
from typing import Union,List

TEETH_num = {18,17,16,15,14,13,12,11,
             28,27,26,25,24,23,22,21,
             38,37,36,35,34,33,32,31,
             48,47,46,45,44,43,42,41}

TEETH_num_cls = {18: 8,
                 17: 7,
                 16: 6,
                 15: 5,
                 14: 4,
                 13: 3,
                 12: 2,
                 11: 1,
                 28:16,
                 27:15,
                 26:14,
                 25:13,
                 24:12,
                 23:11,
                 22:10,
                 21: 9,
                 38:24,
                 37:23,
                 36:22,
                 35:21,
                 34:20,
                 33:19,
                 32:18,
                 31:17,
                 48:32,
                 47:31,
                 46:30,
                 45:29,
                 44:28,
                 43:27,
                 42:26,
                 41:25}

TEETH_cls_num = { 8:18,
                  7:17,
                  6:16,
                  5:15,
                  4:14,
                  3:13,
                  2:12,
                  1:11,
                 16:28,
                 15:27,
                 14:26,
                 13:25,
                 12:24,
                 11:23,
                 10:22,
                  9:21,
                 24:38,
                 23:37,
                 22:36,
                 21:35,
                 20:34,
                 19:33,
                 18:32,
                 17:31,
                 32:48,
                 31:47,
                 30:46,
                 29:45,
                 28:44,
                 27:43,
                 26:42,
                 25:41}

superior_gingival = 33
inferior_gingival = 34

superior_dentition= 35
inferior_dentition= 36

all_classes = 36

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

#各种辅助！！ 
def tooth_lables(labels:torch.Tensor, shape_weight:torch.Tensor) -> List[torch.Tensor]: # b g, b g-> [t,...], [t g,...], [t g,...]
    """
    将编号标签形式转成分类形式. labels和shape_weight的shape相同！
    """ 
    b_s = labels.shape[0]
    b_class_labels = []
    b_mask_labels  = []
    b_shape_weights = []
    for b in range(b_s):
        class_labels = []
        masks = []
        up_or_low = "unknown"
        s_w = shape_weight[b].unsqueeze(0)
        for i in TEETH_num:            
            x = torch.where(labels[b]==i,1,0)
            if x.sum() > 0 :                           # 有这个牙齿的标签！
                #
                x = x.unsqueeze(0)
                masks.append(x)
                #
                cls = TEETH_num_cls[i]                 # 牙编号 -> Class!
                class_labels.append(cls)
                up_or_low="up" if i%10 < 2 else "low"  # 有点浪费, TODO:how?
        #       
        non_tooth_mask = torch.where(labels[b]>0, 0, 1).unsqueeze(0)             # 牙龈
        if non_tooth_mask.sum()>0:
           #
           masks.append(non_tooth_mask)
           #
           class_labels.append(superior_gingival if up_or_low == "up" else inferior_gingival)

        all_tooth_mask = torch.where(labels[b]>0, 1, 0).unsqueeze(0)             # 所有牙齿
        if all_tooth_mask.sum()> 0:
            #
            masks.append(all_tooth_mask)
            #
            class_labels.append(superior_dentition if up_or_low == "up" else inferior_dentition)

        t = len(class_labels)
        class_labels = torch.tensor(class_labels, device= labels.device).long()   # [t,......]   
        b_class_labels.append(class_labels)

        mask_labels = torch.cat(masks, dim=0).float()                             # [t g,...]      # TODO: 每个目标, 都有一个掩码！
        b_mask_labels.append(mask_labels)

        shape_weights = s_w.repeat(t,1).float()                                     # [t g,...]     # TODO: 每个目标， 都有一个权重 
        b_shape_weights.append(shape_weights)

    return b_class_labels, b_mask_labels, b_shape_weights

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
        num_labels:          int =  all_classes
        num_queries:         int = 64             
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