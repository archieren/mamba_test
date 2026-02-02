import math
import torch

from dataclasses import dataclass, asdict
from typing import Union,List, ClassVar

TEETH_num = {18,17,16,15,14,13,12,11,
             28,27,26,25,24,23,22,21,
             38,37,36,35,34,33,32,31,
             48,47,46,45,44,43,42,41,}

# FIXME:牙齿的实例和类究竟是是个什么关系呢？
# 一个牙齿一类!
@dataclass
class TEETH:
    # 牙编号到类号
    TEETH_num_cls :ClassVar[dict[int,int]]  = { 18: 8, 17: 7, 16: 6, 15: 5, 14: 4, 13: 3, 12: 2, 11: 1,
                                                28:16, 27:15, 26:14, 25:13, 24:12, 23:11, 22:10, 21: 9,
                                                38:24, 37:23, 36:22, 35:21, 34:20, 33:19, 32:18, 31:17,
                                                48:32, 47:31, 46:30, 45:29, 44:28, 43:27, 42:26, 41:25}
                                # 类号到牙编号
    TEETH_cls_num  :ClassVar[dict[int,int]] = {  8:18,  7:17,  6:16,  5:15,  4:14,  3:13,  2:12,  1:11,
                                                16:28, 15:27, 14:26, 13:25, 12:24, 11:23, 10:22,  9:21,
                                                24:38, 23:37, 22:36, 21:35, 20:34, 19:33, 18:32, 17:31,
                                                32:48, 31:47, 30:46, 29:45, 28:44, 27:43, 26:42, 25:41}

    # superior_gingival :ClassVar[int] = 33
    # inferior_gingival :ClassVar[int] = 34

    # superior_dentition :ClassVar[int] = 35
    # inferior_dentition :ClassVar[int] = 36

    all_classes :ClassVar[int]  = 32

# 所有牙齿都归为一类
# @dataclass
# class TEETH:
#     # 牙编号到类号
#     TEETH_num_cls :ClassVar[dict[int,int]] ={ 
#                                             18: 1, 17: 1, 16: 1, 15: 1, 14: 1, 13: 1, 12: 1, 11: 1,
#                                             28: 1, 27: 1, 26: 1, 25: 1, 24: 1, 23: 1, 22: 1, 21: 1,
#                                             38: 1, 37: 1, 36: 1, 35: 1, 34: 1, 33: 1, 32: 1, 31: 1,
#                                             48: 1, 47: 1, 46: 1, 45: 1, 44: 1, 43: 1, 42: 1, 41: 1,
#                                             #替牙期的编号,反过来写以示特殊
#                                             51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1,
#                                             61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1, 68: 1,
#                                             71: 1, 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1,
#                                             81: 1, 82: 1, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1
#                                             }

#     superior_gingival :ClassVar[int] = 2
#     inferior_gingival :ClassVar[int] = 2

#     superior_dentition :ClassVar[int] = 3
#     inferior_dentition :ClassVar[int] = 3

#     all_classes :ClassVar[int]  = 6

kp_name_cls =   {'buccal':1,                # 颊(侧)点                               #6    #7       各2个
                'buccal-cusp':2,            # 颊(侧)尖点                 #4    #5                   各1个
                'contact':3,                # ？？          #1 #2   #3  #4    #5     #6             各2个
                'dental-cusp':4,            # 牙尖点                #3                              各1个
                'edge':5,                   # ？？          #1 #2 
                'fossa':6,                  # 窝凹点                     #4    #5     #6    #7      各1个
                'lingual':7,                # 舌(侧)点                                #6    #7      各2个
                'lingual-cusp':8,           # 舌(侧)尖点                 #4    #5                   各1个
                'groove':9,                 # 凹槽点                                  #6            各1个
                }                           #              2-4 2-4 2-3  4-5   4-5    5-8    3-5    N类-M个点

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
        up_or_low = "unknown"                         #FIXME: 还有用吗？
        s_w = shape_weight[b].unsqueeze(0)
        for i in TEETH_num:            
            x = torch.where(labels[b]==i,1,0)
            if x.sum() > 0 :                           # 有这个牙齿的标签！
                #
                x = x.unsqueeze(0)
                masks.append(x)
                #
                cls = TEETH.TEETH_num_cls[i]                 # 牙编号 -> Class!
                class_labels.append(cls)
                up_or_low="up" if i//10 in {1, 2, 5, 6} else "low"  # TODO:how?
        #       
        # non_tooth_mask = torch.where(labels[b]>0, 0, 1).unsqueeze(0)             # 牙龈
        # if non_tooth_mask.sum()>0:
        #    #
        #    masks.append(non_tooth_mask)
        #    #
        #    class_labels.append(TEETH.superior_gingival if up_or_low == "up" else TEETH.inferior_gingival)

        # all_tooth_mask = torch.where(labels[b]>0, 1, 0).unsqueeze(0)             # 所有牙齿
        # if all_tooth_mask.sum()> 0:
        #     #
        #     masks.append(all_tooth_mask)
        #     #
        #     class_labels.append(TEETH.superior_dentition if up_or_low == "up" else TEETH.inferior_dentition)

        t = len(class_labels)
        class_labels = torch.tensor(class_labels, device= labels.device).long()   # [t,......]   
        b_class_labels.append(class_labels)

        mask_labels = torch.cat(masks, dim=0).float()                             # [t g,...]      # FIXME: 每个目标, 都有一个掩码！
        b_mask_labels.append(mask_labels)

        shape_weights = s_w.repeat(t,1).float()                                     # [t g,...]     # FIXME: 每个目标， 都有一个权重 
        b_shape_weights.append(shape_weights)

    return b_class_labels, b_mask_labels, b_shape_weights

@dataclass
class Mamba1Config:     # 抄自 Mamba1的初始化参数!!!
    # 要注意确保 d_model * expand / headdim = multiple of 8
    # d_model: int        # D in paper/comments
    d_state: int = 64   # N in paper/comments -- Default 128
    d_conv:  int = 4
    expand:  int = 2     # E in paper/comments -- Default 2
    headdim: int = 32    # 24<->96 # Default is 64
    # dt_rank:    Union[int, str] = 'auto'
    # dt_min:     float = 0.001
    # dt_max:     float = 0.1
    # dt_init:    str = "random" # "random" or "constant"
    # dt_scale:   float = 1.0
    # dt_init_floor: float = 1e-4

    # conv_bias: bool = True
    # bias:      bool = False
    # use_fast_path: bool = True  # Fused kernel options

    #layer_idx = None
    # device    = None
    # dtype     = None

@dataclass
class PointSISConfig():
        d_cat:        int=1      # 数据的范畴
        in_channels:  int = 7    # 7: coord + normals + cur;  4: normals + cur;  3: normals
        # About SFC
        #Spatial Filling Curve!
        #{"z", "z-trans", "z-reverse","hilbert", "hilbert-trans", "hilbert-reverse"}
        order              = ["hilbert", "hilbert-reverse" ]  # "z", "z-reverse"
        shuffle_orders:bool=False
        mamba_config = asdict(Mamba1Config())
        d_model:      int = 128       # feature_dim pos_dim d_model 是一样的!, 未将d_model放到mamba_config里！

        #Follow MLP 
        # AboutGroup
        group_ratio:  float = 0.09 # 按ratio方式下采样！
        num_group:    int = 32768 # 4096 # 8172 # 16384
        group_size:   int = 11  # 邻居个数       
        # TODO: 加了下采样后，加了一个Stage！
        enc_depths  =   ( 6, 6, 6) 
        enc_channels=   (128, 128, 128)


        #MaskDecoder!
        nhead:               int = 4
        dim_feedforward:     int = 2048
        num_feature_levels:  int = len(enc_depths) - 1       # 
        num_decode_layers:   int = int(num_feature_levels)
        num_labels:          int =  TEETH.all_classes
        num_queries:         int = 64             
        dropout:             float = 0.1
        ## About loss
        class_weight:        float = 2.0
        mask_weight:         float = 5.0
        dice_weight:         float = 5.0
        no_object_weight:    float = 0.2     # 0.1 是原始值！            
        # Prompting

def make_default_config():
    config = PointSISConfig()
    return config