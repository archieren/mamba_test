import torch
import torch.nn as nn

from addict import Dict
from deprecated import deprecated
from pm.sfc_serialization import encode
from pm.utils.misc import offset2batch,batch2offset,offset2bincount

import spconv.pytorch as spconv

class PointCloud(Dict):
    """
    Point Cloud Structure 
    - "coord": 点云的原始坐标! 形如 : (N1+N2+.....+Nb) x 3
        意思是: 第i个点云,有Ni个点.
    - "grid_coord": 网格化的坐标. 形如 : (N1+N2+.....+Nb) x 3
    Point also support the following optional attributes:
    - "offset": 
        应当形如[N1, N1+N2, N1+N2+N3, ..., N1+N2+.....+Nb]
    - "batch": 
        应当形如[N1s 0, N2s 1,..., Nbs (b-1)]
    - "feat": feature of point cloud, default input of model;
        (N1+N2+.....+Nb) x Fd     (Fd是特征的维度)
    - "grid_size": 
        用于将coord 转成 grid_coord.
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** serialized_depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self.batch = offset2batch(self.offset)
            self.batch_bin = offset2bincount(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self.offset = batch2offset(self.batch)
        # 

    def serialization(self, order={"hilbert", "hilbert-trans"}, depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        order指扫描的方式, 是{"z", "z-trans", "hilbert", "hilbert-trans"}的子集
        """
        assert self.batch is not None, "Batch cannot be none!"
        if not self.grid_coord:
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self.grid_coord = torch.div(
                self.coord - self.coord.min(0).values, self.grid_size, rounding_mode="trunc"
            ).int()
        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self.serialized_depth = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # 每一种遍历方式,都有一种排序!!!
        # 故Order
        # 注意对batch的运用!
        code = [
            encode(self.grid_coord, batch=self.batch, depth=depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code,stable=True)
        src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1)
        inverse = torch.zeros_like(order, device=order.device).scatter_(dim=1,index=order,src=src,)  
        # 可以将inverse看成order的逆函数!!! 即: inverse[m, order[m, i]] = i
        # 针对上面,回顾小知识!
        # numpy里面传出来的broadcasting机制!!!
        # Tensor.scatter_和Tensor.gather
        # Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor
        # For a 3-D tensor, self is updated as:
        # self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        # self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
        # For a 2-D tensor, self is updated as:
        # self[index[i][j]][j] = src[i][j]  # if dim == 0
        # self[i][index[i][j]] = src[i][j]  # if dim == 1
        # This is the reverse operation of the manner described in Tensor.gather(dim, index) → Tensor.
        # self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
        # self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
        # self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

        if shuffle_orders:  # 应当是对扫描方式的shuffle!
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self.serialized_code = code
        self.serialized_order = order
        self.serialized_inverse = inverse

    def get_padding_and_inverse(self, patch_size=1024):
        """
        必须在序列化后作！
        TODO :有些条件检测，先暂不考虑！
        返回结果：
            得到多个索引！
        """
        offset = self.offset                 # [N0, N0+N1,......]
        batches = len(offset)
        batch_bin = offset2bincount(offset)  # [N0, N1,......] 
        #batch_pad_bin = (torch.div(batch_bin + patch_size - 1,patch_size,rounding_mode="trunc")* patch_size)  #奇怪的计算方式，没必要！！！
        batch_pad_bin = (torch.ceil(torch.div(batch_bin, patch_size))*patch_size).int() # [N0+P0, N1+P1,......]
        # only pad point when num of points larger than patch_size # TODO， 我假设这种情况不出现！
        _offset = nn.functional.pad(offset, (1, 0))  # [0, N0, N0+N1,......]
        
        offset_pad = torch.cumsum(batch_pad_bin, dim=0) #  [N0+P0, N0+P0+N1+P1,......]
        _offset_pad = nn.functional.pad(offset_pad, (1, 0)) # [0,N0+P0, N0+P0+N1+P1,......]
        
        pad = torch.arange(_offset_pad[-1], device=offset.device) # [0,1,2,..., N0+P0-1, N0+P0,..., N0+P0+N1+P1-1]
        unpad = torch.arange(_offset[-1], device=offset.device)   # [0,1,2,..., N0+N1-1]
        # 要调整pad 和 unpad里面的值！！！
        for i in range(batches):
            Ps = _offset_pad[i] - _offset[i]   #  Ps == ΣPj (j < i)
            unpad[_offset[i] : _offset[i + 1]] += Ps
            if batch_bin[i] != batch_pad_bin[i]:   # ? Pi>0
                remainder = (batch_bin[i] % patch_size)
                pad_num = patch_size -remainder                                 # 要修改的数量
                t_ = _offset_pad[i + 1] - pad_num                               # 要修改段（目标段）的起点
                # s_ = _offset_pad[i + 1] - 2 * patch_size + remainder          # 这个取法要出问题！！！必须要求这个bin 大于两个patch_size!!!
                # pad[ t_: _offset_pad[i + 1]] = pad[s_: _offset_pad[i + 1] - patch_size]
                s_ = _offset_pad[i]                                            # 被提取段的起点。我这改成序列i的起点！x相当于循环补位！
                pad[ t_ : t_ + pad_num] = pad[s_ : s_ + pad_num]               # TODO: 始终有危险！ batch_bin[i] < patch_size/2 就要出问题！
            pad[_offset_pad[i] : _offset_pad[i + 1]] -= Ps

        pad_shift =torch.arange(_offset_pad[-1], device=offset.device)
        pad_shift_back = torch.arange(_offset_pad[-1], device=offset.device)
        for i in range(batches):
            temp = pad_shift[_offset_pad[i]:_offset_pad[i+1]]
            pad_shift[_offset_pad[i]:_offset_pad[i+1]] = temp.roll(int(patch_size/2))
            temp = pad_shift_back[_offset_pad[i]:_offset_pad[i+1]]
            pad_shift_back[_offset_pad[i]:_offset_pad[i+1]] = temp.roll(int(-patch_size/2))        
        # 注意： pad - pad[pad_shift][pad_shift_back] == 0
        return pad, unpad, pad_shift, pad_shift_back


    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.
        点云是稀疏的!这里的"Sparsify"特指为SpConv操作而去准备spconv.SparseConvTensor

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

@torch.no_grad()
@deprecated("# 这个真不能用！, torch_geometric的实现，真是慢。用pointops里的那个，比他快十倍以上！！！")
def group_by_ratio_(xyz_pc:PointCloud, group_size:int, ratio=0.1):  
    '''
        input: 
        xyz_pc.coord  shape 为 (N1+N2+.....+Nb) x 3.
        xyx_pc.batch  应当形如[N1s 0, N2s 1,..., Nbs (b-1)]
        xyz_pc.batch_bin 应为[N1, N2,..., Nb]
        ---------------------------
    '''
    from torch_geometric.nn import fps, knn
    s_idx = fps(xyz_pc.coord, xyz_pc.batch, ratio=ratio)  # 样本点的index. 采样,样本点当作中心.
    s_xyz  = xyz_pc.coord[s_idx]                # 样本点的坐标
    samples_batch = xyz_pc.batch[s_idx]           # 样本点的批号
    patch_idx = knn(xyz_pc.coord, s_xyz, group_size,xyz_pc.batch, samples_batch)  # 很奇怪的一个返回结果!
    patch_idx = patch_idx[1].reshape((-1,group_size))

    return patch_idx, s_idx


@torch.no_grad()
@deprecated("这种返回结果的方式，不好！")
def group_by_group_number_(xyz_pc:PointCloud, 
                   num_group:int,  # 分多少个组                  # 其实取多少点！
                   group_size:int, # 组内多少个元素
                   ): 
    # 不得不面临将点云都搞成传统的Batch模式!
    # 序列分段的方式,是另一思路.先不考虑!
    # 假设xyx_pc已经serialized!
    # 这个地方很花了点时间.基本功不牢呀! index broadcasting 和 scatter_, gather的关系!
    from pointops import knn_query as knn
    from pointops import farthest_point_sampling as fps
    
    # pointops方式 :按量,返回结果还包括距离
    # s_ 解读为 samples, n_ 解读为neighbors, o_解读为ordered.
    # batch_size = xyz_pc.batch[-1] + 1
    s_offset = (torch.ones_like( xyz_pc.batch_bin)* num_group).cumsum(0).int() #[batch_size]
    s_idx = fps(xyz_pc.coord, xyz_pc.offset, s_offset)  # [batch_size*num_group ]  # 幸亏这个fps

    # 此时 还不用考虑mesh提供的norm作为特征的提取基础，直接用坐标和临域关系来构造特征！
    s_xyz  = xyz_pc.coord[s_idx]                                                     # [batch_size*num_group, coord's dim]
    s_n_idx, _dist = knn(group_size, xyz_pc.coord, xyz_pc.offset, s_xyz,s_offset)    ## [batch_size*num_group, group_size ], _
    s_n = xyz_pc.coord[s_n_idx]                                                      # [batch_size*num_group , group_size, coord's dim]
    s_n = s_n - s_xyz.unsqueeze(1)                                                   # [batch_size*num_group , group_size, vector's dim]
    s_n = s_n[:,1:, :]                                                               
    
    #排序,根据原有的SFC遍历序好,获得采样点的各总次序!
    s_order = torch.argsort(xyz_pc.serialized_code[:, s_idx])                                           # 获得样本的各种序列吗, 种类排序! [order_s, batch_size * num_group]
    src=torch.arange(0, s_order.shape[1], device=s_order.device).repeat(s_order.shape[0], 1)
    s_inverse = torch.zeros_like(s_order, device=s_order.device).scatter_(dim=1,index=s_order,src=src,) # [order_s, batch_size * num_group]
    # assert s_inverse[0, s_order[0, i]] == i
    # s_idx[s_order].gather(1, s_inverse)- s_idx 等于零矩阵!!! 注意这个关系!!!

    # s_o_xyz = s_xyz[s_order]      # [order_s, batch_size * num_group , coord's dim]
    # s_o_n = s_n[s_order]          # [order_s, batch_size * num_group , group_size, coord's dim]
    # return s_o_n, s_o_xyz, s_idx, s_xyz, s_order, s_inverse

    # s_idx是样本和数据之间的对应桥梁!!!
    return s_idx, s_n, s_xyz, s_order, s_inverse

@torch.no_grad()
def group_by_ratio(parent_pc:PointCloud, group_size:int, ratio=0.1):  
    '''
        input: 
        parent_pc.coord  shape 为 (N1+N2+.....+Nb) x 3.
        parent_pc.batch  应当形如[N1s 0, N2s 1,..., Nbs (b-1)]
        parent_pc.batch_bin 应为[N1, N2,..., Nb]
        ---------------------------
    '''
    from pointops import knn_query as knn
    from pointops import farthest_point_sampling as fps
    assert ratio > 0.05, "采样比例不能太小"
    s_offset = torch.ceil(parent_pc.batch_bin * ratio).cumsum(0).int() # [batch_size]
    s_idx = fps(parent_pc.coord, parent_pc.offset, s_offset)  # [batch_size*num_group ]  # 幸亏这个fps

    # 此时 还不用考虑mesh提供的norm作为特征的提取基础，直接用坐标和临域关系来构造特征！
    s_xyz  = parent_pc.coord[s_idx]                                                        # [batch_size*num_group, coord's dim]
    s_n_idx, _dist = knn(group_size, parent_pc.coord, parent_pc.offset, s_xyz,s_offset)    # [batch_size*num_group, group_size ], _
    s_n = parent_pc.coord[s_n_idx]                                                         # [batch_size*num_group, group_size, coord's dim]
    s_n = s_n - s_xyz.unsqueeze(1)                                                         # [batch_size*num_group, group_size, vector's dim]
    #s_n = s_n[:,1:, :]                                                                    # TODO: 需不需要,去掉组内第一个vector? 不需要！从两个点集来看？
    
    s_data = Dict(coord=s_xyz, 
                  feat=s_n,
                  offset=s_offset,
                  grid_size=parent_pc.grid_size,                                            # 用父点云的grid_size
                  index_back_to_parent=s_idx)                                               # 用于构造一个新的点集！
    return PointCloud(s_data)

@torch.no_grad()
def group_by_group_number(parent_pc:PointCloud, 
                   num_group:int,  # 分多少个组                  # 其实取多少点！
                   group_size:int, # 组内多少个元素
                   ): 
    from pointops import knn_query as knn
    from pointops import farthest_point_sampling as fps
    # pointops方式 :按量,返回结果还包括距离
    # s_ 解读为 samples, n_ 解读为neighbors, o_解读为ordered.
    # batch_size = parent_pc.batch[-1] + 1
    s_offset = (torch.ones_like( parent_pc.batch_bin)* num_group).cumsum(0).int() #[batch_size]
    s_idx = fps(parent_pc.coord, parent_pc.offset, s_offset)  # [batch_size*num_group ]  # 幸亏这个fps

    # 此时 还不用考虑mesh提供的norm作为特征的提取基础，直接用坐标和临域关系来构造特征！
    s_xyz  = parent_pc.coord[s_idx]                                                        # [batch_size*num_group, coord's dim]
    s_n_idx, _dist = knn(group_size, parent_pc.coord, parent_pc.offset, s_xyz,s_offset)    # [batch_size*num_group, group_size ], _
    s_n = parent_pc.coord[s_n_idx]                                                         # [batch_size*num_group, group_size, coord's dim]
    s_n = s_n - s_xyz.unsqueeze(1)                                                         # [batch_size*num_group, group_size, vector's dim]

    
    s_data = Dict(coord=s_xyz, 
                  feat=s_n,
                  offset=s_offset,
                  grid_size=parent_pc.grid_size,                                            # 用父点云的grid_size
                  index_back_to_parent=s_idx)                                               # 用于构造一个新的点集！
    return PointCloud(s_data)
