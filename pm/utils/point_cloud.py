from addict import Dict
import torch
import torch.nn as nn

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

    def serialization(self, order={"hilbert", "hilbert-trans"}, depth=None, shuffle_orders=True):
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
        order = torch.argsort(code)
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

    # def grouping(self, num_group:int, group_size:int):
    #     self.s_idx, self.s_n, self.s_xyz, self.s_order, self.s_inverse = group_by_count(self, num_group, group_size)


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

# @torch.inference_mode()
# def group_by_ratio(xyz_pc:PointCloud, group_size:int, ratio=0.1):  # 这个应当弃用,效率不高!!!!!!
#     '''
#         input: 
#         xyz_pc.coord  shape 为 (N1+N2+.....+Nb) x 3.
#         xyx_pc.batch  应当形如[N1s 0, N2s 1,..., Nbs (b-1)]
#         xyz_pc.batch_bin 应为[N1, N2,..., Nb]
#         ---------------------------
#     '''
#     from torch_geometric.nn import fps, knn
#     samples_idx = fps(xyz_pc.coord, xyz_pc.batch, ratio=ratio)  # 样本点的index. 采样,样本点当作中心.
#     samples  = xyz_pc.coord[samples_idx]                # 样本点的坐标
#     samples_batch = xyz_pc.batch[samples_idx]           # 样本点的批号
#     patch_idx = knn(xyz_pc.coord, samples, group_size,xyz_pc.batch, samples_batch)  # 很奇怪的一个返回结果!
#     patch_idx = patch_idx[1].reshape((-1,group_size))

#     return patch_idx, samples_idx
