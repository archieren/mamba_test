import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2"
import time
from addict import Dict
import numpy as np
import open3d as o3d

import torch

from pm.utils.misc import offset2batch,batch2offset
from pm.sfc_serialization import encode
from pm.utils.point_cloud import PointCloud
from pm.pointmamba import group_by_fps_knn


device='cuda'
torch.device(device)

def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.2f}秒".format(stop_time - start_time))
    return

def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")


def read_mesh(file_path:str):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.remove_duplicated_vertices()    # Must!
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh

def make_data_dict(upper_stl_path, lower_stl_path):
    # 熟悉一下数据的处理!!!
    mesh_lower : o3d.t.geometry.TriangleMesh = read_mesh(lower_stl_path)
    mesh_upper : o3d.t.geometry.TriangleMesh = read_mesh(upper_stl_path)
    # dtype=torch.float是必要的!!!
    points_lower = torch.asarray(np.asarray(mesh_lower.vertices),device=device, dtype=torch.float)
    normals_lower = torch.asarray(np.asarray(mesh_lower.vertex_normals), device=device, dtype=torch.float) 

    points_upper = torch.asarray(np.asarray(mesh_upper.vertices), device=device, dtype=torch.float)
    normals_upper = torch.asarray(np.asarray(mesh_upper.vertex_normals), device=device,dtype=torch.float) 

    points = torch.cat([points_upper, points_lower])
    normals = torch.cat([normals_upper, normals_lower])
    offset = torch.tensor([points_upper.shape[0],points_lower.shape[0]], device=device).cumsum(0).int() # cumsum就成了浮点数了!
    data = Dict(coord=points,feat=normals, offset=offset, grid_size=1.0e-2)
    return data

def make_test_data_dict():
    # dtype=torch.float是必要的!!!
    BN= [1024, 2048]
    points_lower = torch.randn(BN[0], 3, device=device, dtype=torch.float) 
    normals_lower = torch.randn(BN[0], 3, device=device, dtype=torch.float) 

    points_upper = torch.randn(BN[1], 3, device=device, dtype=torch.float) 
    normals_upper = torch.randn(BN[1], 3, device=device, dtype=torch.float) 

    points = torch.cat([points_upper, points_lower])
    normals = torch.cat([normals_upper, normals_lower])
    offset = torch.tensor([points_upper.shape[0],points_lower.shape[0]], device=device).cumsum(0).int()
    data = Dict(coord=points,feat=normals, offset=offset, grid_size=1.0e-2)
    return data
   
def make_PointCloud(upper_stl_path="./assets/124_upper.stl",lower_stl_path="./assets/124_lower.stl"):    
    data = make_data_dict(upper_stl_path, lower_stl_path)
    pc = PointCloud(data)  
    return pc

def test_PointCloud():
    start_time = time.time() 
    pc = make_PointCloud()
    time_it(start_time)

    start_time = time.time() 
    pc.serialization(depth=16,order={"z","z-trans","hilbert","hilbert-trans"})
    pc.sparsify()
    time_it(start_time)
    print("keys:")
    print(pc.keys())
    print("About Batch:shape")
    print(pc.batch.shape)
    print("About Batch:bincount")
    print(pc.batch.bincount())
    print("serialized_code")
    print(pc.serialized_code.shape)
    print("serialized_order")
    print(pc.serialized_order.shape)
    print("serialized_inverse_order")
    print(pc.serialized_inverse.shape)
    print("Order and inverse")
    print(pc.serialized_order[0, 100:102])
    print(pc.serialized_inverse[0, pc.serialized_order[0, 100:102]])
    print("Sparsity")
    print("About Sparsity:sparse_shape")
    print(pc.sparse_shape)
    print("About sparse_conv_feat: spatial shape")
    print(pc.sparse_conv_feat.spatial_shape)
    print("About sparse_conv_feat: features's shape")
    print(pc.sparse_conv_feat.features.shape)
    print("About sparse_conv_feat: indices's shape")
    print(pc.sparse_conv_feat.indices.shape)
    print("About sparse_conv_feat: batch size")
    print(pc.sparse_conv_feat.batch_size)

def test_grouping_by_fps():
    ###pyg提供的工具,适合在graph上作!!!但性能真不咋地!!
    pc = make_PointCloud()
    pc.serialization()
    start_time = time.time()
    s_idx, s_n, s_xyz, s_order, s_inverse= group_by_fps_knn(pc,4, 3) # (1024*)
    time_it(start_time)
    print("Samples_idx:")
    print(s_idx.shape)

    print(" Samples neighbor and xyz")
    print(s_n.shape)
    print(s_xyz.shape)

    print("Different Order of samples:")
    print(s_order)
    print(s_idx[s_order])
    ddd = s_idx[s_order].gather(1, s_inverse)- s_idx
    print("ddd应当是零矩阵")
    print(ddd)  

    
def test_fps_pointnet2():
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
    pc = torch.randn(4, 1000000, 3, device=device)
    print(pc.shape)
    start_time = time.time()
    fpc = fps(pc, 10000)
    a = fpc[0,118]
    print(a)
    time_it(start_time)

def test_pointmlp():
    from pm.pointmlp import pointMLP
    import torch.autograd.profiler as profiler

    B,C,N,Cls = 4,3,2048,16
    data = torch.rand(B,C,N).to(device)
    norm = torch.rand(B,C,N).to(device)
    cls_label = torch.rand([B, Cls]).to(device)
    print("===> testing modelD ...")
    model = pointMLP(50).to(device)
    out = model(data, norm, cls_label)  # [2,2048,50]
    print(out.shape)

    # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    #     with profiler.record_function("model_forward"):
    #         output = model(data, norm, cls_label)
    #         loss = output.sum()
    #     with profiler.record_function("model_backward"):
    #         loss.backward()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")
    # input()

def test_curvenet():
    from pm.curvenet import CurveNet
    import torch.autograd.profiler as profiler
    data = torch.rand(2, 3, 2048).cuda()
    cls_label = torch.rand([2, 16]).cuda()
    print("===> testing modelD ...")
    model = CurveNet().cuda()
    out = model(data,l=cls_label)  # [2,2048,50]
    print(out.shape)

    with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        with profiler.record_function("model_forward"):
            output = model(data, l=cls_label)
            loss = output.sum()
        with profiler.record_function("model_backward"):
            loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

def test_point_transformer():
    import torch.autograd.profiler as profiler

    from pathlib import Path
    from torch.utils.cpp_extension import CUDA_HOME
    from cumm.nvrtc import get_cudadevrt_path
    from cumm.common import _get_cuda_include_lib

    # cumm.common 的 231-232做了修改的!!!
    # 原代码里的cuda路径有问题!
    print(get_cudadevrt_path())
    print(_get_cuda_include_lib())    
    from pm.point_transformer import PointTransformerV3
    model =PointTransformerV3(3).to(device)
    dc = make_test_data_dict()
    pc = model(dc)
    print(pc.keys())

    with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        with profiler.record_function("model_forward"):
            print("hello")
            output = model(dc)
            loss = output.sparse_conv_feat.features.sum()
        with profiler.record_function("model_backward"):
            loss.backward()
    print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))
    print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")


def test_point_sis():
    import torch.autograd.profiler as profiler

    from pathlib import Path
    from torch.utils.cpp_extension import CUDA_HOME

    from pm.pointmamba import PointSIS, make_default_config
    config = make_default_config()
    model =PointSIS(config).to(device)
    dc = make_data_dict()
    sn = model(dc)
    print(sn.shape)
    # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    #   with profiler.record_function("model_forward"):
    #       output = model(dc)
    #       loss = output.sum()
    #   with profiler.record_function("model_backward"):
    #       loss.backward()
    # print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))
    # print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")
    
    input()


def test_patch():
    import torch.nn as nn
    from pm.utils.misc import offset2bincount

    def get_padding_and_inverse( point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        patch_size = 1024
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + patch_size - 1,
                    patch_size,
                    rounding_mode="trunc",
                )
                * patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                p_ = _offset_pad[i] - _offset[i]
                unpad[_offset[i] : _offset[i + 1]] += p_
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - patch_size
                        + (bincount[i] % patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * patch_size
                        + (bincount[i] % patch_size) : _offset_pad[i + 1]
                        - patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= p_

                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    pc = make_PointCloud(upper_stl_path="./assets/124_upper.stl",lower_stl_path="./assets/124_lower.stl")
    pc.serialization(depth=16,order={"z","z-trans","hilbert","hilbert-trans"},shuffle_orders=False)
    pc.sparsify()

    a = 0
    b = 2**16
    l = 1024
    print(pc.serialized_order.shape)
    print(pc.serialized_order[2,a])
    print(pc.serialized_order[2,b])
    vertices_a = pc.coord[pc.serialized_order[3, a:a+l]].cpu()
    vertices_b = pc.coord[pc.serialized_order[3, b:b+l]].cpu()
    pointSet_a = o3d.geometry.PointCloud()
    pointSet_a.points = o3d.utility.Vector3dVector(vertices_a)
    pointSet_a.paint_uniform_color([1,0.75,0])
    pointSet_b = o3d.geometry.PointCloud()
    pointSet_b.points = o3d.utility.Vector3dVector(vertices_b)
    pointSet_b.paint_uniform_color([0,0.75,1])

    upper_mesh = read_mesh(file_path="./assets/124_upper.stl")
    o3d.visualization.draw_geometries([ pointSet_a,  upper_mesh]) #pointSet_b,
    # x, y, z = get_padding_and_inverse(pc)
    # print(x.shape, y.shape,z.shape)
    # print(z)

# test_PointCloud()
# test_grouping_by_fps()
# test_fps_pointnet2()
# test_pointmlp()
# test_curvenet()

# test_point_transformer()
# test_point_sis()
for i in range(5):
    test_patch()
