import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
import time
from addict import Dict
import numpy as np
import open3d as o3d

import torch

from pm.utils.misc import offset2batch,batch2offset
from pm.sfc_serialization import encode
from pm.utils.point_cloud import PointCloud, Group


device='cuda'
torch.device(device)

def time_it(start_time):
  stop_time = time.time()
  print("耗时: {:.2f}秒".format(stop_time - start_time))
  return


def read_mesh(file_path):
  mesh = o3d.io.read_triangle_mesh(file_path)
  mesh.remove_duplicated_vertices()
  mesh.compute_vertex_normals()
  return mesh

def make_data_dict():
  # 熟悉一下数据的处理!!!
  mesh_lower : o3d.geometry.TriangleMesh = read_mesh("./assets/124_lower.stl")
  mesh_upper : o3d.geometry.TriangleMesh = read_mesh("./assets/124_upper.stl")
  # dtype=torch.float是必要的!!!
  points_lower = torch.asarray(np.asarray(mesh_lower.vertices),device=device, dtype=torch.float)
  normals_lower = torch.asarray(np.asarray(mesh_lower.vertex_normals), device=device, dtype=torch.float) 

  points_upper = torch.asarray(np.asarray(mesh_upper.vertices), device=device, dtype=torch.float)
  normals_upper = torch.asarray(np.asarray(mesh_upper.vertex_normals), device=device,dtype=torch.float) 

  points = torch.cat([points_upper, points_lower])
  normals = torch.cat([normals_upper, normals_lower])
  offset = torch.tensor([points_upper.shape[0],points_lower.shape[0]], device=device).cumsum(0).int()
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
   
def make_PointCloud():    
  data = make_data_dict()
  pc = PointCloud(data)  
  return pc

def test_PointCloud():
  pc = make_PointCloud()

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

def test_fps_pyg():
  ###pyg提供的工具,适合在graph上作!!!
  from torch_geometric.nn import fps
  pc = make_PointCloud()
  start_time = time.time()
  grouper = Group(2, 7)
  c_idx, patch_index = grouper(pc)
  time_it(start_time)
  print(c_idx[0])
  print(pc.coord[c_idx[0]])
  print(pc.batch[c_idx][0:10])
  print(patch_index.shape)
  print(patch_index[1,0:7])
  time_it(start_time)
  #print(pc.coord[patch_index[0,0:7]])

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

  with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("model_forward"):
        output = model(data, norm, cls_label)
        loss = output.sum()
    with profiler.record_function("model_backward"):
        loss.backward()

  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
  print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

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
          output = model(dc)
          loss = output.sum()
      with profiler.record_function("model_backward"):
          loss.backward()
    input()

def test_point_sis():
    import torch.autograd.profiler as profiler

    from pathlib import Path
    from torch.utils.cpp_extension import CUDA_HOME
    from cumm.nvrtc import get_cudadevrt_path
    from cumm.common import _get_cuda_include_lib

    # cumm.common 的 231-232做了修改的!!!
    # 原代码里的cuda路径有问题!
    print(get_cudadevrt_path())
    print(_get_cuda_include_lib())    
    from pm.pointmamba import PointSIS
    model =PointSIS(3).to(device)
    dc = make_data_dict()
    pc = model(dc)
    print(pc.keys())

    # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    #   with profiler.record_function("model_forward"):
    #       output = model(dc)
    #       loss = output.sum()
    #   with profiler.record_function("model_backward"):
    #       loss.backward()
    input()

# test_PointCloud()
# test_fps_pyg()
# test_fps_pointnet2()
# test_pointmlp()
# test_curvenet()

# test_point_transformer()
test_point_sis()
