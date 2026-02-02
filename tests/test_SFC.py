import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2"
import random,time
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from addict import Dict
from pathlib import Path
from pm.utils.misc import offset2batch,batch2offset
from pm.sfc_serialization import encode
from pm.utils.point_cloud import PointCloud,group_by_group_number # ,group_by_ratio,group_by_ratio_



device='cuda'
torch.device(device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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
    points_lower = points_lower - torch.mean(points_lower)
    normals_lower = torch.asarray(np.asarray(mesh_lower.vertex_normals), device=device, dtype=torch.float) 

    points_upper = torch.asarray(np.asarray(mesh_upper.vertices), device=device, dtype=torch.float)
    points_upper = points_upper - torch.mean(points_upper)
    normals_upper = torch.asarray(np.asarray(mesh_upper.vertex_normals), device=device,dtype=torch.float) 

    points = torch.cat([points_upper, points_lower])
    normals = torch.cat([normals_upper, normals_lower])
    offset = torch.tensor([points_upper.shape[0],points_lower.shape[0]], device=device).cumsum(0).int() # cumsum就成了浮点数了!
    data = Dict(coord=points,feat=normals, offset=offset, grid_size=1.0e-2)
    return data

def make_data_dict_(upper_stl_path):
    # 熟悉一下数据的处理!!!
    mesh_upper : o3d.t.geometry.TriangleMesh = read_mesh(upper_stl_path)
    # dtype=torch.float是必要的!!!
    points_upper = torch.asarray(np.asarray(mesh_upper.vertices), device=device, dtype=torch.float)
    points_upper = points_upper - torch.mean(points_upper)
    normals_upper = torch.asarray(np.asarray(mesh_upper.vertex_normals), device=device,dtype=torch.float) 

    points = points_upper
    normals = normals_upper
    offset = torch.tensor([points_upper.shape[0]], device=device).cumsum(0).int() # cumsum就成了浮点数了!
    data = Dict(coord=points,feat=normals, offset=offset, grid_size=1.0e-2)
    return data

def make_test_data_dict():
    # dtype=torch.float是必要的!!!
    BN= [32768, 2048]
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

# def test_grouping_by_ratio():
#     from ctypes.util import find_library
#     pc = make_PointCloud()
#     pc.serialization()
#     for i in range(3):
#         print(f"Grouping_by_ratio_with_pointops:{i}")
#         start_time = time.time()
#         s_pc = group_by_ratio(pc,7, ratio=0.08) # (1024*)
#         time_it(start_time)


def test_grouping_by_fps():
    pc = make_PointCloud()
    pc.serialization()
    for i in range(3):
        print(i)
        start_time = time.time()
        s_pc= group_by_group_number(pc,32768, 11) # (1024*)
        time_it(start_time)

        start_time = time.time()
        s_s_pc= group_by_group_number(s_pc,1024, 11) # (1024*)
        time_it(start_time)

        start_time = time.time()
        s_s_s_pc= group_by_group_number(s_s_pc,8, 11) # (1024*)
        time_it(start_time)         



    
def test_fps_pointnet2():
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
    pc = torch.randn(4, 16384, 3, device=device)
    print(pc.shape)
    start_time = time.time()
    fpc = fps(pc, 8192)
    a = fpc[0,18]
    print(a)
    time_it(start_time)

def test_pointmlp():
    from pm.pointmlp import pointMLP
    from torchinfo import summary
    import torch.autograd.profiler as profiler

    B,C,N,Cls = 4,3,2**11,16
    data = torch.rand(B,C,N).to(device)
    norm = torch.rand(B,C,N).to(device)
    cls_label = torch.rand([B, Cls]).to(device)
    print("===> testing modelD ...")
    model = pointMLP(50).to(device)
    out = model(data, norm, cls_label)  # [2,2048,50]
    print(out.shape)
    for i in range(10):
        with torch.no_grad():
            start_time = time.time()
            out = model(data, norm, cls_label)
            time_it(start_time)

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
    data = torch.rand(2, 3, 2**11).cuda()
    cls_label = torch.rand([2, 16]).cuda()
    print("===> testing modelD ...")
    model = CurveNet().cuda()
    out = model(data,l=cls_label)  # [2,2048,50]
    print(out.shape)
    for i in range(10):
        with torch.no_grad():
            start_time = time.time()
            out = model(data, l=cls_label)
            print(out.shape)
            time_it(start_time)

    # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    #     with profiler.record_function("model_forward"):
    #         output = model(data, l=cls_label)
    #         loss = output.sum()
    #     with profiler.record_function("model_backward"):
    #         loss.backward()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

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
    model =PointTransformerV3(3,enable_flash=True,
                            upcast_attention=False,
                            upcast_softmax=False).to(device)
    dc = make_test_data_dict()
    pc = model(dc)
    #print(pc.keys())
    for i in range(10):
        with torch.no_grad():
            start_time = time.time()
            out = model(dc)
            print(out.shape)
            time_it(start_time)

    # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    #     with profiler.record_function("model_forward"):
    #         print("hello")
    #         output = model(dc)
    #         loss = output.sparse_conv_feat.features.sum()
    #     with profiler.record_function("model_backward"):
    #         loss.backward()
    # print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))
    # print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

def __get_ckpt(name='model_weights.pth')-> Path:
    #Some Dir
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_file = checkpoints_dir.joinpath(name)
    return checkpoints_file

def test_point_sis():
    import torch.autograd.profiler as profiler
    from torch.utils.cpp_extension import CUDA_HOME
    from pm.pointmamba import PointSIS_SEG, make_default_config

    m_config = make_default_config()
    #checkpoints_file = __get_ckpt(name='model_weights_keypoint.pth')

    model =PointSIS_SEG(m_config)
    # if checkpoints_file.exists():
    #     ckpt = torch.load(checkpoints_file)
    #     model.load_state_dict(ckpt)
    #     print("Load a saved model")
    model = model.to(device)
    #dc = make_data_dict(upper_stl_path="./assets/124_upper.stl",lower_stl_path="./assets/124_lower.stl")
    dc = make_data_dict_(upper_stl_path="./assets/124_upper.stl")
    for i in range(10):
        start_time = time.time()
        sn = model(PointCloud(dc))
        time_it(start_time)
        print(sn.shape)



def test_patch():
    import torch.nn as nn
    from pm.utils.misc import offset2bincount

    pc = make_PointCloud(upper_stl_path="./assets/124_upper.stl",lower_stl_path="./assets/124_lower.stl")
    pc.serialization(depth=16,order={"z","z-trans","hilbert","hilbert-trans"},shuffle_orders=False)
    pc.sparsify()

    def draw(pc:PointCloud):
        a = 0
        b = 2**15
        c = 2**16
        l = 2**9
        print(pc.serialized_order.shape)
        print(pc.serialized_order[2,a])
        print(pc.serialized_order[2,b])
        print(pc.serialized_order[2,c])
        vertices_a = pc.coord[pc.serialized_order[2, a:a+l]].cpu()
        vertices_b = pc.coord[pc.serialized_order[2, b:b+l]].cpu()
        vertices_c = pc.coord[pc.serialized_order[2, c:c+l]].cpu()
        pointSet_a = o3d.geometry.PointCloud()
        pointSet_a.points = o3d.utility.Vector3dVector(vertices_a)
        pointSet_a.paint_uniform_color([1,0.75,0])
        pointSet_b = o3d.geometry.PointCloud()
        pointSet_b.points = o3d.utility.Vector3dVector(vertices_b)
        pointSet_b.paint_uniform_color([0,0.75,1])
        pointSet_c = o3d.geometry.PointCloud()
        pointSet_c.points = o3d.utility.Vector3dVector(vertices_c)
        pointSet_c.paint_uniform_color([0.75,1, 0])

        upper_mesh = read_mesh(file_path="./assets/124_upper.stl")
        lower_mesh = read_mesh(file_path="./assets/124_lower.stl")
        o3d.visualization.draw_geometries([ pointSet_a, pointSet_b, pointSet_c, upper_mesh, lower_mesh]) #
    
    draw(pc)

    start_time = time.time()
    pad,unpad, pad_shift, pad_shift_back=pc.get_padding_and_inverse()
    time_it(start_time)
    print(pad.shape, unpad.shape, pad_shift.shape, pad_shift_back.shape)
    #print(pad[:10], unpad[:10], pad_shift[:10], pad_shift_back[:10])



def test_serializedpooling():
    # ？？？
    pc = make_PointCloud(upper_stl_path="./assets/124_upper.stl",lower_stl_path="./assets/124_lower.stl")
    pc.serialization(depth=16,order={"z","z-trans","hilbert","hilbert-trans"},shuffle_orders=False)
    pc.sparsify()

    pooling_depth = 2
    code = pc.serialized_code >> pooling_depth * 3     # 按序列码分组？
    print("code[0]\n", code[0].shape)
    code_, inverse_indices, counts = torch.unique(
            code[0],   #这儿，让人不懂了！！！
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
    print(f"code_\n {code_}")
    print(f"inverse_indices\n {code[0]-code_[inverse_indices]}")   # ?batch  Assert code[0] == code_[inverse_indices]
    print("counts\n",counts)    #?batch_bin
    see_, indices = torch.sort(inverse_indices)  # 这个为什么要排序？
    print("see_\n",see_)
    print("see_\n",see_- inverse_indices[indices])  # Assert see_ == inverse_indices[indices]
    print("indices\n", indices)
    # index pointer for sorted point, for torch_scatter.segment_csr
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    print("idx_ptr",idx_ptr)
    # head_indices of each cluster, for reduce attr e.g. code, batch
    head_indices = indices[idx_ptr[:-1]]
    print("head_indices\n",head_indices)
    # generate down code, order, inverse
    code = code[:, head_indices]
    print("code\n",code)
    order = torch.argsort(code)
    print("order",order)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(0, code.shape[1], device=order.device).repeat(
            code.shape[0], 1
        ),
    )
    print("inverse\n",inverse)
    
def test_remote_pointsis():
    import requests
    http_url = 'http://106.13.35.169:8001'
    mesh = read_mesh("./assets/124_upper.stl")
    response = requests.get(http_url+'/v0/test')
    res = response.json()
    res = np.array(res["response"])
    indx= np.nonzero(res)[0]
    print(indx.shape)
    print(res.shape)

    t = o3d.geometry.PointCloud()
    points = np.asarray(mesh.vertices)
    points = points[indx]
    t.points = o3d.utility.Vector3dVector(points)
    t.paint_uniform_color([0.75,1, 0])
    o3d.visualization.draw_geometries([ t, mesh])

def test_mask_predictor():
    b,q,d = (2, 10, 128)
    _,_,l = (2, 128, 32)
    query_embeddings = torch.randn(b,q,d)
    memory_embeddings = torch.randn(b, d, l)
    out_mask = torch.zeros((b,q,l))
    for c in range(d):
        out_mask += query_embeddings[...,c][...,None] * memory_embeddings[:,None,c]
    out_mask = (out_mask.sigmoid() < 0.5).bool()
    # print(out_mask[0, 0])
    # print(out_mask[0, 0].sum())
    # print(out_mask.sum(-1).shape)
    out_mask[torch.where(out_mask.sum(-1) == out_mask.shape[-1])] = False  # 这是在设什么值！
    print(out_mask)

    # out_mask_p = torch.matmul(query_embeddings, memory_embeddings)
    # print(out_mask_p.shape)

    # inspect = (out_mask - out_mask_p)
    # print(inspect.shape)
    # print(torch.where( inspect > 0.1))

    # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    # transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    # memory = torch.rand(10, 32, 512)
    # tgt = torch.rand(20, 32, 512)
    # out = transformer_decoder(tgt, memory)
    # print(out.shape)



# test_PointCloud()
# test_grouping_by_ratio()
test_grouping_by_fps()
# test_fps_pointnet2()
# test_pointmlp()
# test_curvenet()

#test_point_transformer()
####test_point_sis()
# test_patch()
# test_serializedpooling()
# test_remote_pointsis()
# test_mask_predictor()
input()