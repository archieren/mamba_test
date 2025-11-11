import datasets
import json
import math
import numpy as np
import open3d as o3d
import torch

from addict import Dict
from datasets import Dataset
from pathlib import Path
from pm.pointmamba import make_default_config
from pm.utils.align_the_mesh import align_the_mesh
from pm.utils.point_cloud import Grouper_By_NumGroup
from pm.utils.point_cloud import PointCloud

device='cuda'
torch.device(device)

"""
看来数据集的制作,在刚开始的时候,真的,走了弯路.或许,情有可原,我考虑的是没有下采样的情况.
现在添加一部分,制作下采样的训练集!
将采样后的数据生成数据集，会加快训练的时间！

"""
def get_labeled_data(source_dir:Path, stem:str):   # 假设做了前期处理！！！
    json_item_path = source_dir / (stem + ".json")
    label = json.load(json_item_path.open())

    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()
    return mesh, label

"""

"""
def collect_group_with_aligned_and_sampled_data(source_dir:Path, stems:list[str]):
    def get_data(mesh, label_):
        seg = label_.get("seg")                       # 简单考虑{"seg" -> {"tooth-id"-> [index_of_vertex]}} 或 {"tooth-id"-> [index_of_vertex]}
        label_ = seg if seg is not None else label_    
        label_keys = list(label_.keys())               # {"tooth-id"-> [index_of_vertex]}
        vertices = np.asarray(mesh.vertices,dtype=float)
        triangles = np.asarray(mesh.triangles,dtype=int)
        normals = np.asarray(mesh.vertex_normals,dtype=float)
        label = np.zeros((vertices.shape[0],), dtype=int) #dtype=np.dtype('b'))
        for label_key in label_keys:
            label[label_[label_key]] = int(label_key)     # 
        return vertices, triangles, normals, label
    
    def bi_cls(x, *y):  # 没想明白*y！
        if x > 0 : return int(x)
        else: return int(0)

    def to_numpy(x:torch.Tensor):
        return x.detach().cpu().numpy()

    coord_c = []
    feat_c = []
    label_c =[]
    shape_weight_c = []
    offset_c = []
    name_c = []

    m_config = make_default_config()
    grouper = Grouper_By_NumGroup(num_group=m_config.num_group, group_size=11)
    for stem in stems:
        print(stem)
        mesh, label_ = get_labeled_data(source_dir, stem)

        # 对齐？姿态和尺度？ TODO           
        mesh, _ = align_the_mesh(mesh)
        vertices, _ , normals, label = get_data(mesh, label_)

        offset = torch.tensor([vertices.shape[0]], device=device).cumsum(0).int()
        vertices = torch.tensor(vertices, dtype=torch.float).cuda(device)
        normals  = torch.tensor(normals, dtype=torch.float).cuda(device)
        label    = torch.tensor(label)
        label = label.map_(label,bi_cls).cuda(device)

        data = Dict(coord=vertices,feat=normals, labels=label, offset=offset, grid_size=1.0e-2)
        s_pc = grouper(PointCloud(data))
        # print(s_pc.keys())
        coord, feat, label, shape_weight = s_pc.coord, s_pc.feat, s_pc.labels, s_pc.shape_weight
        coord = to_numpy(coord)
        feat  = to_numpy(feat)
        shape_weight = to_numpy(shape_weight)
        label  = to_numpy(label)
        offset = coord.shape[0]
        # print(coord.shape)
        # print(feat.shape)
        # print(label.shape)
        # print(shape_weight.shape)

        coord_c.append(coord)
        feat_c.append(feat)
        label_c.append(label)
        shape_weight_c.append(shape_weight)
        offset_c.append(offset)
        name_c.append(stem)   # TODO: 要不要搞成s_o_i呢？

    return {"coord":coord_c, "feat":feat_c,"label":label_c,"shape_weight":shape_weight_c,"offset":offset_c,"name":name_c}

def make_parquet_with_aligned_sampled_data(source_dir:Path, out_dir="data", group_size = 400,clx="train"):
    stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
    total_examples = len(stems)
    file_numbers = math.ceil(total_examples/group_size)
    for i in range(file_numbers):
        stems_part = stems[group_size*i:group_size*(i+1)]
        with torch.no_grad():
            data_d = collect_group_with_aligned_and_sampled_data(source_dir, stems_part)
        d = Dataset.from_dict(data_d)
        out_name = f"{out_dir}/{clx}-oralscan-part-{i:04}-of-{file_numbers:04}.parquet"
        d.to_parquet(out_name)

def make_test_dataset_with_aligned_sampled_data(source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns"),clx_a=""):
    out_dir="zby/oral_scan/data"
    group_size = 4
    make_parquet_with_aligned_sampled_data(source_dir=source_dir,out_dir=out_dir,group_size=group_size,clx="test"+clx_a)

def make_train_dataset_with_aligned_sampled_data(source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据"),clx_a=""):
    out_dir="zby/oral_scan/data"
    group_size = 400
    make_parquet_with_aligned_sampled_data(source_dir=source_dir,out_dir=out_dir,group_size=group_size, clx="train"+clx_a)

# make_train_dataset_with_aligned_sampled_data()
# make_test_dataset_with_aligned_sampled_data()
make_train_dataset_with_aligned_sampled_data(source_dir=Path("/home/archie/Projects/data/口扫模型/口扫模型分割新增（有乳牙）"), clx_a="_a")




