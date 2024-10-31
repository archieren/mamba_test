import datasets
import json
import math
import numpy as np
import open3d as o3d

from datasets import Dataset
from pathlib import Path
from pm.utils.align_the_mesh import align_the_mesh

"""
看来数据集的制作,在刚开始的时候,真的,走了弯路.或许,情有可原,我考虑的是没有下采样的情况.
现在添加一部分,制作下采样的训练集!
"""

#这一段，half-manual
def gather_unlabeled_data(source_dir:Path, target_dir:Path):  # 将未标注的 stl移到一个目录！！
    target_dir.mkdir(exist_ok=True)
    for item_stl in source_dir.glob("*.stl"):
        parent = item_stl.parent
        stem = item_stl.stem
        item_json = parent / (stem + ".json")
        if not item_json.exists() :   # 对应的标注文件是否存在？ 不存在，就移到target_dir
            to_item_stl = target_dir / item_stl.name
            item_stl.rename(to_item_stl)  # 重命名的方式，进行移动！

#### Gather_labelled_data
# dir_ = Path("/home/archie/Projects/data/口扫模型/模型测量关键点---/原始数据")
# temp_ = Path("/home/archie/Projects/data/口扫模型/模型测量关键点---/temp")
# gather_unlabeled_data(source_dir=dir_, target_dir= temp_)
#----这一段，half-manual

def get_labeled_data(source_dir:Path, stem:str):   # 假设做了前期处理！！！
    json_item_path = source_dir / (stem + ".json")
    label = json.load(json_item_path.open())

    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()
    return mesh, label

def collect_group_with_aligned_data(source_dir:Path, stems:list[str]):
    import random
    vertices_c = []
    triangles_c = []
    label_c =[]
    name_c = []
    for stem in stems:
        mesh, label_ = get_labeled_data(source_dir, stem)
        # 对齐            
        mesh, _ = align_the_mesh(mesh)

        seg = label_.get("seg")                       # 简单考虑{"seg" -> {"tooth-id"-> [index_of_vertex]}} 或 {"tooth-id"-> [index_of_vertex]}
        label_ = seg if seg is not None else label_    
        label_keys = list(label_.keys())               # {"tooth-id"-> [index_of_vertex]}
        print(label_keys)
        vertices = np.asarray(mesh.vertices,dtype=float)
        triangles = np.asarray(mesh.triangles,dtype=int)
        label = np.zeros((vertices.shape[0],), dtype=int) #dtype=np.dtype('b'))
        for label_key in label_keys:
            label[label_[label_key]] = int(label_key)     # 
        vertices_c.append(vertices)
        triangles_c.append(triangles)
        label_c.append(label)
        name_c.append(stem)

    return {"vertices":vertices_c, "triangles":triangles_c,"label":label_c,"name":name_c}

def make_parquet_with_aligned_data(source_dir:Path, out_dir="data", group_size = 400,clx="train"):
    stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
    total_examples = len(stems)
    group_size = group_size
    file_numbers = math.ceil(total_examples/group_size)
    for i in range(file_numbers):
        stems_part = stems[group_size*i:group_size*(i+1)]
        data_d = collect_group_with_aligned_data(source_dir, stems_part)
        d = Dataset.from_dict(data_d)
        out_name = f"{out_dir}/{clx}-oralscan-part-{i:04}-of-{file_numbers:04}.parquet"
        d.to_parquet(out_name)

def make_test_dataset_with_aligned_data():
    source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns")
    out_dir="zby/oral_scan/data"
    group_size = 4
    make_parquet_with_aligned_data(source_dir=source_dir,out_dir=out_dir,group_size=group_size,clx="test")

def make_train_dataset_with_aligned_data():
    source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据")
    out_dir="zby/oral_scan/data"
    group_size = 400
    make_parquet_with_aligned_data(source_dir=source_dir,out_dir=out_dir,group_size=group_size, clx="train")    

def enumerate_example(ex):
    print(ex["name"])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(ex["vertices"])
    mesh.triangles = o3d.utility.Vector3iVector(ex["triangles"])
    mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh])
    (t_i,)= np.where(ex["label"] > 0)      # 只有一个元素的元组，奇怪的行为！！！
    v =  ex["vertices"][t_i]
    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(v)
    pointSet.paint_uniform_color([1,0.75,0])

    s_mesh = mesh.select_by_index(t_i)

    frame_o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    o3d.visualization.draw_geometries([frame_o, s_mesh], point_show_normal=True, mesh_show_back_face=True)

def noop():
    from datasets import load_dataset, Dataset, IterableDataset
    from torch.utils.data import DataLoader
    ## Method 1
    data_dir = Path.cwd()/'zby'/'oral_scan'         # 注意，没有‘data’这一项！！！
    d = load_dataset(str(data_dir),split="test")   # load_dataset解释太多，这样也可以！
    d.set_format(type="numpy")
    for i in range(2):
        example = d[i]
        enumerate_example(example)

def noop_():
    from datasets import load_dataset, Dataset, IterableDataset
    from torch.utils.data import DataLoader
    ## Method 2
    data_dir = Path.cwd()/'zby'/'oral_scan'/'data'
    data_files={'test': [str(t) for t in data_dir.glob("*.parquet")]}
    d = Dataset.from_parquet(data_files,split='test')
    d.set_format(type='torch')
    d.shuffle(seed=21)
    # for idx, batch in enumerate([234, 120, 1800]):
    #     print(idx)
    #     print(d[batch]['vertices'].shape)
    #     print(d[batch]['name'])
    #     enumerate_example(d[batch])
    for idx, batch in enumerate(d):
        max_v = batch['vertices'].shape[0] if max_v < batch['vertices'].shape[0] else max_v
        print(batch['name'])
        # enumerate_example(batch)    

def noop_ok():
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    datasets_root_name = "datasets_d"
    dataset_id = "zby/oral_scan"
    data_dir =Path.home().joinpath(datasets_root_name, dataset_id) # 终于搞成huggingface的那个样子，
    d = load_dataset(str(data_dir),split="train")   # load_dataset解释太多，这样也可以！
    d.set_format(type="torch")
    print(d)
    example = d[2222]
    enumerate_example(example)

#make_train_dataset_with_aligned_data()
#make_test_dataset_with_aligned_data()
noop()
# noop_()
# noop_ok()

