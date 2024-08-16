import datasets
import json
import numpy as np
import open3d as o3d

from pathlib import Path
teeth = {17,16,15,14,13,12,11,
         27,26,25,24,23,22,21,
         37,36,35,34,33,32,31,
         47,46,45,44,43,42,41}

key_point = {'buccal', 
             'buccal-cusp', 
             'contact', 
             'dental-cusp',
             'edge',
             'fossa',
             'lingual',
             'lingual-cusp',
             'groove'}
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
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh, label

def collect_group(source_dir:Path, stems:list[str]):
    import random
    vertices_c = []
    triangles_c = []
    label_c =[]
    name_c = []
    for stem in stems:        
        mesh, label_ = get_labeled_data(source_dir, stem)
        label_keys = list(label_.keys())

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

## 这种流式方式，似乎不太好？  当然，也可行？
# def gen_(source_dir:Path):
#     for stl_item in source_dir.glob("*.stl"):
#         stem = stl_item.stem
#         mesh, label_ = get_labeled_data(source_dir, stem)
#         label_keys = list(label_.keys())

#         vertices = np.asarray(mesh.vertices,dtype=float)
#         triangles = np.asarray(mesh.triangles,dtype=int)
#         label = np.zeros((vertices.shape[0],), dtype=int) #dtype=np.dtype('b'))
#         for label_key in label_keys:
#             label[label_[label_key]] = int(label_key)
#         yield {"vertices":vertices, "triangles":triangles,"label":label,"name":stem}    


# def make_parquet(source_dir:Path, out_name="test.parquet"):  
#     from datasets import Dataset,IterableDataset
#     d = Dataset.from_generator(gen_, gen_kwargs={"source_dir":source_dir})    
#     d.to_parquet(out_name)

def make_parquet_(source_dir:Path, out_dir="data", group_size = 400,clx="train"):
    import math
    from datasets import Dataset
    stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
    total_examples = len(stems)
    group_size = group_size
    file_numbers = math.ceil(total_examples/group_size)
    for i in range(file_numbers):
        stems_part = stems[group_size*i:group_size*(i+1)]
        data_d = collect_group(source_dir, stems_part)
        d = Dataset.from_dict(data_d)
        # TODO dataset-name-train-0000-of-0004.parquet
        out_name = f"{out_dir}/{clx}-oralscan-part-{i:04}-of-{file_numbers:04}.parquet"
        d.to_parquet(out_name)

def make_test_dataset():
    source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns")
    out_dir="zby/oral_scan/data"
    group_size = 4
    make_parquet_(source_dir=source_dir,out_dir=out_dir,group_size=group_size,clx="test")

def make_train_dataset():
    source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据")
    out_dir="zby/oral_scan/data"
    group_size = 400
    make_parquet_(source_dir=source_dir,out_dir=out_dir,group_size=group_size, clx="train")    

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
    o3d.visualization.draw_geometries([ pointSet, s_mesh])




def noop():
    from datasets import load_dataset, Dataset, IterableDataset
    from torch.utils.data import DataLoader
    ## Method 1
    # data_dir = Path.cwd()/'zby'/'oral_scan'/'data'
    # data_files={'train': [str(t) for t in data_dir.glob("*.parquet")]}
    # d = load_dataset("parquet", data_files=data_files,split="train")
    data_dir = Path.cwd()/'zby'/'oral_scan'         # 注意，没有‘data’这一项！！！
    d = load_dataset(str(data_dir),split="test")   # load_dataset解释太多，这样也可以！
    d.set_format(type="numpy")
    print(d)
    example = d[1]
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

# make_train_dataset()
# make_test_dataset()
# noop()
# noop_()
noop_ok()

