import datasets
import json
import numpy as np
import open3d as o3d

from pathlib import Path

def get_labeled_data(source_dir:Path, stem:str):   # 假设做了前期处理！！！
    json_item_path = source_dir / (stem + ".json")
    label = json.load(json_item_path.open())

    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()
    return mesh, label

def collect_group(source_dir:Path, stems:list[str]):
    import random
    vertices_c = []
    triangles_c = []
    label_c =[]
    name_c = []
    for stem in stems:        
        mesh, label_ori = get_labeled_data(source_dir, stem)

        vertices = np.asarray(mesh.vertices,dtype=float)
        triangles = np.asarray(mesh.triangles,dtype=int)
        seg_labels = label_ori["seg"]                  # "seg"->{"tooth-id"-> [index_of_vertex]}
        key_point_labels = label_ori["kp"]             # "kp" ->{"tooth_id"-> {"keypoint_name" -> [index_of_vertex]}}

        label = np.zeros((vertices.shape[0],), dtype=int) #dtype=np.dtype('b'))
        for tooth_id in list(seg_labels.keys()):
            label[seg_labels[tooth_id]] = int(tooth_id)     # 
        vertices_c.append(vertices)
        triangles_c.append(triangles)
        label_c.append(label)
        name_c.append(stem)

        for tooth_id in list(key_point_labels.keys()):
            keypoints=[]
            for keypoint_name in list(key_point_labels[tooth_id].keys()):
                keypoints += key_point_labels[tooth_id][keypoint_name]
            keypoints = np.asarray(keypoints)
            keypoints = np.unique(keypoints)
            print(f"{tooth_id}: {keypoints.shape}")

    return {"vertices":vertices_c, "triangles":triangles_c,"label":label_c,"name":name_c}

def make_parquet_(source_dir:Path, out_dir="data", group_size = 400,clx="train"):
    import math
    from datasets import Dataset
    stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
    total_examples = len(stems)
    print(total_examples)
    group_size = group_size
    file_numbers = math.ceil(total_examples/group_size)
    for i in range(file_numbers):
        stems_part = stems[group_size*i:group_size*(i+1)]
        data_d = collect_group(source_dir, stems_part)
        # d = Dataset.from_dict(data_d)
        # # TODO dataset-name-train-0000-of-0004.parquet
        # out_name = f"{out_dir}/{clx}-oralscan-part-{i:04}-of-{file_numbers:04}.parquet"
        # d.to_parquet(out_name)

def make_test_dataset():
    source_dir= Path("/home/archie/Projects/data/口扫模型/模型测量关键点---/测试数据")
    out_dir="zby/oral_scan/data"
    group_size = 100
    make_parquet_(source_dir=source_dir,out_dir=out_dir,group_size=group_size,clx="test")

def make_train_dataset():
    source_dir= Path("/home/archie/Projects/data/口扫模型/模型测量关键点---/标注数据")
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
make_test_dataset()
# noop()
# noop_()
# noop_ok()

