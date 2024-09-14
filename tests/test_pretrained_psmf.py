import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2"

import numpy as np
import open3d as o3d
import random
import time
import torch
import torch.nn as nn

from addict import Dict
from datasets import load_dataset
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pm.pointmamba as pms
from pm.pointmamba import PointSIS_Seg_Model, make_default_config
from pm.utils.point_cloud import PointCloud

device='cuda'
torch.device(device)

MODE_CLS = PointSIS_Seg_Model  # PointSIS_SEG

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#参数：TODO
epoches = 1000
batch_size = 1

def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.2f}秒".format(stop_time - start_time))
    return

def bi_cls(x, *y):  # 没想明白*y！
    if x > 0 : return int(x)
    else: return int(0)

def collate_fn(batch, device):
    vertices=[]
    normals=[]
    offset = []
    labels = []
    triangles = []
    for example in batch:
        oral_scan = o3d.geometry.TriangleMesh()
        oral_scan.vertices  = o3d.utility.Vector3dVector(example["vertices"])
        oral_scan.triangles = o3d.utility.Vector3iVector(example["triangles"])
        oral_scan.compute_vertex_normals()
        vertex_normals = np.asarray(oral_scan.vertex_normals)    # TODO：制做数据集时去处理， 这儿就省事了！

        vertices.append(torch.tensor(example["vertices"], dtype=torch.float))
        normals.append(torch.tensor(vertex_normals, dtype=torch.float))
        triangles.append(torch.tensor(example["triangles"], dtype=torch.int))
        offset.append(example["vertices"].shape[0])
        label = torch.tensor(example["label"])
        label = label.map_(label,bi_cls)
        labels.append(label)
    
    triangles = torch.cat(triangles).cuda(device)
    vertices = torch.cat(vertices).cuda(device)
    normals  = torch.cat(normals).cuda(device)
    labels   = torch.cat(labels).cuda(device)
    offset = torch.tensor(offset, device=device).cumsum(0).int()
    return Dict(coord=vertices,
                triangles=triangles,
                feat=normals, 
                labels=labels, 
                offset=offset, 
                grid_size=1.0e-2)

collate_fn = partial(collate_fn, device=device)

def dataloader(split="train"):
    datasets_root_name = "datasets_d"
    dataset_id = "zby/oral_scan"
    data_dir =Path.home().joinpath(datasets_root_name, dataset_id) # 终于搞成huggingface的那个样子，
    d = load_dataset(str(data_dir),split=split)   # load_dataset解释太多，这样也可以！
    d.set_format(type="numpy") # `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`
    loader = DataLoader(d, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


exp_dir = Path('./log/')
exp_dir.mkdir(exist_ok=True)
checkpoints_dir = exp_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
checkpoints_file = checkpoints_dir.joinpath('model_weights.pth')
test_loader = dataloader(split="test")
m_config = make_default_config()
model = MODE_CLS(m_config)

model.load_state_dict(torch.load(checkpoints_file))
print("Load a saved model")

model= model.to(device)

colors_t = [[[0.8, 0.2, 0.4]],
          [[0.9, 0.2, 0.8]]]

def fetch_colors(i:int,feat, pred_index):
    a_pred_mask = feat[:, pred_index].transpose(1,0)
    colors = a_pred_mask[i].detach().sigmoid().unsqueeze(1).repeat(1, 3)
    v = torch.tensor(colors_t[i%2], device=colors.device)
    colors = (colors*v).cpu().numpy()
    return colors

l_m = nn.LogSoftmax(dim=-1)
for i, data in enumerate(test_loader): 
    start_time = time.time()  
    pc =model(PointCloud(data))
    time_it(start_time)
    feat = pc.feat
    pred_probs = pc.pred_probs
    pred_index = l_m(pc.pred_probs)
    pred_index = torch.argmax(pred_index,dim=-1)
    pred_index = pred_index.squeeze(0).nonzero().view(-1)
    print(pred_index.shape)
    points = pc.coord.cpu().numpy()
    triangles = pc.triangles.cpu().numpy()

    #print(colors.shape)
    colors_1 = fetch_colors(5, feat=feat, pred_index=pred_index)
    t_1 = o3d.geometry.PointCloud()
    t_1.points = o3d.utility.Vector3dVector(points)
    t_1.colors = o3d.utility.Vector3dVector(colors_1)

    # colors_2 = fetch_colors(10, feat=feat, pred_index=pred_index)
    # t_2 = o3d.geometry.PointCloud()
    # t_2.points = o3d.utility.Vector3dVector(points)
    # t_2.colors = o3d.utility.Vector3dVector(colors_2)

    oral_scan = o3d.geometry.TriangleMesh()
    oral_scan.vertices  = o3d.utility.Vector3dVector(points)
    oral_scan.triangles = o3d.utility.Vector3iVector(triangles)
    oral_scan.compute_vertex_normals()

    # t.paint_uniform_color([0.75,1, 0])
    o3d.visualization.draw_geometries([t_1, oral_scan])

    
        

        


