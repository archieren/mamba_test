import numpy as np
import open3d as o3d
import random
import time
import torch

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
epoches = 1200
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
    for example in batch:
        oral_scan = o3d.geometry.TriangleMesh()
        oral_scan.vertices  = o3d.utility.Vector3dVector(example["vertices"])
        oral_scan.triangles = o3d.utility.Vector3iVector(example["triangles"])
        oral_scan.compute_vertex_normals()
        vertex_normals = np.asarray(oral_scan.vertex_normals)    # TODO：制做数据集时去处理， 这儿就省事了！

        vertices.append(torch.tensor(example["vertices"], dtype=torch.float))
        normals.append(torch.tensor(vertex_normals, dtype=torch.float))
        offset.append(example["vertices"].shape[0])
        label = torch.tensor(example["label"])
        label = label.map_(label,bi_cls)
        labels.append(label)
    
    vertices = torch.cat(vertices).cuda(device)
    normals  = torch.cat(normals).cuda(device)
    labels   = torch.cat(labels).cuda(device)
    offset = torch.tensor(offset, device=device).cumsum(0).int()
    return Dict(coord=vertices,
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


def loss_fn(pc):
    return sum(pc.loss.values())

def train():
    #Some Dir
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_file = checkpoints_dir.joinpath('model_weights.pth')
    train_loader = dataloader()
    test_loader = dataloader(split="test")
    m_config = make_default_config()
    model = MODE_CLS(m_config)
    if checkpoints_file.exists():
        model.load_state_dict(torch.load(checkpoints_file))
        print("Load a saved model")
    
    model= model.to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,], gamma=0.2)
    
    for epoch in range(epoches):
        model= model.train()
        loss_batch = []
        with tqdm(test_loader) as t:  #
            for i, data in enumerate(t):
                optimizer.zero_grad()    
                pc=model(PointCloud(data))
                loss = loss_fn(pc)
                loss.backward()                
                optimizer.step()
                loss_batch.append(loss.item())
                t.set_description(f"Epoch {epoch+1}/{epoches} : Train_Loss:{loss.item():6f} LearningRate:{optimizer.param_groups[0]['lr']:8f}")
        scheduler.step()
        mean_loss = np.mean(loss_batch)
        print(f"Epoch_{epoch+1}/{epoches}'s meam_loss:{mean_loss}")
        
        # model=model.eval()
        # with torch.no_grad():        
        #     with tqdm(test_loader) as t:
        #         for i,data in enumerate(t):
        #             pc = model(PointCloud(data))                      # prediction
        #             t.set_description(f"Epoch {epoch}/{epoches}: Loss:{loss_fn(pc)}")
        
        torch.save(model.state_dict(), checkpoints_file)
        print("Saved a checkpoints!")
train()
input()
