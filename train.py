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
from pm.pointmamba import PointSIS_SEG, PointSIS_Seg_Model, make_default_config
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


def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.2f}秒".format(stop_time - start_time))
    return

def bi_cls(x, *y):  # 没想明白*y！
    if x > 0 : return 1
    else: return 0

def collate_fn(batch, device):
    vertices=[]
    # normals=[]
    offset = []
    labels = []
    for example in batch:
        # oral_scan = o3d.geometry.TriangleMesh()
        # oral_scan.vertices  = o3d.utility.Vector3dVector(example["vertices"])
        # oral_scan.triangles = o3d.utility.Vector3iVector(example["triangles"])
        # oral_scan.compute_vertex_normals()
        # vertex_normals = np.asarray(oral_scan.vertex_normals)    # TODO：制做数据集时去处理， 这儿就省事了！

        vertices.append(torch.tensor(example["vertices"], dtype=torch.float))
        # normals.append(torch.tensor(vertex_normals, dtype=torch.float))
        offset.append(example["vertices"].shape[0])
        label = torch.tensor(example["label"])
        label = label.map_(label,bi_cls)
        labels.append(label)
    
    vertices = torch.cat(vertices).cuda(device)
    # normals  = torch.cat(normals).cuda(device)
    labels   = torch.cat(labels).cuda(device)
    offset = torch.tensor(offset, device=device).cumsum(0).int()
    return Dict(coord=vertices,
                #feat=normals, 
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
    loader = DataLoader(d, batch_size=2, shuffle=True, collate_fn=collate_fn)
    return loader

loss_fn_f = pms.FocalLoss()
loss_fn_d = pms.DiceLoss()

def loss_fn(pre,gt):
    loss = loss_fn_f(pre,gt) + loss_fn_d(pre,gt)
    return loss

#参数：TODO
epoches = 1

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
    
    for epoch in range(epoches):
        model= model.train()
        loss_batch = []
        with tqdm(test_loader) as t:  #
            for i, data in enumerate(t):
                optimizer.zero_grad()    
                pred=model(PointCloud(data))
                loss = loss_fn(pred, data["labels"])
                loss.backward()                
                optimizer.step()

                loss_batch.append(loss.item())
                t.set_description(f"Epoch {epoch}/{epoches}.Train_Loss:{loss.item():6f}")
        mean_loss = np.mean(loss_batch)
        print(f"Epoch_{epoch}/{epoches}'s meam_loss:{mean_loss}")
        
        model=model.eval()
        with torch.no_grad():        
            with tqdm(test_loader) as t:
                IoU_epoch =[]
                Acc_epoch =[]
                for i,data in enumerate(t):
                    pred = model(PointCloud(data))                      # prediction
                    total = pred.shape[0]
                    pred = torch.argmax(pred, dim=-1)
                    TP = (pred    *data["labels"]).sum().cpu().numpy()       # 对牙齿的正确预测数
                    FP = ((1-pred)*data["labels"]).sum().cpu().numpy()       # 对牙齿的错误预测数 
                    TN = ((1-pred)*(1-data["labels"])).sum().cpu().numpy()   # 对牙龈的正确预测数
                    FN = (pred    *(1-data["labels"])).sum().cpu().numpy()   # 对牙龈的错误预测数

                    IoU = TP/(TP+FP+FN)                                  # TP其实是交集的大小。 TP+FP+FN其实是并集的大小
                    IoU_epoch.append(IoU)
                    Acc = (TP+TN)/(TP+FP+TN+FN)
                    Acc_epoch.append(Acc)
                    t.set_description(f"Epoch {epoch}/{epoches}: IoU:{IoU:6f} Acc:{Acc:6f}")
                    # equals = sn.eq(data["labels"]).sum().cpu().numpy()
                    # tooth_points = data["labels"].sum().cpu().numpy()
            mIoU = np.mean(IoU_epoch)
            mAcc = np.mean(Acc_epoch)   
        print(f"Epoch_{epoch}/{epoches}'s mIoU:{mIoU:6f}, mAcc:{mAcc:6f}")     
        torch.save(model.state_dict(), checkpoints_file)
        print("Saved a checkpoints!")
train()
input()
