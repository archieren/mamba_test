import io,os,tempfile
#os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2"

import numpy as np
import open3d as o3d
import random
import torch
import torch.nn as nn

from addict import Dict
from fastapi import APIRouter
from pathlib import Path

from pm.pointmamba import PointSIS_Seg_Model, make_default_config
from pm.service_api.protocol import SegRequest, SegResponse, str_to_file_bytes
from pm.utils.point_cloud import PointCloud
from pm.utils.align_the_mesh import align_the_mesh,S_O_I
from pm.pointmamba.conifuguration_point_sis import TEETH



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

def load_model(c_f:Path, model:nn.Module):  # TODO:还要打磨一下,主要为了一些模块的参数复用！.先留下思路！
    pre_trained_dict = torch.load(c_f)
    model_dict = model.state_dict()
    # for key in model_dict.keys():
    #     print(key)
    state_dict = {k:v for k,v in pre_trained_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

def __get_ckpt(name='model_weights.pth')-> Path:
    #Some Dir
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_file = checkpoints_dir.joinpath(name)
    return checkpoints_file

m_config = make_default_config()
checkpoints_file = __get_ckpt()

model = MODE_CLS(m_config)
load_model(checkpoints_file, model.model)      # HACK: XXX: FIXME: TODO:

model= model.to(device)
model.train(False)
# pc =model(PointCloud(data))
# pred_probs = pc.pred_probs
# pred_index = l_m(pc.pred_probs)
# pred_index = torch.argmax(pred_index,dim=-1)

def read_oral_scan_mesh_from_stl_in_str(str):
    file_content = io.BytesIO(str_to_file_bytes(str))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
    temp_file.write(file_content.getbuffer())
    temp_file.close()

    oral_scan_mesh = o3d.io.read_triangle_mesh(temp_file.name)
    oral_scan_mesh.remove_duplicated_vertices()
    oral_scan_mesh.remove_duplicated_triangles()
    oral_scan_mesh.compute_vertex_normals()   #!!!TODO
    os.remove(temp_file.name)
    return oral_scan_mesh

def make_data_dict(mesh:o3d.geometry.TriangleMesh, s_o_i:str):
    s_o_i = S_O_I.from_str(s_o_i)
    # dtype=torch.float是必要的!!!
    mesh, _ = align_the_mesh(mesh) 
    points = torch.asarray(np.asarray(mesh.vertices), device=device, dtype=torch.float)
    # points = points - torch.mean(points)  # TODO:好像,我还没将中心统一吧。
    normals = torch.asarray(np.asarray(mesh.vertex_normals), device=device,dtype=torch.float) 
    offset = torch.tensor([points.shape[0]], device=device).cumsum(0).int() # cumsum就成了浮点数了!
    s_o_i = torch.tensor([s_o_i], device=device).float()
    data = Dict(coord=points,feat=normals, offset=offset, s_o_i=s_o_i, grid_size=1.0e-2)
    return data    

def read_result(pc:PointCloud, threshold:float):
    def to_numpy(x:torch.Tensor):
        return x.detach().cpu().numpy()
    l_m = nn.LogSoftmax(dim=-1)
    offset=pc.offset                             # [N0, N0+N1,......]
    _offset = nn.functional.pad(offset, (1, 0))  # [0, N0, N0+N1,......]
    for i in range(len(offset)):                 # 
        feat = pc.feat[_offset[i]:offset[i]].sigmoid()                                      # 第i个点云的分割结果!预测的掩码！
        feat = to_numpy(feat)

        pred_probs = pc.pred_probs[i]                          #  b q l, fetch i -> q l
        pred_cls = torch.argmax(l_m(pred_probs),dim=-1)  # 每个 query预测了那个类！ -> q
        pred_cls_s = pred_cls.to_sparse()
        indices = to_numpy(pred_cls_s.indices()[0])                 # TODO:注意这个零！
        values  = to_numpy(pred_cls_s.values())

        seg_result = {}
        for j in range(len(indices)):
            if values[j] < 33 :
                t_num = TEETH.TEETH_cls_num[values[j]]
                (one_teeth_seg,)= np.where(feat[:, indices[j]] > threshold)
                seg_result[f'{t_num}'] = one_teeth_seg.tolist()
            # if values[j] in {33, 34, 35, 36}: # TODO: 牙龈， 合并的牙齿！
            #     (one_teeth_seg,)= np.where(feat[:, indices[j]] > threshold)
            #     seg_result[f'{values[j]+200}'] = one_teeth_seg.tolist()                
                
    return seg_result

def seg(stl:str, s_o_i:str):
    with torch.no_grad():
        mesh = read_oral_scan_mesh_from_stl_in_str(stl)
        dc = make_data_dict(mesh, s_o_i)
        pred = model(PointCloud(dc))
    return pred

pm_router = APIRouter(prefix="/segment")
@pm_router.post("/file")
def oral_scan_seg(req:SegRequest) -> SegResponse:
    threshhold = req.threshhold
    s_o_i = req.s_o_i
    pred = seg(req.stl, s_o_i)
    result = read_result(pred, threshold=threshhold)
    res = SegResponse(seg_result=result)
    return res
