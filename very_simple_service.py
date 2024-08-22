import json
import numpy as np
import open3d as o3d
import torch

from addict import Dict
from flask import Flask
from flask import request 
from pathlib import Path
from pm.pointmamba import PointSIS_SEG, make_default_config
from pm.utils.point_cloud import PointCloud

device='cuda'
torch.device(device)

def __get_ckpt(name='model_weights.pth')-> Path:
    #Some Dir
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_file = checkpoints_dir.joinpath(name)
    return checkpoints_file

def read_mesh(file_path:str):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.remove_duplicated_vertices()    # Must!
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh

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


m_config = make_default_config()
checkpoints_file = __get_ckpt()

model =PointSIS_SEG(m_config)
ckpt = torch.load(checkpoints_file)
model.load_state_dict(ckpt)
model.to(device=device)



app = Flask(__name__)

@app.route("/v0/test", methods=["GET"])
def test_semantic_segmentation():
    result_dict = {'success': True}
    try: 
        dc = make_data_dict_(upper_stl_path="./assets/124_upper.stl")
        pred = model(PointCloud(dc))
        pred = torch.argmax(pred, dim=-1)
        result_dict["response"] = pred.cpu().numpy().tolist()
    except Exception as e:
            result_dict['success'] = False
            result_dict['response'] = f"Exception: {e.args[0]}"
    finally:
         return json.dumps(result_dict, ensure_ascii=False, default=lambda o: o.__dict__)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8001)