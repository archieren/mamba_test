import json
import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
import numpy as np
import open3d as o3d
import regex as rg
from pathlib import Path
from pm.utils.align_the_mesh import align_the_mesh

cat_case = ["mt1", "radius", "teeth"] 
source_dir = Path("/home/archie/Projects/data/CPsurfcomp/DATA") / cat_case[2] /"meshes"
stems=[ off_item.stem for off_item in source_dir.glob("*.off")]

def get_labeled_data(source_dir:Path, stem:str):   # 假设做了前期处理！！！
    off_item_path = source_dir / (stem + ".off")
    mesh = o3d.io.read_triangle_mesh(str(off_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()
    return mesh

ttt = 0
for stem in stems:
    if ttt > 9 : 
        break
    
    mesh = get_labeled_data(source_dir, stem)
    mesh, _ = align_the_mesh(mesh)
    mesh.paint_uniform_color([0.1, 0.9, 0.9])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([mesh,frame])
    
    ttt += 1
