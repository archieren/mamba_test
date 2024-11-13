import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
import numpy as np
import open3d as o3d
import regex as rg
from pathlib import Path
from pm.utils.align_the_mesh import align_the_mesh

#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/乳牙")
#source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/TestData")
#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据")
source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/temp3")
stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
# stem = "lower0059"

def oral_scan_align(stem:str):
    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()

    # covariance_matrix = np.dot(points.T, points)/points.shape[0]
    # eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # main_direction = eigen_vectors[:, -1]
    # # x = np.dot(points, main_direction)

    mesh, trans_mat= align_the_mesh(mesh)
    # # 创建坐标系网格

    frame_o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0,origin=np.array([1.0,1.0,1.0]))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    frame.rotate(trans_mat)  # 将坐标系网格旋转到主方向

    # 创建并显示可视化窗口
    o3d.visualization.draw_geometries([mesh, frame_o, frame], point_show_normal=True)

    # points = np.asarray(mesh.vertices)
    # #points[:, 0] = 0
    # print(points.max(0), points.min(0))
    # t_1 = o3d.geometry.PointCloud()
    # t_1.points = o3d.utility.Vector3dVector(points)
    # #t_1.colors = o3d.utility.Vector3dVector(colors_1)
    # o3d.visualization.draw_geometries([t_1])

for stem in stems:
    #if stem == "lower0061":
    print(stem)
    oral_scan_align(stem)