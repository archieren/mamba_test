import json
import os,sys
sys.path.append(os.getcwd()) # 先这样!!!
import numpy as np
import open3d as o3d
import regex as rg
from pathlib import Path
from pm.utils.align_the_mesh import align_the_mesh


def get_labeled_data(source_dir:Path, stem:str):   # 假设做了前期处理！！！
    json_item_path = source_dir / (stem + ".json")
    label = json.load(json_item_path.open())

    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()    # 去重是必须的！！
    mesh.remove_duplicated_triangles()   # 去重是必须的！！
    mesh.compute_vertex_normals()
    return mesh, label

def fit_bounding_box(xyz):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # return pcd.get_minimal_oriented_bounding_box()
    b_box = o3d.geometry.OrientedBoundingBox.create_from_points_minimal(o3d.utility.Vector3dVector(xyz))
    b_box.color = (1.0,0.0,1.0)
    return b_box

def fit_ellipsoid(points, method='best_fit'):
    from scipy.linalg import eigh
    # points: n×3 numpy array
    n = points.shape[0]
    center = np.mean(points, axis=0)
    q = points - center
    C = q.T @ q / n
    w, v = eigh(C)  # w: eigenvalues, v: eigenvectors (columns)
    # Order descending (largest eigenvalue first)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    
    # Project points
    r = q @ v
    
    # if method == 'best_fit':
    #     # Best-fit (least squares)
    #     u = r[:, 0]**2
    #     vv = r[:, 1]**2
    #     ww = r[:, 2]**2
    #     A = np.column_stack((u, vv, ww))
    #     d, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
    #     a, b, c = 1 / np.sqrt(np.abs(d))
    # elif method == 'bounding':
        # Bounding ellipsoid (contains all points)
    a = np.max(np.abs(r[:, 0]))
    b = np.max(np.abs(r[:, 1]))
    c = np.max(np.abs(r[:, 2]))
    # else:
    #     raise ValueError("Method must be 'best_fit' or 'bounding'")
    
    return center, v, (a, b, c)  # center, rotation matrix, semi-axes

#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/乳牙")
#source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/TestData")
#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据")
#source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/temp2")
#source_dir = Path("/home/archie/Projects/data/口扫模型/口扫模型分割新增（有乳牙）")
source_dir = Path("/home/archie/Projects/data/口扫模型/HowTo")
stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
# stem = "lower0059"

def create_a_fit_ellipsoid(points, method='best_fit'):
    center, v, (a, b, c) = fit_ellipsoid(points,method=method)

    radii = np.array([a,b,c])    # Semi-axis lengths [a, b, c]
    rotation = v                 # Rotation matrix (columns = principal axes)


    # Create unit sphere (radius=1)
    mesh_elli = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0,
        resolution=100  # Increase for smoother surface
    )
    mesh_elli.compute_vertex_normals()

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation @ np.diag(radii)  # Combine rotation and scaling
    T[:3, 3] = center                     # Translation

    # Apply transformation
    mesh_elli.transform(T)

    # Customize appearance
    mesh_elli.paint_uniform_color([0.1, 0.9, 0.1])  # RGB color (blue)
    mesh_elli.compute_vertex_normals()               # Improve lighting
    
    return mesh_elli
    
def oral_scan_align(stem:str):
    #stl_item_path = source_dir / (stem + ".stl")
    mesh, label = get_labeled_data(source_dir, stem)
    #print(label['seg'].keys())
    mesh, trans_mat= align_the_mesh(mesh)
    # # 创建坐标系网格

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    frame_new = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame.rotate(trans_mat)  # 将坐标系网格旋转到主方向

    points_out = np.asarray(mesh.vertices)
    #mesh_elli = create_a_fit_ellipsoid(points_out, method='bounding')

    geos = [frame_new] # mesh,
    points_ = np.asarray(mesh.vertices)
    for key, indexes_value in label['seg'].items():
        #print(key, indexes_value)
        points_part = points_[indexes_value]
        
        part_elli = create_a_fit_ellipsoid(points_part, method='bounding')
        geos.append(part_elli)
        
        part_box = fit_bounding_box(points_part)
        geos.append(part_box)
    
    points_[:,2] = 0
    project_pc = o3d.geometry.PointCloud()
    project_pc.points = o3d.utility.Vector3dVector(points_)
    geos.append(project_pc)   
    # points_31 = np.asarray(mesh.vertices)[label['seg']['31']]
    # mesh_elli_31 = create_a_fit_ellipsoid(points_31, method='bounding')
    # points_36 = np.asarray(mesh.vertices)[label['seg']['36']]
    # mesh_elli_36 = create_a_fit_ellipsoid(points_36, method='bounding')
    # points_41 = np.asarray(mesh.vertices)[label['seg']['41']]
    # mesh_elli_41 = create_a_fit_ellipsoid(points_41, method='bounding')
    # points_46 = np.asarray(mesh.vertices)[label['seg']['46']]
    # mesh_elli_46 = create_a_fit_ellipsoid(points_46, method='bounding')


    # Visualize
    o3d.visualization.draw_geometries(geos,point_show_normal=True)
    
for stem in stems:
    if stem == "00252_l":
        print(stem)
        oral_scan_align(stem)
        
#