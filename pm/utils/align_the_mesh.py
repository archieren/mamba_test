import numpy as np
import open3d as o3d
import regex as rg

from enum import IntEnum
from sklearn.decomposition import PCA

class S_O_I(IntEnum):
    Superior = 0
    Inferior = 1

    @classmethod
    def from_str(cls, str):
        s_o_i = cls.Superior
        if rg.search("(lower|_l|_d)",str) != None:
            s_o_i = cls.Inferior
        return s_o_i

def is_soi(stem:str):   # TODO:口扫模型的名字,需要约定一下！
    s_o_i = S_O_I.Superior
    if rg.search("(lower|_l|_d)",stem) != None:
        s_o_i = S_O_I.Inferior
    return s_o_i

def scale_to_unit_sphere(points):   # 将处理的坐标归到单位球内！
    #shifts = (points.max(axis=0) + points.min(axis=0)) / 2
    shifts = points.mean(axis=0)
    points = points - shifts
    distances = np.linalg.norm(points, axis=1)
    scale = 1 / np.max(distances)
    points *= scale
    return points
   
def align_the_mesh(mesh:o3d.geometry.TriangleMesh): # 不要, s_o_i:S_O_I):
    """
    这段代码是小林的！致敬!
    """
    assert mesh.vertex_normals is not None, "此Mesh应当有Vertex Normals！"
    # 点集中心
    points = np.asarray(mesh.vertices)
    center = np.mean(points,axis=0)
    points -= center

    #计算法相向量均值!
    normals = mesh.vertex_normals
    mean_nvec = np.mean(np.asarray(normals), axis=0)
    mean_nvec = mean_nvec / np.linalg.norm(mean_nvec)  #点的法向量的归一均值
    pca = PCA(n_components=3, svd_solver='randomized')

    pca.fit(points)
    mat = pca.components_

    #TODO:还未弄清楚，如何解读几个轴的含义.
    #判断是否反向（Z轴）
    if np.dot(mat[2], mean_nvec) < 0:
        z_axis = - mat[2]
    else:
        z_axis = mat[2]

    projs = np.dot(points, mat[1]) / np.dot(mat[1], mat[1])
    projs = np.tile(mat[1], (points.shape[0], 1)) * np.expand_dims(projs, axis=-1)
    distses = np.linalg.norm(points - projs, axis=1) #所有顶点到中等特征值对应特征向量的距离
    min_dists_point = points[np.argmin(distses)] #最小值顶点
    max_dists_point = points[np.argmax(distses)]  #最大值顶点

    #判断是否反向（矢状向量）
    if np.dot(mat[1], max_dists_point - min_dists_point) < 0:
        y_axis = - mat[1]
    else:
        y_axis = mat[1]

    x_axis = np.cross(y_axis,z_axis)
    trans_mat = np.stack([x_axis,y_axis,z_axis],axis=0)

    # 最终决定,还是不用空间来区分，而是用category来区分！
    # if s_o_i == S_O_I.Superior: #如果上牙列,绕z舟转180度!
    #     angle_radians = np.pi
    #     R_z = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
    #                     [np.sin(angle_radians), np.cos(angle_radians), 0],
    #                     [0, 0, 1]])
    #     trans_mat = np.dot(trans_mat, R_z)

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(scale_to_unit_sphere(points))
    #mesh_out.vertices = o3d.utility.Vector3dVector(points)
    mesh_out.triangles = o3d.utility.Vector3iVector(mesh.triangles)
    mesh_out.rotate(trans_mat)
    mesh_out.compute_vertex_normals()
    
    return mesh_out, trans_mat


