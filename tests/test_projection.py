import numpy as np
import vtk
import networkx as nx
from sklearn.decomposition import PCA
from sklearn import preprocessing
def read_data(self, stl_pth):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_pth)
    reader.Update()
    ori_plyd = reader.GetOutput()
    ori_plyd = ori_plyd
    # verts_num = ori_plyd.GetNumberOfPoints()
    x_min, x_max, y_min, y_max, z_min, z_max = ori_plyd.GetBounds()
    bound_cp = np.array([(x_max + x_min) / 2., (y_max + y_min) / 2., (z_max + z_min) / 2.])

    ori_verts = []  # stl所有点集合
    for i in range(ori_plyd.GetNumberOfPoints()):
        ori_verts.append(ori_plyd.GetPoint(i))
    ori_verts = np.array(ori_verts)
    ori_verts = ori_verts - bound_cp

    faces = []  # 三角面的连接点的索引
    for id in range(ori_plyd.GetNumberOfCells()):
        p0_idx = ori_plyd.GetCell(id).GetPointId(0)
        p1_idx = ori_plyd.GetCell(id).GetPointId(1)
        p2_idx = ori_plyd.GetCell(id).GetPointId(2)
        faces.append([p0_idx, p1_idx, p2_idx])

    faces = np.array(faces)

    edge0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)  # 三角形第一条边的集合
    edge1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)  # 三角形第二条边的集合
    edge2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)  # 三角形第三条边的集合
    edges = np.concatenate([edge0, edge1, edge2])  # 三角形所有边的集合

    graph = nx.Graph()  # 定义一个无向图
    graph.add_edges_from(edges)  # 向图中添加边

    vtk_crvt_max = vtk.vtkCurvatures()  # VTK中的vtkCurvatures类实现了四种计算网格模型点曲率的计算方法，四种曲率包括最小曲率、最大曲率、高斯曲率和平均曲率。
    vtk_crvt_max.SetCurvatureTypeToMaximum()  # 最大曲率
    vtk_crvt_max.SetInputData(ori_plyd)
    vtk_crvt_max.Update()
    max_crvts = []  # 点最大曲率集合

    for i in range(vtk_crvt_max.GetOutput().GetNumberOfPoints()):
        crvt = vtk_crvt_max.GetOutput().GetPointData().GetScalars().GetComponent(i, 0)

        max_crvts.append(crvt)
    max_crvts = np.array(max_crvts)
    max_crvts = np.clip(max_crvts, -5., 5.)  # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值

    vtk_crvt_min = vtk.vtkCurvatures()
    vtk_crvt_min.SetCurvatureTypeToMinimum()  # 最小曲率
    vtk_crvt_min.SetInputData(ori_plyd)
    vtk_crvt_min.Update()
    min_crvts = []  # 点最小曲率集合
    for i in range(vtk_crvt_min.GetOutput().GetNumberOfPoints()):
        crvt = vtk_crvt_min.GetOutput().GetPointData().GetScalars().GetComponent(i, 0)
        min_crvts.append(crvt)
    min_crvts = np.array(min_crvts)
    min_crvts = np.clip(min_crvts, -5., 5.)  # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值

    nvec = vtk.vtkPolyDataNormals()  # 计算法向量
    nvec.SetInputData(ori_plyd)
    nvec.SetSplitting(0)
    nvec.SetComputeCellNormals(0)
    nvec.Update()
    nvec_op = nvec.GetOutput().GetPointData().GetNormals()
    nvecs = []  # 法向量集合

    for i in range(nvec.GetOutput().GetNumberOfPoints()):
        nvecs.append([nvec_op.GetComponent(i, 0), nvec_op.GetComponent(i, 1), nvec_op.GetComponent(i, 2)])
    nvecs = np.array(nvecs)
    mean_nvec = np.mean(nvecs, axis=0)
    mean_nvec = mean_nvec / np.linalg.norm(mean_nvec)

    pca = PCA(n_components=3, svd_solver='randomized')
    pca.fit(ori_verts)
    mat = pca.components_  # 特征分解
    # 判断是否反向（Z轴）
    if np.dot(mat[2], mean_nvec) < 0:
        z_axis = - mat[2]
    else:
        z_axis = mat[2]

    projs = np.dot(ori_verts, mat[1]) / np.dot(mat[1], mat[1])
    projs = np.tile(mat[1], (ori_verts.shape[0], 1)) * np.expand_dims(projs, axis=-1)
    distses = np.linalg.norm(ori_verts - projs, axis=1)  # 所有顶点到中等特征值对应特征向量的距离
    min_dists_point = ori_verts[np.argmin(distses)]  # 最小值顶点
    max_dists_point = ori_verts[np.argmax(distses)]  # 最大值顶点

    # 判断是否反向（矢状向量）
    if np.dot(mat[1], max_dists_point - min_dists_point) < 0:
        y_axis = - mat[1]
    else:
        y_axis = mat[1]

    x_axis = np.cross(y_axis, z_axis)  # 横向向量=矢状向量*颌面观向量

    trans_mat = np.stack([x_axis, y_axis, z_axis], axis=0  )  # 特征向量

    """将顶点坐标变换到X轴=横向向量，Y轴=矢状向量，Z轴=颌面观向量的坐标系下。将空间沿X轴和Y轴划分为224*224个格栅柱。
    在每一个格栅柱内选择Z值最大的顶点为每个格栅柱的候选点。
    将格栅柱的索引值作为2D特征图的像素点坐标，候选点的法向量归一化后作为该像素点的三个通道值"""
    vertexs = np.matmul(trans_mat, ori_verts.T).T  # 变换坐标系

    _verts = vertexs.copy()
    _verts[:, 1] = - _verts[:, 1]
    nvecs = np.matmul(trans_mat, nvecs.T).T  # 变换坐标系

    normed_verts = preprocessing.MinMaxScaler().fit_transform(_verts)  # 归一化
    normed_nvecs = preprocessing.MinMaxScaler().fit_transform(nvecs)

    ori_pic_size = 320
    x_cords = np.floor(normed_verts[:, 0] * (ori_pic_size - 1)).astype(int)  # 对元素截断取整
    y_cords = np.floor(normed_verts[:, 1] * (ori_pic_size - 1)).astype(int)

    coo = np.stack([y_cords, x_cords], axis=1)
    verts_coo_view = coo.view(dtype='i,i').reshape(-1)

    image = np.zeros((ori_pic_size, ori_pic_size, 4))  # 初始话特征图零矩阵
    # 对特征图矩阵进行填值
    for idx, w_cord in enumerate(x_cords):
        if image[y_cords[idx], w_cord, 3] < normed_verts[idx, 2]:
            image[y_cords[idx], w_cord, 0:3] = normed_nvecs[idx]
            image[y_cords[idx], w_cord, 3] = normed_verts[idx, 2]


    image = image[:, :, 0:3]  # 2d特征图