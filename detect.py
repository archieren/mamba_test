import torch
from torch_geometric.utils import accuracy

from assembly import graph_net
import numpy as np
import vtkmodules.all as vtk
import networkx as nx
from sklearn.decomposition import PCA
from sklearn import preprocessing
import json
from torch_geometric.utils import subgraph
import matplotlib.pyplot as plt
from assembly.u2net import U2NET
from assembly.nested_unet import NestedUNet
import cfg
import os
import time
import open3d as o3d
from vtkmodules.util import numpy_support

CLS_DIC_LOWER = {8: '31', 7: '32', 6: '33',  5: '34', 4: '35',  3: '36', 2: '37', 1: '38',
                 9: '41', 10: '42', 11: '43', 12: '44', 13: '45', 14: '46', 15: '47', 16: '48', 17: 'other'}

# CLS_DIC_UPPER = {9: '11', 10: '12', 11: '13', 12: '14', 13: '15', 14: '16', 15: '17', 16: '18',
#                  8: '21', 7: '22', 6: '23', 5: '24', 4: '25', 3: '26', 2: '27', 1: '28', 17: 'other'}

CLS_DIC_UPPER = {8: '11', 7: '12', 6: '13', 5: '14', 4: '15', 3: '16', 2: '17', 1: '18',
                 9: '21', 10: '22', 11: '23', 12: '24', 13: '25', 14: '26', 15: '27', 16: '28', 17: 'other'}


class SegDetector:
    def __init__(self):

        self.device = torch.device('cuda:0')

        # self.unet = nested_unet.NestedUNet()
        self.unet = U2NET(3,18)
        self.unet.eval()
        self.unet = self.unet.to(self.device)
        # load_path = 'para\\unet.pt'
        # load_path = r'K:\project\oral_scan_segment2D\unet\para_u2net\640e.pt'
        # load_path = r'D:\ren_work\oral_scan_segment2D\u2net_320_512.pth'
        load_path = os.path.join(cfg.BASE_DIR, 'assembly', 'para','u2net_320_512.pth')
        # load_path = r'K:\project\oral_teeth_segmentation\u2net_320.pth'
        self.unet.load_state_dict(torch.load(load_path))

        self.graph_net = graph_net.Net()
        self.graph_net.eval()
        self.graph_net = self.graph_net.to(self.device)
        # load_path = 'para\\graph_net.pt'
        load_path = os.path.join(cfg.BASE_DIR, 'assembly', 'para', 'graph_net.pt')
        self.graph_net.load_state_dict(torch.load(load_path))

    def read_data(self, stl_pth):
        import time
        t1 = time.time()
        mesh = o3d.io.read_triangle_mesh(stl_pth)
        mesh.remove_duplicated_vertices()
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_pth)
        reader.Update()
        ori_plyd = reader.GetOutput()
        self.ori_plyd = ori_plyd
        # verts_num = ori_plyd.GetNumberOfPoints()
        x_min, x_max, y_min, y_max, z_min, z_max = ori_plyd.GetBounds()
        bound_cp = np.array([(x_max + x_min) / 2., (y_max + y_min) / 2., (z_max + z_min) / 2.])

        ori_verts = np.asarray(mesh.vertices)

        ori_verts = ori_verts - bound_cp

        triangles = np.asarray(mesh.triangles)
        self.faces = np.array(triangles)

        edge0 = np.stack([self.faces[:, 0], self.faces[:, 1]], axis=1) #三角形第一条边的集合
        edge1 = np.stack([self.faces[:, 1], self.faces[:, 2]], axis=1) #三角形第二条边的集合
        edge2 = np.stack([self.faces[:, 2], self.faces[:, 0]], axis=1) #三角形第三条边的集合
        edges = np.concatenate([edge0, edge1, edge2]) #三角形所有边的集合

        self.graph = nx.Graph() #定义一个无向图
        self.graph.add_edges_from(edges) #向图中添加边

        vtk_crvt_max = vtk.vtkCurvatures() #VTK中的vtkCurvatures类实现了四种计算网格模型点曲率的计算方法，四种曲率包括最小曲率、最大曲率、高斯曲率和平均曲率。
        vtk_crvt_max.SetCurvatureTypeToMaximum() #最大曲率
        vtk_crvt_max.SetInputData(ori_plyd)
        vtk_crvt_max.Update()
        max_crvts = [] #点最大曲率集合

        for i in range(vtk_crvt_max.GetOutput().GetNumberOfPoints()):
            crvt = vtk_crvt_max.GetOutput().GetPointData().GetScalars().GetComponent(i, 0)

            max_crvts.append(crvt)
        max_crvts = np.array(max_crvts)
        self.max_crvts = np.clip(max_crvts, -5., 5.) #np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值

        vtk_crvt_min = vtk.vtkCurvatures()
        vtk_crvt_min.SetCurvatureTypeToMinimum() #最小曲率
        vtk_crvt_min.SetInputData(ori_plyd)
        vtk_crvt_min.Update()
        min_crvts = [] #点最小曲率集合
        for i in range(vtk_crvt_min.GetOutput().GetNumberOfPoints()):
            crvt = vtk_crvt_min.GetOutput().GetPointData().GetScalars().GetComponent(i, 0)
            min_crvts.append(crvt)
        min_crvts = np.array(min_crvts)
        self.min_crvts = np.clip(min_crvts, -5., 5.) #np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值

        nvec = vtk.vtkPolyDataNormals() #计算法向量
        nvec.SetInputData(ori_plyd)
        nvec.SetSplitting(0)
        nvec.SetComputeCellNormals(0)
        nvec.Update()
        nvec_op = nvec.GetOutput().GetPointData().GetNormals()
        nvecs = [] #法向量集合

        for i in range(nvec.GetOutput().GetNumberOfPoints()):
            nvecs.append([nvec_op.GetComponent(i, 0), nvec_op.GetComponent(i, 1), nvec_op.GetComponent(i, 2)])
        nvecs = np.array(nvecs)
        mean_nvec = np.mean(nvecs, axis=0)
        mean_nvec = mean_nvec / np.linalg.norm(mean_nvec)
        t2 = time.time()
        pca = PCA(n_components=3, svd_solver='randomized')
        pca.fit(ori_verts)
        mat = pca.components_ #特征分解

        #判断是否反向（Z轴）
        if np.dot(mat[2], mean_nvec) < 0:
            z_axis = - mat[2]
        else:
            z_axis = mat[2]

        projs = np.dot(ori_verts, mat[1]) / np.dot(mat[1], mat[1])
        projs = np.tile(mat[1], (ori_verts.shape[0], 1)) * np.expand_dims(projs, axis=-1)
        distses = np.linalg.norm(ori_verts - projs, axis=1) #所有顶点到中等特征值对应特征向量的距离
        min_dists_point = ori_verts[np.argmin(distses)] #最小值顶点
        max_dists_point = ori_verts[np.argmax(distses)]  #最大值顶点

        #判断是否反向（矢状向量）
        if np.dot(mat[1], max_dists_point - min_dists_point) < 0:
            y_axis = - mat[1]
        else:
            y_axis = mat[1]

        x_axis = np.cross(y_axis, z_axis) #横向向量=矢状向量*颌面观向量

        trans_mat = np.stack([x_axis, y_axis, z_axis], axis=0)#特征向量

        """将顶点坐标变换到X轴=横向向量，Y轴=矢状向量，Z轴=颌面观向量的坐标系下。将空间沿X轴和Y轴划分为224*224个格栅柱。
        在每一个格栅柱内选择Z值最大的顶点为每个格栅柱的候选点。
        将格栅柱的索引值作为2D特征图的像素点坐标，候选点的法向量归一化后作为该像素点的三个通道值"""
        self.vertexs = np.matmul(trans_mat, ori_verts.T).T #变换坐标系

        _verts = self.vertexs.copy()
        _verts[:, 1] = - _verts[:, 1]
        self.nvecs = np.matmul(trans_mat, nvecs.T).T  #变换坐标系

        normed_verts = preprocessing.MinMaxScaler().fit_transform(_verts) #归一化
        normed_nvecs = preprocessing.MinMaxScaler().fit_transform(self.nvecs)

        ori_pic_size = 320
        x_cords = np.floor(normed_verts[:, 0] * (ori_pic_size - 1)).astype(int) #对元素截断取整
        y_cords = np.floor(normed_verts[:, 1] * (ori_pic_size - 1)).astype(int)

        coo = np.stack([y_cords, x_cords], axis=1)
        self.verts_coo_view = coo.view(dtype='int,int').reshape(-1)

        image = np.zeros((ori_pic_size, ori_pic_size, 4)) #初始话特征图零矩阵
        #对特征图矩阵进行填值
        for idx, w_cord in enumerate(x_cords):
            if image[y_cords[idx], w_cord, 3] < normed_verts[idx, 2]:
                image[y_cords[idx], w_cord, 0:3] = normed_nvecs[idx]
                image[y_cords[idx], w_cord, 3] = normed_verts[idx, 2]


        self.image = image[:, :, 0:3] #2d特征图


    def read_data_time(self, stl_pth):
        # Initialize the STL reader and load the data
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_pth)
        reader.Update()
        ori_plyd = reader.GetOutput()
        self.ori_plyd = ori_plyd

        # Calculate bounds and the center point
        x_min, x_max, y_min, y_max, z_min, z_max = ori_plyd.GetBounds()
        bound_cp = np.array([(x_max + x_min) / 2., (y_max + y_min) / 2., (z_max + z_min) / 2.])

        # Get vertices and center them
        ori_verts = np.array([ori_plyd.GetPoint(i) for i in range(ori_plyd.GetNumberOfPoints())])
        ori_verts -= bound_cp
        start_time = time.time()
        # Get faces (triangle connectivity) more efficiently
        faces = np.array([[ori_plyd.GetCell(id).GetPointId(j) for j in range(3)]
                          for id in range(ori_plyd.GetNumberOfCells())])
        print(time.time()-start_time,'....for time....')
        # Create edges for the graph from faces
        edges = np.concatenate([faces[:, [0, 1]],
                                faces[:, [1, 2]],
                                faces[:, [2, 0]]])
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)

        # Calculate maximum curvature
        vtk_crvt_max = vtk.vtkCurvatures()
        vtk_crvt_max.SetCurvatureTypeToMaximum()
        vtk_crvt_max.SetInputData(ori_plyd)
        vtk_crvt_max.Update()
        max_crvts = numpy_support.vtk_to_numpy(vtk_crvt_max.GetOutput().GetPointData().GetScalars())
        self.max_crvts = np.clip(max_crvts, -5., 5.)

        # Calculate minimum curvature
        vtk_crvt_min = vtk.vtkCurvatures()
        vtk_crvt_min.SetCurvatureTypeToMinimum()
        vtk_crvt_min.SetInputData(ori_plyd)
        vtk_crvt_min.Update()
        min_crvts = numpy_support.vtk_to_numpy(vtk_crvt_min.GetOutput().GetPointData().GetScalars())
        self.min_crvts = np.clip(min_crvts, -5., 5.)

        # Calculate normals
        nvec = vtk.vtkPolyDataNormals()
        nvec.SetInputData(ori_plyd)
        nvec.SetSplitting(0)
        nvec.SetComputeCellNormals(0)
        nvec.Update()
        nvecs = numpy_support.vtk_to_numpy(nvec.GetOutput().GetPointData().GetNormals())

        # Mean normal vector
        mean_nvec = np.mean(nvecs, axis=0)
        mean_nvec /= np.linalg.norm(mean_nvec)

        # PCA for dimensionality reduction
        pca = PCA(n_components=3, svd_solver='randomized')
        pca.fit(ori_verts)
        mat = pca.components_
        print(time.time() - start_time,'.......pca time.....')
        # Determine z-axis direction
        z_axis = -mat[2] if np.dot(mat[2], mean_nvec) < 0 else mat[2]

        # Project vertices onto the second principal component (y-axis)
        projs = (np.dot(ori_verts, mat[1]) / np.dot(mat[1], mat[1])).reshape(-1, 1)
        distses = np.linalg.norm(ori_verts - (mat[1] * projs), axis=1)

        min_dists_point = ori_verts[np.argmin(distses)]
        max_dists_point = ori_verts[np.argmax(distses)]

        # Determine y-axis direction
        y_axis = -mat[1] if np.dot(mat[1], max_dists_point - min_dists_point) < 0 else mat[1]
        x_axis = np.cross(y_axis, z_axis)

        # Transformation matrix
        trans_mat = np.stack([x_axis, y_axis, z_axis], axis=0)

        # Transform vertex coordinates
        self.vertexs = np.matmul(trans_mat, ori_verts.T).T
        _verts = self.vertexs.copy()
        _verts[:, 1] = -_verts[:, 1]  # Flip y-coordinates if needed
        self.nvecs = np.matmul(trans_mat, nvecs.T).T

        # Normalize vertices and normals
        normed_verts = preprocessing.MinMaxScaler().fit_transform(_verts)
        normed_nvecs = preprocessing.MinMaxScaler().fit_transform(self.nvecs)

        # Compute pixel coordinates
        ori_pic_size = 320
        x_cords = np.floor(normed_verts[:, 0] * (ori_pic_size - 1)).astype(int)  # 对元素截断取整
        y_cords = np.floor(normed_verts[:, 1] * (ori_pic_size - 1)).astype(int)

        coo = np.stack([y_cords, x_cords], axis=1)
        self.verts_coo_view = coo.view(dtype='int,int').reshape(-1)

        image = np.zeros((ori_pic_size, ori_pic_size, 4))  # 初始话特征图零矩阵
        # 对特征图矩阵进行填值
        for idx, w_cord in enumerate(x_cords):
            if image[y_cords[idx], w_cord, 3] < normed_verts[idx, 2]:
                image[y_cords[idx], w_cord, 0:3] = normed_nvecs[idx]
                image[y_cords[idx], w_cord, 3] = normed_verts[idx, 2]

        self.image = image[:, :, 0:3]  # 2d特征图



    def detecting(self,stl_pth, CLS_DIC,):
        self.read_data(stl_pth)
        image = torch.tensor(self.image)

        image = image.permute(2, 0, 1).float()
        _image = image.cuda()

        _image = torch.unsqueeze(_image, dim=0)
        # output = self.unet(_image)
        output = self.unet(_image)[0]
        cls_op = output.permute(0, 2, 3, 1)
        cls_op = torch.softmax(cls_op, dim=-1)
        pred = cls_op.argmax(dim=-1).squeeze().detach().cpu().numpy().astype(np.int32) #分割结果

        ori_pic_size = 320
        glob_edge_index = torch.tensor(np.array(self.graph.edges), dtype=torch.long).t().contiguous()
        seg_vid_dic = {}
        img_graph = nx.grid_2d_graph(ori_pic_size, ori_pic_size) #定义一个224*224的节点图
        total_tee_sets = []
        #取出单颗牙齿的数据
        for cls in np.unique(pred)[1:]:

            y_coos_pred, x_coos_pred = np.where(pred == cls)


            coos = np.stack([y_coos_pred.astype(np.int32), x_coos_pred.astype(np.int32)], axis=1)
            coos = list(map(tuple, coos))
            _graph = nx.subgraph(img_graph, coos) #子图
            coos = max(nx.connected_components(_graph), key=len) #获取连通分量的节点列表(nx.connected_components(G),包含每个连通图的节点列表)

            if len(coos) < 550:
                # print(len(coos))
                continue
            coos_view = np.array(list(coos), dtype='i,i')
            coos_view = coos_view.reshape(-1) #单颗牙齿的元素索引

            area = np.isin(self.verts_coo_view, coos_view) #判断self.verts_coo_view的元素是否属于coos_view
            area_vids = np.where(area)[0]
  
            area_vids_set = set(area_vids.tolist())
            area_vids_set_copy = area_vids_set.copy()

            #临界扩增（扩大2d分割牙齿范围）
            for k in range(15):
                bound_set = nx.node_boundary(self.graph, area_vids_set) #area_vids_set的节点边界
                area_vids_set = area_vids_set | bound_set #原节点与新节点的并集

            # edge_index, _ = subgraph(list(area_vids_set), glob_edge_index, relabel_nodes=True)
            # if 107208 in list(area_vids_set):
            #     area_vids_set.remove(107208)
            #
            #     edge_index, _ = subgraph(list(area_vids_set), glob_edge_index, relabel_nodes=True)
            # else:
            edge_index, _ = subgraph(list(area_vids_set), glob_edge_index, relabel_nodes=True)

            area_vids = np.array(list(area_vids_set), dtype=np.int32)
            local_verts = self.vertexs[area_vids]
            local_cp = np.mean(local_verts, axis=0)
            local_verts = local_verts - local_cp

            features = np.hstack([local_verts, self.nvecs[area_vids],
                                  np.expand_dims(self.max_crvts, axis=-1)[area_vids],
                                  np.expand_dims(self.min_crvts, axis=-1)[area_vids]])

            features = preprocessing.MinMaxScaler().fit_transform(features)
            x = torch.from_numpy(features).to(torch.float)

            _, sub2_out = self.graph_net(x.cuda(), edge_index.cuda())
            sub2_out = torch.squeeze(sub2_out)
            local_confs = sub2_out.detach().cpu().numpy()

            teeth_vids = area_vids[local_confs > 0.5]
            teeth_vids = set(teeth_vids.tolist())

            # if CLS_DIC[int(cls)] != '31':
            #     continue



            scalars = np.zeros(self.vertexs.shape[0])
            scalars[np.array(list(teeth_vids))] = 1.
            # scalars[area_vids] = sub2_out.detach().cpu().numpy()
            # mlab.triangular_mesh(self.vertexs[:, 0], self.vertexs[:, 1], self.vertexs[:, 2], self.faces, scalars=scalars, colormap='Paired')
            # mlab.view(-90.0, reset_roll=False)
            # mlab.show()

            teeth_vids = self.denoising_padding(area_vids_set, teeth_vids)
            # teeth_vids = self.denoising_padding(area_vids_set_copy, teeth_vids)

            scalars = np.zeros(self.vertexs.shape[0])
            scalars[np.array(list(teeth_vids))] = 1.
            # mlab.triangular_mesh(self.vertexs[:, 0], self.vertexs[:, 1], self.vertexs[:, 2], self.faces, scalars=scalars, colormap='Paired')
            # # mlab.view(-90.0, reset_roll=False)
            # mlab.show()

            teeth_vids = self.open_op(teeth_vids)

            seg_vid_dic[CLS_DIC[int(cls)]] = list(teeth_vids)
            total_tee_sets.append([list(teeth_vids), CLS_DIC[int(cls)]])
        return  seg_vid_dic

    def denoising_padding(self, area_vids_set, vids):
        _graph = nx.subgraph(self.graph, vids)
        vids = max(nx.connected_components(_graph), key=len)
        _graph = nx.subgraph(self.graph, area_vids_set - vids)
        for comp in sorted(nx.connected_components(_graph), key=len)[0:-1]:
            vids = vids | comp
        return vids

    def open_op(self, vids, ratio=0.3):
        ori_vnum = len(vids)
        trg_vnum = int(ori_vnum - ori_vnum * ratio)
        erode_k = 0
        for _ in range(50):
            bound_set = set([edge[0] for edge in nx.edge_boundary(self.graph, vids)])

            vids = vids - bound_set
            erode_k += 1
            if len(vids) < trg_vnum:
                break

        _graph = nx.subgraph(self.graph, vids)
        vids = max(nx.connected_components(_graph), key=len)

        for _ in range(erode_k):
            bound_set = nx.node_boundary(self.graph, vids)
            vids = vids | bound_set

        return vids

    def close_op(self, vids, area_vids_set, ratio=0.2):
        ori_vnum = len(vids)
        trg_vnum = int(ori_vnum + ori_vnum * ratio)
        dilate_k = 0
        for _ in range(50):
            bound_set = nx.node_boundary(self.graph, vids)
            vids = vids | bound_set
            dilate_k += 1
            if len(vids) > trg_vnum:
                break

        _graph = nx.subgraph(self.graph, area_vids_set - vids)
        for comp in sorted(nx.connected_components(_graph), key=len)[0:-1]:
            vids = vids | comp

        for _ in range(dilate_k):
            bound_set = set([edge[0] for edge in nx.edge_boundary(self.graph, vids)])
            vids = vids - bound_set
            if len(vids) <= ori_vnum:
                break

        return vids




