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

        # ori_verts = []  #stl所有点集合
        # for i in range(ori_plyd.GetNumberOfPoints()):
        #     ori_verts.append(ori_plyd.GetPoint(i))
        # ori_verts = np.array(ori_verts)

        ori_verts = np.asarray(mesh.vertices)

        ori_verts = ori_verts - bound_cp

        faces = [] #三角面的连接点的索引

        # for id in range(ori_plyd.GetNumberOfCells()):
        #     p0_idx = ori_plyd.GetCell(id).GetPointId(0)
        #     p1_idx = ori_plyd.GetCell(id).GetPointId(1)
        #     p2_idx = ori_plyd.GetCell(id).GetPointId(2)
        #     faces.append([p0_idx, p1_idx, p2_idx])
        # print(time.time() - t1,'....for time......')
        # self.faces = np.array(faces)
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




        # _img_graph = nx.grid_2d_graph(224, 224) #定义一个224*224的节点图
        # _pred = np.zeros(pred.shape, dtype=np.int32)
        # for cls in np.unique(pred)[1:]:
        #     y_coos_pred, x_coos_pred = np.where(pred == cls)
        #     coos = np.stack([y_coos_pred.astype(np.int32), x_coos_pred.astype(np.int32)], axis=1)
        #     coos = list(map(tuple, coos))
        #     _graph = nx.subgraph(_img_graph, coos)
        #     coos = max(nx.connected_components(_graph), key=len)
        #     if len(coos) < 20:
        #         continue
        #     coos = np.array(list(coos))
        #     _pred[coos[:, 0], coos[:, 1]] = cls

        # plt.subplot(1, 2, 1)
        # # plt.grid(True)
        # plt.imshow(image.permute(1, 2, 0).numpy())
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(_pred)
        # for cls in np.unique(_pred)[1:]:
        #     mask = _pred == cls
        #     ys, xs = np.where(mask)
        #     cx = np.mean(xs)
        #     cy = np.mean(ys)
        #     plt.text(cx, cy, str(cls))

        # plt.subplot(1, 3, 3)
        # plt.imshow(pred)
        # plt.imshow(self.image)
        # for cls in np.unique(pred)[1:]:
        #     mask = pred == cls
        #     ys, xs = np.where(mask)
        #     cx = np.mean(xs)
        #     cy = np.mean(ys)
        #     plt.text(cx, cy, str(cls))
        # plt.show()
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

            # _vids_set = set(area_vids.tolist())
            # face_ids = vtk.vtkIdTypeArray()
            # face_ids.SetNumberOfComponents(1)
            # for face_id in range(self.ori_plyd.GetNumberOfCells()):
            #     p0_idx = self.ori_plyd.GetCell(face_id).GetPointId(0)
            #     p1_idx = self.ori_plyd.GetCell(face_id).GetPointId(1)
            #     p2_idx = self.ori_plyd.GetCell(face_id).GetPointId(2)
            #     if p0_idx in _vids_set or p1_idx in _vids_set or p2_idx in _vids_set:
            #         face_ids.InsertNextValue(face_id)
            #
            # selectionNode = vtk.vtkSelectionNode()
            # selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
            # selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
            # selectionNode.SetSelectionList(face_ids)
            #
            # selection = vtk.vtkSelection()
            # selection.AddNode(selectionNode)
            #
            # extractSelection = vtk.vtkExtractSelection()
            # extractSelection.SetInputData(0, self.ori_plyd)
            # extractSelection.SetInputData(1, selection)
            # extractSelection.Update()
            #
            # # selected = vtk.vtkUnstructuredGrid()
            # # selected.ShallowCopy(extractSelection.GetOutput())
            #
            # extract_op = extractSelection.GetOutput()
            # surface_filter = vtk.vtkDataSetSurfaceFilter()
            # surface_filter.SetInputData(extract_op)
            # surface_filter.Update()
            # plyd = surface_filter.GetOutput()
            #
            # _teeth_verts = []
            # for i in range(plyd.GetNumberOfPoints()):
            #     p = plyd.GetPoint(i)
            #     _teeth_verts.append(p)
            # _teeth_verts = np.array(_teeth_verts, dtype=np.float)
            #
            # _teeth_tri_ids = []
            # for id in range(plyd.GetNumberOfCells()):
            #     p0_idx = plyd.GetCell(id).GetPointId(0)
            #     p1_idx = plyd.GetCell(id).GetPointId(1)
            #     p2_idx = plyd.GetCell(id).GetPointId(2)
            #     _teeth_tri_ids.append([p0_idx, p1_idx, p2_idx])
            # _teeth_tri_ids = np.array(_teeth_tri_ids)
            #
            # import mayavi.mlab as mlab
            # mlab.triangular_mesh(_teeth_verts[:, 0], _teeth_verts[:, 1], _teeth_verts[:, 2], _teeth_tri_ids, color=(0.95, 0.95, 0.95))
            # mlab.show()

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

            # scalars = np.zeros(self.vertexs.shape[0])
            # scalars[np.array(list(teeth_vids))] = 1.
            # mlab.triangular_mesh(self.vertexs[:, 0], self.vertexs[:, 1], self.vertexs[:, 2], self.faces, scalars=scalars, colormap='Paired')
            # mlab.view(-90.0, reset_roll=False)
            # mlab.show()

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




if __name__ == '__main__':
    import glob
    import os
    import repair

    # stl_pth = 'E:\jfl\u.stl'
    stl_pth = r'D:\ren_work\ORAL_SCAN_AI\oral_scan_ai\20231108153358940_upper.stl'
    # stl_pth = r'K:\oral_mesh\upper3026.stl'
    detector = SegDetector()
    # detector.read_data(stl_pth)

    seg_dic = detector.detecting(stl_pth,CLS_DIC_UPPER)
    print(seg_dic)
    # repairer = repair.Repairer(stl_pth, seg_dic)
    # repairer.gen_teeth_obj_dic()
    # repairer.repairing()
    # repairer.save_as_stl(r'D:\ren_work\oral_scan_segment2D\test_stl')


    # def dice(x, y):
    #     """
    #     没做参数有效检查检查,先这样吧！
    #     用重合的点数，来计算Dice系数！
    #     """
    #     x, y = set(x), set(y)
    #     return len(x & y) / len(x | y)
    # # a = [37446, 36722, 37447, 36723, 36724, 37448, 37445, 36721, 35979, 35980, 35981, 35983, 35984, 35986, 36725, 37449, 37450, 37451, 38156, 38157, 38158, 38159, 37444, 36718, 36720, 38155, 35978, 35194, 35195, 35196, 35982, 35199, 35985, 35202, 35203, 35987, 36726, 38160, 38161, 38162, 37452, 38848, 38849, 38850, 38851, 38852, 38853, 37443, 36717, 38152, 38153, 38154, 35977, 36719, 35193, 34433, 35975, 35976, 34436, 35197, 35198, 34441, 35200, 34446, 35204, 35988, 36727, 36728, 38854, 38855, 38856, 38163, 36729, 37453, 38847, 39531, 39532, 39533, 39534, 37442, 38151, 36716, 38843, 38844, 38845, 38846, 36715, 35192, 35974, 34432, 33666, 34434, 34435, 36714, 33667, 34437, 34438, 34439, 33673, 34442, 35201, 34444, 34445, 33678, 34447, 35205, 35989, 35990, 39535, 39536, 39537, 39538, 38857, 38164, 36730, 37454, 39530, 40169, 40170, 40171, 40172, 40173, 40174, 40175, 37441, 37440, 38150, 38149, 38842, 38148, 39527, 39528, 39529, 36713, 35191, 35973, 34431, 33665, 32909, 35972, 32910, 32911, 33668, 33670, 34440, 33672, 32918, 33674, 34443, 33675, 33676, 33677, 32921, 32922, 33679, 34448, 34449, 35206, 35207, 35208, 35991, 40176, 40177, 40178, 40179, 39539, 38858, 38165, 38859, 36731, 37455, 37456, 40168, 40773, 40774, 40775, 40776, 40777, 40778, 40779, 40780, 40781, 40782, 37439, 36711, 36712, 37438, 38841, 38147, 39526, 40165, 40166, 40167, 35190, 34430, 33664, 32908, 32146, 35971, 35969, 36710, 32147, 32912, 33669, 32915, 33671, 32916, 32917, 32154, 32919, 32920, 32157, 32158, 32159, 32160, 32161, 32923, 33680, 34450, 34451, 34452, 35209, 35992, 36732, 40783, 40784, 40785, 40786, 40180, 39540, 39541, 38166, 38860, 39542, 38861, 36733, 37457, 40772, 41379, 41380, 41381, 41382, 41383, 41384, 41385, 41386, 41387, 41388, 37437, 38146, 38840, 38839, 39525, 40164, 40769, 40768, 40770, 40771, 35189, 34429, 33663, 32907, 34428, 32145, 31430, 35970, 35968, 35187, 36709, 31431, 31432, 32148, 32913, 32914, 32151, 32152, 32153, 31440, 32155, 32156, 31442, 31443, 31444, 31445, 32162, 32924, 32925, 33681, 34453, 35210, 35993, 41389, 41390, 40787, 40181, 40182, 39543, 38167, 38168, 38862, 39544, 36734, 37458, 41378, 41967, 41968, 41969, 41970, 41971, 41972, 41973, 41974, 41975, 41976, 41977, 41978, 41979, 41980, 41981, 41982, 41983, 37436, 38145, 39524, 38838, 39523, 40163, 40767, 40162, 41375, 41376, 41377, 35188, 34427, 33662, 32905, 33661, 32906, 33660, 32144, 32143, 31429, 30720, 35967, 35186, 34426, 36708, 37435, 30721, 31433, 32149, 32150, 31434, 31435, 31437, 31438, 31439, 30729, 31441, 30730, 30731, 30732, 30733, 30734, 30735, 31446, 32163, 32926, 32927, 33682, 34454, 34455, 35211, 35994, 35212, 41984, 41985, 41391, 40788, 40789, 40183, 37459, 38169, 38863, 38864, 39545, 40184, 36735, 41966, 42542, 42543, 42544, 42545, 42546, 42547, 42548, 42549, 42550, 42551, 42552, 42553, 42554, 42555, 42556, 38144, 38837, 39522, 40766, 41374, 40161, 41963, 41964, 41965, 32902, 32904, 32142, 33659, 32901, 31428, 30719, 30014, 35966, 36707, 35185, 34425, 37433, 37434, 30722, 30723, 30724, 31436, 30726, 30728, 30020, 30021, 30022, 30023, 30024, 30736, 31447, 32164, 32928, 32165, 33683, 33684, 33685, 34456, 35995, 36736, 35213, 42557, 42558, 42559, 41986, 41392, 41393, 40790, 37460, 38170, 38865, 39546, 39547, 40185, 40791, 37461, 36737, 42541, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 38143, 38836, 39521, 40765, 41372, 41373, 40160, 39519, 41962, 42540, 32138, 32903, 32141, 31427, 33658, 34423, 34424, 32900, 30718, 30013, 29288, 29289, 30015, 35965, 36706, 35184, 37432, 38141, 38142, 30016, 30725, 30018, 30727, 30019, 29296, 29297, 29298, 29300, 30025, 30026, 30027, 30737, 30738, 31448, 32929, 33686, 32166, 34457, 34458, 34459, 34460, 34461, 35214, 35996, 36738, 35997, 43140, 43141, 42560, 41987, 41394, 38171, 38172, 38866, 39548, 40186, 40792, 40793, 41395, 37462, 43124, 42539, 43673, 43674, 43675, 43676, 43677, 43678, 43679, 43680, 43681, 43682, 43683, 43684, 43685, 43686, 38835, 39520, 40764, 41371, 41960, 41961, 40159, 40763, 39518, 38833, 38834, 42538, 32137, 31422, 32139, 32140, 31426, 30717, 33657, 32899, 34422, 32136, 30012, 29287, 27566, 30011, 27567, 29290, 29291, 35964, 36705, 35963, 35183, 35961, 35962, 37431, 38140, 30017, 29294, 29295, 27573, 27575, 27576, 29299, 27577, 27578, 27579, 29301, 29302, 30028, 30739, 31449, 32167, 32930, 33687, 34462, 31450, 35215, 35216, 35217, 35218, 35219, 35998, 36739, 36740, 36741, 36742, 37463, 36743, 36744, 43687, 43688, 42561, 42562, 43142, 41396, 41988, 38867, 38868, 38869, 38870, 38173, 39549, 40187, 40794, 41397, 41398, 38174, 43123, 43122, 43672, 44212, 44213, 44214, 44215, 44216, 44217, 44218, 44219, 44220, 44221, 44222, 44223, 41370, 41959, 42535, 42534, 42533, 42536, 42537, 40158, 40762, 39517, 38832, 38139, 31421, 30711, 30712, 31423, 31424, 31425, 30716, 33656, 32898, 34421, 32135, 32897, 29286, 27564, 27565, 25869, 30010, 29285, 30715, 25870, 27568, 29292, 36704, 37429, 37430, 35182, 35960, 36703, 29293, 27572, 25874, 27574, 25877, 25878, 25879, 25880, 25881, 27580, 29303, 30029, 30030, 30031, 30740, 32168, 32931, 33688, 34463, 33689, 31451, 35999, 36000, 36001, 36002, 35220, 36745, 36746, 36759, 36758, 36757, 36756, 36755, 37466, 36754, 37464, 44224, 43689, 42563, 43143, 41989, 41990, 42564, 39550, 39551, 39552, 39553, 38871, 40188, 40795, 41399, 37467, 38175, 43121, 43670, 43671, 44211, 45305, 45304, 45299, 45306, 45307, 45308, 45309, 45310, 45311, 45312, 45313, 45314, 45315, 45316, 45317, 45318, 45319, 45320, 45321, 45322, 45323, 45324, 41369, 41958, 41367, 41956, 42532, 43119, 43120, 40157, 40761, 40760, 41368, 39516, 38831, 38137, 38138, 31420, 30710, 30008, 30713, 30714, 33655, 32896, 33654, 34420, 32134, 32895, 27563, 25866, 25867, 25868, 24189, 30009, 29283, 29284, 25871, 27569, 27571, 27570, 37428, 36702, 35181, 35958, 35959, 25873, 24193, 25875, 25876, 24195, 24196, 24197, 24198, 24199, 24201, 25882, 27581, 29304, 29305, 30032, 30741, 32169, 32932, 32933, 34464, 33690, 34465, 30742, 31452, 32170, 36747, 35221, 36003, 36748, 36749, 36752, 36751, 36750, 36013, 36014, 36760, 36011, 36010, 36761, 36753, 36009, 37465, 43690, 44225, 45325, 45326, 43144, 43145, 43691, 41991, 42565, 43146, 40189, 40190, 40191, 40192, 39554, 38872, 40796, 41400, 41992, 41993, 37468, 38176, 43669, 44206, 44210, 45303, 49326, 45300, 45700, 45701, 45702, 45703, 45704, 45705, 45706, 45707, 45708, 41957, 41366, 41955, 41365, 42531, 41954, 43118, 43117, 43668, 40156, 39515, 40759, 40758, 38830, 38136, 31419, 30709, 30006, 30007, 29281, 29282, 34419, 33653, 34418, 32133, 31417, 31418, 32894, 27562, 25865, 24185, 24186, 24188, 22562, 22563, 24190, 27561, 24192, 25872, 37427, 37426, 36701, 35956, 35957, 35180, 22567, 24194, 22568, 22569, 22570, 22571, 22572, 22573, 24200, 22576, 22577, 24202, 25883, 27582, 27583, 29306, 30033, 30743, 32934, 35222, 33691, 34466, 35223, 35224, 31453, 32171, 32172, 36004, 36006, 36007, 36005, 36012, 35233, 35234, 36015, 35232, 35231, 36016, 36762, 36008, 35230, 44226, 45327, 45709, 45710, 45711, 43692, 43693, 43147, 42566, 40797, 40798, 40799, 40194, 40193, 39555, 38873, 41401, 41994, 42567, 37469, 38177, 44208, 44207, 44205, 43659, 45302, 44204, 45301, 46080, 46081, 46082, 46083, 46084, 46085, 41364, 40757, 42530, 43116, 41953, 42529, 43664, 43667, 40155, 39514, 40154, 40153, 38829, 38135, 30708, 30005, 29280, 27559, 33652, 34417, 32132, 31416, 32129, 32130, 32131, 32893, 33651, 25864, 27560, 24184, 22559, 22560, 24187, 22561, 20991, 22564, 24191, 22565, 22566, 36700, 37425, 38134, 35955, 35179, 20994, 20995, 20996, 20997, 20998, 20999, 22574, 22575, 21003, 21004, 22578, 24203, 25884, 25885, 25886, 27584, 29307, 29308, 30034, 30744, 30745, 32935, 33692, 34467, 34468, 33693, 35225, 31454, 32173, 35226, 35227, 35228, 34473, 34474, 34476, 35235, 36017, 34472, 36763, 36018, 37470, 37471, 35229, 44227, 45328, 45329, 45712, 46086, 46087, 46088, 44228, 43694, 44229, 43148, 43695, 41402, 40800, 39557, 40195, 39556, 38874, 41995, 41996, 41997, 42568, 43149, 38178, 43661, 44209, 43660, 43658, 43104, 47921, 47922, 47923, 47924, 47925, 41363, 40756, 40152, 43114, 43115, 41952, 42528, 43663, 43107, 43665, 43666, 39513, 38828, 30707, 31415, 30004, 30706, 29279, 27558, 25863, 34416, 33650, 32892, 32128, 32890, 32891, 33649, 24183, 22558, 20987, 20988, 20989, 20990, 19469, 20992, 20993, 36699, 37424, 38132, 38133, 38827, 35954, 35178, 36698, 19470, 19471, 19472, 19473, 19474, 19475, 21000, 21001, 21002, 19478, 21005, 22579, 22580, 24204, 25887, 27585, 27586, 27587, 25888, 29309, 30035, 31455, 30746, 30036, 32936, 34469, 33694, 32937, 32174, 32175, 33696, 33698, 33699, 34475, 33701, 34477, 35236, 36019, 34471, 36764, 38179, 38180, 37472, 34470, 45330, 45713, 45714, 45715, 47926, 47927, 46089, 44230, 45331, 43696, 43150, 44231, 44232, 41403, 40801, 41404, 38875, 39558, 40196, 40197, 40802, 40803, 42569, 42570, 42571, 42572, 41998, 43106, 43662, 43105, 48061, 48062, 41362, 41951, 40755, 40151, 43113, 42525, 42526, 42527, 42518, 43108, 43109, 39512, 31414, 32127, 30003, 30705, 29278, 27557, 25862, 34415, 35177, 33647, 33648, 32888, 32889, 24182, 22557, 20986, 19467, 19468, 17983, 17984, 37422, 37423, 38130, 38131, 38825, 38826, 39510, 39508, 39511, 35953, 35176, 36697, 17985, 17986, 17987, 17988, 17989, 19476, 19477, 17993, 17994, 19479, 21006, 24205, 24206, 22581, 25889, 29310, 27588, 27589, 27594, 29311, 29312, 30747, 31456, 30038, 30037, 29316, 29317, 32938, 33695, 32176, 32939, 32940, 33697, 32942, 33700, 32944, 33702, 34478, 35237, 36020, 36765, 38876, 38877, 37473, 38181, 45716, 45332, 45717, 43697, 44233, 43151, 44236, 41999, 41405, 42000, 38878, 39559, 39560, 40198, 41406, 41407, 40804, 41408, 43152, 43153, 43154, 42573, 42517, 41361, 40754, 41949, 40150, 40149, 43111, 42524, 41948, 43112, 41939, 42519, 43110, 31413, 32126, 32887, 30002, 30704, 29277, 30001, 27556, 25861, 34414, 35175, 33646, 34412, 34413, 32886, 24181, 25860, 22556, 20985, 19465, 19466, 17982, 16558, 16559, 16560, 37421, 37420, 38129, 38128, 38822, 38823, 38824, 39509, 39507, 38821, 40147, 40148, 35952, 36696, 16561, 16562, 16563, 16564, 17990, 17991, 17992, 16568, 17995, 19480, 21007, 22582, 24207, 25890, 27591, 27590, 27593, 25894, 27595, 29313, 30748, 31457, 32177, 30039, 29315, 27598, 27599, 29318, 32941, 32179, 32943, 32180, 32181, 32182, 32945, 33703, 34479, 35238, 36021, 36022, 36766, 37474, 38182, 37475, 43698, 44234, 43699, 42574, 42001, 38879, 39561, 40199, 40805, 42002, 42003, 41409, 41410, 43174, 43173, 43171, 43155, 41938, 41360, 40753, 41950, 41358, 40146, 40752, 42522, 42523, 41945, 41946, 41947, 41345, 41940, 41941, 42520, 42521, 31412, 32125, 30703, 30000, 29276, 27555, 29275, 35173, 35174, 33645, 32885, 34411, 32123, 32124, 24180, 22555, 25859, 20984, 22554, 19464, 17978, 17979, 17981, 16557, 15193, 15194, 15195, 15196, 36695, 37418, 38127, 38819, 38820, 39506, 40145, 35951, 36694, 15197, 15198, 16565, 16566, 16567, 21008, 22583, 24208, 24209, 25891, 24211, 25892, 27592, 25893, 24213, 25895, 27596, 29314, 30749, 30750, 31458, 32178, 29319, 30040, 27597, 25896, 25897, 27600, 31459, 31460, 31461, 32183, 32946, 33704, 34480, 35239, 36023, 36024, 36767, 37476, 37477, 37478, 38183, 38880, 44235, 43703, 42575, 43156, 39562, 40200, 39563, 40806, 42004, 42576, 42577, 41411, 42594, 42595, 43175, 43172, 43170, 42593, 43157, 41359, 40750, 40751, 41357, 40747, 40748, 40143, 41943, 41944, 41353, 41355, 41356, 40730, 41346, 41347, 41942, 31411, 30702, 29999, 29274, 27554, 35172, 35950, 33644, 32884, 33643, 34410, 32122, 31408, 24179, 22552, 22553, 25858, 24178, 20983, 19463, 17977, 16554, 16555, 17980, 16556, 15192, 13871, 36693, 37419, 37417, 36691, 38126, 38818, 39504, 39505, 39501, 40144, 35948, 15199, 21009, 22584, 24210, 22585, 22586, 22587, 24212, 22588, 24214, 24215, 24216, 30751, 29320, 30041, 30042, 24217, 24218, 25898, 25899, 27601, 30752, 31462, 32184, 32947, 33705, 33706, 34481, 34482, 35240, 36025, 36768, 37479, 37480, 38184, 38185, 38186, 38881, 43704, 43158, 38882, 39564, 40201, 40807, 41412, 42005, 42578, 43159, 43160, 43161, 42006, 42007, 41413, 42019, 42020, 42021, 42022, 42596, 43176, 43169, 42592, 43700, 42018, 40749, 40746, 40138, 40139, 40140, 40142, 40141, 41351, 41352, 40738, 41354, 40741, 40742, 40744, 40745, 40731, 41348, 41350, 31410, 30701, 31409, 29998, 29273, 27553, 35171, 34409, 35949, 32883, 33642, 33641, 32121, 31407, 22551, 20981, 20982, 24176, 24177, 25857, 19462, 17976, 16553, 15191, 17974, 17975, 36692, 37416, 38124, 38125, 36690, 35946, 35947, 38816, 38122, 39503, 38815, 39500, 38809, 39502, 35169, 21014, 21015, 21016, 21017, 21018, 22589, 22590, 22591, 22592, 29321, 30043, 30753, 22593, 22594, 24219, 25900, 27602, 29322, 31463, 30754, 32185, 32948, 33707, 32949, 33708, 34483, 35241, 36026, 36027, 36769, 37481, 38187, 38883, 38884, 38885, 39565, 40202, 40808, 42579, 43701, 43702, 43162, 43165, 42580, 41414, 42008, 40809, 41428, 41429, 43168, 42591, 42017, 41427, 40137, 39498, 39499, 40737, 40128, 40739, 40740, 40132, 40743, 40135, 40136, 40732, 41349, 30700, 29997, 31405, 31406, 29272, 27552, 35170, 34408, 32882, 33639, 33640, 32120, 32879, 32880, 32881, 22550, 20980, 24175, 25856, 27551, 19461, 16552, 16551, 17973, 15190, 37415, 38123, 36689, 35945, 35167, 35168, 38115, 38817, 38121, 37414, 38813, 38814, 38808, 38111, 38810, 38812, 34406, 19490, 21019, 21020, 21021, 21022, 21023, 30044, 30045, 21024, 22595, 24220, 24221, 25901, 27603, 29323, 30755, 30757, 31464, 32186, 32187, 32950, 33709, 34484, 35242, 36028, 36770, 37482, 38188, 39566, 38886, 39567, 40203, 40204, 40205, 40810, 43166, 42581, 43163, 43164, 42586, 41415, 41418, 42009, 40824, 40825, 43167, 42590, 42016, 41425, 41426, 39497, 38807, 40736, 40126, 40127, 39484, 39485, 40129, 40131, 39490, 39491, 40133, 40134, 39495, 39496, 40124, 40733, 40125, 30699, 29996, 31404, 32118, 32119, 29271, 29270, 34407, 33637, 33638, 32878, 33636, 33635, 22549, 20979, 22548, 24173, 24174, 25855, 27550, 19460, 17970, 17971, 15187, 15188, 16550, 36688, 37412, 37413, 35944, 35166, 34404, 34405, 38114, 37405, 38116, 38117, 38120, 38113, 38110, 37402, 38112, 38805, 38806, 38811, 19491, 19492, 19493, 30756, 30046, 21025, 22596, 24222, 25902, 27604, 29324, 30758, 31465, 31466, 32188, 32951, 33710, 33711, 33712, 34485, 35243, 36029, 36771, 37483, 38189, 38190, 38191, 38887, 38888, 39568, 40811, 40812, 40207, 40206, 42582, 42583, 42584, 42585, 42014, 42587, 41416, 41417, 40815, 41419, 42010, 40823, 40220, 40221, 42588, 42589, 42015, 41424, 40820, 39483, 38788, 38789, 39486, 40130, 39489, 38797, 38798, 39492, 39494, 38804, 40123, 39482, 40110, 30698, 29995, 29267, 31403, 32117, 32116, 29268, 29269, 34403, 32877, 32115, 33634, 34402, 20978, 19459, 22547, 24172, 25854, 24171, 27549, 17969, 16545, 16546, 17972, 16549, 15186, 36687, 37411, 38118, 38119, 35943, 35165, 35942, 35164, 37404, 36680, 36681, 37406, 37408, 37403, 38109, 37401, 36677, 21026, 19494, 30047, 30759, 30760, 30761, 22597, 22598, 22599, 24223, 25903, 27605, 29325, 29326, 31467, 32189, 32952, 32954, 33713, 32955, 34486, 35244, 36030, 36772, 37484, 37485, 37487, 38192, 38889, 39569, 40813, 40814, 39570, 39571, 40208, 42011, 42012, 42013, 41423, 40213, 40816, 41420, 40821, 40219, 39588, 39589, 39590, 40819, 40216, 38787, 38091, 38092, 38790, 39487, 38791, 38795, 38796, 38101, 38102, 38799, 39493, 38803, 40122, 39481, 38786, 40111, 30697, 29994, 29266, 27548, 31402, 31401, 32876, 32114, 31400, 33633, 34401, 20977, 19458, 17968, 22546, 24170, 25853, 24169, 16544, 15182, 15183, 16547, 16548, 15185, 36686, 35941, 37410, 35163, 35162, 34400, 36678, 36679, 35933, 36682, 37407, 36684, 37409, 38108, 37400, 38802, 36676, 35931, 36675, 21027, 19495, 30048, 31468, 30762, 24224, 24225, 24226, 22600, 21028, 25904, 27606, 29327, 30049, 32190, 32953, 32192, 33714, 32956, 33715, 34487, 35245, 36031, 36773, 36774, 37486, 36775, 36776, 36777, 37488, 38193, 38890, 40209, 40212, 38891, 39572, 41421, 41422, 40818, 39577, 39578, 39579, 40214, 40817, 40217, 40822, 40218, 39586, 39587, 38913, 38914, 38915, 39591, 40215, 39582, 38089, 38090, 37382, 37383, 38093, 38094, 39488, 38794, 38792, 38100, 37393, 38103, 38800, 38104, 38801, 40121, 40112, 40734, 39480, 38785, 38784, 39479, 30696, 30695, 29993, 29265, 27547, 32875, 32874, 33632, 32113, 31399, 30694, 34399, 20976, 19457, 22545, 17967, 24168, 25852, 25851, 16543, 15180, 15181, 13858, 13859, 13860, 15184, 36685, 35939, 35940, 35161, 35932, 35148, 35149, 35934, 36683, 35935, 38107, 37399, 38106, 35928, 35929, 35930, 35147, 36674, 21029, 21030, 30763, 31469, 31470, 31471, 25905, 25906, 25907, 24227, 25908, 22601, 27607, 29328, 29329, 29330, 30050, 32191, 32193, 32957, 33716, 34488, 34489, 35246, 36032, 36033, 36034, 36035, 36778, 37489, 38194, 38195, 38196, 38892, 38893, 40210, 39573, 40211, 39576, 38901, 38902, 39580, 39581, 39583, 39584, 39585, 38911, 38912, 38220, 38221, 38222, 38905, 38906, 38088, 37380, 37381, 36653, 36654, 37384, 37386, 38095, 38793, 38099, 37392, 36666, 37394, 37395, 38105, 37396, 40120, 40735, 38783, 39478, 40118, 40119, 29992, 29264, 27544, 27545, 27546, 32112, 32873, 33630, 33631, 34398, 31398, 30693, 34395, 20975, 19456, 22544, 24167, 17966, 25850, 16542, 17965, 15179, 13856, 13857, 35938, 35160, 34394, 34378, 34379, 35150, 35156, 35937, 37397, 37398, 36673, 35927, 35144, 35145, 35146, 34375, 35926, 22602, 30764, 31472, 32194, 32195, 27608, 27609, 24228, 25909, 25910, 25911, 27610, 27611, 29331, 29332, 30051, 30052, 30053, 32958, 33717, 34490, 34492, 35247, 35248, 36036, 36037, 36779, 37490, 37491, 37492, 38894, 38895, 38199, 38197, 38896, 38897, 39574, 39575, 38900, 38206, 38207, 38208, 38903, 38904, 38907, 38908, 38909, 38910, 38218, 38219, 37518, 37519, 38210, 38087, 37379, 36652, 35902, 35903, 36655, 37385, 36656, 36657, 37387, 38096, 38098, 37391, 36665, 35920, 36667, 36668, 36669, 36671, 38782, 38086, 39477, 38781, 40117, 39474, 29991, 29990, 29263, 27543, 25848, 25849, 32111, 31397, 32872, 32110, 33629, 34397, 34396, 30692, 33628, 20974, 19455, 22543, 24166, 19454, 16541, 17964, 19453, 15178, 16540, 35159, 34393, 33627, 34377, 33612, 34376, 34380, 35151, 35155, 34391, 35157, 35936, 36672, 35143, 34374, 33610, 33611, 35925, 35141, 35142, 24229, 22603, 30765, 31473, 32196, 32959, 32960, 29333, 29334, 25912, 24230, 25913, 27612, 30054, 29336, 30766, 30767, 30768, 33718, 34491, 34493, 35249, 36038, 36780, 37493, 36787, 38198, 37494, 38200, 38201, 37496, 38202, 38203, 38204, 38898, 38899, 38205, 37503, 37504, 37505, 37506, 37507, 38209, 38211, 38212, 38213, 38214, 38215, 38217, 37516, 37517, 36818, 36819, 37520, 37508, 37378, 36651, 35901, 35121, 35904, 35905, 35907, 36658, 37388, 37389, 38097, 37390, 36664, 36663, 35919, 35137, 35921, 35923, 36670, 38085, 37374, 37375, 39476, 39475, 38780, 38084, 29262, 29989, 30691, 27542, 25847, 29261, 24165, 31394, 31395, 31396, 32871, 32109, 32870, 20973, 22541, 22542, 20971, 20972, 17963, 19451, 19452, 20970, 16538, 16539, 35158, 34392, 33626, 32854, 33613, 34381, 34382, 35152, 35153, 34389, 34390, 33625, 34373, 33609, 35924, 35140, 34371, 34372, 22604, 31474, 32197, 32198, 32199, 33719, 32961, 27616, 29335, 24231, 25914, 27613, 29337, 29338, 30055, 30769, 27617, 31475, 30770, 34494, 35250, 36039, 36040, 36781, 36786, 36788, 36789, 36054, 36055, 36056, 37495, 37497, 37499, 36791, 36792, 37500, 37501, 37502, 36803, 36804, 36805, 36806, 37509, 37510, 37511, 38216, 37514, 37515, 36816, 36817, 36083, 36085, 36820, 37521, 37522, 37377, 37376, 36650, 35900, 35120, 36649, 34353, 35122, 35906, 35125, 35908, 36659, 36661, 36662, 35917, 35918, 35135, 35136, 34368, 35138, 35922, 35139, 37373, 36648, 38779, 38078, 38083, 29988, 30690, 27541, 29257, 29258, 29259, 29260, 25845, 24164, 25846, 31393, 30688, 32108, 32869, 22540, 24163, 22538, 22539, 17962, 19450, 20966, 20967, 20968, 20969, 22536, 22537, 17961, 32853, 32085, 32855, 32852, 33608, 33614, 34383, 35154, 34388, 33624, 32868, 33607, 34370, 33602, 33603, 33605, 32962, 32963, 31476, 32200, 32964, 33720, 34495, 27615, 25918, 25919, 25915, 24232, 27614, 29339, 29340, 30056, 27618, 29341, 30771, 35251, 36041, 36782, 36783, 36050, 36784, 36785, 36053, 36057, 36790, 35280, 35281, 35285, 36793, 37498, 36795, 36796, 36059, 36797, 36798, 36802, 36068, 36069, 36070, 36807, 36808, 37512, 37513, 36814, 36815, 36082, 35314, 36084, 35315, 36086, 36821, 35899, 35119, 35123, 34354, 35124, 34357, 35126, 35909, 35910, 36660, 35912, 35911, 35915, 35916, 35132, 35133, 35134, 34366, 34367, 33600, 34369, 37372, 29987, 30689, 27540, 29256, 29984, 29983, 29985, 29986, 25844, 24162, 27539, 31392, 30687, 32107, 32867, 24160, 24161, 19449, 20964, 22531, 22532, 22533, 22534, 22535, 24158, 24159, 19448, 32083, 32084, 31364, 32086, 32856, 33615, 32851, 33616, 34384, 34385, 33623, 33622, 33606, 33601, 32846, 33604, 32848, 33721, 33722, 30772, 31477, 32201, 32965, 34496, 25917, 24233, 25916, 30057, 30058, 30059, 30060, 30061, 35252, 36042, 36049, 35275, 35276, 36051, 36052, 35286, 36058, 35279, 34530, 34531, 35282, 35284, 34536, 36060, 36794, 36061, 36062, 36063, 35288, 36064, 36799, 36801, 36067, 35300, 35302, 36071, 36072, 36809, 36811, 36813, 36081, 35313, 34558, 34559, 35316, 35317, 36087, 36822, 36823, 34356, 34355, 33587, 34358, 34359, 35127, 35914, 35913, 35130, 35131, 34364, 34365, 33599, 32844, 29255, 27538, 29982, 30686, 30685, 25843, 31391, 31390, 32106, 32866, 25842, 25841, 20965, 20963, 19441, 22530, 24151, 24152, 24153, 24154, 24155, 24156, 24157, 25838, 25839, 25840, 32082, 31363, 32850, 30651, 30652, 31365, 32087, 32857, 33617, 34386, 33621, 34387, 32849, 32845, 32077, 32847, 32079, 34497, 33723, 30773, 31478, 32202, 32966, 33724, 35253, 30774, 30775, 30776, 36043, 36048, 35274, 34524, 34525, 34526, 35277, 35278, 35287, 34527, 34529, 33759, 33760, 33761, 34532, 35283, 34534, 34535, 33767, 33768, 33769, 34537, 35289, 35291, 35292, 35293, 35294, 36065, 36800, 36066, 35299, 34546, 35301, 34548, 35303, 36073, 36075, 36810, 36076, 36812, 36079, 36080, 35312, 34557, 33791, 35311, 34560, 34562, 34563, 35318, 36088, 33586, 32833, 33588, 33589, 33590, 34360, 35128, 35129, 34362, 34363, 33597, 33598, 32843, 32074, 29254, 27537, 29981, 30684, 31389, 32105, 32865, 33620, 27536, 19442, 20962, 22529, 19440, 20961, 24150, 25832, 25830, 25829, 25828, 25827, 25834, 25835, 25836, 25837, 27533, 27534, 27535, 32081, 31361, 31362, 30650, 29946, 30653, 31366, 30655, 32088, 32858, 33618, 33619, 32080, 32076, 31355, 32078, 31357, 34498, 35254, 34499, 31479, 31480, 32203, 32967, 32968, 32969, 33725, 36044, 31481, 31482, 31483, 36047, 36045, 35273, 34523, 33753, 33754, 33755, 33756, 34528, 33757, 33758, 33010, 33011, 33012, 33013, 33762, 34533, 33764, 33765, 33766, 33019, 33020, 33021, 33770, 34538, 35290, 34539, 34540, 35295, 35296, 35298, 34545, 33777, 33778, 34547, 33780, 34549, 35304, 36074, 35306, 36077, 36078, 35309, 35310, 34556, 33790, 33043, 33792, 34555, 34561, 32063, 32834, 32835, 33591, 34361, 33595, 33596, 32841, 32842, 32073, 31352, 32075, 29253, 29979, 29978, 29980, 30683, 31388, 32102, 32103, 32104, 32864, 32862, 22528, 24149, 19439, 20960, 25833, 25831, 24147, 24142, 24143, 24141, 25826, 27528, 27527, 27529, 27530, 27531, 27532, 29250, 29251, 29252, 31360, 30649, 29945, 29221, 29947, 30654, 31367, 32089, 29949, 30656, 32859, 32860, 32861, 31358, 31354, 30641, 31356, 30643, 30645, 35255, 34500, 32204, 32970, 33726, 33727, 33728, 33729, 34501, 32205, 32206, 36046, 35272, 34522, 35271, 33003, 33005, 33007, 33008, 33763, 33014, 33017, 33018, 32257, 32258, 32259, 33022, 33771, 34541, 34542, 35297, 34544, 33776, 33030, 33031, 33779, 33032, 33033, 33781, 34550, 35305, 34551, 34552, 35307, 35308, 33788, 33789, 33042, 34554, 32064, 32836, 33592, 33594, 32840, 32071, 32072, 31351, 30639, 31353, 29977, 30682, 30681, 31387, 32101, 32094, 32100, 32092, 22527, 24144, 24145, 22520, 24140, 25825, 24138, 27526, 25824, 27525, 29247, 29248, 29249, 29974, 29975, 29976, 31359, 30648, 29944, 29220, 27501, 29222, 29943, 29948, 30657, 31368, 32090, 29950, 30658, 32091, 30640, 29934, 30642, 29935, 30644, 29937, 29940, 30646, 35256, 35257, 32971, 33730, 34502, 34503, 34504, 34505, 34506, 32972, 32973, 35270, 34520, 34521, 34518, 33004, 33015, 33016, 32256, 31527, 31528, 31529, 32260, 33023, 33772, 33773, 34543, 33774, 33029, 32266, 32268, 32269, 32270, 33034, 33782, 33784, 34553, 33787, 33040, 33786, 33041, 33785, 32837, 32065, 33593, 32839, 32070, 31350, 30638, 29933, 30680, 31386, 31385, 31384, 32093, 31374, 31375, 32095, 32863, 32096, 32099, 32098, 31371, 31372, 22521, 22519, 20947, 24139, 24137, 22518, 25823, 24136, 27524, 25822, 29246, 29971, 29972, 29973, 29219, 29218, 29941, 29942, 27500, 25792, 27502, 29223, 30647, 31369, 31370, 29951, 30659, 29211, 29212, 29936, 29213, 29938, 29939, 29217, 35269, 35258, 33731, 34507, 35259, 35260, 35261, 35262, 32974, 34517, 32254, 32255, 31526, 30815, 30816, 31530, 32261, 33024, 33025, 33775, 33028, 32265, 32267, 31538, 31540, 31541, 32271, 33035, 33783, 33036, 33039, 33038, 33037, 32066, 32838, 31345, 32068, 32069, 31349, 30637, 29932, 29931, 30679, 31383, 30678, 31373, 30665, 30666, 31376, 32097, 31380, 31381, 30660, 30661, 20946, 22517, 24135, 22515, 22516, 27523, 29244, 29245, 25821, 24134, 29970, 30676, 30677, 27496, 27497, 27498, 27499, 25791, 24106, 25793, 27503, 29224, 29952, 29210, 27490, 27491, 27492, 29214, 29215, 29216, 34516, 33732, 34508, 35268, 49329, 33733, 31531, 32262, 33026, 33027, 32264, 31537, 31539, 32272, 32273, 31346, 32067, 31344, 30634, 31347, 31348, 30636, 29209, 29930, 29208, 30662, 30664, 29958, 30667, 31377, 31379, 30673, 31382, 29953, 20945, 22514, 22513, 24132, 24133, 27522, 29243, 29242, 29968, 29969, 25820, 30675, 30674, 27495, 25787, 25789, 25790, 24105, 22476, 24107, 25794, 27504, 25795, 29225, 29226, 27488, 27489, 25781, 25782, 27493, 27494, 32263, 30635, 30633, 29929, 27487, 29207, 30663, 29957, 29230, 29959, 30668, 31378, 30672, 29966, 29954, 29955, 22512, 24131, 22511, 27521, 25819, 27520, 29241, 27519, 29967, 25788, 24104, 22475, 22477, 22478, 24108, 25796, 27505, 29227, 25783, 29956, 29229, 27508, 29231, 29233, 29960, 30669, 30671, 30670, 29965, 29240, 29228, 24130, 22509, 22510, 25818, 27518, 29239, 24103, 22479, 24109, 25797, 25798, 27506, 27507, 25802, 25803, 27509, 29232, 27510, 27512, 29234, 29961, 29964, 29238, 22508, 24129, 25817, 27517, 25799, 24110, 25800, 25801, 24113, 25804, 25805, 25806, 27511, 25812, 27513, 29235, 29962, 29963, 29237, 25816, 27516, 29236, 24111, 24112, 27514, 25813, 27515]
    # # b = [37449, 37450, 29309, 37451, 29310, 37452, 29311, 37453, 29312, 37454, 29313, 37455, 29314, 37456, 29315, 37457, 29316, 37458, 37459, 29960, 37460, 37461, 37462, 32834, 32835, 32836, 32837, 37463, 32839, 32840, 32841, 32842, 32843, 32844, 32845, 32846, 32847, 32848, 32849, 32850, 32851, 32852, 32853, 32854, 32855, 32856, 32857, 32858, 32859, 32860, 32861, 32862, 32863, 32864, 32865, 32866, 32867, 32868, 32869, 32870, 32871, 32872, 32873, 32874, 32875, 32876, 32877, 32878, 32879, 32880, 32881, 32882, 32883, 32884, 32885, 32886, 32887, 32888, 32889, 32890, 32891, 32892, 32893, 32894, 32895, 32896, 32897, 32898, 32899, 32900, 32901, 32902, 32903, 32904, 32905, 32906, 32907, 32908, 32909, 32910, 32911, 32912, 32913, 32914, 32915, 32916, 32917, 32918, 32919, 32920, 32921, 32922, 32923, 32924, 32925, 32926, 32927, 32928, 32929, 32930, 32931, 32932, 32933, 32934, 32935, 32936, 32937, 32938, 32939, 32940, 32941, 32942, 32943, 32944, 32945, 32946, 32947, 32948, 32949, 32950, 32951, 37484, 37485, 32952, 32954, 32955, 37486, 32953, 37487, 32956, 32957, 32958, 32959, 37488, 32960, 32961, 32962, 32963, 37489, 32964, 32965, 32966, 32967, 37490, 32968, 32969, 32970, 32971, 37491, 17954, 37494, 37495, 37496, 37497, 17959, 37498, 33017, 17960, 33019, 33020, 33021, 33022, 17961, 37500, 33024, 33023, 33018, 33025, 37501, 33029, 33030, 33032, 33033, 37502, 33034, 33031, 33035, 33036, 33037, 33038, 33043, 17966, 17967, 17968, 17969, 37508, 17970, 37509, 17971, 17972, 17973, 17974, 17975, 17976, 17977, 17978, 17979, 37518, 17980, 37519, 17981, 37520, 17982, 37521, 17983, 37522, 17984, 37523, 17985, 29961, 17986, 41344, 41345, 17987, 41346, 41347, 41349, 41350, 41351, 41352, 41353, 41354, 41355, 41356, 41357, 41358, 41359, 41360, 41361, 41362, 41363, 41364, 41365, 41366, 41367, 41368, 41369, 41370, 41371, 41372, 41373, 41374, 41375, 41376, 41377, 41378, 41379, 41380, 41381, 41382, 41383, 41384, 41385, 41386, 41387, 41388, 41389, 41390, 41391, 41392, 41393, 41394, 41395, 41396, 41397, 41398, 41399, 41400, 41401, 41402, 37465, 41403, 41404, 41405, 41406, 41407, 41408, 41409, 41410, 41411, 41412, 41413, 41414, 41415, 41416, 41418, 41417, 41419, 41420, 37466, 41423, 41421, 41422, 41424, 41425, 41426, 41427, 41428, 41429, 41430, 37467, 29962, 37468, 37469, 45701, 37470, 45702, 45703, 45704, 45705, 45706, 37471, 45707, 45708, 45709, 45710, 45711, 45712, 37472, 45713, 45714, 45715, 29963, 37473, 29333, 37474, 37475, 37476, 37477, 29964, 37478, 37479, 34352, 34353, 34354, 33584, 37480, 34355, 33587, 33588, 33589, 33590, 33591, 33592, 33593, 33594, 33595, 33596, 33597, 33598, 33599, 33600, 33601, 33602, 33603, 33604, 33605, 33606, 33607, 33608, 33609, 33610, 33611, 33612, 33613, 33614, 33615, 33616, 33617, 33618, 33619, 33620, 33621, 33622, 33623, 33624, 33625, 33626, 33627, 33628, 33629, 33630, 33631, 33632, 33633, 33634, 33635, 33636, 33637, 33638, 33639, 33640, 33641, 33642, 33643, 33644, 33645, 33646, 33647, 33648, 33649, 33650, 33651, 33652, 33653, 33654, 33655, 33656, 33657, 33658, 33659, 33660, 33661, 33662, 33663, 33664, 33665, 33666, 33667, 33668, 33669, 33670, 33671, 33672, 33673, 33674, 33675, 33676, 33677, 33678, 33679, 33680, 33681, 33682, 33683, 33684, 33685, 33686, 33687, 33688, 33689, 33690, 33691, 33692, 33693, 33694, 33695, 33696, 33697, 33698, 33699, 33700, 33701, 33702, 33703, 33704, 33705, 33706, 33707, 33708, 33709, 33710, 33711, 33712, 33713, 33714, 33715, 33716, 33717, 33718, 33719, 33720, 33721, 33722, 33723, 33724, 33725, 33726, 33727, 33728, 33729, 33730, 30648, 41937, 41938, 41939, 41940, 41941, 41942, 41943, 41944, 41945, 41946, 41947, 41948, 41949, 41950, 41951, 41952, 41953, 41954, 41955, 41956, 41957, 41958, 41959, 41960, 41961, 41962, 41963, 41964, 41965, 41966, 41967, 41968, 41969, 41970, 33770, 33771, 41973, 41971, 41972, 41976, 41977, 41978, 41979, 41980, 41981, 41982, 41983, 41984, 41985, 41986, 41987, 41988, 41989, 41990, 41991, 41992, 41993, 41994, 41995, 41996, 41997, 41998, 41999, 42000, 42001, 42002, 42003, 42004, 42005, 42006, 42007, 42008, 42009, 42010, 42011, 42012, 42013, 42014, 42015, 42016, 42017, 42018, 42019, 42020, 42021, 42022, 42546, 42547, 42548, 42549, 42555, 42556, 22513, 42557, 42558, 42559, 36046, 36047, 42560, 42561, 22514, 42562, 42563, 42564, 36052, 42565, 42566, 22515, 42567, 42568, 42569, 42570, 42571, 22516, 42572, 42573, 42574, 25787, 25789, 42575, 25791, 25792, 25793, 25794, 42576, 25795, 25796, 25797, 25798, 42577, 22517, 25799, 25800, 44206, 42578, 25801, 25802, 25803, 44207, 42579, 25806, 25812, 44208, 42580, 25816, 25817, 25818, 44209, 42581, 25821, 25822, 25823, 25824, 42582, 25825, 25827, 25826, 25828, 42583, 25829, 44211, 25830, 25834, 25835, 25836, 25837, 25838, 25839, 25840, 25841, 25842, 25843, 25844, 25845, 25846, 25847, 25848, 25849, 25850, 25851, 25852, 25853, 25854, 25855, 25856, 25857, 25858, 25859, 25860, 25861, 25862, 25863, 25864, 25865, 25866, 25867, 25868, 25869, 25870, 25871, 25872, 25873, 25874, 25875, 25876, 25877, 25878, 25879, 25880, 25881, 25882, 25883, 25884, 25885, 25886, 25887, 44222, 44223, 25888, 25889, 42594, 16543, 16544, 44224, 25890, 25897, 25898, 25899, 25900, 25901, 25902, 30662, 25903, 25904, 16545, 44225, 44226, 25905, 44227, 16546, 25906, 25907, 16547, 44228, 16548, 25908, 16549, 44229, 16550, 44230, 30663, 16551, 44231, 32833, 16552, 44232, 16553, 44233, 16554, 44234, 16555, 44235, 30664, 32838, 16556, 44236, 16557, 16558, 16559, 16560, 30665, 16561, 16562, 16563, 16564, 16565, 30666, 16566, 16567, 16568, 16569, 16570, 30667, 30668, 22528, 30669, 22529, 30670, 22530, 30671, 42516, 42517, 42518, 42519, 42520, 42521, 42522, 42523, 42524, 42525, 42526, 42527, 42528, 42529, 42530, 42531, 42532, 42533, 42534, 42535, 22531, 30672, 42538, 42539, 42540, 42541, 42542, 42543, 17964, 42544, 42545, 17962, 17963, 17965, 42550, 42551, 42552, 42553, 42554, 34363, 34364, 34365, 34366, 34367, 34368, 34369, 34370, 34371, 34372, 34373, 34374, 34375, 34376, 34377, 34378, 34379, 34380, 34381, 34382, 34383, 34384, 34385, 34386, 34387, 34388, 34389, 34390, 34391, 34392, 34393, 34394, 34395, 34396, 34397, 34398, 34399, 34400, 34401, 34402, 34403, 34404, 34405, 34406, 34407, 34408, 34409, 34410, 34411, 34412, 34413, 34414, 34415, 34416, 34417, 34418, 34419, 34420, 34421, 34422, 34423, 34424, 34425, 34426, 34427, 34428, 34429, 34430, 34431, 34432, 34433, 34434, 34435, 34436, 34437, 34438, 34439, 34440, 34441, 34442, 34443, 34444, 34445, 34446, 34447, 34448, 34449, 34450, 34451, 34452, 34453, 34454, 34455, 34456, 34457, 34458, 34459, 34460, 34461, 34462, 34463, 34464, 34465, 34466, 34467, 34468, 34469, 34470, 34471, 34472, 34473, 34474, 34475, 34476, 34477, 34478, 34479, 34480, 34481, 34482, 34483, 34484, 34485, 34486, 34487, 34488, 34489, 34490, 34491, 34492, 34493, 34494, 34495, 34496, 34497, 34498, 34499, 34500, 34501, 34502, 34503, 34504, 34505, 34506, 34523, 34524, 34525, 34526, 34527, 34529, 34530, 34531, 34532, 34533, 34534, 34535, 34536, 34537, 34538, 34539, 34540, 34541, 34542, 34543, 34544, 34545, 34546, 34547, 34548, 34549, 34550, 34551, 34552, 34553, 34554, 34555, 34556, 34557, 34558, 34559, 34560, 34561, 34562, 17988, 17989, 17990, 39481, 31342, 31343, 17991, 31344, 31345, 37464, 31346, 31347, 31348, 17992, 31349, 31350, 31351, 31352, 31353, 17993, 31354, 31355, 31356, 31357, 31358, 17994, 31359, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147, 43148, 43149, 43150, 43151, 43152, 43153, 43154, 43155, 43156, 43157, 43158, 43159, 43160, 43161, 43162, 43163, 43164, 43165, 43166, 43167, 43168, 43169, 43170, 43171, 43172, 43173, 43174, 43175, 33026, 33028, 35119, 35120, 35121, 35122, 35123, 35124, 35125, 35126, 35127, 35128, 35129, 35130, 35131, 35132, 35133, 35134, 35135, 35136, 35137, 35138, 35139, 35140, 35141, 35142, 35143, 35144, 35145, 35146, 35147, 35148, 35149, 35150, 35151, 35152, 35153, 35154, 35155, 35156, 35157, 35158, 35159, 35160, 35161, 35162, 35163, 35164, 35165, 35166, 35167, 35168, 35169, 35170, 35171, 35172, 35173, 35174, 35175, 35176, 35177, 35178, 35179, 35180, 35181, 35182, 35183, 35184, 35185, 35186, 35187, 35188, 35189, 35190, 35191, 35192, 35193, 35194, 35195, 35196, 35197, 35198, 35199, 35200, 35201, 35202, 35203, 35204, 35205, 35206, 35207, 35208, 35209, 35210, 35211, 35212, 35213, 35214, 35215, 35216, 35217, 35218, 35219, 35220, 35221, 35222, 35223, 35224, 35225, 35226, 35227, 35228, 35229, 35230, 35231, 35232, 35233, 35234, 35235, 35236, 35237, 35238, 35239, 35240, 35241, 39568, 35242, 39569, 35243, 35244, 39570, 35245, 35246, 35247, 35248, 39571, 35249, 35250, 35251, 35252, 39572, 35253, 35254, 35255, 35256, 39573, 35257, 35258, 35259, 35260, 39574, 35269, 35270, 35271, 39575, 35273, 35274, 35275, 35276, 39576, 35277, 35278, 35280, 35281, 39577, 35282, 35284, 35285, 35286, 39578, 35288, 35289, 35290, 35291, 39579, 35293, 35292, 35294, 35287, 39580, 35295, 35296, 35298, 35299, 39581, 35302, 35303, 35304, 35305, 39582, 35300, 35301, 35306, 35307, 39583, 35312, 35313, 35314, 35308, 35309, 39584, 35310, 35311, 35315, 17995, 39585, 35316, 35317, 39586, 39587, 39588, 39589, 39590, 17996, 43659, 43660, 43661, 43662, 43663, 43664, 43665, 43666, 43667, 43668, 43669, 43670, 43671, 43672, 43673, 43674, 43675, 43676, 43677, 43678, 43679, 43680, 43681, 43682, 43683, 43684, 43685, 43686, 43687, 43688, 43689, 43690, 43691, 43692, 43693, 43694, 43695, 43696, 43697, 43698, 43699, 43700, 43701, 43702, 43703, 35279, 27490, 27491, 27492, 27493, 27494, 27495, 27496, 27497, 27498, 27499, 27500, 27501, 27502, 27503, 27504, 27505, 27506, 27507, 27508, 27509, 27510, 27511, 27512, 27513, 27516, 27517, 27518, 27519, 27520, 27521, 27522, 27523, 27524, 27525, 27526, 27527, 27528, 27529, 27530, 27531, 27532, 27533, 27534, 27535, 27536, 27537, 27538, 27539, 27540, 27541, 27542, 27543, 27544, 27545, 27546, 27547, 27548, 27549, 27550, 27551, 27552, 27553, 27554, 27555, 27556, 27557, 27558, 27559, 27560, 27561, 27562, 27563, 27564, 27565, 27566, 27567, 27568, 27569, 27570, 27571, 27572, 27573, 27574, 27575, 27576, 27577, 27578, 27579, 27580, 27581, 27582, 27583, 27584, 27585, 27586, 27587, 27588, 27589, 27590, 27591, 27592, 27593, 27594, 27595, 27596, 27597, 27598, 27599, 27600, 27601, 27602, 27603, 27604, 27605, 27606, 27607, 27608, 27609, 19440, 19441, 19442, 19443, 19444, 19445, 19447, 19448, 19449, 19450, 19451, 19452, 19453, 19454, 19455, 19456, 19457, 19458, 19459, 19460, 19461, 39556, 19462, 19463, 19464, 19465, 19467, 19466, 19468, 19469, 19470, 19471, 19472, 19473, 19474, 19475, 19476, 19478, 19477, 19479, 19480, 19481, 38222, 38223, 29929, 29930, 35899, 35900, 35901, 35902, 35903, 35904, 35905, 35906, 35907, 35908, 35909, 35910, 35911, 35912, 35913, 35914, 35915, 35916, 35917, 35918, 35919, 35920, 35921, 35922, 35923, 35924, 35925, 35926, 35927, 35928, 35929, 35930, 35931, 35932, 35933, 35934, 35935, 35936, 35937, 35938, 35939, 35940, 35941, 35942, 35943, 35944, 35945, 35946, 35947, 35948, 35949, 35950, 35951, 35952, 35953, 35954, 35955, 35956, 35957, 35958, 35959, 35960, 35961, 35962, 35963, 35964, 35965, 35966, 35967, 35968, 35969, 35970, 35971, 35972, 35973, 35974, 35975, 35976, 35977, 35978, 35979, 35980, 35981, 35982, 35983, 35984, 35985, 35986, 35987, 35988, 35989, 35990, 35991, 35992, 35993, 35994, 35995, 35996, 35997, 35998, 35999, 36000, 36001, 36002, 36003, 36004, 36005, 36006, 36007, 36008, 36009, 36010, 36011, 36012, 36013, 36014, 36015, 36016, 36017, 36018, 36019, 36020, 36021, 36022, 36023, 36024, 36025, 36026, 36027, 36028, 44221, 36029, 36030, 36031, 36032, 36033, 36034, 36035, 36036, 36037, 36038, 36039, 36040, 36041, 36042, 36043, 36044, 29958, 29959, 36045, 36049, 36050, 36048, 36051, 36053, 36054, 36055, 36056, 36057, 36058, 36059, 36060, 36061, 36062, 36063, 36064, 36065, 36066, 36067, 36068, 36069, 36070, 36071, 36072, 36073, 36074, 36075, 36076, 36077, 36078, 29966, 36079, 36081, 36082, 36080, 29967, 36083, 36084, 36085, 36086, 29968, 36087, 39558, 39559, 38136, 38155, 38156, 38157, 38158, 38159, 38160, 38161, 38162, 38163, 38164, 38165, 38166, 38167, 38168, 38169, 38170, 38171, 38172, 38173, 38174, 38175, 38176, 38177, 38178, 38179, 38180, 38181, 38182, 38183, 38184, 38185, 38186, 38187, 38188, 38189, 38190, 38191, 38192, 38193, 38194, 38195, 38196, 38197, 38198, 38199, 38200, 38201, 38202, 39552, 38203, 38204, 38205, 38206, 38207, 39553, 38208, 38209, 38210, 38211, 38212, 39554, 38213, 38214, 38215, 38216, 38217, 39555, 38218, 38219, 38220, 36648, 36649, 36650, 36651, 36652, 36653, 36654, 36655, 36656, 36657, 36658, 36659, 36660, 36661, 36662, 36663, 36664, 36665, 36666, 36667, 36668, 36669, 36670, 36671, 36672, 36673, 36674, 36675, 36676, 36677, 36678, 36679, 36680, 36681, 36682, 36683, 36684, 36685, 36686, 36687, 36688, 36689, 36690, 36691, 36692, 36693, 36694, 36695, 36696, 36697, 36698, 36699, 36700, 36701, 36702, 36703, 36704, 36705, 36706, 36707, 36708, 36709, 36710, 36711, 36712, 36713, 36714, 36715, 36716, 36717, 36718, 36719, 36720, 36721, 36722, 36723, 36724, 36725, 36726, 36727, 36728, 36729, 36730, 36731, 36732, 36733, 36734, 36735, 36736, 36737, 36738, 36739, 36740, 36741, 36742, 36743, 36744, 36745, 36746, 36747, 36748, 36749, 36750, 36751, 36752, 36753, 36754, 36755, 36756, 36757, 36758, 36759, 36760, 36761, 36762, 36763, 36764, 39560, 36765, 36766, 36767, 36768, 36769, 36770, 36771, 36772, 36773, 36774, 36775, 36776, 36777, 36778, 36779, 36780, 36781, 36782, 36783, 39561, 36786, 36787, 36788, 36789, 36790, 36791, 36792, 36793, 36794, 36795, 36796, 36797, 36798, 36799, 36800, 36801, 36802, 36803, 36804, 36805, 36806, 36807, 36808, 36809, 39562, 36811, 36810, 36812, 36814, 36815, 36816, 36813, 36817, 36818, 36819, 36820, 36821, 36822, 36823, 36824, 39563, 21002, 38912, 39564, 21003, 38913, 39565, 21004, 38914, 39566, 21005, 34356, 38915, 39567, 21006, 34357, 30775, 38916, 25891, 21007, 34358, 38917, 25892, 37481, 21008, 34359, 38918, 25893, 30643, 21009, 34360, 25894, 34361, 25895, 34362, 25896, 12528, 37482, 12529, 45305, 45312, 45313, 45314, 45315, 45316, 45317, 45318, 45319, 45320, 45321, 45322, 45323, 45324, 45325, 45326, 30644, 45327, 45328, 45329, 45330, 45331, 45332, 45333, 29965, 37483, 30645, 20945, 20946, 20947, 20961, 20962, 20963, 20964, 20965, 20966, 20967, 20968, 20969, 20970, 20971, 20972, 20973, 20974, 20975, 20976, 20977, 20978, 20979, 20980, 20981, 20982, 20983, 20984, 20985, 20986, 20987, 20988, 20989, 20990, 20991, 20992, 20993, 20994, 30646, 37380, 37381, 37382, 37383, 37384, 37385, 37386, 37387, 37388, 37389, 37390, 37391, 37392, 37393, 37394, 37395, 37396, 37397, 37398, 37399, 37400, 37401, 37402, 37403, 37404, 37405, 37406, 37407, 37408, 37409, 37410, 37411, 37412, 37413, 37414, 37415, 37416, 37417, 37418, 37419, 37420, 37421, 37422, 37423, 37424, 37425, 37426, 37427, 37428, 37429, 37430, 37431, 37432, 37433, 37434, 37435, 37436, 29245, 29246, 29247, 37440, 37441, 29248, 37443, 29249, 29250, 29251, 29252, 29253, 29254, 29255, 29256, 29257, 29258, 29259, 29260, 29264, 29265, 29266, 29267, 29268, 29269, 29270, 29271, 29272, 29273, 29274, 29275, 29276, 29277, 29278, 29279, 29280, 29281, 29282, 29283, 29284, 29285, 29286, 29287, 29288, 29289, 29290, 29291, 29292, 29293, 29294, 29295, 29296, 29297, 29298, 29299, 29300, 29301, 29302, 29303, 29304, 29305, 29306, 29307, 29308, 37492, 37493, 37503, 37504, 37505, 37506, 37507, 37499, 29317, 29318, 29319, 29320, 29321, 29322, 29323, 29324, 29325, 29326, 37510, 37511, 37512, 37513, 37514, 37515, 37516, 37517, 29327, 29328, 29329, 29330, 29331, 29332, 45716, 45717, 29334, 29336, 29338, 29337, 29340, 29339, 29931, 36784, 36785, 29932, 38224, 29933, 46083, 46084, 46085, 46086, 46087, 46088, 42536, 33586, 42537, 29934, 40121, 40122, 40123, 40125, 40126, 29935, 38085, 38086, 38087, 38088, 38089, 38090, 38091, 38092, 38093, 38094, 38095, 38096, 38097, 38098, 38099, 38100, 38101, 38102, 38103, 38104, 38105, 38106, 38107, 38108, 38109, 38110, 38111, 38112, 38113, 38114, 38115, 38116, 38117, 38118, 38119, 38120, 38121, 38122, 38123, 38124, 38125, 38126, 38127, 38128, 38129, 38130, 38131, 38132, 38133, 38134, 38135, 29944, 29945, 29946, 29947, 29948, 29949, 29950, 38137, 38138, 38139, 38140, 38142, 38143, 38144, 38141, 38145, 38146, 38147, 38148, 38149, 38150, 38151, 38152, 38153, 38154, 29969, 29970, 29971, 29972, 29973, 29974, 29975, 29976, 29977, 29978, 29979, 29980, 29981, 29982, 29983, 29984, 29985, 29986, 29987, 29988, 29989, 29990, 29991, 29992, 29993, 29994, 29995, 29996, 29997, 29998, 29999, 30000, 30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010, 30011, 30012, 30013, 30014, 30015, 30016, 30017, 30018, 30019, 30020, 30021, 30022, 30023, 30024, 30025, 30026, 30027, 30028, 30029, 30030, 30031, 30032, 30033, 30034, 30035, 30036, 30037, 30038, 30039, 30040, 30041, 30042, 30043, 30044, 30045, 30046, 30047, 30048, 30049, 30050, 30051, 30052, 30053, 30054, 30055, 30056, 30057, 30058, 30059, 35283, 33760, 35297, 33761, 33762, 33780, 33763, 33764, 33765, 13855, 13856, 13857, 13858, 13859, 33766, 33767, 33781, 33768, 33769, 33772, 33782, 33773, 33774, 33775, 33776, 29225, 33777, 29224, 33784, 33753, 33754, 33783, 33778, 33755, 33759, 33779, 38782, 38783, 38784, 38785, 38786, 38787, 38788, 38789, 38790, 38791, 38792, 38793, 38794, 38795, 38796, 38797, 38798, 38799, 38800, 38801, 38802, 38803, 38804, 38805, 38806, 38807, 38808, 38809, 38810, 38811, 38812, 38813, 38814, 38815, 38816, 38817, 38818, 38819, 38820, 38821, 38822, 38823, 38824, 38825, 38826, 38827, 38828, 38829, 38830, 38831, 38832, 38833, 38834, 38835, 38836, 38837, 38838, 38839, 38840, 38841, 38842, 38843, 38844, 38845, 30649, 38847, 30650, 30651, 30652, 30653, 30654, 30655, 30656, 30657, 30658, 30659, 30660, 30661, 38851, 38852, 38853, 38854, 38855, 30673, 30674, 30675, 30676, 30677, 30678, 30679, 30680, 30681, 30682, 30683, 30684, 30685, 30686, 30687, 30688, 30689, 30690, 30691, 30692, 30693, 30694, 30695, 30696, 30697, 30698, 30699, 30700, 30701, 30702, 30703, 30704, 30705, 30706, 30707, 30708, 30709, 30710, 30711, 30712, 30713, 30714, 30715, 30716, 30717, 30718, 30719, 30720, 30721, 30722, 30723, 30724, 22533, 22534, 22535, 22536, 22537, 22538, 22539, 22540, 22541, 22542, 22543, 22544, 22545, 22546, 22547, 22548, 22549, 22550, 22551, 22552, 22553, 22554, 22555, 22556, 22557, 22558, 22559, 30743, 30744, 30745, 30746, 30747, 30748, 30749, 30750, 30751, 30752, 30753, 30754, 30755, 30756, 30757, 22575, 22576, 22577, 22578, 30762, 30763, 30764, 30765, 22579, 22580, 30768, 30769, 30771, 30772, 30773, 22582, 30774, 22592, 22593, 22594, 29228, 22596, 22597, 22589, 22591, 22595, 22598, 22518, 29226, 29229, 33785, 33788, 30815, 30816, 38226, 29230, 33789, 44210, 29231, 33790, 29942, 29232, 33791, 24147, 44212, 29233, 42584, 37374, 33792, 22519, 29227, 44213, 33786, 42585, 37375, 41974, 44214, 42586, 41975, 37376, 16536, 44215, 29236, 42587, 16537, 44216, 42588, 16538, 44217, 42589, 22520, 16539, 44218, 42590, 16540, 44219, 42591, 16541, 44220, 42592, 16542, 42593, 24149, 22476, 22521, 42595, 42596, 30634, 30635, 30636, 30637, 30638, 30639, 30640, 30641, 39478, 39479, 30642, 39480, 39482, 39483, 39484, 39485, 39486, 39487, 39488, 39489, 39490, 39491, 39492, 39493, 39494, 39495, 39496, 39497, 39498, 39499, 39500, 39501, 39502, 39503, 39504, 39505, 39506, 39507, 39508, 39509, 39510, 39511, 39512, 39513, 39514, 39515, 39516, 39517, 39518, 39519, 39520, 39521, 39522, 39523, 39524, 39525, 39526, 39527, 39528, 39529, 39530, 39531, 39532, 39533, 39534, 39535, 39536, 39537, 39538, 39539, 39540, 39541, 39542, 39543, 39544, 39545, 39546, 39547, 39548, 39549, 39550, 39551, 31360, 31361, 31362, 31363, 31364, 31365, 31366, 31367, 31368, 31369, 31370, 31371, 31372, 31373, 31374, 31375, 31376, 31377, 31378, 31379, 31380, 31381, 31382, 31383, 31384, 31385, 31386, 31387, 31388, 31389, 31390, 31391, 31392, 31393, 31394, 31395, 31396, 31397, 31398, 31399, 31400, 31401, 31402, 31403, 31404, 31405, 31406, 31407, 31408, 31409, 31410, 31411, 31412, 31413, 31414, 31415, 31416, 31417, 31418, 31419, 31420, 31421, 31422, 31423, 31424, 31425, 31426, 31427, 31428, 31429, 31430, 31431, 31432, 31433, 31434, 31435, 31436, 31437, 31438, 31439, 31440, 31441, 31442, 31443, 31444, 31445, 31446, 31447, 31448, 31449, 31450, 31451, 31452, 31453, 31454, 31455, 31456, 31457, 31458, 31459, 31460, 31461, 31462, 31463, 31464, 31465, 31466, 31467, 31468, 31469, 31470, 31471, 31472, 31473, 31474, 31475, 31476, 31477, 31478, 31479, 31480, 31481, 31526, 31527, 31528, 31529, 31530, 47922, 47923, 47924, 47925, 31540, 31538, 31541, 15177, 15178, 15179, 15180, 15181, 15182, 15183, 15186, 15187, 15191, 15193, 15194, 15201, 22560, 15202, 15203, 22561, 22562, 25819, 22563, 25820, 38846, 22564, 22565, 38848, 22566, 38849, 22567, 38850, 22568, 22569, 22570, 22571, 22572, 22573, 38856, 22574, 25831, 38857, 25832, 38858, 25833, 38859, 38860, 38861, 38862, 38863, 22581, 24209, 38864, 24210, 38865, 22583, 38866, 30725, 24212, 22584, 38867, 30726, 24213, 22585, 38868, 30727, 24214, 22586, 38869, 30728, 22587, 38870, 30729, 22588, 38871, 30730, 38872, 30731, 22590, 38873, 30732, 38874, 30733, 38875, 30734, 38876, 30735, 38221, 38877, 30736, 38878, 30737, 38879, 30738, 30739, 38880, 30740, 38881, 30741, 38882, 30742, 38883, 38884, 38885, 38886, 38887, 22532, 38888, 38889, 38890, 38891, 38892, 38893, 38894, 38895, 38896, 38897, 38898, 38899, 30758, 30647, 29951, 38900, 30759, 41348, 38901, 30760, 38902, 30761, 38903, 38904, 38905, 20995, 38906, 20996, 38907, 30766, 20997, 38908, 30767, 40109, 20998, 40110, 38909, 20999, 38910, 40118, 40119, 21000, 40120, 38911, 30770, 40124, 21001, 40127, 40128, 40129, 40130, 40131, 40132, 40133, 40134, 40135, 40136, 40137, 40138, 40139, 40140, 40141, 40142, 40143, 40144, 40145, 40146, 40147, 40148, 40149, 40150, 40151, 40152, 40153, 40154, 40155, 40156, 40157, 40158, 40159, 40160, 40161, 40162, 40163, 40164, 40165, 40166, 40167, 40168, 40169, 40170, 40171, 40172, 40173, 40174, 40175, 40176, 40177, 40178, 40179, 40180, 40181, 40182, 40183, 40184, 40185, 40186, 40187, 40188, 40189, 40190, 40191, 40192, 40193, 40194, 40195, 40196, 40197, 40198, 40199, 40200, 40201, 40202, 40203, 40204, 40205, 40206, 40207, 40208, 40209, 40210, 40211, 40212, 40213, 40214, 40215, 40216, 40217, 40218, 40219, 40220, 40221, 21022, 21023, 21024, 21025, 32063, 32064, 32065, 32066, 32067, 32068, 32069, 32070, 32071, 32072, 32073, 32074, 32075, 32076, 32077, 32078, 32079, 32080, 32081, 32082, 32083, 32084, 32085, 32086, 32087, 32088, 32089, 32090, 32091, 32092, 32093, 32094, 32095, 32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104, 32105, 32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125, 32126, 32127, 32128, 32129, 32130, 32131, 32132, 32133, 32134, 32135, 32136, 32137, 32138, 32139, 32140, 32141, 32142, 32143, 32144, 32145, 32146, 32147, 32148, 32149, 32150, 32151, 32152, 32153, 32154, 32155, 32156, 32157, 32158, 32159, 32160, 32161, 32162, 32163, 32164, 32165, 32166, 32167, 32168, 32169, 32170, 32171, 32172, 32173, 32174, 32175, 32176, 32177, 32178, 32179, 32180, 32181, 32182, 32183, 32184, 32185, 32186, 32187, 32188, 32189, 32190, 32191, 32192, 32193, 32194, 32195, 32196, 32197, 32198, 32199, 32200, 32201, 32202, 32203, 32204, 32205, 29936, 29937, 32256, 32257, 32258, 32259, 32260, 32261, 32262, 38225, 29938, 29208, 32266, 32267, 32268, 32269, 32270, 32271, 32272, 29209, 29210, 29211, 29212, 29939, 29213, 29214, 24105, 24106, 24107, 29215, 24108, 24109, 29216, 29217, 29940, 29218, 29219, 24130, 24131, 24132, 24133, 29220, 24134, 24135, 24136, 24137, 29221, 24138, 24141, 24139, 24140, 29222, 24142, 24143, 29941, 24144, 29223, 24150, 24151, 24152, 24153, 24154, 24155, 24156, 24157, 24158, 24159, 24160, 24161, 24162, 24163, 24164, 24165, 24166, 24167, 24168, 24169, 24170, 24171, 24172, 24173, 24174, 24175, 24176, 24177, 24178, 24179, 24180, 24181, 24182, 24183, 24184, 24185, 24186, 24187, 24188, 24189, 24190, 24191, 24192, 24193, 24194, 24195, 24196, 24197, 24198, 24199, 24200, 24201, 24202, 24203, 24204, 29943, 24205, 24206, 39557, 24207, 29234, 29235, 24208, 24211, 37377, 24215, 24216, 24217, 24218, 24219, 37378, 24221, 24222, 24223, 37379, 24224, 29238, 29237, 24220, 29239, 24225, 24226, 29240, 29241, 29242, 29243, 29244, 29261, 29262, 40730, 40731, 40732, 40733, 29263, 40736, 40737, 40738, 40739, 40740, 40741, 40742, 40743, 40744, 40745, 40746, 40747, 40748, 40749, 40750, 40751, 40752, 40753, 40754, 40755, 40756, 40757, 40758, 40759, 40760, 40761, 40762, 40763, 40764, 40765, 40766, 40767, 40768, 40769, 40770, 40771, 40772, 40773, 40774, 40775, 40776, 40777, 40778, 40779, 40780, 40781, 40782, 40783, 40784, 40785, 40786, 40787, 40788, 40789, 40790, 40791, 40792, 40793, 40794, 40795, 40796, 40797, 40798, 40799, 40800, 40801, 40802, 40803, 40804, 40805, 40806, 40807, 40808, 40809, 40810, 40811, 40812, 40813, 40814, 40815, 40816, 40817, 29952, 40818, 40819, 40820, 40821, 40822, 40823, 40824, 40825, 29953, 29954, 29955, 37437, 37438, 37439, 29956, 37442, 37444, 29957, 37445, 37446, 37447, 37448, 25790]
    # #
    # # print(dice(a,b))
    # # exit()
    # detector = SegDetector()
    # # for stl_pth in glob.glob(r'M:\invis_ori\*.stl'):
    # stl_pth = r'L:\mesh\lower2190.stl'
    # # stl_pth = r'K:\oral_mesh\20150474_shell_occlusion_l.stl'
    # stl_pths = r'L:\zhengya\stl\*.stl'
    # # stl_pths = r'H:\teethseg_test_100\mesh\*.stl'
    # for kk,stl_path in enumerate(glob.glob(stl_pths)):
    #     print(stl_path)
    #     stl_path = r'L:\zhengya\stl\7_d.stl'
    #     stl_path =  r'E:\jfl\u.stl'
    #     detector.read_data(stl_path)
    #     seg_dic = detector.detecting(CLS_DIC_UPPER,kk)






