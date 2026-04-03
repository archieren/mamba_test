"""
Open3D Quadratic Error Metric (QEM) 网格简化示例

使用 Open3D 的 simplify_quadric_decimation 实现 STL 模型简化。
该算法基于 Garland & Heckbert 1997 的二次误差度量方法。

Wayland/GLFW 问题解决：
- 如果遇到 GLFW 错误，使用 export XDG_SESSION_TYPE=x11 切换到 X11 会话。
"""

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import time
import torch
import torch.linalg as LA

from pathlib import Path

#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/乳牙")
#source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/TestData")
#source_dir= Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/标注数据")
#source_dir= Path("/home/archie/Projects/data/TestSet/ATA-TestSample/Separate/temp2")
#source_dir = Path("/home/archie/Projects/data/口扫模型/口扫模型分割新增（有乳牙）")
#source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
#source_dir = Path("/home/archie/Projects/data/口扫模型/HowTo")
# stems=[ stl_item.stem for stl_item in source_dir.glob("*.stl")]
#stem = "lower0059"

def time_it(start_time, label=""):
    stop_time = time.time()
    print(f"{label}耗时: {stop_time - start_time :.6f}秒")
    return

def load_stl_mesh(source_dir:Path, stem: str) -> o3d.geometry.TriangleMesh:
    """加载 STL 模型"""
    #mesh = o3d.io.read_triangle_mesh(stl_path)
    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    print(f"原始模型: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")
    return mesh


def simplify_mesh_qem(mesh: o3d.geometry.TriangleMesh,
                      target_triangle_count: int) -> o3d.geometry.TriangleMesh:
    """使用 QEM 算法简化网格"""
    current_triangles = len(mesh.triangles)
    if target_triangle_count >= current_triangles:
        print(f"目标三角形数 >= 当前三角形数")
        return mesh
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
    simplified.compute_vertex_normals()
    print(f"简化后: {len(simplified.vertices)} 顶点, {len(simplified.triangles)} 三角形")
    return simplified


def simplify_mesh_by_ratio(mesh: o3d.geometry.TriangleMesh,
                           simplification_ratio: float) -> o3d.geometry.TriangleMesh:
    """按比例简化网格 (0.1 表示保留 10%)"""
    target_count = int(len(mesh.triangles) * simplification_ratio)
    return simplify_mesh_qem(mesh, target_count)

# def create_sample_mesh() -> o3d.geometry.TriangleMesh:
#     """创建示例网格"""
#     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=50)
#     mesh.compute_vertex_normals()
#     mesh.paint_uniform_color([0.3, 0.7, 1.0])
#     return mesh


def visualize_comparison(original: o3d.geometry.TriangleMesh,
                         simplified: o3d.geometry.TriangleMesh):
    """并排可视化原始和简化后的网格"""
    orig_vis = o3d.geometry.TriangleMesh(original)
    simp_vis = o3d.geometry.TriangleMesh(simplified)
    orig_vis.paint_uniform_color([0.7, 0.7, 0.7])
    simp_vis.paint_uniform_color([1.0, 0.5, 0.0])
    simp_vis.translate([0.0, 0, 0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    o3d.visualization.draw_geometries(
        [orig_vis, simp_vis, frame], #orig_vis, 
        window_name="原始网格 (左) vs 简化网格 (右)",
        width=1280, height=720
    )

def build_cotan_laplacian(mesh: o3d.geometry.TriangleMesh, device='cuda') -> torch.sparse.FloatTensor:
    """构建 Cotangent Laplacian 矩阵 (PyTorch 稀疏张量)"""
    v = torch.tensor(np.asarray(mesh.vertices,dtype=float), dtype=torch.float).to(device)
    f = torch.tensor(np.asarray(mesh.triangles,dtype=int), dtype=torch.long).to(device)
   
    # 1. 提取三角形顶点
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0, v1, v2 = v[i0], v[i1], v[i2]
    
    # 2. 计算边向量和面积
    # 边向量：e0 是 v2-v1 的对边，以此类推
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0
    
    # 叉乘计算面积：Area = 0.5 * |e1 x e2|
    cross_prod = LA.cross(e1, e2, dim=1)
    areas = 0.5 * LA.norm(cross_prod, dim=1)
    # 防止退化三角形导致除零
    areas = torch.clamp(areas, min=1e-10)
    
    # 3. 计算余切权重 (利用点积和叉乘模长的比值)
    # cot(theta) = (a·b) / |a x b|
    # 对于顶点0处的角：cot0 = (-e1 · e2) / (2 * Area)
    cot0 = -torch.sum(e1 * e2, dim=1) / (4.0 * areas)
    cot1 = -torch.sum(e2 * e0, dim=1) / (4.0 * areas)
    cot2 = -torch.sum(e0 * e1, dim=1) / (4.0 * areas)
    
    # 4. 构建稀疏对称权重矩阵 W
    # 每条边 (i,j) 对应两个三角形的余切值之和
    I = torch.cat([i1, i2, i2, i0, i0, i1])
    J = torch.cat([i2, i1, i0, i2, i1, i0])
    W_values = torch.cat([cot0, cot0, cot1, cot1, cot2, cot2])
    
    num_v = v.shape[0]
    # 使用 coalesce() 合并重复边的权重 (即合并相邻三角形的贡献)
    W_sparse = torch.sparse_coo_tensor(torch.stack([I, J]), W_values, (num_v, num_v)).coalesce()
    
    # 5. 构建拉普拉斯矩阵 L = D - W
    # 计算度矩阵 D (对角线元素)
    deg_values = torch.sparse.sum(W_sparse, dim=1).to_dense()
    
    # 构建 L (同样以稀疏格式存储)
    indices = W_sparse.indices()
    # 非对角线元素是 -W
    L_values = -W_sparse.values()
    
    # 添加对角线元素 D
    diag_indices = torch.arange(num_v, device=device).repeat(2, 1)
    L_indices = torch.cat([indices, diag_indices], dim=1)
    L_values = torch.cat([L_values, deg_values])
    
    L = torch.sparse_coo_tensor(L_indices, L_values, (num_v, num_v)).coalesce()
    
    # 6. 构造质量矩阵 M (Lumped Mass Matrix)
    # 顶点面积 = 相邻三角形面积之和 / 3
    M_values = torch.zeros(num_v, device=device)
    M_values.scatter_add_(0, i0, areas / 3.0)
    M_values.scatter_add_(0, i1, areas / 3.0)
    M_values.scatter_add_(0, i2, areas / 3.0)
    
    return L, M_values

def compute_eigenpairs(L: torch.sparse.FloatTensor, M_values: torch.Tensor, k=100):
    """计算广义特征值问题 L x = λ M x 的前 k 个特征值和特征向量"""
    # 注意：torch.lobpcg 需要稀疏矩阵 L 和对角矩阵 M 的形式
    # 计算 L 的对角线逆作为预处理器
    diag_L = L.indices()[0] == L.indices()[1] # 找到对角线索引
    inv_diag = 1.0 / L.values()[diag_L]
    M_inv = torch.diag(inv_diag) # 简单的预处理矩阵

    # 在 lobpcg 中传入 M (注意：此 M 是预处理器，不是质量矩阵)
    eigenvalues, eigenvectors = torch.lobpcg(L, k=k, B=torch.diag(M_values),iK=M_inv, largest=False)
    return eigenvalues, eigenvectors

def compute_wks(eigenvalues, eigenvectors, num_scales=100, sigma=0.06):
    """
    eigenvalues: (k,) 升序排列的特征值
    eigenvectors: (N, k) 特征向量，每一列是一个模态
    num_scales: 能量采样的数量（特征向量的维度）
    """
    # 1. 预处理特征值
    # 排除接近 0 的第一个特征值（对应常数模态）
    # print("Eigenvalues (前20):", eigenvalues[:20])
    evals = eigenvalues[1:].unsqueeze(0)  # (1, k-1)
    evecs = eigenvectors[:, 1:]           # (N, k-1)
    
    # 2. 在对数空间定义能量采样点 e
    log_evals = torch.log(torch.abs(evals))
    e_min = log_evals.min() + 2 * sigma
    e_max = log_evals.max() - 2 * sigma
    e = torch.linspace(e_min, e_max, num_scales, device=evals.device).view(-1, 1) # (S, 1)
    
    # 3. 计算带通滤波器 (Gauss Kernel)
    # 矩阵化计算：(S, 1) 与 (1, k-1) 广播得到 (S, k-1)
    weights = torch.exp(-torch.pow(e - log_evals, 2) / (2 * sigma**2))
    
    # 4. 组合特征向量平方
    # WKS = sum( phi_n^2 * weights_n )
    # evecs_sq: (N, k-1)
    evecs_sq = torch.pow(evecs, 2)
    
    # 5. 生成描述子
    # (S, k-1) @ (k-1, N) -> (S, N)
    wks = torch.matmul(weights, evecs_sq.T).T # 结果维度 (N, S)
    
    # 6. 归一化 (L1 归一化使描述子对尺度不敏感)
    wks_sum = torch.sum(wks, dim=1, keepdim=True)
    wks = wks / wks_sum
    
    return wks

def compute_fixed_cheb_features(L_sparse, K=5):
    """
    L_sparse: 预先构造好的归一化拉普拉斯稀疏矩阵 [N, N]
    K: 阶数。阶数越高，捕捉的几何结构越“宏观”
    """
    device = L_sparse.device
    num_v = L_sparse.shape[0]
    
    # 初始信号：使用常数信号或随机信号作为种子
    # 这里我们使用简单的单位向量，代表每个顶点的初始“能量”
    x = torch.ones((num_v, 1), device=device)
    
    # 递归计算切比雪夫多项式 T_k(L) * x
    # T0 = I, T1 = L
    T0 = x
    T1 = torch.sparse.mm(L_sparse, x)
    
    features = [T0, T1]
    
    for k in range(2, K + 1):
        # 递推公式: Tk = 2 * L * T_{k-1} - T_{k-2}
        Tk = 2 * torch.sparse.mm(L_sparse, features[-1]) - features[-2]
        features.append(Tk)
        
    # 将不同阶数的特征拼接：[N, K+1]
    # 每一列代表了不同频率下的几何响应
    cheb_feats = torch.cat(features, dim=1)
    
    # 归一化，方便后续聚类
    cheb_feats = (cheb_feats - cheb_feats.mean(0)) / (cheb_feats.std(0) + 1e-6)
    
    return cheb_feats

def overview():
    """主函数: 演示网格简化流程

    Args:
        headless: 无头模式，只保存文件不显示窗口
        use_browser: 使用浏览器显示 (兼容 Wayland)
    """
    print("创建示例网格...")
    source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
    stem = "20182059_shell_occlusion_u"
    mesh = load_stl_mesh(source_dir, stem)
    print(f"示例网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")

    # 测试按比例简化
    print("\n=== 按比例简化测试 ===")
    simplification_ratios = [0.5, 0.25, 0.1, 0.05, 0.01]
    simplified_meshes_by_ratio = []
    for ratio in simplification_ratios:
        print(f"\n简化至 {ratio*100:.0f}%...")
        
        start_time = time.time()
        simplified = simplify_mesh_by_ratio(mesh, ratio)
        time_it(start_time)
        
        simplified_meshes_by_ratio.append(simplified)

    # 测试按面数简化
    """测试按具体面数简化"""
    original_triangles = len(mesh.triangles)
    print(f"\n=== 按面数简化测试 ===")
    print(f"原始面数: {original_triangles}")

    # 测试不同的目标面数
    target_counts = [2**16, 2**15, 2**14, 2**13, 2**12]

    simplified_meshes_by_count = []
    for target in target_counts:
        if target >= original_triangles:
            print(f"目标面数 {target} >= 原始面数 {original_triangles}，跳过")
            continue

        start_time = time.time()
        simplified = simplify_mesh_qem(mesh, target)
        time_it(start_time)
        
        actual = len(simplified.triangles)
        reduction = (1 - actual / original_triangles) * 100
        print(f"目标: {target:5d} → 实际: {actual:5d} 面片 (压缩率: {reduction:.1f}%)")
        simplified_meshes_by_count.append(simplified)

    output_dir = Path("./tests/simplified_meshes")
    output_dir.mkdir(exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_dir / "original.stl"), mesh)
    for ratio, simp in zip(simplification_ratios, simplified_meshes_by_ratio):
        o3d.io.write_triangle_mesh(
            str(output_dir / f"simplified_{int(ratio*100)}percent.stl"), simp
        )
        
    for count, simp in zip(target_counts, simplified_meshes_by_count):
        o3d.io.write_triangle_mesh(
            str(output_dir / f"simplified_to_{count}face.stl"), simp
        )

    print(f"\n简化结果已保存至: {output_dir.absolute()}")

    #visualize_comparison(mesh, simplified_meshes_by_count[4])
    start_time = time.time()
    L, M_values  = build_cotan_laplacian(simplified_meshes_by_count[0])
    time_it(start_time)
    print(f"0 - ",L.shape, M_values.shape, "\n")
    
    for i in reversed(range(len(simplified_meshes_by_count))):
        start_time = time.time()
        L, M_values = build_cotan_laplacian(simplified_meshes_by_count[i])
        time_it(start_time)
        print(f"{i} - ",L.shape, M_values.shape, "\n")

    input("按 Enter 键退出...")

def visualize_wks_pca(vertices, faces, wks):
    """
    使用 Open3D 高效渲染 WKS 的 PCA 降维特征图
    vertices: (N, 3) 数组或 tensor
    faces: (M, 3) 数组或 tensor
    wks: (N, S) 预计算的 WKS 描述子 (通常 S=100)
    """
    # 1. 创建 Open3D 网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy() if torch.is_tensor(faces) else faces)
    
    # 2. 执行 PCA 降维 (从 100维 降到 3维: R, G, B)
    wks_np = wks.cpu().detach().numpy() if torch.is_tensor(wks) else wks
    pca = PCA(n_components=3)
    # wks_pca 维度: (N, 3)
    wks_pca = pca.fit_transform(wks_np)
    
    # 3. 将 PCA 结果归一化到 [0, 1] 范围作为 RGB 颜色
    # 这一步确保颜色值在 Open3D 要求的合法范围内
    wks_min = wks_pca.min(axis=0)
    wks_max = wks_pca.max(axis=0)
    # 防止除以零
    wks_color = (wks_pca - wks_min) / (wks_max - wks_min + 1e-10)
    
    # 4. 将颜色赋值给顶点
    mesh.vertex_colors = o3d.utility.Vector3dVector(wks_color)
    
    # 5. 渲染设置
    mesh.compute_vertex_normals() # 计算法线以开启光照，增加立体感
    print("正在渲染 PCA 特征图... 颜色相似的区域具有相似的局部几何结构。")
    o3d.visualization.draw_geometries([mesh], window_name="WKS PCA Feature Visualization",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=True)

def visualize_wks_similarity(vertices, faces, wks, query_v_idx):
    """
    使用 Open3D 高效渲染 WKS 相似性
    vertices: (N, 3) 数组或 tensor
    faces: (M, 3) 数组或 tensor
    wks: (N, S) 预计算的 WKS 描述子
    query_v_idx: 参考点的索引
    """
    # 1. 创建 Open3D 网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy() if torch.is_tensor(faces) else faces)
    
    # 2. 计算 WKS 欧氏距离 (在 GPU/CPU 上计算)
    query_feat = wks[query_v_idx:query_v_idx+1, :]
    # dist 维度: (N,)
    dist = torch.norm(wks - query_feat, dim=1).cpu().numpy()
    
    # 3. 距离归一化并映射到颜色映射表 (Colormap)
    # 距离越小（越相似），颜色越深/亮
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-10)
    print(f"参考点索引: {query_v_idx}, 最小距离: {dist.min():.4f}, 最大距离: {dist.max():.4f}")
    # 使用 matplotlib 的 viridis 映射表生成 RGB 颜色
    cmap = plt.get_cmap('viridis')
    # 我们取 1-dist_norm，使得相似的地方（距离小）呈现高亮色
    vertex_colors = cmap(1.0 - dist_norm)[:, :3] # 取 RGB 部分，忽略 alpha 通道
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # 4. 渲染设置
    mesh.compute_vertex_normals() # 开启光照效果
    print(f"正在渲染... 参考点索引: {query_v_idx}")
    o3d.visualization.draw_geometries([mesh], window_name="WKS Similarity Visualization",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=True)
    
def segment_by_wks():
    print("创建示例网格...")
    source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
    stem = "20182059_shell_occlusion_u"
    mesh = load_stl_mesh(source_dir, stem)

    target = 2**15 
    start_time = time.time()  
    simplified = simplify_mesh_qem(mesh, target)
    time_it(start_time, label="QEM网格简化")
    # Waste time!
    L, M_values = build_cotan_laplacian(simplified)
    
    start_time = time.time()
    L, M_values = build_cotan_laplacian(simplified)
    time_it(start_time, label="构建拉普拉斯矩阵")
    # {- 1
    start_time = time.time()
    eigenvalues, eigenvectors = compute_eigenpairs(L, M_values, k=64)
    time_it(start_time, label="计算特征值和特征向量")
    start_time = time.time()
    wks = compute_wks(eigenvalues, eigenvectors, num_scales=128, sigma=0.06)
    time_it(start_time, label="计算 WKS")
    # 1 -}
    # {- 2
    # start_time = time.time()
    # wks = compute_fixed_cheb_features(L, K=16)
    # time_it(start_time, label="计算切比雪夫特征")
    # 2 -}
    print("WKS 特征维度:", wks.shape)
    input("按 Enter 键继续...")
    print(wks[:5, :5])  # 打印前5个顶点的 WKS 特征示例
    # visualize_wks_similarity(
    #     vertices=np.asarray(simplified.vertices),
    #     faces=np.asarray(simplified.triangles),
    #     wks=wks,
    #     query_v_idx=8192  # 选择一个顶点作为参考点
    # )
    # visualize_wks_pca(
    #     vertices=np.asarray(simplified.vertices),
    #     faces=np.asarray(simplified.triangles),
    #     wks=wks
    # )

if __name__ == "__main__":
    # overview()
    segment_by_wks()