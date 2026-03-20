"""
Probabilistic Laplacian 和网格随机游走

基于 Cotangent 权重构建概率转移矩阵，实现多种网格遍历策略：
- 纯随机游走 (Random Walk)
- 自回避游走 (Self-Avoiding Walk)
- 几何引导游走 (Geometry-Guided Walk)
- 覆盖最大化游走 (Coverage-Maximizing Walk)

应用场景：
- 网格采样、特征提取、路径规划、数据增强、图神经网络
"""

import numpy as np
import open3d as o3d
import torch
import torch.linalg as LA
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# 输出目录配置
OUTPUT_DIR = Path("./tests/probabilistic_laplacian")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# 核心：构建 Probabilistic Laplacian
# ============================================================================

def build_cotangent_weights(mesh: o3d.geometry.TriangleMesh, device='cuda'):
    """
    构建 Cotangent 权重矩阵

    Args:
        mesh: Open3D 三角网格
        device: 计算设备

    Returns:
        W_sparse: 稀疏权重矩阵 (N, N)
        deg: 度向量 (N,)
        vertices: 顶点坐标 (N, 3)
    """
    v = torch.tensor(np.asarray(mesh.vertices, dtype=float), dtype=torch.float).to(device)
    f = torch.tensor(np.asarray(mesh.triangles, dtype=int), dtype=torch.long).to(device)

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0, v1, v2 = v[i0], v[i1], v[i2]

    # 边向量
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    # 三角形面积
    cross_prod = LA.cross(e1, e2, dim=1)
    areas = torch.clamp(0.5 * LA.norm(cross_prod, dim=1), min=1e-10)

    # Cotangent 权重（取绝对值确保非负，用于概率）
    cot0 = torch.abs(-torch.sum(e1 * e2, dim=1) / (4.0 * areas))
    cot1 = torch.abs(-torch.sum(e2 * e0, dim=1) / (4.0 * areas))
    cot2 = torch.abs(-torch.sum(e0 * e1, dim=1) / (4.0 * areas))

    # 构建稀疏权重矩阵
    num_v = v.shape[0]
    I = torch.cat([i1, i2, i2, i0, i0, i1])
    J = torch.cat([i2, i1, i0, i2, i1, i0])
    W_values = torch.cat([cot0, cot0, cot1, cot1, cot2, cot2])

    W_sparse = torch.sparse_coo_tensor(
        torch.stack([I, J]), W_values, (num_v, num_v)
    ).coalesce()

    # 度矩阵（行和）
    deg = torch.sparse.sum(W_sparse, dim=1).to_dense()
    deg = torch.clamp(deg, min=1e-10)

    return W_sparse, deg, v.cpu().numpy()


def build_transition_matrix(W_sparse, deg):
    """
    从权重矩阵构建转移概率矩阵 P

    P[i,j] = W[i,j] / D[i] 表示从顶点 i 转移到 j 的概率

    Args:
        W_sparse: 稀疏权重矩阵
        deg: 度向量

    Returns:
        neighbors: 每个顶点的邻居列表
        neighbor_probs: 每个顶点到邻居的转移概率
        P_sparse: 稀疏转移概率矩阵
    """
    indices = W_sparse.indices()
    values = W_sparse.values()
    num_v = deg.shape[0]

    # 归一化得到转移概率
    inv_deg = 1.0 / deg
    P_values = values * inv_deg[indices[0]]
    P_sparse = torch.sparse_coo_tensor(indices, P_values, (num_v, num_v)).coalesce()

    # 构建邻接表（方便采样）
    neighbors = []
    neighbor_probs = []

    for i in range(num_v):
        mask = indices[0] == i
        if mask.sum() == 0:
            # 孤立点，添加自环
            neighbors.append(np.array([i]))
            neighbor_probs.append(np.array([1.0]))
        else:
            nbrs = indices[1][mask].cpu().numpy()
            probs = P_values[mask].cpu().numpy()
            neighbors.append(nbrs)
            neighbor_probs.append(probs / probs.sum())

    return neighbors, neighbor_probs, P_sparse


def build_probabilistic_laplacian(mesh: o3d.geometry.TriangleMesh, device='cuda'):
    """
    完整构建 Probabilistic Laplacian

    Random Walk Laplacian: L_rw = I - P

    Args:
        mesh: Open3D 三角网格
        device: 计算设备

    Returns:
        neighbors: 邻居列表
        neighbor_probs: 转移概率
        L_rw: Random Walk Laplacian (稀疏矩阵)
        P: 转移概率矩阵 (稀疏矩阵)
        stationary_dist: 稳态分布 π[i] = D[i] / ΣD
        vertices: 顶点坐标
    """
    W_sparse, deg, vertices = build_cotangent_weights(mesh, device)
    neighbors, neighbor_probs, P_sparse = build_transition_matrix(W_sparse, deg)

    num_v = len(vertices)

    # Random Walk Laplacian: L_rw = I - P
    # 创建稀疏单位矩阵（兼容不同 PyTorch 版本）
    diag_indices = torch.arange(num_v, device=device).unsqueeze(0).repeat(2, 1)
    I_values = torch.ones(num_v, device=device)
    I_sparse = torch.sparse_coo_tensor(diag_indices, I_values, (num_v, num_v)).coalesce()
    L_rw = I_sparse - P_sparse

    # 稳态分布
    stationary_dist = (deg / deg.sum()).cpu().numpy()

    return neighbors, neighbor_probs, L_rw, P_sparse, stationary_dist, vertices


# ============================================================================
# 随机游走策略
# ============================================================================

def random_walk(neighbors, neighbor_probs, start_idx=0, num_steps=5000, seed=None):
    """
    纯随机游走

    每一步按转移概率随机选择下一个顶点

    Args:
        neighbors: 邻居列表
        neighbor_probs: 转移概率
        start_idx: 起始顶点索引
        num_steps: 游走步数
        seed: 随机种子

    Returns:
        sequence: 顶点索引序列 [x_0, x_1, ..., x_T]
    """
    if seed is not None:
        np.random.seed(seed)

    sequence = [start_idx]
    current = start_idx

    for _ in range(num_steps):
        nbrs = neighbors[current]
        probs = neighbor_probs[current]
        current = np.random.choice(nbrs, p=probs)
        sequence.append(current)

    return np.array(sequence)


def self_avoiding_walk(neighbors, neighbor_probs, num_steps=5000, seed=None, penalty=0.1):
    """
    自回避游走 (Self-Avoiding Walk)

    降低已访问节点的访问概率，倾向于探索新区域

    Args:
        neighbors: 邻居列表
        neighbor_probs: 转移概率
        num_steps: 游走步数
        seed: 随机种子
        penalty: 对已访问节点的惩罚因子

    Returns:
        sequence: 顶点索引序列
    """
    if seed is not None:
        np.random.seed(seed)

    num_v = len(neighbors)
    start_idx = np.random.randint(num_v)
    sequence = [start_idx]
    visited = {start_idx}
    current = start_idx

    for _ in range(num_steps):
        nbrs = neighbors[current]
        probs = neighbor_probs[current].copy()

        # 惩罚已访问的邻居
        for i, nbr in enumerate(nbrs):
            if nbr in visited:
                probs[i] *= penalty

        probs = probs / probs.sum()
        current = np.random.choice(nbrs, p=probs)

        sequence.append(current)
        visited.add(current)

    return np.array(sequence)


def geometry_guided_walk(neighbors, neighbor_probs, feature_scores,
                         num_steps=5000, seed=None, alpha=1.0):
    """
    几何引导游走

    根据几何特征（如曲率、WKS等）引导游走方向

    Args:
        neighbors: 邻居列表
        neighbor_probs: 转移概率
        feature_scores: (N,) 每个顶点的特征得分
        num_steps: 游走步数
        seed: 随机种子
        alpha: 引导强度（0=纯随机，越大越强引导）

    Returns:
        sequence: 顶点索引序列
    """
    if seed is not None:
        np.random.seed(seed)

    num_v = len(neighbors)
    # 从特征最强处开始
    start_idx = np.argmax(feature_scores)
    sequence = [start_idx]
    current = start_idx

    for _ in range(num_steps):
        nbrs = neighbors[current]
        base_probs = neighbor_probs[current].copy()

        # 融合几何特征
        nbr_features = feature_scores[nbrs]
        # 特征归一化
        nbr_features_norm = (nbr_features - nbr_features.min()) / (nbr_features.max() - nbr_features.min() + 1e-10)
        guided_probs = base_probs * (1 + alpha * nbr_features_norm)

        guided_probs = guided_probs / guided_probs.sum()
        current = np.random.choice(nbrs, p=guided_probs)
        sequence.append(current)

    return np.array(sequence)


def coverage_maximizing_walk(neighbors, num_v, num_steps=5000, seed=None):
    """
    覆盖最大化游走

    倾向于访问未探索区域，最大化网格覆盖率

    Args:
        neighbors: 邻居列表
        num_v: 顶点总数
        num_steps: 游走步数
        seed: 随机种子

    Returns:
        sequence: 顶点索引序列
        visit_counts: 每个顶点的访问次数
    """
    if seed is not None:
        np.random.seed(seed)

    start_idx = 0
    sequence = [start_idx]
    visit_count = np.zeros(num_v, dtype=int)
    visit_count[start_idx] = 1
    current = start_idx

    for _ in range(num_steps):
        nbrs = neighbors[current]

        # 选择访问次数最少的邻居
        min_visits = visit_count[nbrs].min()
        candidates = nbrs[visit_count[nbrs] == min_visits]
        current = np.random.choice(candidates)

        sequence.append(current)
        visit_count[current] += 1

    return np.array(sequence), visit_count


def node2vec_style_walk(neighbors, neighbor_probs, num_walks=10, walk_length=80,
                        p=1.0, q=1.0, seed=None):
    """
    Node2Vec 风格的游走

    通过参数 p 和 q 控制游走策略：
    - p: 回参参数（控制返回前节点的概率）
    - q: 进出参数（控制探索DFS还是BFS）

    Args:
        neighbors: 邻居列表
        neighbor_probs: 转移概率
        num_walks: 每个节点开始的游走次数
        walk_length: 游走长度
        p: 回参参数
        q: 进出参数
        seed: 随机种子

    Returns:
        walks: 所有游走序列的列表
    """
    if seed is not None:
        np.random.seed(seed)

    num_v = len(neighbors)
    walks = []

    # 首先构建完整的邻接字典
    adjacency = {i: list(neighbors[i]) for i in range(num_v)}

    for node in range(num_v):
        for _ in range(num_walks):
            walk = [node]
            curr = node
            prev = None

            for _ in range(walk_length - 1):
                nbrs = adjacency[curr]

                if prev is None:
                    # 第一步：随机选择
                    next_node = np.random.choice(nbrs)
                else:
                    # Node2Vec 转移概率
                    probs = []
                    for nbr in nbrs:
                        if nbr == prev:
                            # 返回前节点
                            probs.append(1.0 / p)
                        elif nbr in adjacency.get(prev, []):
                            # 与前节点共同邻居（1-hop）
                            probs.append(1.0)
                        else:
                            # 探索新节点
                            probs.append(1.0 / q)

                    probs = np.array(probs)
                    probs = probs / probs.sum()
                    next_node = np.random.choice(nbrs, p=probs)

                walk.append(next_node)
                prev, curr = curr, next_node

            walks.append(walk)

    return walks


# ============================================================================
# 统计分析
# ============================================================================

def analyze_walk_sequence(sequence, num_v):
    """
    分析游走序列的统计特性

    Args:
        sequence: 顶点索引序列
        num_v: 顶点总数

    Returns:
        stats: 统计信息字典
    """
    visit_counts = Counter(sequence)
    unique_visited = len(visit_counts)
    coverage = unique_visited / num_v * 100

    # 回访时间分析
    first_visit = {}
    return_times = []

    for i, idx in enumerate(sequence):
        if idx not in first_visit:
            first_visit[idx] = i
        else:
            return_times.append(i - first_visit[idx])

    stats = {
        'total_steps': len(sequence),
        'unique_visited': unique_visited,
        'coverage': coverage,
        'mean_return_time': np.mean(return_times) if return_times else 0,
        'visit_counts': visit_counts
    }

    print(f"=== 游走统计 ===")
    print(f"总步数: {stats['total_steps']}")
    print(f"唯一访问顶点: {stats['unique_visited']} / {num_v}")
    print(f"覆盖率: {stats['coverage']:.2f}%")
    if return_times:
        print(f"平均回访时间: {stats['mean_return_time']:.2f} 步")
    print(f"最常访问顶点: {visit_counts.most_common(5)}")

    return stats


# ============================================================================
# 可视化
# ============================================================================

def visualize_walk_sequence(vertices, triangles, sequence, title="Random Walk",
                            output_path=None, show_mesh=True):
    """
    可视化游走序列

    Args:
        vertices: 顶点坐标 (N, 3)
        triangles: 三角面片 (M, 3)
        sequence: 游走序列
        title: 图标题
        output_path: 保存路径（可选）
        show_mesh: 是否显示底层网格

    Returns:
        fig: matplotlib 图形对象
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制底层网格（半透明）
    if show_mesh:
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles, alpha=0.15, color='gray',
            edgecolor='none', linewidth=0
        )

    # 绘制游走路径（彩色渐变）
    walk_vertices = vertices[sequence]
    colors = plt.cm.turbo(np.linspace(0, 1, len(sequence)))

    # 绘制连接线
    for i in range(len(sequence) - 1):
        ax.plot(
            [walk_vertices[i, 0], walk_vertices[i+1, 0]],
            [walk_vertices[i, 1], walk_vertices[i+1, 1]],
            [walk_vertices[i, 2], walk_vertices[i+1, 2]],
            color=colors[i], alpha=0.7, linewidth=1.5
        )

    # 标记关键点
    ax.scatter(*walk_vertices[0], color='lime', s=150, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=10)
    ax.scatter(*walk_vertices[-1], color='red', s=150, marker='X',
               edgecolors='black', linewidths=2, label='End', zorder=10)

    # 标记中点
    mid_idx = len(sequence) // 2
    ax.scatter(*walk_vertices[mid_idx], color='yellow', s=100, marker='*',
               edgecolors='black', linewidths=1.5, label='Mid', zorder=10)

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_title(f'{title} ({len(sequence)} steps)', fontsize=14)

    # 设置视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存至: {output_path}")

    return fig


def visualize_visit_heatmap(vertices, triangles, visit_counts, title="Visit Frequency",
                            output_path=None):
    """
    可视化访问频率热力图

    Args:
        vertices: 顶点坐标 (N, 3)
        triangles: 三角面片 (M, 3)
        visit_counts: 每个顶点的访问次数
        title: 图标题
        output_path: 保存路径（可选）

    Returns:
        fig: matplotlib 图形对象
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 归一化访问次数
    visits = np.array([visit_counts.get(i, 0) for i in range(len(vertices))])
    visits_norm = (visits - visits.min()) / (visits.max() - visits.min() + 1e-10)

    # 绘制网格，顶点颜色表示访问频率
    surf = ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        facecolors=plt.cm.hot(visits_norm),
        edgecolor='none', alpha=0.9, shade=True
    )

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.hot)
    mappable.set_array(visits)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Visit Count', fontsize=10)

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存至: {output_path}")

    return fig


# ============================================================================
# 示例和测试
# ============================================================================

def load_stl_mesh(source_dir: Path, stem: str, target_triangles: int = 8096) -> o3d.geometry.TriangleMesh:
    """
    加载并简化 STL 模型

    Args:
        source_dir: 模型目录
        stem: 模型文件名（不含扩展名）
        target_triangles: 目标三角形数（使用 QEM 简化）

    Returns:
        mesh: 简化后的网格
    """
    stl_item_path = source_dir / (stem + ".stl")
    mesh = o3d.io.read_triangle_mesh(str(stl_item_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    original_triangles = len(mesh.triangles)
    print(f"原始网格: {len(mesh.vertices)} 顶点, {original_triangles} 三角形")

    # QEM 简化
    if original_triangles > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh.compute_vertex_normals()
        reduction = (1 - len(mesh.triangles) / original_triangles) * 100
        print(f"简化后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形 (压缩率: {reduction:.1f}%)")
    else:
        mesh.compute_vertex_normals()
        print(f"网格已符合要求，无需简化")

    return mesh


def demo_basic_walk():
    """基础演示：随机游走"""
    print("=" * 60)
    print("演示 1: 基础随机游走")
    print("=" * 60)

    # 加载网格
    source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
    stem = "20182059_shell_occlusion_u"
    mesh = load_stl_mesh(source_dir, stem)

    # 构建 Probabilistic Laplacian
    start_time = time.time()
    neighbors, neighbor_probs, L_rw, P, stationary_dist, vertices = build_probabilistic_laplacian(mesh)
    print(f"构建 Probabilistic Laplacian 耗时: {time.time() - start_time:.3f} 秒")
    print(f"L_rw 形状: {L_rw.shape}")
    print(f"转移矩阵 P 非零元素: {P.values().shape[0]}")

    # 纯随机游走
    print("\n--- 纯随机游走 ---")
    seq_random = random_walk(neighbors, neighbor_probs, start_idx=0, num_steps=5000, seed=42)
    analyze_walk_sequence(seq_random, len(vertices))

    # 可视化
    triangles = np.asarray(mesh.triangles)
    visualize_walk_sequence(
        vertices, triangles, seq_random,
        title="Pure Random Walk",
        output_path=str(OUTPUT_DIR / "random_walk.png")
    )

    return neighbors, neighbor_probs, vertices, triangles


def demo_comparison():
    """对比演示：不同游走策略"""
    print("\n" + "=" * 60)
    print("演示 2: 游走策略对比")
    print("=" * 60)

    # 加载网格
    source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
    stem = "20182059_shell_occlusion_u"
    mesh = load_stl_mesh(source_dir, stem)

    neighbors, neighbor_probs, _, _, _, vertices = build_probabilistic_laplacian(mesh)
    triangles = np.asarray(mesh.triangles)
    num_v = len(vertices)

    strategies = []

    # 1. 纯随机游走
    print("\n--- 纯随机游走 ---")
    seq1 = random_walk(neighbors, neighbor_probs, start_idx=0, num_steps=5000, seed=42)
    stats1 = analyze_walk_sequence(seq1, num_v)
    strategies.append(("Random Walk", seq1))

    # 2. 自回避游走
    print("\n--- 自回避游走 ---")
    seq2 = self_avoiding_walk(neighbors, neighbor_probs, num_steps=5000, seed=42, penalty=0.1)
    stats2 = analyze_walk_sequence(seq2, num_v)
    strategies.append(("Self-Avoiding Walk", seq2))

    # 3. 覆盖最大化游走
    print("\n--- 覆盖最大化游走 ---")
    seq3, visit_counts = coverage_maximizing_walk(neighbors, num_v, num_steps=5000, seed=42)
    stats3 = analyze_walk_sequence(seq3, num_v)
    strategies.append(("Coverage-Maximizing Walk", seq3))

    # 对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    for idx, (name, seq) in enumerate(strategies):
        ax = axes[idx]

        # 绘制网格
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles, alpha=0.1, color='gray'
        )

        # 绘制路径
        walk_verts = vertices[seq]
        colors = plt.cm.turbo(np.linspace(0, 1, len(seq)))

        for i in range(len(seq) - 1):
            ax.plot(
                [walk_verts[i, 0], walk_verts[i+1, 0]],
                [walk_verts[i, 1], walk_verts[i+1, 1]],
                [walk_verts[i, 2], walk_verts[i+1, 2]],
                color=colors[i], alpha=0.6, linewidth=1
            )

        ax.scatter(*walk_verts[0], color='green', s=80, label='Start')
        ax.scatter(*walk_verts[-1], color='red', s=80, label='End')
        ax.set_title(f"{name}\nCoverage: {analyze_walk_sequence(seq, num_v)['coverage']:.1f}%")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "walk_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存对比图至: {OUTPUT_DIR / 'walk_comparison.png'}")

    return strategies


def demo_guided_walk():
    """演示：几何引导游走（需要特征）"""
    print("\n" + "=" * 60)
    print("演示 3: 几何引导游走")
    print("=" * 60)

    # 加载网格
    source_dir = Path("/home/archie/Projects/data/口扫模型/牙齿分割标注数据---/牙齿分割10个-ns/")
    stem = "20182059_shell_occlusion_u"
    mesh = load_stl_mesh(source_dir, stem)

    neighbors, neighbor_probs, _, _, _, vertices = build_probabilistic_laplacian(mesh)
    triangles = np.asarray(mesh.triangles)

    # 计算简单的几何特征：高斯曲率近似
    # 使用顶点邻域的角度亏损作为曲率代理
    v = torch.tensor(vertices, dtype=torch.float)
    f = torch.tensor(triangles, dtype=torch.long)

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0, v1, v2 = v[i0], v[i1], v[i2]

    # 计算每个三角形的角度
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
    cos_angle0 = torch.sum(e1 * e2, dim=1) / (LA.norm(e1, dim=1) * LA.norm(e2, dim=1) + 1e-10)
    cos_angle1 = torch.sum(e2 * e0, dim=1) / (LA.norm(e2, dim=1) * LA.norm(e0, dim=1) + 1e-10)
    cos_angle2 = torch.sum(e0 * e1, dim=1) / (LA.norm(e0, dim=1) * LA.norm(e1, dim=1) + 1e-10)

    angle0 = torch.acos(torch.clamp(cos_angle0, -1, 1))
    angle1 = torch.acos(torch.clamp(cos_angle1, -1, 1))
    angle2 = torch.acos(torch.clamp(cos_angle2, -1, 1))

    # 角度亏损（高曲率区域角度亏损大）
    angle_deficit = torch.zeros(len(vertices))
    angle_deficit.scatter_add_(0, i0, angle0)
    angle_deficit.scatter_add_(0, i1, angle1)
    angle_deficit.scatter_add_(0, i2, angle2)
    angle_deficit = 2 * np.pi - angle_deficit  # 理想和为 2π

    feature_scores = angle_deficit.numpy()

    # 对比游走
    print("\n--- 纯随机游走 ---")
    seq_random = random_walk(neighbors, neighbor_probs, num_steps=5000, seed=42)
    analyze_walk_sequence(seq_random, len(vertices))

    print("\n--- 曲率引导游走 ---")
    seq_guided = geometry_guided_walk(
        neighbors, neighbor_probs, feature_scores,
        num_steps=5000, seed=42, alpha=2.0
    )
    analyze_walk_sequence(seq_guided, len(vertices))

    # 对比可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})

    for idx, (seq, title) in enumerate([(seq_random, "Random"), (seq_guided, "Curvature-Guided")]):
        ax = axes[idx]

        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles, alpha=0.1, color='gray'
        )

        walk_verts = vertices[seq]
        colors = plt.cm.turbo(np.linspace(0, 1, len(seq)))

        for i in range(len(seq) - 1):
            ax.plot(
                [walk_verts[i, 0], walk_verts[i+1, 0]],
                [walk_verts[i, 1], walk_verts[i+1, 1]],
                [walk_verts[i, 2], walk_verts[i+1, 2]],
                color=colors[i], alpha=0.6, linewidth=1
            )

        ax.scatter(*walk_verts[0], color='green', s=80)
        ax.scatter(*walk_verts[-1], color='red', s=80)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "guided_walk_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存至: {OUTPUT_DIR / 'guided_walk_comparison.png'}")

    return seq_random, seq_guided


def main():
    """主函数"""
    print("Probabilistic Laplacian 和网格随机游走")
    print("=" * 60)

    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "basic"

    if mode == "basic":
        demo_basic_walk()
    elif mode == "compare":
        demo_comparison()
    elif mode == "guided":
        demo_guided_walk()
    elif mode == "all":
        demo_basic_walk()
        demo_comparison()
        demo_guided_walk()
    else:
        print(f"未知模式: {mode}")
        print("可用模式: basic, compare, guided, all")

    print("\n完成!")


if __name__ == "__main__":
    main()
