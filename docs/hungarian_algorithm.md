# 匈牙利匹配算法（Hungarian Algorithm）详解

## 算法概述

**匈牙利算法**（Hungarian Algorithm），也称为 **Kuhn-Munkres 算法**，是一种用于解决**二分图最大/最小权完美匹配**问题的经典算法。

### 核心问题

给定一个二分图，其中：
- 左部有 $n$ 个节点，右部有 $n$ 个节点
- 每条边都有一个权重（成本）
- 目标是找到一个**完美匹配**（每个节点都恰好匹配一次），使得**总权重最小**（或最大）

## 算法思想

匈牙利算法的核心思想是：
1. **增广路径**：通过不断寻找增广路径来增加匹配数
2. **标记技术**：使用标记（labeling）来指导搜索方向
3. **等价变换**：通过调整权重矩阵，保持最优解不变

## 算法步骤（最小权匹配）

### 步骤 1：初始化
- 构建成本矩阵 $C$，大小为 $n \times n$
- 初始化两个标记数组：$u[i]$（左部节点标记）和 $v[j]$（右部节点标记）
- 初始化匹配数组：$match[j]$ 表示右部节点 $j$ 匹配的左部节点

### 步骤 2：初始化标记
```python
# 对于最小权匹配，初始化左部标记为每行的最小值
u[i] = min(C[i][j] for j in range(n))
v[j] = 0  # 右部标记初始化为 0
```

### 步骤 3：寻找增广路径
使用 DFS 或 BFS 在**相等子图**中寻找增广路径：
- **相等子图**：只包含满足 $C[i][j] = u[i] + v[j]$ 的边
- **增广路径**：从未匹配的左部节点开始，交替经过匹配边和非匹配边，到达未匹配的右部节点

### 步骤 4：更新标记
如果找不到增广路径，调整标记：
```python
delta = min(C[i][j] - u[i] - v[j] for i in S, j not in T)
# 其中 S 是已访问的左部节点，T 是已访问的右部节点

for i in S:
    u[i] += delta
for j in T:
    v[j] -= delta
```

### 步骤 5：重复步骤 3-4
直到找到完美匹配为止

## 时间复杂度

- **朴素实现**：$O(n^4)$
- **优化实现**（使用 BFS/DFS 和标记技术）：$O(n^3)$
- **稀疏图优化**：可以进一步优化

## 算法伪代码

```
function Hungarian(C):
    n = size of C
    u = [min(C[i]) for i in range(n)]  # 左部标记
    v = [0] * n                         # 右部标记
    match = [-1] * n                    # 匹配数组
    
    for i in range(n):
        while True:
            visited_left = [False] * n
            visited_right = [False] * n
            parent = [-1] * n
            
            # 寻找增广路径
            if dfs(i, visited_left, visited_right, parent):
                break
            
            # 更新标记
            delta = find_min_delta(visited_left, visited_right)
            update_labels(u, v, visited_left, visited_right, delta)
    
    return match, total_cost
```

## 应用场景

### 1. 任务分配问题
- **场景**：$n$ 个工人，$n$ 个任务，每个工人完成每个任务有不同的成本
- **目标**：找到总成本最小的分配方案

### 2. 目标跟踪
- **场景**：视频中多目标跟踪，需要将当前帧的检测框与上一帧的轨迹匹配
- **成本**：通常是 IoU 距离或欧氏距离

### 3. 数据关联
- **场景**：传感器融合、多目标跟踪中的观测与预测关联
- **应用**：卡尔曼滤波中的数据关联步骤

### 4. 图像匹配
- **场景**：特征点匹配、图像配准
- **成本**：特征描述符之间的距离

### 5. 资源分配
- **场景**：云计算中的任务调度、网络资源分配

## Python 实现示例

### 简化版本（用于理解）

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm_simple(cost_matrix):
    """
    使用 scipy 实现的匈牙利算法（最小权匹配）
    
    Args:
        cost_matrix: n x n 的成本矩阵
    
    Returns:
        row_indices: 左部节点的匹配索引
        col_indices: 右部节点的匹配索引
        total_cost: 总成本
    """
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()
    return row_indices, col_indices, total_cost

# 示例
cost_matrix = np.array([
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2]
])

row_ind, col_ind, cost = hungarian_algorithm_simple(cost_matrix)
print(f"匹配: {list(zip(row_ind, col_ind))}")
print(f"总成本: {cost}")
```

### 完整实现（教学版本）

```python
import numpy as np

class HungarianAlgorithm:
    def __init__(self, cost_matrix):
        self.cost = np.array(cost_matrix, dtype=float)
        self.n = len(cost_matrix)
        self.u = np.zeros(self.n)  # 左部标记
        self.v = np.zeros(self.n)  # 右部标记
        self.match = [-1] * self.n  # match[j] = i 表示右部节点j匹配左部节点i
        self.parent = [-1] * self.n
    
    def solve(self):
        """求解最小权完美匹配"""
        # 初始化左部标记为每行的最小值
        for i in range(self.n):
            self.u[i] = np.min(self.cost[i])
        
        # 为每个左部节点寻找匹配
        for i in range(self.n):
            while True:
                visited_left = [False] * self.n
                visited_right = [False] * self.n
                self.parent = [-1] * self.n
                
                # 尝试寻找增广路径
                if self._dfs(i, visited_left, visited_right):
                    break
                
                # 更新标记
                delta = self._find_delta(visited_left, visited_right)
                self._update_labels(visited_left, visited_right, delta)
        
        # 计算总成本
        total_cost = sum(self.cost[i, self.match[i]] 
                        for i in range(self.n) if self.match[i] != -1)
        return self.match, total_cost
    
    def _dfs(self, i, visited_left, visited_right):
        """DFS 寻找增广路径"""
        visited_left[i] = True
        
        for j in range(self.n):
            if visited_right[j]:
                continue
            
            # 检查是否在相等子图中
            if abs(self.cost[i, j] - self.u[i] - self.v[j]) < 1e-10:
                visited_right[j] = True
                self.parent[j] = i
                
                # 如果j未匹配，或能找到增广路径
                if self.match[j] == -1 or self._dfs(self.match[j], 
                                                    visited_left, 
                                                    visited_right):
                    self.match[j] = i
                    return True
        
        return False
    
    def _find_delta(self, visited_left, visited_right):
        """找到最小的 delta 值用于更新标记"""
        delta = float('inf')
        for i in range(self.n):
            if visited_left[i]:
                for j in range(self.n):
                    if not visited_right[j]:
                        delta = min(delta, 
                                   self.cost[i, j] - self.u[i] - self.v[j])
        return delta
    
    def _update_labels(self, visited_left, visited_right, delta):
        """更新标记"""
        for i in range(self.n):
            if visited_left[i]:
                self.u[i] += delta
        for j in range(self.n):
            if visited_right[j]:
                self.v[j] -= delta

# 使用示例
cost_matrix = [
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2]
]

hungarian = HungarianAlgorithm(cost_matrix)
match, total_cost = hungarian.solve()
print(f"匹配结果: {match}")
print(f"总成本: {total_cost}")
```

## 在目标跟踪中的应用

### 多目标跟踪中的匹配

```python
def match_detections_to_tracks(detections, tracks, max_distance=0.5):
    """
    使用匈牙利算法将检测框与轨迹匹配
    
    Args:
        detections: 当前帧的检测框列表
        tracks: 上一帧的轨迹列表
        max_distance: 最大匹配距离
    
    Returns:
        matches: 匹配对列表 [(det_idx, track_idx), ...]
        unmatched_detections: 未匹配的检测框索引
        unmatched_tracks: 未匹配的轨迹索引
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))
    
    # 计算成本矩阵（IoU 距离）
    cost_matrix = compute_iou_distance(detections, tracks)
    
    # 如果检测数和轨迹数不等，需要填充
    n = max(len(detections), len(tracks))
    cost_matrix_padded = np.full((n, n), max_distance * 2)
    cost_matrix_padded[:len(detections), :len(tracks)] = cost_matrix
    
    # 使用匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix_padded)
    
    # 过滤掉成本过高的匹配
    matches = []
    unmatched_detections = set(range(len(detections)))
    unmatched_tracks = set(range(len(tracks)))
    
    for r, c in zip(row_ind, col_ind):
        if r < len(detections) and c < len(tracks):
            if cost_matrix[r, c] < max_distance:
                matches.append((r, c))
                unmatched_detections.discard(r)
                unmatched_tracks.discard(c)
    
    return matches, list(unmatched_detections), list(unmatched_tracks)
```

## 算法变种

### 1. 最大权匹配
将成本矩阵取负，然后使用最小权匹配算法

### 2. 非完美匹配
允许某些节点不匹配，可以通过添加虚拟节点实现

### 3. 非方阵匹配
当左右节点数不等时，可以填充虚拟节点

## 优化技巧

1. **稀疏图优化**：对于稀疏图，只考虑存在的边
2. **增量更新**：当成本矩阵小幅变化时，可以增量更新匹配
3. **并行化**：某些步骤可以并行处理
4. **近似算法**：对于大规模问题，可以使用近似算法

## 与其他算法的比较

| 算法 | 时间复杂度 | 适用场景 |
|------|-----------|---------|
| 匈牙利算法 | $O(n^3)$ | 二分图完美匹配，精确解 |
| 贪心算法 | $O(n^2 \log n)$ | 快速近似，可能不是最优 |
| 最小费用流 | $O(n^2 m)$ | 更通用的网络流问题 |
| 动态规划 | $O(2^n)$ | 小规模问题 |

## 总结

匈牙利算法是解决二分图匹配问题的经典算法，具有以下特点：

✅ **优点**：
- 保证找到最优解
- 时间复杂度相对较低（$O(n^3)$）
- 实现相对简单
- 应用广泛

⚠️ **限制**：
- 要求完美匹配（左右节点数相等）
- 对于非方阵需要填充
- 对于超大规模问题可能较慢

## 参考资料

1. Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
2. Munkres, J. (1957). Algorithms for the assignment and transportation problems.
3. scipy.optimize.linear_sum_assignment 文档
