# 点云分割中的边界损失计算方法

## 目录
1. [什么是边界损失](#什么是边界损失)
2. [边界检测方法](#边界检测方法)
3. [边界损失计算公式](#边界损失计算公式)
4. [点云分割中的实现](#点云分割中的实现)
5. [代码示例](#代码示例)
6. [与现有损失的结合](#与现有损失的结合)

---

## 什么是边界损失

### 定义
**边界损失（Boundary Loss）** 是一种专门关注分割边界的损失函数，用于提升实例分割任务中的边界精度。

### 为什么需要边界损失？

```
问题：传统损失（如 Dice Loss、BCE）主要关注整体区域重叠

示例：
┌─────────────────────┐
│  Ground Truth       │  ██████（真实牙齿）
│  ██████             │
│                     │
┌─────────────────────┐
│  Prediction         │  ██████████（预测偏大）
│  ████████           │
│                     │
┌─────────────────────┐
│  Overlap Analysis   │
│  ██████     ██████  │  ← 内部重叠好
│  ██████     ██████  │
│  ═════════════════  │  ← 边界不准
└─────────────────────┘

结果：Dice Loss 很高（内部重叠好），但边界不准确！
```

### 边界损失的优势

| 特性 | 传统损失 (Dice/BCE) | 边界损失 |
|------|---------------------|----------|
| 内部区域 | ✅ 很好关注 | ⚠️ 关注较少 |
| 边界区域 | ⚠️ 关注不足 | ✅ 专门优化 |
| 对小边界敏感 | ❌ 不敏感 | ✅ 高敏感 |
| 训练稳定性 | ✅ 稳定 | ⚠️ 需要平衡权重 |

---

## 边界检测方法

### 方法 1：基于邻接关系的边界检测（点云专用）

```python
def compute_boundary_point_cloud(points, mask, k=20):
    """
    基于邻接关系检测点云边界

    参数:
        points: (N, 3) 点云坐标
        mask: (N,) 二值 mask (0 或 1)
        k: 邻居数量

    返回:
        boundary_mask: (N,) 边界 mask
    """
    from sklearn.neighbors import NearestNeighbors

    # 找 k 近邻
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    boundary_mask = torch.zeros_like(mask, dtype=torch.float32)

    for i in range(len(points)):
        if mask[i] == 0:  # 只考虑前景点
            continue

        # 获取邻居的标签
        neighbor_labels = mask[indices[i][1:]]  # 排除自己

        # 如果邻居中有不同标签，则该点是边界点
        if (neighbor_labels != mask[i]).any():
            boundary_mask[i] = 1.0

    return boundary_mask
```

### 方法 2：基于形态学操作的边界检测

```python
def compute_boundary_morphology(mask, kernel_size=3):
    """
    基于形态学操作检测边界（适用于 2D/3D 体素化点云）

    参数:
        mask: (H, W) 或 (D, H, W) 的 mask
        kernel_size: 膨胀核大小

    返回:
        boundary_mask: 边界 mask
    """
    import torch.nn.functional as F

    # 膨胀操作
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
    dilated = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0).float(),
        kernel,
        padding=kernel_size//2
    ) > 0

    # 边界 = 膨胀 - 原始
    boundary = (dilated.squeeze() != mask).float()

    return boundary
```

### 方法 3：基于距离变换的边界检测

```python
def compute_boundary_distance(mask, threshold=2.0):
    """
    基于距离变换检测边界

    参数:
        mask: (N,) 二值 mask
        threshold: 距离阈值

    返回:
        boundary_mask: (N,) 边界 mask
    """
    from scipy.ndimage import distance_transform_edt

    # 计算前景到背景的距离
    distance_map = distance_transform_edt(mask.cpu().numpy())

    # 距离小于阈值的是边界
    boundary_mask = (distance_map < threshold).astype(np.float32)

    return torch.from_numpy(boundary_mask).to(mask.device)
```

### 方法 4：基于曲率的边界检测（高级）

```python
def compute_boundary_curvature(points, mask, curvature_threshold=0.3):
    """
    基于曲率检测边界（适用于牙齿等有明确边界的对象）

    参数:
        points: (N, 3) 点云坐标
        mask: (N,) 二值 mask
        curvature_threshold: 曲率阈值

    返回:
        boundary_mask: (N,) 边界 mask
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    nbrs = NearestNeighbors(n_neighbors=20).fit(points.cpu().numpy())
    _, indices = nbrs.kneighbors(points.cpu().numpy())

    curvatures = []
    boundary_mask = torch.zeros_like(mask, dtype=torch.float32)

    for i in range(len(points)):
        if mask[i] == 0:
            continue

        # 计算局部曲率（简化版：使用邻居点的协方差）
        neighbor_points = points[indices[i][1:]]
        centered = neighbor_points - neighbor_points.mean(dim=0)
        cov = torch.mm(centered.T, centered) / len(neighbor_points)

        # 曲率与最小特征值相关
        eigenvalues, _ = torch.linalg.eigh(cov)
        curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-8)

        if curvature > curvature_threshold:
            boundary_mask[i] = 1.0

    return boundary_mask
```

---

## 边界损失计算公式

### 1. **Boundary Dice Loss**（推荐）

最常用，对边界不平衡鲁棒

$$
\mathcal{L}_{\text{boundary-dice}} = 1 - \frac{2 \sum_{i} B_i^{(p)} B_i^{(gt)} + \epsilon}{\sum_{i} B_i^{(p)} + \sum_{i} B_i^{(gt)} + \epsilon}
$$

其中：
- $B_i^{(p)}$ 是预测边界的第 $i$ 个点
- $B_i^{(gt)}$ 是真实边界的第 $i$ 个点
- $\epsilon$ 是平滑项（防止除零）

**代码实现：**
```python
def boundary_dice_loss(pred_boundary, target_boundary, eps=1e-8):
    """
    Boundary Dice Loss

    参数:
        pred_boundary: (N,) 预测边界 mask (0 或 1)
        target_boundary: (N,) 真实边界 mask (0 或 1)

    返回:
        loss: 标量损失值
    """
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum()

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice
```

### 2. **Boundary BCE Loss**

逐点二元交叉熵损失

$$
\mathcal{L}_{\text{boundary-bce}} = -\frac{1}{N}\sum_{i} \left[ B_i^{(gt)} \log(\sigma(B_i^{(p)})) + (1-B_i^{(gt)}) \log(1-\sigma(B_i^{(p)})) \right]
$$

**代码实现：**
```python
def boundary_bce_loss(pred_logits, target_boundary):
    """
    Boundary BCE Loss

    参数:
        pred_logits: (N,) 预测 logit（未 sigmoid）
        target_boundary: (N,) 真实边界 mask

    返回:
        loss: 标量损失值
    """
    return F.binary_cross_entropy_with_logits(
        pred_logits,
        target_boundary,
        reduction='mean'
    )
```

### 3. **Boundary Focal Loss**

关注难分的边界点

$$
\mathcal{L}_{\text{boundary-focal}} = -\frac{1}{N}\sum_{i} \alpha_t (1-p_t)^\gamma \log(p_t)
$$

其中：
- $p_t = \begin{cases} p & \text{if } y=1 \\ 1-p & \text{if } y=0 \end{cases}$
- $\alpha_t = \begin{cases} \alpha & \text{if } y=1 \\ 1-\alpha & \text{if } y=0 \end{cases}$

**代码实现：**
```python
def boundary_focal_loss(pred_logits, target_boundary, alpha=0.25, gamma=2.0):
    """
    Boundary Focal Loss

    参数:
        pred_logits: (N,) 预测 logit
        target_boundary: (N,) 真实边界 mask
        alpha: 平衡参数
        gamma: 聚焦参数

    返回:
        loss: 标量损失值
    """
    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_logits,
        target_boundary,
        reduction='none'
    )

    p_t = p * target_boundary + (1 - p) * (1 - target_boundary)
    alpha_t = alpha * target_boundary + (1 - alpha) * (1 - target_boundary)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = (focal_weight * ce_loss).mean()
    return loss
```

### 4. **Hausdorff Distance Loss**（高级）

基于 Hausdorff 距离的边界损失

$$
\mathcal{L}_{\text{hausdorff}} = \max\left( \sup_{x \in \partial P} \inf_{y \in \partial GT} d(x,y), \sup_{y \in \partial GT} \inf_{x \in \partial P} d(x,y) \right)
$$

**近似实现（更高效）：**
```python
def hausdorff_distance_loss(pred_points, target_points, num_samples=1024):
    """
    近似 Hausdorff 距离损失

    参数:
        pred_points: (N, 3) 预测边界点坐标
        target_points: (M, 3) 真实边界点坐标
        num_samples: 采样数量（加速计算）

    返回:
        loss: 标量损失值
    """
    # 随机采样（加速）
    if pred_points.shape[0] > num_samples:
        idx = torch.randperm(pred_points.shape[0])[:num_samples]
        pred_points = pred_points[idx]
    if target_points.shape[0] > num_samples:
        idx = torch.randperm(target_points.shape[0])[:num_samples]
        target_points = target_points[idx]

    # 计算距离矩阵
    dist_matrix = torch.cdist(pred_points, target_points)

    # 双向 Hausdorff
    h1 = dist_matrix.min(dim=1)[0].max()  # pred -> target
    h2 = dist_matrix.min(dim=0)[0].max()  # target -> pred

    return (h1 + h2) / 2
```

---

## 点云分割中的实现

### 完整的边界损失模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class BoundaryLoss(nn.Module):
    """
    点云分割的边界损失

    支持的边界检测方法:
        - 'adjacency': 基于邻接关系（推荐用于点云）
        - 'distance': 基于距离变换

    支持的损失类型:
        - 'dice': Boundary Dice Loss（推荐）
        - 'bce': Boundary BCE Loss
        - 'focal': Boundary Focal Loss
        - 'hausdorff': Hausdorff Distance Loss
    """

    def __init__(
        self,
        boundary_method='adjacency',
        loss_type='dice',
        k_neighbors=20,
        distance_threshold=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_weight=1.0,
        **kwargs
    ):
        super().__init__()
        self.boundary_method = boundary_method
        self.loss_type = loss_type
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight

    def compute_boundary(self, points, mask):
        """
        计算 mask 的边界

        参数:
            points: (B, N, 3) 点云坐标
            mask: (B, N) 二值 mask

        返回:
            boundary: (B, N) 边界 mask
        """
        batch_size, num_points = points.shape[:2]
        boundary_list = []

        for b in range(batch_size):
            pts = points[b].cpu().numpy()  # (N, 3)
            msk = mask[b].cpu().numpy()    # (N,)

            if self.boundary_method == 'adjacency':
                # 方法 1: 邻接关系
                boundary = self._boundary_by_adjacency(pts, msk)
            elif self.boundary_method == 'distance':
                # 方法 2: 距离变换
                boundary = self._boundary_by_distance(msk)
            else:
                raise ValueError(f"Unknown boundary method: {self.boundary_method}")

            boundary_list.append(torch.from_numpy(boundary).float())

        return torch.stack(boundary_list).to(points.device)

    def _boundary_by_adjacency(self, points, mask):
        """基于邻接关系的边界检测"""
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(points)
        _, indices = nbrs.kneighbors(points)

        boundary = np.zeros_like(mask, dtype=np.float32)

        for i in range(len(points)):
            if mask[i] == 0:
                continue

            # 检查邻居是否有不同标签
            neighbor_labels = mask[indices[i][1:]]
            if (neighbor_labels != mask[i]).any():
                boundary[i] = 1.0

        return boundary

    def _boundary_by_distance(self, mask):
        """基于距离变换的边界检测"""
        from scipy.ndimage import distance_transform_edt

        # 将 mask reshape 为体素网格（简化版）
        # 实际应用中可能需要更复杂的体素化
        distance_map = distance_transform_edt(mask.reshape(-1, 1))
        distance_map = distance_map.reshape(-1)

        boundary = (distance_map < self.distance_threshold).astype(np.float32)
        return boundary

    def forward(self, pred_mask, target_mask, points=None):
        """
        计算边界损失

        参数:
            pred_mask: (B, N) 预测 mask (logits)
            target_mask: (B, N) 真实 mask
            points: (B, N, 3) 点云坐标（某些边界检测方法需要）

        返回:
            loss: 标量损失值
        """
        # 计算 target 的边界
        if points is not None and self.boundary_method == 'adjacency':
            target_boundary = self.compute_boundary(points, target_mask)
        else:
            # 如果没有 points，使用简单的距离方法
            target_boundary = self.compute_boundary(points, target_mask)

        # 计算 pred 的边界
        pred_probs = torch.sigmoid(pred_mask)
        pred_binary = (pred_probs > 0.5).float()

        if points is not None and self.boundary_method == 'adjacency':
            pred_boundary = self.compute_boundary(points, pred_binary)
        else:
            pred_boundary = self.compute_boundary(points, pred_binary)

        # 计算边界损失
        if self.loss_type == 'dice':
            loss = self._dice_loss(pred_boundary, target_boundary)
        elif self.loss_type == 'bce':
            loss = self._bce_loss(pred_mask, target_boundary)
        elif self.loss_type == 'focal':
            loss = self._focal_loss(pred_mask, target_boundary)
        elif self.loss_type == 'hausdorff':
            loss = self._hausdorff_loss(pred_mask, target_boundary, points)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss * self.boundary_weight

    def _dice_loss(self, pred, target):
        """Boundary Dice Loss"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice

    def _bce_loss(self, pred_logits, target):
        """Boundary BCE Loss"""
        return F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

    def _focal_loss(self, pred_logits, target):
        """Boundary Focal Loss"""
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        loss = (focal_weight * ce_loss).mean()
        return loss

    def _hausdorff_loss(self, pred_mask, target_boundary, points):
        """Hausdorff Distance Loss（简化版）"""
        # 获取边界点坐标
        pred_boundary = (torch.sigmoid(pred_mask) > 0.5).float()

        pred_boundary_points = []
        target_boundary_points = []

        for b in range(pred_mask.shape[0]):
            if points is None:
                continue

            pred_boundary_points.append(points[b][pred_boundary[b] == 1])
            target_boundary_points.append(points[b][target_boundary[b] == 1])

        if len(pred_boundary_points) == 0 or len(target_boundary_points) == 0:
            return torch.tensor(0.0, device=pred_mask.device)

        # 计算平均 Hausdorff 距离
        total_loss = 0.0
        count = 0

        for pred_pts, target_pts in zip(pred_boundary_points, target_boundary_points):
            if pred_pts.shape[0] == 0 or target_pts.shape[0] == 0:
                continue

            dist_matrix = torch.cdist(pred_pts, target_pts)
            h1 = dist_matrix.min(dim=1)[0].max()
            h2 = dist_matrix.min(dim=0)[0].max()
            total_loss += (h1 + h2) / 2
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred_mask.device)

        return total_loss / count
```

---

## 代码示例

### 示例 1：在你的 PointSIS 项目中使用

```python
# 在 pm/pointmamba/point_sis_masked_former.py 中添加

from pm.pointmamba.boundary_loss import BoundaryLoss

class PMLoss(nn.Module):
    def __init__(self, config: PointSISConfig):
        super().__init__()
        # ... 现有代码 ...

        # 添加边界损失
        self.boundary_loss = BoundaryLoss(
            boundary_method='adjacency',  # 使用邻接关系
            loss_type='dice',             # 使用 Dice Loss
            k_neighbors=20,
            boundary_weight=1.0           # 权重
        )

        # 损失权重
        self.boundary_weight = config.boundary_weight  # 在配置中添加

    def loss_masks(
        self,
        inputs: torch.Tensor,
        target_masks: torch.Tensor,
        points: torch.Tensor,  # 添加点云坐标参数
        indices: torch.Tensor,
        num_masks: int,
    ):
        """计算 mask 相关损失（包括边界损失）"""
        # ... 现有代码 ...

        # 原有损失
        losses = {
            "loss_mask": self.mask_weight * loss_mask,
            "loss_dice": self.dice_weight * loss_dice,
        }

        # 添加边界损失
        if indices.numel() > 0 and num_masks > 0:
            # 获取匹配后的预测和目标
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)

            # 对每个 query 计算边界损失
            boundary_losses = []
            for i in range(len(src_idx[0])):
                pred_idx = src_idx[0][i], src_idx[1][i]
                tgt_idx_i = tgt_idx[0][i], tgt_idx[1][i]

                pred_mask = inputs[pred_idx]
                target_mask = target_masks[tgt_idx_i]
                pts = points[src_idx[0][i]]  # (N, 3)

                boundary_loss = self.boundary_loss(pred_mask, target_mask, pts)
                boundary_losses.append(boundary_loss)

            if len(boundary_losses) > 0:
                losses["loss_boundary"] = (
                    self.boundary_weight * torch.stack(boundary_losses).mean()
                )

        return losses
```

### 示例 2：训练时使用

```python
# 训练循环
for batch in dataloader:
    points = batch['points']      # (B, N, 3)
    masks = batch['masks']        # (B, N, num_instances)
    labels = batch['labels']      # (B, num_instances)

    # 前向传播
    outputs = model(points)
    pred_masks = outputs['pred_masks']  # (B, num_queries, N)

    # 计算损失（包含边界损失）
    losses = criterion(pred_masks, masks, labels, points)

    # 反向传播
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

    # 记录损失
    print(f"Total Loss: {losses['total_loss']:.4f}")
    print(f"Boundary Loss: {losses.get('loss_boundary', 0):.4f}")
```

### 示例 3：配置文件

```python
# 在 pm/pointmamba/conifuguration_point_sis.py 中添加

@dataclass
class PointSISConfig:
    # ... 现有配置 ...

    # 边界损失配置
    use_boundary_loss: bool = True
    boundary_weight: float = 1.0
    boundary_method: str = 'adjacency'  # 'adjacency' or 'distance'
    boundary_loss_type: str = 'dice'    # 'dice', 'bce', 'focal', 'hausdorff'
    boundary_k_neighbors: int = 20
```

---

## 与现有损失的结合

### 推荐的损失组合

```python
# 总损失
total_loss = (
    class_weight * loss_cross_entropy +    # 类别损失
    mask_weight * loss_mask +              # Mask BCE
    dice_weight * loss_dice +              # Dice
    geo_weight * loss_geo +                # 几何损失（关键点）
    boundary_weight * loss_boundary        # 边界损失（新增）
)
```

### 损失权重建议

```python
# 初始训练阶段（前 50% epochs）
class_weight = 2.0
mask_weight = 5.0
dice_weight = 5.0
geo_weight = 1.0
boundary_weight = 0.5   # 边界损失从较小权重开始

# 稳定训练阶段（后 50% epochs）
class_weight = 2.0
mask_weight = 5.0
dice_weight = 5.0
geo_weight = 1.0
boundary_weight = 1.0   # 逐渐增加边界损失权重
```

### 课程学习策略

```python
def get_boundary_weight(epoch, max_epochs, warmup_ratio=0.3):
    """
    动态调整边界损失权重

    参数:
        epoch: 当前 epoch
        max_epochs: 总 epoch 数
        warmup_ratio: 预热比例

    返回:
        boundary_weight: 边界损失权重
    """
    warmup_epochs = int(max_epochs * warmup_ratio)

    if epoch < warmup_epochs:
        # 预热阶段：从 0 逐渐增加到 0.5
        return 0.5 * (epoch / warmup_epochs)
    else:
        # 主训练阶段：逐渐增加到 1.0
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.5 + 0.5 * progress

# 使用示例
for epoch in range(max_epochs):
    boundary_weight = get_boundary_weight(epoch, max_epochs)
    # 更新损失权重
    criterion.boundary_weight = boundary_weight
```

---

## 总结

### 关键要点

1. **边界检测方法选择**
   - ✅ **点云任务**：使用邻接关系方法（`boundary_method='adjacency'`）
   - ✅ **体素化数据**：使用形态学或距离变换方法
   - ✅ **计算效率**：邻接关系最直接，适合点云

2. **边界损失类型选择**
   - ✅ **推荐**：Boundary Dice Loss（`loss_type='dice'`）
   - ✅ **边界不平衡**：使用 Dice Loss
   - ✅ **难样本多**：使用 Focal Loss
   - ⚠️ **高精度要求**：使用 Hausdorff Loss（计算成本高）

3. **权重设置**
   - 初始训练：`boundary_weight=0.5`（避免干扰整体学习）
   - 稳定训练：`boundary_weight=1.0`（增强边界精度）
   - 调试建议：从 0.5 开始，逐步增加

4. **与现有损失的关系**
   - 边界损失**补充** Dice/BCE 损失，不替代
   - Dice Loss 关注整体重叠
   - Boundary Loss 关注边界精度
   - 两者结合可以获得更好的分割效果

### 实现检查清单

- [ ] 边界检测方法是否正确实现
- [ ] 边界损失权重是否合理设置
- [ ] 是否添加了损失监控（`loss_boundary`）
- [ ] 是否验证了边界计算的正确性（可视化边界点）
- [ ] 是否考虑了课程学习策略

### 下一步

1. **实现边界损失模块**（参考上面的 `BoundaryLoss` 类）
2. **集成到现有损失函数**（在 `PMLoss` 中添加）
3. **调整权重和超参数**
4. **监控训练效果**（对比有无边界损失的性能）
5. **可视化边界**（验证边界检测是否正确）

---

**参考文献**：
- [1] Kervadec et al. "Boundary loss for highly unbalanced segmentation." MICCAI 2021.
- [2] Siddique et al. "A survey on deep learning based boundary detection for medical image segmentation." IEEE Access 2021.
- [3] Ma et al. "Boundary-aware networks for semantic segmentation." CVPR 2022.
