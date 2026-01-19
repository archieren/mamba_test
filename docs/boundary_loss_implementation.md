# 点云下采样场景的边界损失实现

## 背景

在你的项目中：
- **原始点云**：N 个点（可能几十万点）
- **下采样后**：G 个点（比如 32768 个点，通过 FPS 或 GridPooling）
- **模型预测**：在下采样的 G 个点上预测 mask 和类别
- **最终输出**：通过 `FeatPropagation` 插值回原始点云

**边界损失应该在下采样的点上计算**，因为：
1. 模型在这些点上做预测
2. 这些点已经代表了点云的结构
3. 计算效率更高

## 实现方案

### 1. 边界点检测函数

```python
from pointops import knn_query as knn
import torch
import torch.nn.functional as F

def compute_point_boundary(
    labels: torch.Tensor,           # (B*G,) 或 (B, G) - 每个点的标签（类别ID）
    coords: torch.Tensor,            # (B*G, 3) - 点的坐标
    offset: torch.Tensor,             # (B,) - batch 的 offset，用于 KNN 查询
    k: int = 6,                      # KNN 的邻居数
    boundary_threshold: float = 0.7   # 边界判断阈值：如果 < 70% 的邻居是同类别，认为是边界点
) -> torch.Tensor:
    """
    在下采样的点云上检测边界点
    
    Args:
        labels: 点标签，形状可以是 (B*G,) 或 (B, G)
        coords: 点坐标 (B*G, 3)
        offset: batch offset (B,)
        k: KNN 邻居数
        boundary_threshold: 边界判断阈值（邻居中同类别比例 < threshold 认为是边界）
    
    Returns:
        boundary_mask: (B*G,) 或 (B, G)，1 表示边界点，0 表示非边界点
    """
    # 确保 labels 是 1D
    if labels.dim() == 2:
        labels = labels.view(-1)
    
    # KNN 查询：找到每个点的 k 个邻居
    # knn 返回: (neighbor_indices, distances)
    # neighbor_indices: (B*G, k)
    neighbor_indices, _ = knn(k, coords, offset, coords, offset)
    
    # 获取邻居的标签
    neighbor_labels = labels[neighbor_indices]  # (B*G, k)
    
    # 当前点的标签，扩展维度以便比较
    current_labels = labels.unsqueeze(1)  # (B*G, 1)
    
    # 计算每个点的邻居中，有多少个是同类别
    same_class = (neighbor_labels == current_labels).float()  # (B*G, k)
    same_ratio = same_class.mean(dim=1)  # (B*G,)
    
    # 如果同类别比例 < threshold，认为是边界点
    boundary_mask = (same_ratio < boundary_threshold).float()
    
    return boundary_mask
```

### 2. 边界损失实现

```python
def boundary_loss(
    pred_mask_logits: torch.Tensor,    # (B, Q, G) - 预测的 mask logits
    target_mask: torch.Tensor,         # (B, G) 或 (T, G) - 真实 mask（二值）
    target_labels: torch.Tensor,       # (B, G) 或 (T,) - 真实标签（用于检测边界）
    coords: torch.Tensor,              # (B*G, 3) - 点坐标
    offset: torch.Tensor,               # (B,) - batch offset
    indices: List[Tuple],               # 匈牙利匹配的结果
    k: int = 6,
    boundary_threshold: float = 0.7,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    计算边界损失
    
    Args:
        pred_mask_logits: 预测的 mask，形状 (B, Q, G)
        target_mask: 真实的 mask，可能是 (B, G) 或匹配后的 (T, G)
        target_labels: 真实标签，用于检测边界点
        coords: 点坐标 (B*G, 3)
        offset: batch offset (B,)
        indices: 匈牙利匹配结果
        k: KNN 邻居数
        boundary_threshold: 边界判断阈值
        loss_weight: 损失权重
    
    Returns:
        loss: 边界损失标量
    """
    # 1. 根据匹配结果，获取匹配后的预测和标签
    src_idx = _get_predictions_permutation_indices(indices)
    pred_masks = pred_mask_logits[src_idx]  # (T, G) - T 是匹配后的数量
    
    # 2. 获取匹配后的目标 mask 和 labels
    target_masks = torch.cat([
        target[target_indices] 
        for target, (_, target_indices) in zip(target_mask, indices)
    ])  # (T, G)
    
    target_labels_matched = torch.cat([
        labels[target_indices]
        for labels, (_, target_indices) in zip(target_labels, indices)
    ])  # (T, G) 或 (T,)
    
    # 3. 检测边界点（基于真实标签）
    if target_labels_matched.dim() == 2:
        # 如果是 (T, G)，需要展平
        target_labels_flat = target_labels_matched.view(-1)  # (T*G,)
        coords_expanded = coords.repeat_interleave(
            target_labels_matched.size(0), dim=0
        )  # (T*G, 3) - 每个目标重复 G 次
        offset_expanded = torch.arange(
            target_labels_matched.size(0), 
            device=coords.device
        ) * target_labels_matched.size(1)  # (T,)
    else:
        target_labels_flat = target_labels_matched  # (T,)
        coords_expanded = coords
        offset_expanded = offset
    
    boundary_mask = compute_point_boundary(
        target_labels_flat,
        coords_expanded,
        offset_expanded,
        k=k,
        boundary_threshold=boundary_threshold
    )  # (T*G,) 或 (T, G)
    
    if boundary_mask.dim() == 1 and boundary_mask.size(0) == target_labels_matched.numel():
        boundary_mask = boundary_mask.view(target_labels_matched.shape)
    
    # 4. 只在边界点上计算损失
    # 将 boundary_mask 应用到 pred_masks 和 target_masks
    boundary_mask_expanded = boundary_mask.unsqueeze(0)  # (1, T, G) 或 (1, T)
    
    # 只对边界点计算 BCE
    pred_sigmoid = torch.sigmoid(pred_masks)  # (T, G)
    target_masks_float = target_masks.float()
    
    # 边界损失：只在边界点上计算
    bce = F.binary_cross_entropy_with_logits(
        pred_masks, 
        target_masks_float, 
        reduction='none'
    )  # (T, G)
    
    # 只保留边界点的损失
    boundary_bce = bce * boundary_mask_expanded  # (T, G)
    
    # 归一化：除以边界点数量
    num_boundary_points = boundary_mask_expanded.sum()
    if num_boundary_points > 0:
        loss = boundary_bce.sum() / (num_boundary_points + 1e-8)
    else:
        # 如果没有边界点，返回 0
        loss = torch.tensor(0.0, device=pred_masks.device)
    
    return loss * loss_weight
```

### 3. 简化版本（更实用）

考虑到你的项目结构，这里是一个**更实用的简化版本**：

```python
def boundary_loss_simple(
    pred_mask_logits: torch.Tensor,    # (T, G) - 匹配后的预测 mask
    target_mask: torch.Tensor,         # (T, G) - 匹配后的真实 mask
    target_labels: torch.Tensor,       # (T, G) - 匹配后的真实标签
    coords: torch.Tensor,              # (B*G, 3) - 原始坐标（需要根据匹配调整）
    offset: torch.Tensor,              # (B,) - batch offset
    k: int = 6,
    boundary_threshold: float = 0.7,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    简化版边界损失
    
    注意：这个版本假设 pred_mask_logits 和 target_mask 已经是匹配后的形状 (T, G)
    """
    T, G = pred_mask_logits.shape
    
    # 1. 检测边界点（基于真实标签）
    # 将 (T, G) 展平为 (T*G,)
    target_labels_flat = target_labels.view(-1)  # (T*G,)
    
    # 坐标也需要对应展平（每个目标重复 G 次）
    # 这里假设 coords 是 (B*G, 3)，需要根据实际情况调整
    # 简化：假设每个 batch 的坐标相同，重复 T 次
    coords_flat = coords.repeat(T, 1)  # (T*G, 3) - 简化处理
    
    # 构造 offset（每个目标一个 batch）
    offset_flat = torch.arange(T, device=coords.device) * G  # (T,)
    
    # 检测边界点
    boundary_mask = compute_point_boundary(
        target_labels_flat,
        coords_flat,
        offset_flat,
        k=k,
        boundary_threshold=boundary_threshold
    )  # (T*G,)
    
    boundary_mask = boundary_mask.view(T, G)  # (T, G)
    
    # 2. 计算边界损失（只在边界点上）
    pred_sigmoid = torch.sigmoid(pred_mask_logits)  # (T, G)
    target_float = target_mask.float()  # (T, G)
    
    # Dice loss on boundary
    boundary_pred = pred_sigmoid * boundary_mask  # (T, G)
    boundary_target = target_float * boundary_mask  # (T, G)
    
    intersection = (boundary_pred * boundary_target).sum(dim=1)  # (T,)
    union = boundary_pred.sum(dim=1) + boundary_target.sum(dim=1)  # (T,)
    
    dice = (2 * intersection + 1.0) / (union + 1.0 + 1e-8)  # (T,)
    loss = (1 - dice).mean()  # 平均
    
    return loss * loss_weight
```

### 4. 集成到 PMLoss 中

在 `losses.py` 的 `PMLoss` 类中添加：

```python
class PMLoss(nn.Module):
    def __init__(self, config: PointSISConfig):
        super().__init__()
        # ... 现有代码 ...
        
        # 边界损失配置
        self.use_boundary_loss = getattr(config, 'use_boundary_loss', False)
        self.boundary_weight = getattr(config, 'boundary_weight', 1.0)
        self.boundary_k = getattr(config, 'boundary_k', 6)
        self.boundary_threshold = getattr(config, 'boundary_threshold', 0.7)
    
    def loss_masks(self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
        shape_weight: torch.Tensor = None,
        coords: torch.Tensor = None,      # 新增：点坐标
        offset: torch.Tensor = None,      # 新增：batch offset
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks"""
        # ... 现有代码 ...
        
        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
            "loss_geo": geo_loss(pred_masks, target_masks, num_masks, target_shape_weight),
        }
        
        # 添加边界损失
        if self.use_boundary_loss and coords is not None and offset is not None:
            # 需要获取匹配后的标签用于边界检测
            target_labels_matched = torch.cat([
                labels[target_indices]
                for labels, (_, target_indices) in zip(class_labels, indices)
            ])
            
            losses["loss_boundary"] = boundary_loss_simple(
                pred_masks,
                target_masks,
                target_labels_matched,
                coords,
                offset,
                k=self.boundary_k,
                boundary_threshold=self.boundary_threshold,
                loss_weight=self.boundary_weight
            )
        
        return losses
    
    def forward(self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        labels: torch.Tensor,
        shape_weight: torch.Tensor = None,
        coords: torch.Tensor = None,      # 新增
        offset: torch.Tensor = None,      # 新增
    ) -> Dict[str, torch.Tensor]:
        # ... 现有代码 ...
        
        losses = {
            **self.loss_masks(
                masks_queries_logits, mask_labels, indices, num_masks, 
                shape_weights, coords, offset  # 传递坐标信息
            ),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        
        return losses
```

### 5. 在模型 forward 中传递坐标信息

在 `point_sis_masked_former.py` 中：

```python
def forward(self, s_pc:PointCloud):
    # ... 现有代码 ...
    
    if "labels" in s_pc.keys():
        labels = rearrange(s_pc.labels, "(b g) -> b g", b=b_s)
        shape_weight = rearrange(s_pc.shape_weight, "(b g) -> b g", b=b_s) if s_pc.shape_weight is not None else None
        
        # 传递坐标和 offset 给 loss
        m_i = self.loss(
            pred_mask, pred_probs, labels, shape_weight,
            coords=s_pc.coord,      # 新增
            offset=s_pc.offset       # 新增
        )
        s_pc.loss = m_i
```

## 使用建议

1. **参数调优**：
   - `k=6`：KNN 邻居数，可以尝试 4-10
   - `boundary_threshold=0.7`：边界判断阈值，可以尝试 0.6-0.8
   - `boundary_weight=1.0`：边界损失权重，建议从 0.5 开始

2. **性能考虑**：
   - KNN 查询有计算开销，如果点很多，可以考虑：
     - 减少 `k` 的值
     - 或者只在训练时使用，推理时不用

3. **效果验证**：
   - 可视化边界点，看看检测是否合理
   - 对比有无边界损失的训练效果

## 总结

这个实现：
- ✅ 适合点云下采样场景
- ✅ 使用 KNN 检测边界点
- ✅ 只在边界点上计算损失
- ✅ 可以集成到现有的 PMLoss 中

---

# 点云聚集性损失（Clustering Loss）

## 为什么需要聚集性损失？

**边界损失**关注的是：
- ✅ 不同实例之间的边界准确性
- ✅ 边缘区域的分割质量

**但边界损失无法解决**：
- ❌ 同一实例内部的离散点（孤立点）
- ❌ 实例内部的孔洞（不连续性）
- ❌ 预测mask的空间连续性问题

**聚集性损失**的作用：
- 🎯 确保同一颗牙齿的点在3D空间中连续分布
- 🎯 减少孤立噪声点
- 🎯 提高形状完整性

**两者互补，缺一不可！**

---

## 方案1：连通性损失（推荐）

### 核心思想
对于每个点，查看它的 k 个最近邻：
- 如果当前点是前景（牙齿），那么它的邻居也应该倾向于前景
- 鼓励邻近点有相似的预测，提高空间连续性

### 代码实现

```python
def connectivity_loss(
    pred_mask: torch.Tensor,      # (T, G) - 预测 mask（匹配后）
    coords: torch.Tensor,         # (B*G, 3) - 点坐标
    offset: torch.Tensor,         # (B,) - batch offset
    k: int = 6,                   # KNN 邻居数
    loss_weight: float = 0.5
) -> torch.Tensor:
    """
    连通性损失：鼓励同一实例内的邻近点有相似的预测

    核心思想：
    - 对于每个点，查看它的 k 个最近邻
    - 如果当前点是前景，邻居也应该倾向于前景
    - 使用 MSE 惩罚不一致的预测
    """
    from pointops import knn_query as knn

    T, G = pred_mask.shape
    B = offset.size(0)

    # 获取预测概率
    pred_prob = torch.sigmoid(pred_mask)  # (T, G)

    total_loss = 0.0
    count = 0

    # 对每个实例和每个 batch 处理
    for t in range(T):
        for b in range(B):
            # 获取当前 batch 的坐标和预测
            start_idx = offset[b] if b == 0 else 0
            end_idx = offset[b] if b < B - 1 else G

            batch_coords = coords[start_idx:end_idx]  # (G, 3)
            batch_prob = pred_prob[t, start_idx:end_idx]  # (G,)

            # KNN 查询
            # 构造简单的 offset
            batch_offset = torch.arange(1, device=coords.device)

            neighbor_idx, _ = knn(
                k, batch_coords,
                batch_offset,
                batch_coords,
                batch_offset
            )  # (G, k)

            # 获取邻居的预测概率
            neighbor_prob = batch_prob[neighbor_idx]  # (G, k)

            # 计算每个点与其邻居预测的差异
            current_prob_expanded = batch_prob.unsqueeze(1)  # (G, 1)

            # MSE：希望当前点和邻居的预测一致
            prob_diff = (current_prob_expanded - neighbor_prob) ** 2  # (G, k)
            prob_diff = prob_diff.mean(dim=1)  # (G,)

            # 只对前景区域（高置信度）计算
            # 避免背景区域的干扰
            foreground_mask = (batch_prob > 0.3)

            if foreground_mask.sum() > 0:
                loss = prob_diff[foreground_mask].mean()
                total_loss += loss
                count += 1

    if count > 0:
        return total_loss / count * loss_weight
    else:
        return torch.tensor(0.0, device=pred_mask.device)
```

---

## 方案2：聚类损失（更简单）

### 核心思想
惩罚前景点的孤立性：
- 对于每个前景点，计算它的 k 个邻居中有多少也是前景
- 如果一个前景点的邻居都是背景，说明它是孤立的，应该惩罚

### 代码实现

```python
def clustering_loss(
    pred_mask: torch.Tensor,      # (T, G) - 预测 mask（匹配后）
    coords: torch.Tensor,         # (B*G, 3) - 点坐标
    offset: torch.Tensor,         # (B,) - batch offset
    k: int = 6,
    loss_weight: float = 0.5
) -> torch.Tensor:
    """
    聚类损失：惩罚前景点的孤立性

    核心思想：
    - 对于每个前景点，计算它的 k 个邻居中有多少也是前景
    - 如果前景点的邻居都是背景，说明它是孤立的，应该惩罚
    """
    from pointops import knn_query as knn

    T, G = pred_mask.shape
    B = offset.size(0)

    pred_prob = torch.sigmoid(pred_mask)  # (T, G)
    pred_binary = (pred_prob > 0.5).float()  # (T, G)

    total_loss = 0.0
    count = 0

    for t in range(T):
        for b in range(B):
            start_idx = offset[b] if b == 0 else 0
            end_idx = offset[b] if b < B - 1 else G

            batch_coords = coords[start_idx:end_idx]  # (G, 3)
            batch_binary = pred_binary[t, start_idx:end_idx]  # (G,)

            # KNN 查询
            batch_offset = torch.arange(1, device=coords.device)

            neighbor_idx, _ = knn(
                k, batch_coords,
                batch_offset,
                batch_coords,
                batch_offset
            )  # (G, k)

            # 获取邻居的标签
            neighbor_binary = batch_binary[neighbor_idx]  # (G, k)

            # 对于每个前景点，计算邻居中前景的比例
            current_binary = batch_binary.unsqueeze(1)  # (G, 1)

            # 只对前景点计算
            foreground_mask = (batch_binary == 1)

            if foreground_mask.sum() > 0:
                # 前景点的邻居中，前景的比例
                # 取出所有前景点的邻居信息
                foreground_neighbor_labels = neighbor_binary[foreground_mask]  # (N_fg, k)

                # 计算每个前景点的邻居中前景的比例
                foreground_ratio = foreground_neighbor_labels.float().mean(dim=1)  # (N_fg,)

                # 如果比例低，说明前景点是孤立的，需要惩罚
                # loss = 1 - ratio，比例越低，loss越高
                loss = (1.0 - foreground_ratio).mean()

                total_loss += loss
                count += 1

    if count > 0:
        return total_loss / count * loss_weight
    else:
        return torch.tensor(0.0, device=pred_mask.device)
```

---

## 方案3：组合损失（边界 + 聚集性）

### 核心思想
同时关注边界和内部完整性：
- 边界损失：提高边界定位准确性
- 聚集性损失：提高内部连续性

### 代码实现

```python
def shape_consistency_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    target_labels: torch.Tensor,
    coords: torch.Tensor,
    offset: torch.Tensor,
    boundary_weight: float = 1.0,
    clustering_weight: float = 0.5,
    k: int = 6,
    boundary_threshold: float = 0.7
) -> torch.Tensor:
    """
    组合损失：边界损失 + 聚集性损失

    Args:
        pred_mask: 预测 mask (T, G)
        target_mask: 真实 mask (T, G)
        target_labels: 真实标签 (T, G)
        coords: 点坐标 (B*G, 3)
        offset: batch offset (B,)
        boundary_weight: 边界损失权重
        clustering_weight: 聚集性损失权重
        k: KNN 邻居数
        boundary_threshold: 边界判断阈值

    Returns:
        total_loss: 组合损失
    """
    # 1. 边界损失（关注边界）
    loss_boundary = boundary_loss_simple(
        pred_mask, target_mask, target_labels,
        coords, offset,
        k=k,
        boundary_threshold=boundary_threshold,
        loss_weight=1.0
    )

    # 2. 聚集性损失（关注内部）
    loss_clustering = clustering_loss(
        pred_mask, coords, offset,
        k=k,
        loss_weight=1.0
    )

    # 3. 组合
    total_loss = loss_boundary * boundary_weight + loss_clustering * clustering_weight

    return total_loss
```

---

## 方案4：基于距离的加权聚集性损失（高级版）

### 核心思想
考虑邻居的距离信息：
- 距离越近的邻居，权重应该越大
- 使用高斯核函数加权距离

### 代码实现

```python
def distance_weighted_clustering_loss(
    pred_mask: torch.Tensor,      # (T, G)
    coords: torch.Tensor,         # (B*G, 3)
    offset: torch.Tensor,         # (B,)
    k: int = 6,
    sigma: float = 0.1,           # 高斯核的带宽
    loss_weight: float = 0.5
) -> torch.Tensor:
    """
    基于距离的加权聚集性损失

    核心思想：
    - 距离越近的邻居，影响应该越大
    - 使用高斯核函数加权距离
    """
    from pointops import knn_query as knn

    T, G = pred_mask.shape
    B = offset.size(0)

    pred_prob = torch.sigmoid(pred_mask)  # (T, G)

    total_loss = 0.0
    count = 0

    for t in range(T):
        for b in range(B):
            start_idx = offset[b] if b == 0 else 0
            end_idx = offset[b] if b < B - 1 else G

            batch_coords = coords[start_idx:end_idx]  # (G, 3)
            batch_prob = pred_prob[t, start_idx:end_idx]  # (G,)

            # KNN 查询（获取距离）
            batch_offset = torch.arange(1, device=coords.device)

            neighbor_idx, distances = knn(
                k, batch_coords,
                batch_offset,
                batch_coords,
                batch_offset
            )  # neighbor_idx: (G, k), distances: (G, k)

            # 获取邻居的预测概率
            neighbor_prob = batch_prob[neighbor_idx]  # (G, k)

            # 当前点的预测
            current_prob = batch_prob.unsqueeze(1)  # (G, 1)

            # 计算预测差异
            prob_diff = (current_prob - neighbor_prob) ** 2  # (G, k)

            # 基于距离的权重：距离越近，权重越大
            # 使用高斯核：weight = exp(-distance^2 / (2 * sigma^2))
            weights = torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))  # (G, k)

            # 加权损失
            weighted_loss = (prob_diff * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # (G,)

            # 只对前景区域计算
            foreground_mask = (batch_prob > 0.3)

            if foreground_mask.sum() > 0:
                loss = weighted_loss[foreground_mask].mean()
                total_loss += loss
                count += 1

    if count > 0:
        return total_loss / count * loss_weight
    else:
        return torch.tensor(0.0, device=pred_mask.device)
```

---

## 集成到 PMLoss 中

### 完整的集成代码

```python
class PMLoss(nn.Module):
    def __init__(self, config: PointSISConfig):
        super().__init__()
        # ... 现有代码 ...

        # 边界损失配置
        self.use_boundary_loss = getattr(config, 'use_boundary_loss', False)
        self.boundary_weight = getattr(config, 'boundary_weight', 1.0)
        self.boundary_k = getattr(config, 'boundary_k', 6)
        self.boundary_threshold = getattr(config, 'boundary_threshold', 0.7)

        # 聚集性损失配置（新增）
        self.use_clustering_loss = getattr(config, 'use_clustering_loss', False)
        self.clustering_weight = getattr(config, 'clustering_weight', 0.5)
        self.clustering_k = getattr(config, 'clustering_k', 6)
        self.clustering_type = getattr(config, 'clustering_type', 'connectivity')  # 'connectivity' or 'clustering'

    def loss_masks(self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
        shape_weight: torch.Tensor = None,
        coords: torch.Tensor = None,
        offset: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks"""
        # ... 现有代码，获取 pred_masks 和 target_masks ...

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
            "loss_geo": geo_loss(pred_masks, target_masks, num_masks, target_shape_weight),
        }

        # 1. 添加边界损失
        if self.use_boundary_loss and coords is not None and offset is not None:
            target_labels_matched = torch.cat([
                labels[target_indices]
                for labels, (_, target_indices) in zip(class_labels, indices)
            ])

            losses["loss_boundary"] = boundary_loss_simple(
                pred_masks,
                target_masks,
                target_labels_matched,
                coords,
                offset,
                k=self.boundary_k,
                boundary_threshold=self.boundary_threshold,
                loss_weight=self.boundary_weight
            )

        # 2. 添加聚集性损失（新增）
        if self.use_clustering_loss and coords is not None and offset is not None:
            if self.clustering_type == 'connectivity':
                losses["loss_clustering"] = connectivity_loss(
                    pred_masks,
                    coords,
                    offset,
                    k=self.clustering_k,
                    loss_weight=self.clustering_weight
                )
            elif self.clustering_type == 'clustering':
                losses["loss_clustering"] = clustering_loss(
                    pred_masks,
                    coords,
                    offset,
                    k=self.clustering_k,
                    loss_weight=self.clustering_weight
                )
            elif self.clustering_type == 'distance_weighted':
                losses["loss_clustering"] = distance_weighted_clustering_loss(
                    pred_masks,
                    coords,
                    offset,
                    k=self.clustering_k,
                    sigma=0.1,
                    loss_weight=self.clustering_weight
                )

        return losses

    def forward(self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        labels: torch.Tensor,
        shape_weight: torch.Tensor = None,
        coords: torch.Tensor = None,
        offset: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # ... 现有代码 ...

        losses = {
            **self.loss_masks(
                masks_queries_logits, mask_labels, indices, num_masks,
                shape_weights, coords, offset
            ),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }

        return losses
```

---

## 配置文件示例

```yaml
# 在 config.yaml 或 config.py 中添加

# 边界损失配置
use_boundary_loss: True
boundary_weight: 1.0
boundary_k: 6
boundary_threshold: 0.7

# 聚集性损失配置
use_clustering_loss: True
clustering_weight: 0.5  # 聚集性损失权重通常低于边界损失
clustering_k: 6
clustering_type: 'connectivity'  # 可选: 'connectivity', 'clustering', 'distance_weighted'
```

---

## 实验策略

### 阶段1：基线（无额外损失）
```python
config.use_boundary_loss = False
config.use_clustering_loss = False
```
观察基线性能。

### 阶段2：只加边界损失
```python
config.use_boundary_loss = True
config.boundary_weight = 1.0
config.use_clustering_loss = False
```
观察边界损失的效果。

### 阶段3：边界 + 聚集性
```python
config.use_boundary_loss = True
config.boundary_weight = 1.0
config.use_clustering_loss = True
config.clustering_weight = 0.5
```
观察组合效果。

### 阶段4：调参
```python
# 调整聚集性损失权重
config.clustering_weight = 0.3  # 如果过度平滑
config.clustering_weight = 0.7  # 如果离散点仍然存在

# 调整 KNN 邻居数
config.clustering_k = 4  # 更局部
config.clustering_k = 10  # 更全局

# 尝试不同类型
config.clustering_type = 'distance_weighted'  # 考虑距离权重
```

---

## 参数调优建议

### 1. 聚集性损失权重（clustering_weight）
- **0.1 - 0.3**：轻微约束，适用于已经较好的结果
- **0.5 - 0.7**：中等约束，推荐起始值
- **1.0**：强约束，可能过度平滑

### 2. KNN 邻居数（k）
- **k=4**：关注非常局部的连续性
- **k=6**：平衡，推荐值
- **k=10**：考虑更大的邻域，更全局的连续性

### 3. 聚集性损失类型选择
| 类型 | 适用场景 | 计算开销 | 效果 |
|------|---------|---------|------|
| **connectivity** | 通用 | 中等 | 推荐 |
| **clustering** | 离散点严重 | 低 | 简单 |
| **distance_weighted** | 需要精细控制 | 高 | 最优 |

---

## 效果评估方法

### 1. 定量评估
```python
# 计算以下指标
- 孤立点数量：预测mask中，邻居都是背景的前景点数量
- 连通区域数量：使用连通分量分析，数量越少越好
- 平均孔洞面积：前景区域内的背景孔洞
```

### 2. 可视化检查
```python
# 可视化预测mask
- 用不同颜色标注孤立点
- 可视化孔洞区域
- 对比使用聚集性损失前后的差异
```

---

## 总结对比

| 损失类型 | 关注点 | 解决的问题 | 推荐权重 |
|---------|-------|-----------|---------|
| **边界损失** | 边界区域 | 边界定位不准确 | 1.0 |
| **聚集性损失** | 内部区域 | 离散点、孔洞 | 0.5 |
| **Dice Loss** | 整体重叠度 | 整体分割不准确 | 1.0 |
| **Cross Entropy** | 像素级分类 | 分类错误 | 1.0 |

### 推荐配置
```python
# 保守配置（从这开始）
use_boundary_loss = True
boundary_weight = 1.0
use_clustering_loss = True
clustering_weight = 0.3
clustering_type = 'connectivity'

# 激进配置（如果离散点严重）
use_boundary_loss = True
boundary_weight = 1.0
use_clustering_loss = True
clustering_weight = 0.7
clustering_type = 'clustering'
```

---

## 注意事项

1. **不要过度约束**：
   - 聚集性损失权重过高可能导致过度平滑
   - 牙齿的某些区域（如牙根分叉）本身就不是完全连通的

2. **计算开销**：
   - KNN 查询有额外计算开销
   - 如果训练太慢，可以只在训练时使用，推理时不用

3. **与现有损失的关系**：
   - Dice loss 本身已经隐含了一些连续性约束
   - 聚集性损失是对 Dice loss 的补充，而非替代

4. **调试技巧**：
   - 先在小的验证集上测试
   - 可视化边界点和孤立点
   - 逐步增加权重，观察效果

需要我帮你把聚集性损失集成到代码里吗？
