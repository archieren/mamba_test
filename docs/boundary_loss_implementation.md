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

需要我帮你集成到代码里吗？
