# Query Diversity Loss 实现文档

## 概述

本文档描述了 **Query Diversity Loss** 的实现，用于防止 PointSIS 分割模型中的 query 坍缩问题。Query 坍缩指的是多个 query 最终都聚焦在同一个目标（如同一颗牙齿）上，导致预测结果冗余。

## 问题描述

### 现象

训练过程中，多个 query 收敛到识别同一颗牙齿，表现为：
- 尽管有多个 query，却只检测到一颗牙齿
- Mask 预测的多样性低
- 场景中目标覆盖率低

### 根本原因

1. **随机 Query 初始化**：所有 query 从随机向量开始，初始就没有多样性
2. **位置编码不对称**：Cross-Attention 中，位置编码只加在 query 上，没有加在 encoder keys 上
3. **Self-Attention 强化效应**：相似的 query 在自注意力层中会变得更相似
4. **缺少显式多样性约束**：分割 loss 只惩罚错误预测，不惩罚相似预测

## 解决方案：Query Diversity Loss

### 原理

Diversity loss 通过最小化归一化 query 之间的成对余弦相似度，来促使不同 query 具有不同的特征表示。

优化目标：

$$\min \sum_{i \neq j} \text{sim}(q_i, q_j)$$

其中余弦相似度：

$$\text{sim}(q_i, q_j) = \frac{q_i \cdot q_j}{\|q_i\| \|q_j\|}$$

### 数学形式

```python
def _compute_query_diversity_loss(query_emb):
    # 归一化 query 向量
    q_norm = F.normalize(query_emb, p=2, dim=-1)  # [b, q, d]

    # 计算成对余弦相似度
    sim = torch.bmm(q_norm, q_norm.transpose(1, 2))  # [b, q, q], 范围 [-1, 1]

    # 遮盖对角线（自身与自身的相似度）
    mask = torch.eye(q_norm.shape[1], device=q_norm.device).bool()
    sim = sim.masked_fill(mask, 0)

    # 使用 sim^2.mean()：sim→0 时 loss 最小（鼓励多样性），sim→±1 时 loss 最大（惩罚坍缩）
    return sim.pow(2).mean()
```

其中优化目标为：

$$\mathcal{L}_{div} = \frac{1}{q(q-1)} \sum_{i \neq j} \text{sim}(q_i, q_j)^2$$

- 当 $\text{sim} = \pm 1$ 时（完全相同或完全相反），$\mathcal{L}_{div} = 1$（最大，惩罚坍缩）
- 当 $\text{sim} = 0$ 时（正交/多样），$\mathcal{L}_{div} = 0$（最小，鼓励多样性）
- Loss 值域为 $[0, 1]$，非负

## 实现细节

### 修改的文件

1. **`pm/pointmamba/losses.py`**
   - `PMLoss.forward()` 新增 `query_embeddings` 参数
   - 新增 `_compute_query_diversity_loss()` 方法
   - 将 diversity loss 整合到 loss 字典中

2. **`pm/pointmamba/point_sis_masked_former.py`**
   - 调用 `self.loss()` 时传入 `query` 参数

### 代码改动

#### losses.py - forward() 函数签名

```python
def forward(
    self,
    masks_queries_logits: torch.Tensor,        # b q g
    class_queries_logits: torch.Tensor,        # b q l
    labels: torch.Tensor,                      # b g
    shape_weight: torch.Tensor=None,           # b g
    is_mp_query: torch.Tensor=None,            # b q
    query_embeddings: torch.Tensor=None        # b q d (新增)
) -> Dict[str, Tensor]:
```

#### losses.py - diversity loss 计算

```python
if query_embeddings is not None and is_mp_query is not None:
    is_original = ~is_mp_query
    original_queries = query_embeddings[:, is_original[0], :]
    div_loss = self._compute_query_diversity_loss(original_queries)
    losses["loss_diversity"] = div_loss * 0.1  # 权重 = 0.1
```

#### losses.py - 新增方法

```python
def _compute_query_diversity_loss(self, query_emb: torch.Tensor) -> torch.Tensor:
    """
    鼓励不同 query 关注不同特征，防止 query collapse。
    query_emb: [b, q, d]
    """
    q_norm = F.normalize(query_emb, p=2, dim=-1)
    sim = torch.bmm(q_norm, q_norm.transpose(1, 2))
    mask = torch.eye(q_norm.shape[1], device=q_norm.device).bool()
    sim.masked_fill_(mask, 0)
    return sim.pow(2).mean()
```

#### point_sis_masked_former.py - loss 调用

```python
# 修改前
m_i = self.loss(pred_mask, pred_probs, labels, shape_weight, is_mp_query)

# 修改后
m_i = self.loss(pred_mask, pred_probs, labels, shape_weight, is_mp_query, query)
```

## Loss 权重

Diversity loss 的权重默认为 **0.1**：

```python
losses["loss_diversity"] = div_loss * 0.1
```

### 调参建议

| 权重范围 | 效果 |
|---------|------|
| $0.01 \sim 0.05$ | 轻微多样性鼓励，可能无法防止坍缩 |
| $0.1 \sim 0.3$ | 平衡（推荐起始值） |
| $0.5 \sim 1.0$ | 强制多样性，可能损害分割精度 |
| $> 1.0$ | 可能导致 query 过度分散，影响主 loss |

### 调整方法

如果 query 仍然坍缩，尝试增大权重：

```python
losses["loss_diversity"] = div_loss * 0.3  # 或 0.5
```

如果分割精度下降，尝试减小权重：

```python
losses["loss_diversity"] = div_loss * 0.05  # 或 0.01
```

## 作用范围：仅限原始 Query

Diversity loss **只对原始 query 计算**，不作用于 MP (Mask-Piloted) query：

```python
is_original = ~is_mp_query  # True 为原始 query，False 为 MP query
original_queries = query_embeddings[:, is_original[0], :]
```

**原因**：MP query 是从带噪声的 mask 生成，已有的明确监督信号。如果对它们也加强制多样性约束，可能会干扰其学习正确的特征。

## 预期效果

实现 diversity loss 后：
1. Query 应该分散到不同的牙齿/目标上
2. Attention 可视化应显示多样化的关注区域
3. Hungarian 匹配应有更多有效匹配
4. 分割精度应有所提升

## 其他方案

### 1. 基于空间位置的 Query 初始化

不采用随机初始化，而是从点云空间通过 FPS 采样：

```python
q_idx = fps(s_pc.coord, s_pc.offset, q_offset)
query = gathered_feats[0][q_idx]
```

### 2. Margin-Based 多样性

强制 query 之间的最小距离：

$$\mathcal{L} = \text{ReLU}(\text{margin} - (1 - \text{sim}(q_i, q_j)))$$

```python
loss = F.relu(margin - (1 - similarity)).mean()
```

### 3. Attention 熵 Loss

鼓励 attention 分散到多个位置：

$$\mathcal{L} = -\sum_h \sum_q \sum_g p_{hqg} \log p_{hqg}$$

其中 $p_{hqg}$ 是第 $h$ 个 head、第 $q$ 个 query 对第 $g$ 个位置的 attention 权重。

```python
attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(-1)
loss = -attn_entropy.mean()  # 最大化熵 = 鼓励分散
```

## 参考资料

- **MoCo / SimCLR**：对比学习用于特征多样性
- **DETR**：匈牙利匹配实现一对一分配
- **Mask2Former**：中间预测的辅助 loss