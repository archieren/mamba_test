# 论文解读: A Fine-grained Orthodontics Segmentation Model for 3D Intraoral Scan Data

## 1. 论文基本信息

| 项目 | 内容 |
|------|------|
| 标题 | A fine-grained orthodontics segmentation model for 3D intraoral scan data |
| 作者 | Juncheng Li, Bodong Cheng, Najun Niu, Guangwei Gao, Shihui Ying, Jun Shi, Tieyong Zeng |
| 发表期刊 | Computers in Biology and Medicine, Volume 168, 2024 |
| DOI | 10.1016/j.compbiomed.2023.107821 |
| PMID | 38064844 |
| GitHub | https://github.com/MIVRC/Fast-TGCN |
| 引用次数 | 14 次 (截至 2026.01) |

**第一作者单位**: 上海大学通信与信息工程学院  
**通讯作者**: 李俊程 (junchengli@shu.edu.cn)

---

## 2. 研究背景与动机

### 2.1 数字化口腔医学的崛起

随着数字口腔医学在口腔疾病诊疗中的广泛应用，精确分割口内扫描数据中的牙齿成为越来越重要的研究课题。分割精度直接影响后续诊断质量。

### 2.2 现有方法的局限性

论文指出此前研究存在的两大核心问题：

1. **数据集来源局限**: 几乎所有 3D 口内扫描数据集都是**石膏模型的间接扫描**，而非真实的口内扫描数据
2. **异常牙样本不足**: 现有数据集仅包含有限的异常牙齿样本，难以应用于正畸治疗等复杂临床场景

这些问题导致已有的牙齿分割模型在真实的正畸诊疗环境中表现不佳。

### 2.3 研究目标

- 构建一个**统一标准化**的牙齿分割数据集，用于分析和验证牙齿分割方法的有效性
- 特别关注**变形牙（deformed teeth）**的分割问题
- 提出一种快速、准确的 3D 牙齿分割网络

---

## 3. 核心贡献

### 3.1 3D-IOSSeg 数据集

论文提出了一个精细的 3D 口内扫描分割数据集 **3D-IOSSeg**，具有以下特点：

- **数据规模**: 200+ 名患者的真实 3D 口内扫描数据
- **标注精度**: 每个样本都标注了**精细的 mesh 单元（fine-grained mesh unit）**
- **分类体系**: 上下颌的每颗牙齿都进行了细致分类
- **临床导向**: 特别包含复杂结构的牙齿，适用于正畸治疗场景

数据集获取链接: https://reurl.cc/0vjLXY

### 3.2 Fast-TGCN 模型

提出了一种用于 3D 牙齿分割的快速图卷积网络 **Fast-TGCN**（Fast Graph Convolutional Network），核心设计理念是：

> 通过原生邻接矩阵（naive adjacency matrix）直接建立相邻 mesh 单元之间的关系，更好地提取牙齿的局部几何特征。

---

## 4. 算法详解

### 4.1 整体架构

```
输入: 3D mesh
  ├── 坐标特征 (coor): xyz 坐标
  └── 法向量特征 (nor): 顶点法向量

  ↓ 两路并行分支

图卷积层堆叠 (GCN × 多层)
  ├── 坐标流 GCN
  └── 法向量流 GCN

  ↓ 特征融合 (AFF - Attentive Feature Fusion)

多层感知机分类头 (MLP)

  ↓

输出: 每个顶点的类别 (17 类: 上颌 8 颗牙 + 下颌 8 颗牙 + 牙龈)
```

### 4.2 邻接矩阵构建

这是图卷积的拓扑基础。从 mesh 的 `face`（三角形面片）信息构建邻接矩阵：

```python
def Adj_matrix_gen(face):
    B, N = face.shape[0], face.shape[1]  # B=batch, N=顶点数
    # 检查每对顶点是否共享同一个三角形
    adj_1_1 = (face[:, :, 0].unsqueeze(2) == face[:, :, 0].unsqueeze(1))  # 顶点i的第一个索引 == 顶点j的第一个索引
    adj_1_2 = (face[:, :, 0].unsqueeze(2) == face[:, :, 1].unsqueeze(1))
    adj_1_3 = (face[:, :, 0].unsqueeze(2) == face[:, :, 2].unsqueeze(1))
    adj_2_1 = (face[:, :, 1].unsqueeze(2) == face[:, :, 0].unsqueeze(1))
    adj_2_2 = (face[:, :, 1].unsqueeze(2) == face[:, :, 1].unsqueeze(1))
    adj_2_3 = (face[:, :, 1].unsqueeze(2) == face[:, :, 2].unsqueeze(1))
    adj_3_1 = (face[:, :, 2].unsqueeze(2) == face[:, :, 0].unsqueeze(1))
    adj_3_2 = (face[:, :, 2].unsqueeze(2) == face[:, :, 1].unsqueeze(1))
    adj_3_3 = (face[:, :, 2].unsqueeze(2) == face[:, :, 2].unsqueeze(1))
    
    adj = adj_1_1 + adj_1_2 + adj_1_3 + adj_2_1 + adj_2_2 + adj_2_3 + adj_3_1 + adj_3_2 + adj_3_3
    adj = (adj >= 1).float()  # 共享至少一个三角形则为邻接
    return adj
```

**物理意义**: 如果两个顶点出现在**同一个三角形面片**中，则它们是几何相邻的。矩阵中 `adj[i,j] = 1` 表示顶点 i 和顶点 j 相邻。

```python
# 在前向传播中可选地扩展邻域
adj = adj @ adj  # 二次扩展，引入二阶邻居信息
```

### 4.3 双路特征提取

输入 3D mesh 被分为两路并行处理：

| 分支 | 输入 | 处理 |
|------|------|------|
| 坐标流 (coor) | 顶点 xyz 坐标 | Conv1d → 多层 GCN |
| 法向量流 (nor) | 顶点法向量 | Conv1d → 多层 GCN |

两路结构对称但**权重独立**，确保能分别学习几何和法向特征。

### 4.4 图卷积模块 (GCN)

每层 GCN 的核心计算：

```python
class graph(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size):
        super(graph, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(outchannel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(outchannel, outchannel, kernel_size=1)
        )
    
    def forward(self, x, adj):
        x = self.conv(x) @ adj  # 矩阵乘法: (C_out, N) @ (N, N) → (C_out, N)
        return x
```

**图卷积的本质**:

```
(GCN · X)(i) = Σ_{j∈N(i)} W · X(j)
```

即：每个顶点接收其所有邻居顶点的特征，加权求和后通过线性变换。

### 4.5 特征融合 (AFF - Attentive Feature Fusion)

两路特征通过 AFF 模块进行自适应融合：

```python
class AFF(nn.Module):
    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        
        self.global_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
    
    def forward(self, x, y):
        xl = self.local_att(x)
        xg = self.global_att(y)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)
        
        xo = x * wei + y * (1 - wei)  # 自适应加权融合
        return xo
```

**融合机制**: 
- 局部注意力从 x 自身提取权重
- 全局注意力从 y 提取权重
- `sigmoid` 输出的权重决定最终融合比例

### 4.6 分类头

网络末端使用多层感知机进行逐点分类：

```python
self.pred1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(negative_slope=0.2))
self.pred2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(negative_slope=0.2))
self.pred3 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.2))
self.pred4 = nn.Sequential(nn.Linear(128, output_channels))  # output_channels=17

score = F.log_softmax(score, dim=2)  # Log-Softmax 输出
```

**17 类分类**:
- 上颌: 8 颗牙齿 (1-8, FDI notation)
- 下颌: 8 颗牙齿 (31-38 or 41-48, FDI notation)
- 牙龈: 1 类

### 4.7 完整前向传播流程

```python
def forward(self, x, index_face):
    adj = Adj_matrix_gen(index_face)
    adj = adj @ adj  # 二次扩展邻接矩阵
    
    coor = x[:, :12, :]   # 坐标特征
    nor = x[:, 12:, :]    # 法向量特征
    
    # 第一层 GCN
    coor1 = self.gcn_coor_1_1(coor, adj)
    coor1 = self.gcn_coor_1_2(coor1, adj)
    coor1 = self.gcn_coor_1_3(coor1, adj)
    nor1 = self.gcn_nor_1_1(nor, adj)
    coor_nor1 = self.aff_1(coor1, nor1)  # 特征融合
    
    # 中间层 (类似结构...)
    coor_nor2 = self.aff_2(coor2, nor2)
    coor_nor3 = self.aff_3(coor3, nor3)
    coor_nor4 = self.aff_4(coor4, nor4)
    
    # 特征拼接与融合
    x = torch.cat((coor_nor1, coor_nor2), dim=1)
    x = self.fu_1(x)
    # ... 继续分类
    
    score = self.pred4(x)
    return F.log_softmax(score, dim=2)
```

---

## 5. 关键公式与代码对照

| 数学概念 | 代码实现 | 说明 |
|----------|----------|------|
| 邻接矩阵 A | `Adj_matrix_gen(face)` | 从 mesh face 构建 |
| 图卷积 | `conv(x) @ adj` | 特征与邻接矩阵相乘 |
| 邻域聚合 | `x @ adj` | 每行是邻居特征之和 |
| 邻接扩展 | `adj @ adj` | 引入二阶邻居 |
| 特征融合 | `x * wei + y * (1-wei)` | AFF 加权融合 |
| 分类输出 | `F.log_softmax(score, dim=2)` | 逐点类别对数概率 |

---

## 6. 与其他方法的对比

| 方法 | 核心思路 | 优势 | 劣势 |
|------|----------|------|------|
| PointNet | 直接处理点云坐标 | 简单高效 | 忽略拓扑结构 |
| PointNet++ | 分层点云聚合 | 局部特征强 | 需额外学习邻域 |
| MeshSegNet | 专用 mesh 图卷积 | 针对 mesh 设计 | 复杂度较高 |
| **Fast-TGCN (本文)** | 原生邻接矩阵建立拓扑关系 | 轻量快速、精度高 | 依赖 mesh 质量 |

**论文的核心创新点**: 
- 用**原生邻接矩阵**直接编码 mesh 的拓扑关系，无需复杂的图构建过程
- 双路架构分别提取**几何**和**法向**特征
- AFF 注意力融合模块自适应整合两路信息

---

## 7. 训练配置

```python
# 数据集
train_dataset = plydataset("data/train-L", 'train', 'meshsegnet')
test_dataset = plydataset("data/test-L", 'test', 'meshsegnet')

# 模型
model = Baseline(in_channels=12, output_channels=17)  # 17 类分割
model.cuda()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 损失函数
criterion = F.nll_loss  # Negative Log Likelihood Loss

# 训练
for epoch in range(301):
    for data in train_loader:
        pred = model(coordinate, index_face)
        loss = F.nll_loss(pred, label_face)
        loss.backward()
        optimizer.step()
```

---

## 8. 总结与评价

### 8.1 主要贡献

1. **数据集层面**: 填补了正畸场景下精细牙齿分割数据集的空白，3D-IOSSeg 包含 200+ 患者的真实口内扫描数据

2. **方法层面**: 提出了利用 mesh 拓扑结构（图邻接矩阵）进行图卷积的分割模型 Fast-TGCN，兼具效率和精度

3. **工程层面**: 代码开源 (GitHub: MIVRC/Fast-TGCN)，便于复现和跟进

### 8.2 局限性

- 依赖高质量的 mesh 邻接矩阵，对噪声敏感的扫描数据可能效果下降
- 17 类分类体系相对固定，扩展到更多牙位或异常牙类别需要额外工作

### 8.3 启示

这篇论文展示了**如何利用 3D mesh 的拓扑信息**（邻接矩阵）进行有效的几何学习，与当前流行的点云 Transformer 方法形成互补。对于口腔扫描、牙科 CAD 等领域具有直接的临床应用价值。

---

## 9. 参考文献

1. Li, J., Cheng, B., Niu, N., Gao, G., Ying, S., Shi, J., & Zeng, T. (2024). A fine-grained orthodontics segmentation model for 3D intraoral scan data. *Computers in Biology and Medicine*, 168, 107821. https://doi.org/10.1016/j.compbiomed.2023.107821
