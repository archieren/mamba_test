# Mamba2 中 seq_idx 算法深度分析

> 基于 mamba_ssm v2.2.2 源代码分析

## 一句话概括

**`seq_idx` 告诉 Mamba2："在哪些位置需要重置 SSM 状态，让不同的序列互不干扰。"**

---

## 1. 核心原理：SSM 状态更新公式

### 1.1 普通 SSM 状态更新

```
h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t
```

### 1.2 加入 seq_idx 后

```python
if seq_idx[t] == seq_idx[t-1]:
    h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t  # 正常传播状态
else:
    h_t = 0 * h_{t-1} + dt * B_t * x_t            # 重置状态，从头开始
```

**本质**：当 `seq_idx` 发生变化时，把之前的状态清零，实现序列隔离。

---

## 2. 直观图解

```
时间步:    t=0   t=1   t=2   t=3   t=4   t=5   t=6   t=7
seq_idx:   [0,    0,    0,    1,    1,    2,    2,    2]
           └─── 序列0 ───┘└─ 序列1 ─┘└──── 序列2 ────┘

状态传播:
h0 → h1 → h2 ✗ h3 → h4 ✗ h5 → h6 → h7
              ↑         ↑
           边界重置   边界重置
```

在边界处（t=2→3, t=4→5），状态被重置为 0，不继承之前的信息。

---

## 3. 数据流与调用链

```
Mamba2.forward(u, seq_idx=seq_idx)
    │
    ▼
mamba_split_conv1d_scan_combined(..., seq_idx=seq_idx)  [ssd_combined.py]
    │
    ▼
MambaChunkScanCombinedFn.apply(x, dt, A, B, C, ..., seq_idx=seq_idx, ...)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  四个关键 Triton Kernels (并行执行)                           │
│  ├── _chunk_scan_fwd_kernel     (输出计算)                   │
│  ├── _chunk_state_fwd_kernel    (状态计算)                   │
│  ├── _state_passing_fwd_kernel  (状态传递)                   │
│  └── _bmm_chunk_fwd_kernel      (C*B 矩阵乘)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 四个关键 Kernel 的 seq_idx 处理

### 4.1 Chunk Scan Kernel (`ssd_chunk_scan.py`)

**作用**：计算输出，合并 chunk 内计算和跨 chunk 状态传播

```triton
# 加载当前和前一个位置的 seq_idx
if HAS_SEQ_IDX:
    seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                        mask=offs_m < chunk_size_limit, other=-1)

# 决定是否使用前一个 chunk 的状态
if not HAS_SEQ_IDX:
    scale_m = tl.exp(dA_cs_m)
else:
    scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
    #                                               ↑ 相同序列    ↑ 不同序列：状态置零
```

### 4.2 Chunk State Kernel (`ssd_chunk_state.py`)

**作用**：计算每个 chunk 的 SSM 状态

```triton
# 只累加同一序列内的状态贡献
if not HAS_SEQ_IDX:
    scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
else:
    scale = tl.where(seq_idx_k == seq_idx_last,
                     tl.exp((dA_cs_last - dA_cs_k)) * dt_k,  # 相同序列
                     0.0)                                     # 不同序列：贡献为零
b *= scale[:, None]
```

### 4.3 State Passing Kernel (`ssd_state_passing.py`) ⭐ 最核心

**作用**：在 chunk 之间顺序传递状态

```triton
if HAS_SEQ_IDX:
    seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
    scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
    #                                        ↑ 相同    ↑ 不同：scale=0，状态重置
    seq_idx = seq_idx_new
states = scale * states + new_states
```

### 4.4 BMM Kernel (`ssd_bmm.py`)

**作用**：计算 C*B 矩阵（类似 attention 的掩码机制）

```triton
if HAS_SEQ_IDX:
    seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, ...)  # 行的 seq_idx
    seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, ...)  # 列的 seq_idx
    # 创建掩码：只有相同 seq_idx 的位置才能互相 attend
    acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :],
                   acc,   # 相同序列：保留
                   0.0)   # 不同序列：掩码掉
```

---

## 5. 关键文件位置

| 文件 | 作用 |
|------|------|
| `mamba_ssm/modules/mamba2.py` | 高层 Mamba2 模块，`forward()` 接受 `seq_idx` |
| `mamba_ssm/ops/triton/ssd_combined.py` | 组合操作，将 `seq_idx` 传递给各 kernel |
| `mamba_ssm/ops/triton/ssd_chunk_scan.py` | Chunk scan kernel，输出计算 |
| `mamba_ssm/ops/triton/ssd_chunk_state.py` | Chunk state kernel，状态计算 |
| `mamba_ssm/ops/triton/ssd_state_passing.py` | State passing kernel，状态传递 |
| `mamba_ssm/ops/triton/ssd_bmm.py` | BMM kernel，C*B 矩阵乘 |

---

## 6. 在点云处理中的应用

### 6.1 多重序列化场景

```python
# 假设有 4 种排序策略，每种 1024 个点
num_orders = 4
num_points = 1024
d_model = 256

# 构造输入数据
x = torch.randn(batch_size, num_orders * num_points, d_model)  # (B, 4096, 256)

# 构造 seq_idx: 标识不同的排序策略
seq_idx = torch.tensor([
    [0]*1024 + [1]*1024 + [2]*1024 + [3]*1024  # 每种排序连续
] * batch_size).view(batch_size, -1)  # (B, 4096)

# Mamba2 处理
mamba2 = Mamba2(d_model=256, d_state=16, d_conv=4)
output = mamba2(x, seq_idx=seq_idx)

# 结果: 每种排序被独立处理，避免了不同排序间的状态泄露
```

### 6.2 本项目中的实际使用

```python
# pm/pointmamba/point_sis_masked_former.py
seq_idx = s_pc.batch.unsqueeze(0).int()  # (1, batch_size * num_groups)
seq_output = self.mixer_layers(seq_input, seq_idx=seq_idx)
```

每个点云组获得独立的序列 ID，确保不同组的 SSM 计算完全隔离。

### 6.3 多重序列化的优势

```
┌─────────────────────────────────────────────────────────┐
│  seq_idx 确保以下优势：                                   │
│                                                         │
│  1. 每种排序策略独立处理，不相互干扰                        │
│  2. 同一点的不同排序特征可以后续融合                        │
│  3. 避免了因为序列混杂导致的信息污染                        │
└─────────────────────────────────────────────────────────┘

处理流程:
  排序0 (Hilbert)  ──→  独立 SSM 计算  ──→  特征0
  排序1 (Morton)   ──→  独立 SSM 计算  ──→  特征1
  排序2 (Random)   ──→  独立 SSM 计算  ──→  特征2
  排序3 (Grid)     ──→  独立 SSM 计算  ──→  特征3
                                              │
                                              ▼
                                      特征融合 → 最终输出
```

---

## 7. 反向传播中的 seq_idx 处理

反向传播的 kernel 同样包含 `seq_idx` 处理，确保梯度也是隔离的：

```triton
# _state_passing_bwd_kernel
if HAS_SEQ_IDX:
    seq_idx_new = tl.load(seq_idx_ptr + ...)
    scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
```

这保证了不同序列之间的梯度不会相互影响。

---

## 8. 性能优化建议

### 8.1 seq_idx 设计原则

1. **连续性**: 同一个序列的 seq_idx 应该连续存储
2. **简洁性**: 使用从 0 开始的小整数
3. **局部性**: 相关的数据使用相近的 seq_id

### 8.2 正确 vs 错误示例

```python
# ❌ 错误 (低效，随机大数)
seq_idx_bad = torch.randint(0, 1000, (batch_size, seqlen))

# ✅ 正确 (高效，连续小整数)
seq_idx_good = torch.arange(num_sequences).repeat_interleave(seqlen // num_sequences)
# 结果: [0,0,0,..., 1,1,1,..., 2,2,2,...]
```

---

## 9. 总结

### seq_idx 的核心价值

| 特性 | 说明 |
|------|------|
| **状态隔离** | 确保 SSM 计算的正确性，避免不同序列间的状态污染 |
| **序列边界识别** | 自动检测序列切换点，实现状态重置 |
| **并行友好** | 支持多种序列化策略的并行处理 |
| **通用性** | 适用于文本、点云、时间序列等各种变长数据 |

### 实现机制总结

```
核心公式: scale = where(seq_same, scale, 0.0)

四个位置:
  1. Chunk Scan   → 输出计算时隔离
  2. Chunk State  → 状态累加时隔离
  3. State Pass   → 状态传递时隔离 (最核心)
  4. BMM          → 类 attention 掩码隔离
```

这使得 Mamba2 成为一个既能处理长序列，又能支持复杂序列结构的强大模型。
