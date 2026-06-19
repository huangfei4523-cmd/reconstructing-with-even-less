# GNN 增强攻击方法 — 设计与原理分析

## 1. 设计动机与问题分析

### 1.1 原始方法的局限

论文 *Reconstructing with Even Less* 的核心攻击流程：

```
范围查询响应
  ↓ fast_augment_responses: 两两取交集放大信息
  ↓ reduce_to_domain_points: 将点映射到最小响应集
  ↓ find_prime_responses(size=2): 筛出大小为 2 的素数响应
  ↓ make_simple_graph: 建图
```

**核心瓶颈：** 仅利用大小为 2 的直接交集推断邻接关系。在低采样率（p=1%）下，大多数点对可能从未同时出现在恰好 2 个点的响应中，导致大量真实边无法恢复。

### 1.2 GNN 方法的改进思路

GNN 方法的核心改进：**从所有响应大小中共现模式进行统计学习**。

```
原始:  只关注 size=2 的确定性交集 → Recall 低
GNN:   学习所有 size 的共现模式 → 更多信息利用
```

具体来说，即使点 A 和 B 从未同时出现在 size=2 的响应中（所以原始方法无法发现它们），它们在多条 size>2 的响应中频繁共现意味着空间上接近——GNN 可以从这种共现模式的统计特征中推断出邻接关系。

## 2. 架构演进

### 2.1 v0 架构（矩阵输入，已被废弃）

```
共现矩阵 [N, N]
    ↓
node_encoder: Linear(N → 64) → ReLU → Linear(64 → 32)
    ↑ 权重矩阵 [64, N]，N 变化则形状不匹配
    ↓
消息传递: Linear(32→32) @ (k-NN_adj @ h)
    ↓
边预测: concat(emb_i, emb_j) → MLP
    ↓
edge_logits [N, N]
```

**致命缺陷:** `Linear(N, 64)` 的权重矩阵形状随数据集点数 N 变化。训练用 10×10 合成网格（N≈80），推理用 cali_50（N≈1000），维度必然不匹配。

### 2.2 v1 架构（固定特征输入，当前版本）

```
共现矩阵 [N, N]
    ↓
┌─ extract_node_features ────────────────────────┐
│  统计量提取: 均值、标准差、稀疏度、中心性等      │
│  输出: [N, 16] 固定维度特征 (与 N 无关)          │
└────────────────────────────────────────────────┘
    ↓
┌─ build_message_passing_graph ──────────────────┐
│  余弦相似度 k-NN 图 → edge_index [2, E] 稀疏格式 │
└────────────────────────────────────────────────┘
    ↓
┌─ EdgePredictionGNN ────────────────────────────┐
│                                                 │
│  NodeEncoder:                                   │
│    Linear(16→64) → BatchNorm → ReLU             │
│    Linear(64→32) → BatchNorm → ReLU             │
│                                                 │
│  GraphSAGE MessagePassing ×2:                   │
│    h_i = Linear([h_i, mean_{j∈N(i)} h_j])       │
│          → LayerNorm → ReLU                     │
│                                                 │
│  EdgePredictor:                                 │
│    MLP([emb_i, emb_j, edge_feat_ij])            │
│    68 → 64 → 32 → 16 → 1                       │
│                                                 │
│  所有 Linear 层输入维度固定！                    │
└────────────────────────────────────────────────┘
    ↓
边概率 → threshold → 建图
```

**关键改进:** 所有 `nn.Linear` 输入维度与 N 解耦。16 维统计特征包含了共现向量中的结构信息，GraphSAGE 消息传递进一步从邻居聚合中恢复空间上下文，4 维边特征在边预测时增强 pairwise 判别力。

## 3. 数据流

### 3.1 训练阶段

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ grid_2d  │   │random_2d │   │random_3d │  场景配置
│ (40%)    │   │ (30%)    │   │ (30%)    │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     └───────────────┼───────────────┘
                     ▼
            ┌─────────────────┐
            │ 随机生成点集     │ 坐标 (x,y) 或 (x,y,z)
            │ 分配随机 token   │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 枚举范围查询     │ 四重/六重循环
            │ 5% 采样模拟      │
            │ "Even Less"     │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 计算共现矩阵     │ [N, N] float32
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ extract_node    │ → [N, 16] 节点特征
            │ _features()     │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ Focal Loss 训练  │ pos_weight + (1-pt)²
            │ Adam + CosLR    │ grad_clip=1.0
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 保存 checkpoint  │ feature_dim, hidden_dim,
            │                 │ emb_dim, num_mp_layers,
            │                 │ model_state_dict
            └─────────────────┘
```

### 3.2 推理阶段

```
[数据集加载]
     │
     ├── scale_points + make_database_from_points
     │
     ▼
[响应生成 + 采样] ← process_database（与原版兼容）
     │
     ▼
┌─────────────────────────────────────┐
│ compute_cooccurrence_matrix()       │
│   → cooc_normalized [N,N]           │
│   → all_points, token_to_idx        │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
extract_node          build_message
_features()           _passing_graph()
    │                     │
[node_feat]           [edge_index]
[N,16]                [2,E] 稀疏
    │                     │
    └──────────┬──────────┘
               ▼
    ┌──────────────────────┐
    │ EdgePredictionGNN    │
    │  .forward()          │
    │  .predict_all_pairs()│
    └──────────┬───────────┘
               ▼
         边概率矩阵 [N,N]
               │
          threshold=0.5
               │
               ▼
         构建 NetworkX 图
               │
               ▼
    ┌──────────────────────┐
    │ check_accuracy       │  Precision / Recall
    │ _with_edges()        │
    └──────────────────────┘
```

## 4. 关键算法

### 4.1 节点特征提取 `extract_node_features(cooc_matrix)`

**输入:** [N, N] 共现矩阵  
**输出:** [N, 16] 固定维度统计特征

对每个点 i，从共现向量 `vec = cooc_matrix[i, :]` 提取：

| 特征 | 计算方法 | 含义 |
|------|----------|------|
| F1 | vec.mean() / N | 平均共现度 |
| F2 | vec.max() / N | 最大共现度 |
| F3 | nonzero(vec) / N | 共现稀疏度 |
| F4 | vec.std() / N | 标准差 |
| F5 | nonzero(vec) / N | 响应频次(近似) |
| F6 | sum(vec) / N² | 总强度归一化 |
| F7 | nonzero(vec) / N | 度中心性(近似) |
| F8 | 0.0 | 聚类系数(保留) |
| F9 | 1 / (nonzero + eps) | 稀疏度倒数 |
| F10 | sum(vec) / N | 总强度 |
| F11 | sum(top3) / total | 共现集中度 |
| F12 | entropy(norm_vec) | 自信息/熵 |
| F13-F16 | 0.0 | 保留扩展位 |

### 4.2 边特征提取 `extract_edge_features(cooc_matrix, responses, edge_index)`

**输入:** 共现矩阵 + 响应集合 + 边索引 [2, E]  
**输出:** [E, 4] 边特征

| 特征 | 计算方法 |
|------|----------|
| E1 | 2·cooc(i,j) / (total_i + total_j) — 归一化共现计数 |
| E2 | |R(i) ∩ R(j)| / |R(i) ∪ R(j)| — Jaccard 系数 |
| E3 | cosine(vec_i, vec_j) — 余弦相似度 |
| E4 | Σ 1/log(deg(k)) for k ∈ N(i)∩N(j) — Adamic-Adar |

### 4.3 GraphSAGE 消息传递

```
For layer l in 0..num_mp_layers-1:
    For each node i:
        msg = h[neighbors(i)]           # [deg(i), emb_dim]
        agg = mean(msg, dim=0)          # [emb_dim]
        h_i_new = Linear([h_i, agg])    # [2*emb_dim → emb_dim]
        h_i_new = LayerNorm(h_i_new)
        h_i_new = ReLU(h_i_new)
```

GraphSAGE 风格使用 **拼接自身+邻居均值** 而非简单加权和，保留更多自身信息。`mean` 聚合适配不同度数的节点，避免度数依赖。

### 4.4 Focal Loss 训练

```
对于每条训练边 (i,j):
    p   = sigmoid(logit_ij)
    y   = 1 if adjacent else 0
    pt  = p if y=1 else (1-p)
    
    BCE  = -[y·log(p)·pos_weight + (1-y)·log(1-p)]
    FL   = BCE · (1 - pt)²                ← Focal weight
    pos_weight = neg_count / pos_count     ← 不平衡修正
```

- **pos_weight**: 正负样本比加权，处理大多数点对不相邻的极端不平衡
- **Focal (1-pt)²**: 对已置信样本降权，让模型关注难分样本

## 5. 与原始方法对比

| 维度 | 原始方法 | GNN 增强方法 |
|------|----------|-------------|
| **信息利用方式** | 仅 size=2 的确定性交集 | 全共现矩阵的统计模式 |
| **对采样率的鲁棒性** | 低采样率时 size=2 响应极少 | 利用所有大小的响应，统计上更鲁棒 |
| **跨数据集泛化** | 无状态，天然兼容 | 需多样化训练数据，已覆盖 2D/3D |
| **计算复杂度** | O(R²) 交集运算 | O(N²) 共现 + O(E·F) GNN |
| **输入依赖** | 无 | 需要预训练模型 |
| **已知限制** | 信息利用不充分 | 模型泛化依赖训练数据多样性 |
| **边判定方式** | 硬阈值（交集非空） | 软阈值（学习到的概率） |
