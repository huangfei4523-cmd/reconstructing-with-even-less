## 上下文

`gnn_attack/gnn_model.py` 中 `EdgePredictionGNN` 的第一层是 `nn.Linear(input_dim, hidden_dim)`，其中 `input_dim = N`（数据集点数）。训练使用 10×10 合成网格（N≈80）、推理使用 cali_50（N≈1000），维度不匹配导致模型无法加载。

根本原因：将整张共现矩阵的行向量作为节点「特征」，特征维度 = N，与数据集耦合。

### 当前架构（需重构的部分）

```
共现矩阵 [N, N] → node_encoder: Linear(N→64)→ReLU→Linear(64→32)
                         ↑  N 变化则权重矩阵形状变化
```

## 目标 / 非目标

**目标：**

1. 重写 `EdgePredictionGNN`，使所有 `nn.Linear` 层的输入维度为固定值 F（与 N 无关）
2. 扩展训练数据生成器，支持 2D 规则网格、2D 不规则点云、3D 点云
3. 修复 `data_loader.py` 路径、`_get_correct_edges_at_scale` 自环 Bug、`sympy` 缺失
4. 产出 `docs/gnn-attack-design.md` 原理分析文档

**非目标：**

- 不引入 PyTorch Geometric 等新重量级依赖
- 不改变原始 `attack.py` 的代码
- 不改变 `process_database.py` 的查询生成和采样逻辑
- 不修改 checkpoint 的向后兼容性（旧 checkpoint 直接废弃）

## 决策

### D1. 输入编码：从矩阵行向量到固定维度统计特征

**选择：** 从共现矩阵中提取固定维度 F≈16 的统计特征作为节点输入，而不是直接使用行向量。

**理由：** 共现矩阵行向量维度 = N，但行向量中蕴含的结构信息可以通过统计量（均值、标准差、稀疏度、中心性等）以固定维度压缩，F 与 N 无关。GNN 的消息传递层可以进一步从邻居聚合中恢复空间结构。

**替代方案考虑：**
- **方案 B (PyG GNN)**：使用 PyTorch Geometric 的 GCNConv/GATConv 处理稀疏图。排除理由：增加重量级依赖，安装复杂，对项目规模过度。
- **方案 C (Padding/Truncation)**：训练时固定 max_N，推理时截断或 padding。排除理由：信息丢失，且 cali_50 点数远大于训练网格。
- **方案 D (Reshape/插值)**：用插值将 cooc 矩阵缩放至固定大小。排除理由：破坏空间结构，边预测精度下降。

### D2. 节点特征设计（16 维固定）

```
F1:  平均共现度      mean(cooc_vec) / N
F2:  最大共现度      max(cooc_vec) / N
F3:  共现稀疏度      count_nonzero(cooc_vec) / N
F4:  标准差          std(cooc_vec)
F5:  响应频次        该点出现在多少条采样响应中 / total_responses
F6:  平均响应大小    avg(|r| for responses containing this point) / N
F7:  度中心性        共现图的出度 / (N-1)
F8:  局部聚类系数    cooc 图中邻居之间的连接密度
F9:  特征向量中心性  近似（power iteration 1 步）
F10: PageRank        近似（damping=0.85, 1 步）
F11: 共现集中度      (sum top-3 cooc) / total_cooc
F12: 自信息          -log(p) where p = frequency in responses
F13-F16: 保留位      初始为 0，供后续扩展
```

每个点 16 维特征向量 → `node_features[N, 16]`。

**理由：** 16 维在表达能力和计算效率之间取平衡。统计特征在网格、不规则点云、3D 数据上均有区分度。保留位允许在不改变模型架构的前提下增加新特征。

### D3. 边特征设计（4 维）

```
E1: 归一化共现计数   cooc[i,j] / (sum(cooc[i,:])+sum(cooc[j,:]))
E2: Jaccard 系数     |responses(i) ∩ responses(j)| / |responses(i) ∪ responses(j)|
E3: 余弦相似度        cosine(cooc_vec_i, cooc_vec_j)
E4: Adamic-Adar      Σ_{k∈N(i)∩N(j)} 1/log(deg(k))
```

边特征在 edge_predictor MLP 中与节点嵌入拼接 → 输入维度 = 32+32+4 = 68（固定）。

### D4. GNN 架构：GraphSAGE 风格消息传递

```
输入: node_features[N, 16], edge_index[2, E]

┌─ NodeEncoder ───────────────────────────────────┐
│  Linear(16→64) → BatchNorm → ReLU               │
│  Linear(64→32) → BatchNorm → ReLU               │
│  → node_emb[N, 32]                              │
└─────────────────────────────────────────────────┘

┌─ MessagePassing ×2 (GraphSAGE-style) ───────────┐
│  For each layer l:                              │
│    msg_j = Linear(h_j)  for j ∈ N(i)           │
│    h_i = Linear(h_i + mean_{j∈N(i)} msg_j)     │
│    h_i = LayerNorm(h_i) → ReLU                 │
│  → node_emb[N, 32]                             │
└─────────────────────────────────────────────────┘

┌─ EdgePredictor ─────────────────────────────────┐
│  For edge (i,j):                                │
│    feat = [emb_i, emb_j, edge_feat_ij]          │
│    MLP: 68→64→32→16→1                          │
│  → edge_logit                                  │
│  → sigmoid → edge_prob                         │
└─────────────────────────────────────────────────┘
```

**关键：** `edge_index` 来自 k-NN 图（k=10，基于共现向量的余弦相似度），与当前 `_build_knn_graph` 逻辑一致，但改为稀疏格式 `[2, E]`。消息传递中的 `mean` 聚合保证对不同度的节点鲁棒。

### D5. 训练数据生成器 v2

```
generate_training_data_v2(num_samples, configs):
    Config 类型:
      {"type": "grid_2d",     "grid": (H,W),   "density": D}
      {"type": "random_2d",   "N_points": N,   "range": (X,Y)}
      {"type": "shape_2d",    "shape": "circle"|"L"|"line", "size": S}
      {"type": "random_3d",   "N_points": N,   "range": (X,Y,Z)}
      {"type": "grid_3d",     "grid": (H,W,D), "density": D}

    按比例混合采样（默认: 40% grid_2d, 20% random_2d, 10% shape_2d, 20% random_3d, 10% grid_3d）
```

返回 `[(node_features, edge_index, edge_labels, edge_features), ...]`，不再是 `[(cooc_matrix, adj_matrix)]`。

### D6. data_loader 路径修复

**选择：** 使用 `sys.path` 动态添加项目根目录，然后 `from dataset import cali_all, cali_self, nh`。

```
# data_loader.py 修复后
import sys, os
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
from dataset import cali_all, cali_self, nh
```

**理由：** 不需要 exec() 解析源码，不需要正则匹配变量，直接用 Python import。与现有的 `dataset.py` 文件天然配合。

### D7. 分阶段实施（Phase）

#### Phase 1: 核心架构重建（3 个验证门禁）

1. 实现 `extract_node_features()` + `extract_edge_features()`
2. 重写 `EdgePredictionGNN`（GraphSAGE 风格）
3. 重写 `generate_training_data_v2()`

**门禁：** 使用合成数据训练模型，在 15×15 网格上推理，Precision > 0.3 且 Recall > 0.1 证明架构可行。

#### Phase 2: 推理管线适配（2 个验证门禁）

4. 重写 `_build_knn_graph()` → `build_message_passing_graph()`（稀疏格式）
5. 适配 `predict_edges_from_cooccurrence()` 和 `gnn_range_attack()`

**门禁：** 从 Phase 1 训练的模型在 cali_self（5 点十字）上推理，不报维度错误且生成有效图。

#### Phase 3: Bug 修复 + 兼容性（3 个验证门禁）

6. 修复 `data_loader.py` 路径
7. 修复 `_get_correct_edges_at_scale` 自环 Bug
8. 补充 `sympy` 到 `requirements.txt`

**门禁：** `python gnn_attack.py -points=cali_50 -dist=uniform -p=10 --model <model> --baseline` 全流程无报错。

#### Phase 4: 文档产出

9. 撰写 `docs/gnn-attack-design.md`（原理、架构图、数据流、算法说明）

**门禁：** 文档包含架构图（ASCII）、关键算法的伪代码描述、与原始方法的对比表。

## 风险 / 权衡

| 风险 | 缓解措施 |
|------|----------|
| 16 维统计特征可能不足以捕捉细粒度的共现模式 | 保留 F13-F16 扩展位；Phase 1 中若精度不达标则增加特征维度和消融实验 |
| 3D 训练数据生成增加了生成时间 | 默认 3D 占比仅 30%，可在 CLI 参数调整 |
| GraphSAGE mean 聚合在极端稀疏图上可能坍缩 | k-NN 图 k=10 保证最小连通性；如果 N < 10 则 k = N-1 |
| 旧 checkpoint 完全不兼容 | 在 README 中标注此变更的 breaking 性质；旧模型需重新训练 |
| 训练后的模型对全新数据集泛化未知 | Phase 2 门禁要求不同数据集上可运行；Phase 1 训练数据已覆盖多种分布 |

## 开放问题

- Q1: 是否需要在边预测中引入注意力机制（GAT 风格）？当前 GraphSAGE mean 聚合已足够基础，可后续扩展。
- Q2: 训练时的 `response_sampling_ratio` 是否需要与推理时对齐？当前设计默认训练用 5%，推理不限制，需验证跨采样率的泛化能力。
- Q3: 文档 `gnn-attack-design.md` 是否需要中英双语？提案默认中文。
