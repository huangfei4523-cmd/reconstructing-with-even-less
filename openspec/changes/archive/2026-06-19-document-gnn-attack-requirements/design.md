## 上下文

`gnn_attack/` 是基于论文 *Reconstructing with Even Less: Leakage Amplification and Graph Drawing* 的改进版本。原始攻击通过"范围查询响应 → 集合交集放大 → 找大小为 2 的素数响应 → 建图"来重建数据点的空间邻接关系。GNN 增强方法试图用图神经网络学习从共现矩阵到邻接关系的映射，以利用更多统计信息。

当前状态：模块代码已存在但未经完整验证，代码审查发现多个阻断性 Bug。

## 目标 / 非目标

**目标：**
- 复原模块的架构设计，用文档记录数据流和组件关系
- 记录每个源文件在系统中的角色和职责
- 识别核心设计决策、数据结构和关键接口
- 列出已知缺陷和需求缺口

**非目标：**
- 不修复任何代码 Bug
- 不在设计层面提议新功能
- 不评估 GNN 方法的有效性（仅记录设计意图）

## 决策

### D1. 模块架构：两条攻击路径 + 训练管线

```
┌─────────────────────────────────────────────────┐
│              gnn_attack 顶层                     │
│                    │                            │
│    ┌───────────────┼───────────────┐            │
│    ▼               ▼               ▼            │
│ 训练管线         GNN 攻击路径      原始攻击路径    │
│ (合成数据)       (共现→GNN→边)    (交集→素数)    │
│    │               │               │            │
│ train_gnn.py   gnn_range_attack  range_attack   │
│    │               │               │            │
│ gnn_model.py   gnn_model.py        (独立)       │
│                    │                            │
│               gnn_attack.py (主入口)              │
│                    │                            │
│    ┌───────────────┼───────────────┐            │
│    ▼               ▼               ▼            │
│ data_loader    process_database  评估/可视化      │
└─────────────────────────────────────────────────┘
```

**理由：** 分离训练和推理阶段。训练在合成网格数据上完成，产出模型权重文件；推理加载权重后在真实数据集上运行。原始攻击路径作为 baseline 保留在独立的 `range_attack.py` 中。

**替代方案考虑：** 合并两个 `range_attack.py`（原版 + 副本），选择保留独立副本以避免跨模块导入的路径问题。

### D2. 核心数据结构：共现矩阵 (Co-occurrence Matrix)

共现矩阵是 GNN 路径的核心中间表示：

```
 响应 R₁ = {A, B, C}
 响应 R₂ = {B, C, D}
 响应 R₃ = {A, C}

 共现矩阵 C[N×N]：
     A  B  C  D
 A [ 0  1  2  0 ]   A 和 C 在 2 条响应中共现
 B [ 1  0  2  1 ]   B 和 C 在 2 条响应中共现
 C [ 2  2  0  1 ]
 D [ 0  1  1  0 ]

 归一化：每行除以行和 → [N, N] float32 矩阵
```

**理由：** 共现矩阵编码了所有响应大小中的点对关系，不像原始方法只利用 size=2 的响应。这是 GNN 方法比原始方法能利用更多信息的核心原因。

**设计缺陷：** 共现矩阵的维度 N 取决于数据集的点数。如果训练和推理的数据集点数不同，模型的第一层 `nn.Linear(input_dim, hidden_dim)` 维度不匹配。

### D3. GNN 模型：EdgePredictionGNN

```
输入 [N, N] 共现矩阵
  │
  ├─ node_encoder: Linear(N→64) → ReLU → Linear(64→32)
  │     ↓ [N, 32] 节点特征
  ├─ msg_pass_1: Linear(32→32) @ (k-NN_adj @ h) → LayerNorm → ReLU
  │     ↓ [N, 32]
  ├─ msg_pass_2: Linear(32→32) @ (k-NN_adj @ h) → LayerNorm → ReLU
  │     ↓ [N, 32] 节点嵌入
  └─ edge_predictor: MLP([64→hidden→hidden/2→1])
       输入: concat(emb_i, emb_j) → [N, N, 64]
       输出: edge_logits [N, N] → sigmoid → edge_probs
```

**k-NN 消息传递图：** 基于共现矩阵行向量的余弦相似度构建 k=10 的最近邻图，用作 GCN 风格的邻接矩阵进行消息传递。

**训练损失：** Focal Loss + 正样本按 (负样本数/正样本数) 加权，缓解类别不平衡（大多数点对不相邻）。

### D4. 训练数据生成策略

合成 2D 网格上模拟攻击场景：
1. 在 grid_size 网格上随机生成点（每个格子 1~max_points_per_cell 个点）
2. 枚举所有范围查询响应（四重循环 min0..max0, min1..max1）
3. 按 response_sampling_ratio 采样（默认 5%）模拟 "Even Less" 场景
4. 计算共现矩阵（特征）和曼哈顿距离 ≤1 的邻接矩阵（标签）

**设计局限：** 仅使用规则 2D 网格训练，无法覆盖 `crg`（3D）、`cali`（不规则形状）等数据集的分布特征。模型的泛化能力未经验证。

### D5. 数据集加载策略

`data_loader.py` 通过以下方式加载数据集：
- `cali_50`：从原版 `attack.py` 中提取 `cali_all` 变量（正则 + exec）
- `grid`：动态生成
- `dg`, `crg`, `boat`：从 `datasets/` 目录加载 pickle 文件
- `nh`：从原版 `attack.py` 提取 `nh` 变量

**已知缺陷：** `cali_50` 和 `nh` 的路径解析依赖 `../reconstructing-with-even-less/attack.py`，在 monorepo 结构下此路径可能无效。

## 风险 / 权衡

| 风险 | 影响 | 缓解方向 |
|------|------|----------|
| input_dim 训练/推理不匹配 | 模型无法加载，攻击完全失败 | 改为固定维度编码（如 padding）或使用图神经网络原语（PyG） |
| data_loader 路径错误 | cali_50/nh 数据集加载失败 | 使用 `importlib` 动态加载或重构为 package 导入 |
| max_correct 自环 Bug | recall 评估指标无意义 | 修复为 `dictionarry[nt]` 而非 `coord` |
| sympy 未声明依赖 | baseline 模式崩溃 | 添加 `sympy` 到 requirements.txt |
| 训练数据局限性 | 对不规则数据集泛化能力差 | 增加多样化训练数据（不同形状、密度、维度） |
| 大 N 时 O(N²) 张量 | 边缘预测阶段可能 OOM | 分批次预测或使用稀疏边预测 |
