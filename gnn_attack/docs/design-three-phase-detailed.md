# 三阶段可搜索加密 GNN 攻击 — 详细设计

## Phase 1: 预训练（学「空间相邻 ≈ 高频共现」）

### 1.1 输入输出

| | 说明 |
|---|------|
| **数据来源** | 攻击者自己生成——虚构的点集坐标 + 模拟加密范围查询 |
| **输入** | 无，生成器内部提供所有数据 |
| **标签** | 有：攻击者知道所有坐标，可计算任意点对的曼哈顿距离 → 相邻=距离≤1 |
| **输出** | GNN checkpoint：`model_state_dict`, `feature_dim=16`, `hidden_dim`, `emb_dim`, `num_mp_layers`, `train_loss`, `val_loss` |

### 1.2 Phase 1 训练流程（四步）

```
Step 1: 定义参数空间
  ├─ 维度 1: N = 20, 40, 80, 150, 300, 500, 800
  ├─ 维度 2: 形状 = grid, random, circle, contour, line
  ├─ 维度 3: p = 1%, 3%, 5%, 10%, 20%, 50%
  └─ 组合: 7×5×6 = 210 种配置，每种 5-10 样本 ≈ 1500 样本

Step 2: 生成合成数据 (GenerateTrainingSample)
  按 Step 1 的配置逐种生成：
    生成点集 → 模拟加密查询 → 采样 p% → 共现矩阵 → 标签(曼哈顿≤1)

Step 3: 训练 GNN
  用 Step 2 生成的全部数据训练 (Focal Loss, Adam, CosineAnnealingLR)

Step 4: 验证 + 反馈
  留出验证集 AUC > 0.7
  找出 AUC 低的配置 → 增补该配置样本 → 重新训练
  Phase 2 自训练发散 → 回到 Step 1 补充训练数据多样性
```

#### Step 2 详细算法：训练样本生成

```
算法: GenerateTrainingSample

输入: 场景配置 (形状类型, N, 采样率 p)
输出: (node_features[N,16], adj_labels[N,N], responses[N,])

步骤:
  2a. 根据形状类型生成点集坐标 map[token→(x,y)]
      - grid:     在 W×H 网格中每个格子随机撒 M 个点
      - random:   在 (X_range,Y_range) 内随机生成 N 个点
      - circle:   在半径 R 的圆边界上均匀分布 N 个点
      - contour:  从预定义轮廓上采样 N 个点
      - line:     在直线上等距分布 N 个点

  2b. 模拟加密范围查询:
      for min0 in 1..N0:
        for max0 in min0..N0:
          for min1 in 1..N1:
            for max1 in min1..N1:
              查询矩形 [min0,max0]×[min1,max1]
              response = {token | (x,y) 在矩形内}
              if |response| ≥ 2: 保存此响应

  2c. 采样: 随机选 int(|responses| × p/100) 条响应

  2d. 计算共现矩阵 C[N,N]:
      for 每条采样响应 r:
        for (a,b) ∈ r×r, a≠b:
          C[a,b] += 1; C[b,a] += 1

  2e. 提取节点特征: node_feat = ExtractNodeFeatures(C)
      16维统计量: 均值、最大值、稀疏度、标准差、偏度、
                  强度、峰度、集中度、熵等（全部除以N保证尺度不变）

  2f. 计算标签: adj[i,j]=1 if 曼哈顿距离(i,j)≤1 else 0

  返回 (node_feat, adj, sampled_responses)
```

#### Step 1 参数空间设计理由

| 参数维度 | 关键值 | 为什么必须覆盖 |
|----------|--------|---------------|
| N=20~50 | 小N | 共现密集，正负比高 — 提供「清晰信号」的基线 |
| N=60~150 | 中N | 过渡区 — 桥接小N和大N的特征分布 |
| N=200~800 | 大N | 共现稀疏，贴近真实攻击场景 — 保证 Phase 2 可启动 |
| grid | 栅格 | 规则结构，高邻接比(~1:5) — 基础拓扑 |
| random | 随机散点 | 不规则结构，低邻接比(~1:50) — 极端不平衡 |
| circle/contour/line | 轮廓 | 1D嵌入在2D中的结构 — 模拟真实不规则形状 |
| p=1%~5% | 低采样率 | 信噪比极低 — 训练模型在稀疏信号下区分 |
| p=10%~50% | 高采样率 | 信噪比高 — 让模型先学会「容易的」再泛化到「难的」 |

### 1.3 共现图消息传递设计

**设计选择：用共现矩阵本身作为加权消息传递图，而非基于共现向量的 k-NN。**

```
当前 k-NN 方案的问题:
  k-NN 基于 cooc 向量余弦相似度建图
  → 「相似」≠「相邻」（统计相关 vs 几何相关）
  → 训练推理图拓扑不一致（训练用前6维特征，推理用完整cooc）

共现加权图方案:
  消息传递图 = {(i,j) | C[i,j] > 0}   ← 所有共现对
  消息权重 w_ij = normalize(C[i,j])    ← 高共现→高权重
  
  注意力聚合: 
    h_i^(l+1) = Linear([h_i^(l), Σ_{j∈N(i)} α_ij · h_j^(l)])
    α_ij = softmax(1 / log(1 + C[i,j]))   ← 共现越高，注意力越高
  
  直觉: 如果两个点频繁共现→它们空间上接近→在图上传更多消息是合理的
```

### 1.4 验证标准

预训练完成后必须通过以下验证才能进入 Phase 2：

1. **AUC 验证**：在独立的验证集（含 ≥30% N≥200 场景）上，边预测 AUC > 0.7
2. **多阈值 P/R**：输出 5 个阈值(0.1~0.5)下的 P/R，作为 Phase 2 阈值选择参考
3. **尺度一致性**：验证小 N（20）和大 N（500）场景的平均 sigmoid 输出在同一数量级

---

## Phase 2: 自训练（贴合目标加密数据分布）

### 2.1 输入输出

| | 说明 |
|---|------|
| **数据来源** | 目标加密服务：从监听到的加密查询响应计算共现矩阵 |
| **输入** | 共现矩阵 C_target[N,N]（纯数值，无坐标） + Phase 1 的 GNN_pre 模型参数 |
| **标签** | 无真实标签。用模型预测的高置信度边作伪标签 |
| **输出** | GNN_fine 模型参数 + 推断边集合 E_hat = {(i,j,prob), prob∈[0,1]} |
| **约束** | 禁止使用明文坐标、真实标签、源代码或数据集特定逻辑 |

### 2.2 伪标签筛选

```
算法: SelectPseudoLabels

输入: 边概率矩阵 P[N,N]
输出: pseudo_pos, pseudo_neg

步骤:
  1. 取上三角概率值 (i<j) → 排序
  2. pseudo_pos = 概率最高的 K 条边 (K ≈ 2×N)
     多样性约束: 确保选出的边覆盖 ≥ 80% 的节点
  3. pseudo_neg = 概率最低的 L 条边 (L ≈ 10×N)
     排除与 pseudo_pos 中边共享节点的对
     （避免将潜在邻居误标为负）
  4. 中间区域: 丢弃（模型不确定，不参与训练）
  5. 返回 pseudo_pos, pseudo_neg
```

### 2.3 迭代微调

```
算法: SelfTrainingLoop

输入: GNN_pre, C_target, max_iterations=20
输出: GNN_fine, E_hat

初始化: GNN_0 = GNN_pre

for t = 0 to max_iterations:
  1. 推理: P_t = GNN_t.forward(C_target)

  2. 筛选伪标签: pos_t, neg_t = SelectPseudoLabels(P_t)

  3. 微调 GNN_t:
     损失 = BCE(pos_t, 1.0) + BCE(neg_t, 0.0)
           + λ · L_consistency
     
     一致性正则化:
       C' = PerturbCooc(C_target, drop_pct=0.05)
       P' = GNN_t.forward(C')
       L_consistency = |P_t - P'|²     ← 要求对扰动鲁棒

     优化器: Adam(lr=0.0001, ≤ 预训练lr/10)
     冻结: 仅微调边预测器和最后层消息传递参数

  4. 收敛判定:
     E_t = pos_t 中的边集合
     E_{t-1} = 上一轮的 pos 边集合
     overlap = |E_t ∩ E_{t-1}| / max(|E_t|, |E_{t-1}|)
     if overlap ≥ 0.90: 
        break  ← 收敛

  5. 发散检测:
     if 连续 3 轮 overlap 持续下降:
        break  ← 发散，取历史最佳

最终输出:
  E_hat = pos_{final}[:K_out] (K_out ≈ 2N)
  GNN_fine = GNN_final
```

### 2.4 收敛判定

| 条件 | 判定 | 动作 |
|------|------|------|
| 边重合率 ≥ 90% | 收敛 | 停止，输出 |
| 连续 3 轮重合率下降 | 发散 | 停止，取历史最佳 |
| 达到最大轮数 (20) | 超时 | 停止，输出当前 |

---

## Phase 3: 形状重建（从拓扑到几何）

### 3.1 输入输出

| | 说明 |
|---|------|
| **数据来源** | Phase 2 输出的 E_hat |
| **输入** | 边集合 E_hat = {(i,j,prob)} |
| **标签** | 不需要 |
| **输出** | 推测的 2D 坐标 pos[N,2] |
| **可选输入** | 少数已知锚点坐标（如 3-5 对攻击者已知的人造点） |

### 3.2 力导向布局

```
算法: ForceDirectedReconstruction

输入: E_hat = {(i,j,prob)}, 可选锚点 anchors={(k, (x,y))}
输出: pos[N,2]

步骤:
  1. 建有权图:
     G = Graph(N), edges=E_hat
     edge_weight[i,j] = prob    ← GNN 给出的概率作为弹簧强度

  2. 初始化位置:
     if 有锚点: 锚点在固定坐标，其余随机
     else: 所有节点在单位圆上等距分布

  3. 迭代优化:
     重复直到 max(displacement) < ε:
       能量函数:
         E = Σ_{(i,j)} w_ij · |pos_i - pos_j|²        ← 弹簧势能
           + Σ_{i≠j} k / |pos_i - pos_j|               ← 库仑斥力
       
       对每个节点 i:
         F_spring_i = Σ_{j∈N(i)} 2·w_ij·(pos_j - pos_i)
         F_coulomb_i = Σ_{j≠i} k·(pos_i - pos_j) / |pos_i - pos_j|³
         pos_i += η · (F_spring_i + F_coulomb_i)
       
       锚点: 位置不更新

  4. 输出 pos[N,2]
```

### 3.3 可选评估

```
Procrustes 对齐（有锚点时）:
  用已知锚点坐标计算最优旋转/缩放/平移
  将推测坐标对齐到真实坐标系

Hausdorff 距离评估:
  H(pos_pred, pos_true) = max(h(pos_pred,pos_true), h(pos_true,pos_pred))
  h(A,B) = max_{a∈A} min_{b∈B} |a-b|
  值越小 → 重建越精确
```

### 3.4 重建失败检测

| 条件 | 含义 | 建议 |
|------|------|------|
| 边数 < N/2 | 拓扑不连通，信息不足 | Phase 2 可能未收敛，增大迭代轮数 |
| 节点方差 < (边数×avg_weight)²/100 | 所有点坍缩到一处 | 边推断可能完全错误，检查预训练 AUC |
| 连通分量数 > N/4 | 图高度碎片化 | 采样率 p 太低，无法恢复全局拓扑 |

---

## 代码实现关联要求

后续 gnn-attack 模块代码实现时，必须对齐以下设计文档中明确定义的接口点：

| 接口点 | 定义位置 | 对齐要求 |
|--------|----------|----------|
| ExtractNodeFeatures 统计量集合 | §1.2 Step 2e | 16维特征名称和计算方式必须一致 |
| Phase 1 训练场景配置格式 | §1.2 Step 1 | 形状类型、N范围、采样率枚举值必须匹配 |
| 共现图消息传递的边定义 | §1.3 | 禁止使用 k-NN；必须用 cooc>0 的全边集 + 共现注意力权重 |
| Phase 1 Checkpoint 输出字段 | §1.1 | model_state_dict, feature_dim, hidden_dim, emb_dim, num_mp_layers |
| Phase 2 输入格式 | §2.1 | 仅接受 C[N,N] 矩阵 + GNN_pre 参数，禁止依赖数据集特定加载逻辑 |
| 伪标签选取公式 | §2.2 | K=2N, L=10N, 多样性约束, 中间区域丢弃 |
| 一致性正则化实现 | §2.3 | λ 权重, 扰动率 5%, L_consistency 公式 |
| Phase 2 收敛/发散判定 | §2.4 | 90% 重合率, 连续3轮下降=发散, max_iter=20 |
| Phase 2 输出格式 | §2.1 | E_hat = {(i,j,prob)}，prob 为 float |
| 力导向能量函数 | §3.2 | 弹簧项和库仑项的公式和参数 |
| 重建失败判定 | §3.4 | 三个判定条件的阈值 |
