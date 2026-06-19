## 上下文

前序变更 `fix-gnn-architecture` 完成后，`gnn_attack/` 模块运行时暴露了两个新问题（adj_gt batch 维度索引越界、scheduler 条件化失效），全面代码审查又发现了 7 个残留缺陷。当前模块处于可运行但不健壮的状态——训练可能不稳定、特征维度有冗余、缺乏自动化测试覆盖。

## 目标 / 非目标

**目标：**
1. 修复所有已知的代码缺陷（冗余特征、scheduler、训练/推理不一致、dead code）
2. 新增测试套件覆盖核心功能路径
3. 3D 训练数据生成优化（大网格采样替代全枚举）

**非目标：**
- 不改变模型架构（feature_dim、emb_dim、num_mp_layers 不变）
- 不新增外部依赖
- 不修改 checkpoint 格式

## 决策

### D1. 冗余特征 F3/F5/F7 去重

**选择:** 保留 F3（共现稀疏度），将 F5 改为「共现偏度」、F7 改为「共现峰度」。

```
Before (冗余):                After (去重):
F3 = nonzero/N               F3 = nonzero/N            (保留)
F4 = std                      F4 = std                  (不变)
F5 = nonzero/N    ← 同 F3    F5 = skew(vec)            (偏度)
F7 = nonzero/N    ← 同 F3    F7 = kurtosis(vec)        (峰度)
```

**理由:** 偏度和峰度提供了共现分布的形状信息（尾部大小、集中程度），与稀疏度互补，0 额外计算开销。

### D2. Scheduler 无条件 step

**选择:** 将 `scheduler.step()` 从 `if val_loader:` 块中移出，放在 epoch 循环末尾（验证块之后）。

**理由:** `CosineAnnealingLR` 需要每个 epoch 更新学习率以正常衰减。当 `val_loader=None`（纯训练模式）时调度器从未更新，导致学习率恒定。**替代方案:** 在无验证时也检查 and step——但 `if val_loader:` 条件本身就不该限制 scheduler。

### D3. 训练时也使用边特征

**选择:** 训练时在 `train_gnn_model` 中调用 `extract_edge_features` 计算真实的边特征，而不是传 `None` 用零替代。

**前提:** 训练数据生成时需要同时产出 `responses` 集合。

**改动:** `generate_training_data_v2` 返回三元组 `(node_feat, adj_gt, responses)`，`CooccurrenceDataset` 存储 responses，`train_gnn_model` 在每一步从 responses 计算 edge_features。

**理由:** 当前训练时 edge_features=None（零填充），推理时传入真实边特征，模型从未学到如何利用边特征。训练推理一致才能让边特征发挥作用。

**替代方案:** 完全移除边特征，让 edge_predictor 输入维度退回 `emb_dim*2`。放弃方案，因为边特征提供了有用的 pairwise 判别信息。

### D4. 3D 查询生成采样

**选择:** 当 3D 查询组合数超过 5000 时，随机采样替代全枚举。

```python
if N0*N1*N2 > 5000:  # 估算查询数
    query_sampled = random.sample(all_queries, min(5000, len(all_queries)))
```

**理由:** 3D 查询数随 grid 立方增长（8×8×8 = 21,952），全枚举在训练数据生成阶段占绝大部分时间。采样 5000 条足以覆盖丰富的空间模式。

### D5. 测试套件设计

4 个测试文件，覆盖核心模块：

| 文件 | 覆盖函数 | 验证点 |
|------|----------|--------|
| `test_extract_features.py` | `extract_node_features`, `extract_edge_features` | 输出 shape、值范围、边界 N<4 |
| `test_graph_build.py` | `build_message_passing_graph` | 稀疏格式、k 参数、对称性 |
| `test_edge_prediction.py` | `EdgePredictionGNN.forward`, `predict_all_pairs` | 输入输出 shape、设备兼容性 |
| `test_accuracy.py` | `_get_correct_edges_at_scale` | 自环检查、邻接边正确性 |

运行方式：`python -m pytest tests/ -v` 或 `python -m unittest discover tests/`

## 风险 / 权衡

| 风险 | 缓解 |
|------|------|
| 边特征训练启用后训练时间增加 10-20% | 可接受的代价，换取训练/推理一致性 |
| 3D 查询采样可能丢失部分空间模式 | 采样 5000 条覆盖足够多样性，且当前训练 config 中 3D 占比仅 30% |
| 特征去重后已有 checkpoint 精度可能下降 | 已有 checkpoint 仍可加载——旧特征的微小冗余不影响功能，仅推荐重新训练获得更好精度 |
