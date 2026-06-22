# 设计文档

## 上下文

Phase 1 预训练模型输出恒为 0.5（entropy=0.693），导致 Phase 2 自训练后 P/R 仅 3-7%。诊断发现两个致命 Bug 在同一调用链上：

1. **消息传递图用错误的数据构建**（`gnn_model.py:574`）：`cooc_sim = node_feat[:,:3] @ node_feat[:,:3].T` 是特征列 F1-F3 的内积，与真正的共现矩阵无关。真正的 `cooc` 矩阵在 `_make_sample_from_points()` 第 256-264 行计算但被丢弃。
2. **训练/验证间边特征不一致**（`train_gnn.py:79`）：训练循环传入 `edge_feat`，验证循环传 `None`。

## 目标 / 非目标

**目标：**
- 共现矩阵 `cooc` 从生成 → 数据集 → 训练/验证循环的完整传递
- 训练和验证使用相同的消息传递图构建方式和边特征

**非目标：**
- 不改变模型架构（EdgePredictionGNN）
- 不改变特征提取函数（extract_node_features / extract_edge_features）
- 不改变损失函数（Focal Loss + pos_weight）
- 不改变训练超参数（lr, epochs, batch size）

## 决策

### D1：共现矩阵作为第 4 个返回值

**选择：** `_make_sample_from_points()` 返回 `(node_feat, adj_gt, sampled, cooc)` 四元组。

**理由：** `cooc` 已在函数内计算（第 256-264 行），只差返回。添加返回值比通过其他方式重建更简单可靠。

**替代方案（拒绝）：** 在 `train_gnn_model()` 中从 `responses` 重建共现矩阵。拒绝原因：需要重复昂贵的双循环 O(|r|²) 操作，且 `sampled` 本身是 frozensets 列表，转换开销大。

### D2：CooccurrenceDataset 存储 cooc_list

**选择：** `CooccurrenceDataset.__init__` 接受第 4 个参数 `cooc_list`，`__getitem__` 返回 `(node_feat, adj_gt, resp, cooc)`。

**理由：** 最小侵入性修改。数据集类本身是一个简单的 tuple holder，加一个字段是最自然的方式。

### D3：用真正的 cooc 矩阵构建消息传递图

**选择：** 训练和验证循环中，`build_cooc_message_graph(cooc)` 替代 `build_cooc_message_graph(cooc_sim)`。

**理由：** `cooc` 是真正的点对共现计数矩阵，编码了空间邻接信息。`cooc_sim` 只是特征内积，与邻接无关。

### D4：验证循环传入边特征

**选择：** 验证循环中传入 `extract_edge_features(cooc, responses, edge_idx)` 而非 `None`。

**理由：** 训练-验证一致性是机器学习的基本原则。训练时模型学习利用边特征，验证时也必须提供。

**替代方案（拒绝）：** 训练时也不传边特征（都填零）。拒绝原因：边特征（4 维 pairwise 统计）包含有用的共现差异信息，丢弃会降低模型能力。

### D5：训练循环中边特征用真正的 cooc 计算

**选择：** `extract_edge_features(cooc, responses, edge_index)` 替代 `extract_edge_features(cooc_sim, responses, edge_index)`。

**理由：** 第一个参数 `extract_edge_features` 要求的是共现矩阵。传入 `cooc_sim`（特征内积）毫无意义——它跟共现完全没有关系。

## 数据流变更

```
修复前（错误）:
  _make_sample_from_points() → (node_feat, adj_gt, sampled)
                                 ↓
  generate_training_data_v2() → (feat_list, adj_list, resp_list)
                                 ↓
  CooccurrenceDataset → (node_feat, adj_gt, resp)
                                 ↓
  train_gnn_model():
    cooc_sim = node_feat[:,:3] @ node_feat[:,:3].T  ← BUG!
    edge_index = build_cooc_message_graph(cooc_sim)  ← 错误输入
    edge_feat = extract_edge_features(cooc_sim, ...) ← 错误输入
    model(node_feat, edge_index, edge_feat, edge_w)

修复前（验证 - 额外 Bug）:
  train_phase1() 验证循环:
    cooc_sim = node_feat[:,:3] @ node_feat[:,:3].T  ← 同 Bug
    model(node_feat, edge_idx, None, edge_w)        ← edge_feat=None!

修复后（正确）:
  _make_sample_from_points() → (node_feat, adj_gt, sampled, cooc)  ← +cooc
                                 ↓
  generate_training_data_v2() → (feat_list, adj_list, resp_list, cooc_list) ← +cooc_list
                                 ↓
  CooccurrenceDataset → (node_feat, adj_gt, resp, cooc)             ← +cooc
                                 ↓
  train_gnn_model():
    edge_index = build_cooc_message_graph(cooc)       ← 真正的共现矩阵
    edge_feat = extract_edge_features(cooc, ...)      ← 真正的共现矩阵
    model(node_feat, edge_index, edge_feat, edge_w)

  验证循环（一致）:
    edge_index = build_cooc_message_graph(cooc)
    edge_feat = extract_edge_features(cooc, ...)      ← 一致了!
    model(node_feat, edge_idx, edge_feat, edge_w)
```

## 风险 / 权衡

- **[风险]** `cooc` 矩阵大小 O(N²) 随 N 增大。N=800 时为 800²×4 bytes ≈ 2.5MB，在可接受范围内。
  → 缓解：当前最大 N=800，500 个样本的内存开销约 1.2GB，合理。
- **[风险]** 修复后 Phase 1 模型是否能学到有效模式尚不确定——共现矩阵本身是否是好的输入信号需要实验验证。
  → 缓解：通过 Phase 1 验证集 P/R 总结（上一变更已添加）直接观察效果。如果 P/R 仍然低，再调查特征质量和模型容量。

## 验证门禁

1. Phase 1 训练 2 epochs + 10 samples 后，验证集 entropy < 0.60（当前 0.693 = 随机）
2. 验证集最佳 F1 > 0.30（当前 ≈ 0）
3. 全测试 PASS
