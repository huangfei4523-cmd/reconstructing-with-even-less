# 修复 Phase 1 预训练质量问题

## 为什么

Phase 1 预训练模型对所有点对输出 ~0.5（entropy=0.693, pos_conf≈0.506, neg_conf≈0.500），导致 Phase 2 自训练后 Precision/Recall 仅 3-7%——等同于随机猜测。诊断发现 4 个致命 Bug 在同一调用链上：消息传递图使用错误的特征内积矩阵替代真正的共现矩阵，且训练-验证间边特征传入不一致。整个训练管线从根上就是错误的。

## 变更内容

1. **在 `_make_sample_from_points()` 中返回真正的共现矩阵 `cooc`**——当前计算了但丢弃了
2. **`CooccurrenceDataset` 存储并返回共现矩阵**——第 4 个返回值
3. **训练和验证循环使用真正的共现矩阵构建消息传递图**——替代错误的 `node_feat[:,:3] @ node_feat[:,:3].T`
4. **验证循环传入边特征**——修复训练/验证间 `edge_features=None` vs `edge_features=...` 的不一致
5. **训练循环中 `build_cooc_message_graph` 的输入从 `cooc_sim`（特征内积）改为真正的 `cooc_matrix`**

## 功能 (Capabilities)

### 新增功能

- `cooc-matrix-passthrough`: 真正的共现矩阵从数据生成 → 数据集 → 训练/验证循环的完整传递链路

### 修改功能

- `gnn-pretraining`: 消息传递图构建方式从特征内积改为真正的共现矩阵；训练/验证循环统一传入边特征

## 影响

| 文件 | 改动 |
|------|------|
| `gnn_model.py` | `_make_sample_from_points()` 返回值加 `cooc`；`generate_training_data_v2()` 返回值加 `cooc_list`；`CooccurrenceDataset` 存储 `cooc_list`；`train_gnn_model()` 用 `cooc` 构建消息图 |
| `train_gnn.py` | `train_phase1()` 适配新返回值；验证循环传入 `edge_feat` |
