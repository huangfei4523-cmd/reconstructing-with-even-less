## 1. 共现矩阵传递链路

- [x] 1.1 修改 `_make_sample_from_points()` 返回 `(node_feat, adj_gt, sampled, cooc)` 四元组（第 282 行）
- [x] 1.2 修改 `generate_training_data_v2()` 收集 `cooc_list` 并作为第 4 个返回值（第 327-365 行）
- [x] 1.3 修改 `CooccurrenceDataset.__init__` 接受第 4 个参数 `cooc_list`，`__getitem__` 返回 `(node_feat, adj_gt, resp, cooc)`（第 372-385 行）

## 2. 训练循环修复

- [x] 2.1 修改 `train_gnn_model()` 训练循环：从 batch 中获取 `cooc`，用 `build_cooc_message_graph(cooc)` 替代 `build_cooc_message_graph(cooc_sim)`，用 `extract_edge_features(cooc, ...)` 替代 `extract_edge_features(cooc_sim, ...)`（第 565-617 行）
- [x] 2.2 删除 `cooc_sim` 中间变量和 `node_feat[:,:3]` 切片逻辑（第 574-575 行）

## 3. 验证循环修复

- [x] 3.1 修改 `train_phase1()` 验证循环：从 val_loader 获取 `cooc`，用 `build_cooc_message_graph(cooc)` 替代 `build_cooc_message_graph(cooc_sim)`（第 70-83 行）
- [x] 3.2 验证循环传入 `edge_feat`（通过 `extract_edge_features(cooc, responses, edge_idx)`），不再传 `None`（第 79 行）

## 4. 外部适配

- [x] 4.1 修改 `train_phase1()` 中 `CooccurrenceDataset` 构造传入 `cooc_list`（第 54-55 行）

## 5. 验证

- [x] 5.1 运行 `python train_gnn.py --epochs 2 --samples 10 --save results/test_phase1.pth` 确认无报错且 entropy < 0.60
- [x] 5.2 运行 `python attack_gnn.py --phase1-model results/test_phase1.pth --target-data cali_self --p 100` 确认 Phase 2 的 pos_conf 不再等于 0.500
- [x] 5.3 运行 `pytest tests/ -v` 全测试 PASS
