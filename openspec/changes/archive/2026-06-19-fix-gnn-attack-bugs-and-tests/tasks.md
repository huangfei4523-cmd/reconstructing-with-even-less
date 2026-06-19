## 1. 缺陷修复

- [x] 1.1 修复 `extract_node_features` F3/F5/F7 冗余 — F5 改为偏度 `skew(vec)`，F7 改为峰度 `kurtosis(vec)`
- [x] 1.2 修复 `train_gnn_model` 中 `scheduler.step()` 位置 — 从 `if val_loader:` 块内移到 epoch 循环末尾（无条件执行）
- [x] 1.3 修复训练时边特征传递 — `generate_training_data_v2` 返回三元组 `(node_feat, adj_gt, responses)`，`CooccurrenceDataset` 存储 responses，`train_gnn_model` 中调用 `extract_edge_features()` 替代 `None`
- [x] 1.4 优化 `_make_sample_from_points` 3D 查询生成 — 查询组合数 > 5000 时随机采样
- [x] 1.5 清理代码 — 移除 `defaultdict` 导入（未使用）、移除 `all_tokens` 死变量

> **验证门禁:** `python train_gnn.py --epochs 5 --samples 20 --save test.pth` 无报错无 warning，训练 loss 正常下降，epoch 打印显示 scheduler 生效。

## 2. 测试套件

- [x] 2.1 创建 `tests/__init__.py` 和 `tests/test_extract_features.py` — 验证 `extract_node_features` shape=[N,16]、N<4 边界、`extract_edge_features` shape=[E,4]
- [x] 2.2 创建 `tests/test_graph_build.py` — 验证 `build_message_passing_graph` 输出 edge_index [2,E] 稀疏格式，无自环，对称性
- [x] 2.3 创建 `tests/test_edge_prediction.py` — 验证 `EdgePredictionGNN.forward` 输出 shape，`predict_all_pairs` 对称性
- [x] 2.4 创建 `tests/test_accuracy.py` — 验证 `_get_correct_edges_at_scale` 无自环、正确识别邻接边

> **验证门禁:** `python -m pytest tests/ -v` 全部 PASS。

## 3. 完整性验证

- [x] 3.1 运行 `python train_gnn.py --epochs 10 --samples 100 --save test.pth` 全流程无错
- [x] 3.2 运行 `python gnn_attack.py -points=cali_self -dist=uniform -p=100 --model test.pth` 无报错
- [x] 3.3 确保 `python gnn_attack.py -points=cali_self -dist=uniform -p=100 --model test.pth --baseline` 全流程无 `ModuleNotFoundError`

> **验证门禁:** 以上三条命令全部成功。
