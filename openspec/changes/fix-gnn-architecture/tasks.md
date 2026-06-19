## 1. Phase 1: 核心架构重建

- [x] 1.1 在 `gnn_model.py` 新增 `extract_node_features(cooc_matrix)` — 输入 `[N,N]` 共现矩阵，输出 `[N,16]` 固定维度特征，包含 F1-F12 统计量，F13-F16 保留为 0
- [x] 1.2 在 `gnn_model.py` 新增 `extract_edge_features(cooc_matrix, responses, edge_index)` — 输入共现矩阵、响应集合、边索引 `[2,E]`，输出 `[E,4]` 边特征（归一化共现、Jaccard、余弦相似度、Adamic-Adar）
- [x] 1.3 重写 `EdgePredictionGNN.__init__` — 将 `input_dim` 参数改为 `feature_dim=16`，`node_encoder` 改为 `Linear(16→64)→BatchNorm→ReLU→Linear(64→32)`
- [x] 1.4 重写 `EdgePredictionGNN.forward` — 改为接受 `(node_features, edge_index, edge_features)`，使用 GraphSAGE 风格消息传递（mean 聚合 + Linear 更新），边预测拼接 `[emb_i, emb_j, edge_feat_ij]`
- [x] 1.5 新增 `EdgePredictionGNN.predict_all_pairs(node_emb)` — 对 N>500 分批计算全连接边概率以避免 OOM
- [x] 1.6 重写 `generate_training_data()` 为 `generate_training_data_v2(num_samples, configs)` — `configs` 支持 `grid_2d`、`random_2d`、`shape_2d`、`random_3d`、`grid_3d` 五种类型，返回 `[(node_feat, edge_index, edge_labels, edge_feat), ...]`
- [x] 1.7 重写 `_build_knn_graph()` 为 `build_message_passing_graph(cooc_matrix, k=10)` — 返回稀疏格式 `edge_index[2,E]` 而非稠密 `[N,N]`

> **验证门禁 Phase 1**：`python train_gnn.py --epochs 20 --samples 200 --grid 15 15` 训练完成后，在 15×15 网格上推理，Precision > 0.3 且 Recall > 0.1。

## 2. Phase 2: 推理管线适配

- [x] 2.1 重写 `predict_edges_from_cooccurrence()` — 输入改为 `(model, cooc_matrix, responses, device, threshold)`，增加特征提取和消息传递图构建步骤
- [x] 2.2 适配 `gnn_range_attack.compute_cooccurrence_matrix()` 返回签名 — 确保与新的特征提取函数输出格式兼容
- [x] 2.3 适配 `gnn_range_attack.gnn_range_attack()` — 在共现矩阵计算后插入 `extract_node_features()`、`build_message_passing_graph()`、`extract_edge_features()` 三步
- [x] 2.4 适配 `train_gnn.py` 中数据集和模型初始化代码 — 使用新的 `generate_training_data_v2` 和 `EdgePredictionGNN(feature_dim=16, ...)`
- [x] 2.5 更新 `_load_model()` 中 checkpoint 字段 — `input_dim` 改为 `feature_dim`，新增 `num_message_layers` 字段

> **验证门禁 Phase 2**：`python gnn_attack.py -points=cali_self -dist=uniform -p=100 --model <phase1_model>` 不报维度错误，生成有效图。

## 3. Phase 3: Bug 修复与兼容性

- [x] 3.1 修复 `data_loader.py` 中硬编码数据集的路径解析 — 移除 `_extract_variable_from_py` 和 `exec()` 方式，改为 `sys.path` + `from dataset import cali_all, nh`
- [x] 3.2 修复 `gnn_attack.py` 中 `_get_correct_edges_at_scale` 自环 Bug — 第 102-104 行 `tuple(coord)` 改为 `tuple(dictionarry[nt])`
- [x] 3.3 在 `requirements.txt` 中添加 `sympy>=1.9`
- [x] 3.4 更新 `gnn_attack/README.md` — 标注 checkpoint 格式变更（破坏性），更新命令示例中的参数

> **验证门禁 Phase 3**：`python gnn_attack.py -points=cali_50 -dist=uniform -p=10 --model <model> --baseline` 全流程无报错，无 `ModuleNotFoundError`。

## 4. Phase 4: 文档产出

- [x] 4.1 创建 `docs/` 目录和 `docs/gnn-attack-design.md`
- [x] 4.2 撰写"设计动机与问题分析"章节 — 说明原始方法局限性和 GNN 改进点
- [x] 4.3 撰写"架构演进"章节 — v0（矩阵输入）→ v1（固定特征输入）的对比 ASCII 图，解释 input_dim 耦合问题的根源和解决方案
- [x] 4.4 撰写"数据流与组件关系"章节 — 训练阶段和推理阶段的完整 ASCII 数据流图
- [x] 4.5 撰写"关键算法"章节 — `extract_node_features`、`extract_edge_features`、GraphSAGE 消息传递、Focal Loss 的算法步骤说明
- [x] 4.6 撰写"与原始方法对比"章节 — 表格对比 GNN 和 Original 在信息利用、采样率鲁棒性、泛化能力、计算复杂度、已知限制五个维度

> **验证门禁 Phase 4**：文档包含架构对比 ASCII 图、至少 3 个算法的伪代码/步骤说明、1 张对比表。
