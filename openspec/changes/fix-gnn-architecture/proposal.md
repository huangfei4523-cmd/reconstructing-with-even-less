## 为什么

前序代码审查已确认 `gnn_attack/` 模块存在 6 个缺陷，其中 3 个为阻断性 Bug（模型 input_dim 训练/推理不匹配、data_loader 路径错误、max_correct 评估自环）。核心矛盾在于：当前 GNN 模型将共现矩阵的行向量直接输入 Linear 层（`nn.Linear(N, 64)`），导致模型参数与数据集点数 N 绑定，训练和推理的 N 必须一致——这一约束在合成网格训练与真实数据集推理之间必然违反。必须从根本上重新设计 GNN 的输入编码方式，从「固定 N 的矩阵输入」切换到「固定维度 F 的图节点特征输入」。同时需要产出该实现的原理与流程分析文档。

## 变更内容

### 重构

- **BREAKING**: 重写 `EdgePredictionGNN` 架构——从 `nn.Linear(N, 64)` 矩阵输入改为固定维度 `node_features[N, F]` 图节点特征 + 消息传递 + 边预测，`F≈16` 为统计特征数，与 N 无关
- **BREAKING**: 重写 `generate_training_data()` 为 v2 版本——新增 3D 点云、不规则形状、随机点云等多样化训练场景，覆盖 cali/nh/crg/boat 等真实数据集分布
- 重写 `_build_knn_graph()` 为基于稀疏邻接的 `build_message_passing_graph()`
- 新增 `extract_node_features()` 模块——从共现矩阵提取统计特征
- 新增 `extract_edge_features()` 模块——计算 pairwise 特征用于边预测增强
- 修复 `data_loader.py` 路径——从 `exec()` 解析原 attack.py 改为包内导入 dataset.py
- 修复 `_get_correct_edges_at_scale` 自环 Bug——`tuple(coord)` 改为 `tuple(dictionarry[nt])`
- 在 `requirements.txt` 中添加缺失的 `sympy` 依赖

### 新增

- 产出 `docs/gnn-attack-design.md`——GNN 攻击方法的原理、架构、数据流分析文档

## 功能 (Capabilities)

### 修改功能

- `gnn-model-training`: EdgePredictionGNN 架构从矩阵输入改为图节点特征输入，训练数据生成器扩展为 2D/3D/不规则多场景
- `gnn-inference`: 共现矩阵计算保持不变，新增节点特征提取和边特征提取步骤，推理流程改为图消息传递
- `gnn-attack-pipeline`: 主入口的 data_loader 改为包内导入，评估逻辑修复自环 Bug
- `baseline-comparison`: requirements.txt 补全 sympy 依赖

### 新增功能

- `gnn-analysis-doc`: GNN 攻击方法的原理分析文档，包含设计动机、架构演进、数据流、关键算法说明

## 影响

- 受影响目录：`gnn_attack/` 全部源码文件
- 受影响主规范：`gnn-model-training`、`gnn-inference`、`gnn-attack-pipeline`、`baseline-comparison`
- 新增主规范：`gnn-analysis-doc`
- 模型 checkpoint 格式变更（新增 `feature_dim` 等字段），旧 checkpoint 不兼容
- 外部依赖不变（无需 PyG），sympy 已补充
