## 为什么

`docs/design-three-phase-summary.md` 和 `docs/design-three-phase-detailed.md` 已定义了完整的三阶段可搜索加密 GNN 攻击方案。当前 `gnn_attack/` 模块仅实现了其中的预训练+单次推理部分（且已证实对 cali_50 完全无效）。本变更根据两份设计文档实现完整的三阶段攻击代码，并在实现过程中进行设计-代码比对 review 和测试验证。

## 变更内容

### 代码实现

- 重写 `gnn_model.py`：共现加权图消息传递、Phase 1 四步训练流程（参数空间覆盖→数据生成→训练→验证反馈）
- 新增 `gnn_self_training.py`：Phase 2 自训练模块——伪标签筛选、迭代微调、一致性正则化、收敛/发散判定
- 新增 `gnn_reconstruction.py`：Phase 3 形状重建模块——力导向布局、Procrustes 对齐、重建失败检测
- 新增 `train_gnn_v2.py`：Phase 1 完整训练脚本，基于参数空间网格采样的训练数据生成
- 新增 `attack_gnn_v2.py`：三阶段攻击主入口脚本——串联 Phase 1→2→3 全流程

### 设计比对 Review

- 建立「代码→设计」追溯表：每个接口点在代码中的位置对照设计文档中的定义章节
- 实现过程中逐项检查代码对齐情况

### 测试

- 新增 Phase 1 单样本生成测试（不同配置维度组合）
- 新增 Phase 2 伪标签筛选正确性测试
- 新增 Phase 3 力导向布局收敛性测试
- 新增三阶段集成测试（小规模 cali_self 端到端）

## 功能 (Capabilities)

### 修改功能

- `gnn-pretraining`: 从现有单阶段实现升级为设计文档定义的 Phase 1 四步流程
- `gnn-model-training`: 配合 Phase 1 重写的训练管线

### 新增功能

- `gnn-self-training`: Phase 2 自训练模块代码实现
- `shape-reconstruction`: Phase 3 形状重建模块代码实现
- `design-code-review`: 设计文档与代码实现的逐接口比对 review
- `integration-testing`: 三阶段集成测试套件

## 影响

- 修改文件：`gnn_model.py`（共现图消息传递 + Phase 1 流程）
- 新增文件：`gnn_self_training.py`, `gnn_reconstruction.py`, `train_gnn_v2.py`, `attack_gnn_v2.py`
- 新增测试：`tests/test_phase1_data.py`, `tests/test_phase2_pseudo.py`, `tests/test_phase3_layout.py`, `tests/test_integration.py`
- 新增 review 产出：`docs/design-code-review.md`（代码-设计对齐追溯表）
- 实现依据：`docs/design-three-phase-detailed.md`（权威参考）
