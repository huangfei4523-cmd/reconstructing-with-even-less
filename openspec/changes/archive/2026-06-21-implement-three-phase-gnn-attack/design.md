## 上下文

`docs/design-three-phase-detailed.md` 定义了 Phase 1-3 的完整算法流程、接口契约和验证标准。当前 `gnn_attack/` 模块仅部分实现了 Phase 1（且与设计文档有偏差：k-NN 图而非共现图、训练数据未覆盖参数空间等）。本设计给出如何按照设计文档实现完整三阶段的方案，以及如何确保代码与设计保持一致。

## 目标 / 非目标

**目标：**
- 按照详细设计的 §1.2 Step 1-4 实现 Phase 1 四步训练流程
- 按照详细设计的 §2 实现 Phase 2 自训练
- 按照详细设计的 §3 实现 Phase 3 形状重建
- 建立设计-代码追溯表，逐接口验证对齐
- 编写测试覆盖关键路径

**非目标：**
- 不改动 `range_attack.py`、`process_database.py`、`data_loader.py`
- 不修改已有 checkpoint 格式

## 决策

### D1. 模块拆分

```
gnn_attack/
├── gnn_model.py          ← Phase 1: 模型 + 共现图消息传递 + 训练
├── gnn_self_training.py  ← Phase 2: 自训练（NEW）
├── gnn_reconstruction.py ← Phase 3: 力导向布局（NEW）
├── train_gnn_v2.py       ← Phase 1 训练入口（NEW）
├── attack_gnn_v2.py      ← 三阶段攻击入口（NEW）
├── gnn_range_attack.py   ← 保留（兼容旧版本）
├── gnn_attack.py         ← 保留（兼容旧版本）
├── ...
```

**理由：** 每个阶段独立模块，便于单独测试和迭代。旧版脚本保留以确保向后兼容。

### D2. 设计-代码追溯机制

每个实现模块的内部文档注释必须标注对应设计文档章节：

```python
# §1.2 Step 1: 参数空间定义
PARAM_GRID = {...}  # ← 设计文档详细设计 §1.2 Step 1
```

review 产出 `docs/design-code-review.md` 以表格形式追溯。

### D3. 共现图消息传递的实现方式

```
设计文档 §1.3 要求:
  消息传递图 = {(i,j) | C[i,j] > 0}
  消息权重 α_ij = softmax(1/log(1+C[i,j]))

实现: 修改 EdgePredictionGNN.forward:
  - 移除 k-NN 图构建
  - 接受 edge_index（所有共现对）和 edge_weights（归一化共现值）
  - 消息聚合时按 edge_weights 做加权求和
```

### D4. 渐进式实现顺序

```
Phase A: 模型层改动（不影响其余代码）
  → gnn_model.py: 共现图消息传递 + Phase 1 四步流程
  → 测试: test_phase1_data.py

Phase B: Phase 2 + Phase 3 独立模块
  → gnn_self_training.py, gnn_reconstruction.py
  → 测试: test_phase2_pseudo.py, test_phase3_layout.py

Phase C: 集成入口 + 全流程测试
  → train_gnn_v2.py, attack_gnn_v2.py
  → 测试: test_integration.py

Phase D: 设计比对 review
  → docs/design-code-review.md
```

**理由：** 逐层构建，每层独立可测，避免大规模重构引入不可调试的 bug。

## 风险 / 权衡

| 风险 | 缓解 |
|------|------|
| Phase 2 自训练在 cali_50 上可能仍无法收敛 | 先用 cali_self（已知可控）验证，再逐步扩大 |
| 力导向布局计算量大（对 N=1000 可能是瓶颈）| 用 NetworkX 内置的 spring_layout（C 优化），或降采样边 |
| 删除旧代码可能破坏已有测试 | 旧版脚本保留不动，新增 v2 脚本，逐步迁移 |
| 设计文档后期更新导致代码脱节 | 设计-代码 review 表格在每次代码改动后同步更新 |
