## 上下文

`implement-three-phase-gnn-attack` 完成后，新老代码并存。依赖关系分析如下：

```
新架构 (v2):                    旧架构 (v0/v1):
train_gnn_v2.py ──┐            train_gnn.py (孤立, 无引用)
attack_gnn_v2.py ─┤            gnn_attack.py → gnn_range_attack.py → range_attack.py
gnn_self_training  ├─→ gnn_model.py ←─ (互相形成闭环, 新代码不引用)
gnn_reconstruction─┘            │
                                └── 含 _get_correct_edges_at_scale / check_accuracy_with_edges

测试文件:
  test_phase1/2/3/integration.py → 新架构 ✓
  test_graph_build.py → build_message_passing_graph (deprecated) ⚠
  test_extract_features.py → extract_node_features (仍有效) ✓
  test_edge_prediction.py → EdgePredictionGNN (仍有效) ✓
  test_accuracy.py → _get_correct_edges_at_scale (将从 gnn_attack 移动) ⚠
```

## 目标 / 非目标

**目标：**
- 删除所有不被新架构引用的旧 `.py` 文件
- 将仍被测试引用的有价值函数迁移到 `gnn_model.py`
- 统一输出路径到 `results/`
- 提供 `--train` 复合入口一键跑通全流程

**非目标：**
- 不修改任何模型架构或训练算法
- 不新增数据集或采样方式
- 不修改 `implement-three-phase-gnn-attack` 中已完成的 Phase 2/3 逻辑

## 决策

### D1: 评估函数迁移到 `gnn_model.py`

`_get_correct_edges_at_scale` 和 `check_accuracy_with_edges` 从 `gnn_attack.py` 提取到 `gnn_model.py` 的 §7 评估工具区。理由：这两函数是纯工具函数，与 GNN 模型共存于同一模块便于复用，也避免为它们单独创建新的模块文件。

替代方案：创建 `evaluation.py` 独立模块 — 拒绝，因为只有 2 个函数不值得单独成模块。

### D2: 旧训练+推理入口直接删除，不保留别名

`train_gnn.py`、`gnn_range_attack.py` 不保留任何向后兼容别名。所有新代码已切换到 `train_gnn_v2.py`、`gnn_self_training.py`。

### D3: `build_message_passing_graph` 和 `build_message_passing_graph_from_features` 标记为 deprecated 后删除

这两个函数在上一次 `implement-three-phase-gnn-attack` 中已被 `build_cooc_message_graph` 替代并标记为 deprecated。仅旧 `test_graph_build.py` 还在测试它们。将该测试文件改为测试 `build_cooc_message_graph` 后，删除 deprecated 函数。

### D4: 复合入口 `--train` 参数

`attack_gnn_v2.py` 新增 `--train` + `--train-epochs` + `--train-samples` 参数。当指定 `--train` 时：
1. 内部调用 `generate_training_data_v2` + 训练循环（复用 `train_gnn_v2.py` 的 `train_phase1` 函数，提取为可导入函数）
2. 训练完成自动衔接到 Phase 2 自训练 + Phase 3 重建

注意 `train_gnn_v2.py` 目前是一个独立脚本，需要将其训练逻辑提取为 `def train_phase1(args) -> model_path` 函数，供 `attack_gnn_v2.py` 调用。

### D5: 统一输出路径

| 文件类型 | 旧路径 | 新路径 |
|---------|--------|--------|
| 模型 checkpoint | `gnn_model.pth`, `test.pth` (根目录) | `results/<name>.pth` |
| 攻击结果 JSON | `results/results_*.json` | `results/result_*.json` |
| 重建可视化 | `results_v2/recon_*.png` | `results/recon_*.png` |

## 风险 / 权衡

- [风险] 删除 `range_attack.py` 后无法做 baseline 对比 → 新架构尚未实现 baseline 对比功能，这是后续迭代的需求
- [风险] `test_graph_build.py` 改为测试 `build_cooc_message_graph` 后可能需要调整用例 → 按新函数签名修改即可
