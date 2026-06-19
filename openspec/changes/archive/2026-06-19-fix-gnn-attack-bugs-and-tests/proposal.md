## 为什么

`fix-gnn-architecture` 变更完成后，训练过程中暴露了新的运行时问题（`IndexError` 数组越界、训练推理不一致），同时全面代码审查发现了残留缺陷：冗余特征维度削弱模型表达力、学习率调度器条件化失效、边特征训练/推理路径不一致、未使用导入等。必须逐一修复并增加测试用例防止回归。

## 变更内容

### 修复

- 修复 `train_gnn_model` 中 `scheduler.step()` 仅在有验证集时才执行的问题——改为每个 epoch 无条件调用
- 修复 `extract_node_features` 中 F3/F5/F7 三个特征维度计算公式完全相同（均为 `nonzero/N`）的冗余——改为 3 个不同的统计量
- 修复训练时 `edge_features=None` 但推理时传入边特征的训练/推理不一致——训练时也计算并传入边特征
- 移除 `defaultdict`（未使用 import）和 `all_tokens`（死变量）等代码清理
- 优化 `_make_sample_from_points` 中 3D 场景的查询生成循环——对较大网格使用采样替代全枚举

### 新增

- 新增 `tests/` 目录和测试套件：`test_extract_features.py`、`test_graph_build.py`、`test_edge_prediction.py`、`test_accuracy.py`

## 功能 (Capabilities)

### 修改功能

- `gnn-model-training`: 修复 scheduler 无条件 step、训练时启用边特征、3D 查询采样优化
- `gnn-inference`: `extract_node_features` 特征去冗余

### 新增功能

- `gnn-test-suite`: 自动化测试覆盖——验证特征提取维度、图构建稀疏格式、边预测推理、精度计算

## 影响

- 受影响文件：`gnn_model.py`、`gnn_attack.py`
- 新增文件：`tests/__init__.py`、`tests/test_extract_features.py`、`tests/test_graph_build.py`、`tests/test_edge_prediction.py`、`tests/test_accuracy.py`
- 模型架构不变（feature_dim 仍为 16），已有 checkpoint 兼容
- 特征 F3/F5/F7 含义变更——重新训练可获得更好精度，但已有模型仍可加载
