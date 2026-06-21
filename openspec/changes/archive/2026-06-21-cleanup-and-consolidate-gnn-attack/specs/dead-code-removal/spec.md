## 新增需求

### 需求:删除所有不被新架构引用的源代码文件
删除 `gnn_attack/` 目录下不被 v2 架构任何文件（`train_gnn_v2.py`、`attack_gnn_v2.py`、`gnn_self_training.py`、`gnn_reconstruction.py`、`gnn_model.py`、`data_loader.py`、`process_database.py`、测试文件） import 的 `.py` 文件。

#### 场景:旧训练脚本被删除
- **当** 检查 `gnn_attack/train_gnn.py` 是否存在
- **那么** 该文件不存在

#### 场景:旧推理引擎被删除
- **当** 检查 `gnn_attack/gnn_range_attack.py` 是否存在
- **那么** 该文件不存在

#### 场景:原始方法实现被删除
- **当** 检查 `gnn_attack/range_attack.py` 是否存在
- **那么** 该文件不存在

#### 场景:旧入口脚本被删除且评估函数已迁移
- **当** 检查 `gnn_attack/gnn_attack.py` 是否存在
- **那么** 该文件不存在
- **当** 在 `gnn_model.py` 中搜索 `_get_correct_edges_at_scale` 和 `check_accuracy_with_edges`
- **那么** 两个函数均在 `gnn_model.py` 中可用

### 需求:删除 deprecated 函数
`gnn_model.py` 中不得包含 `build_message_passing_graph` 和 `build_message_passing_graph_from_features` 两个 deprecated 函数。

#### 场景:deprecated 函数已移除
- **当** 在 `gnn_model.py` 中搜索 `build_message_passing_graph(` 和 `build_message_passing_graph_from_features(`
- **那么** 两个函数均不存在

### 需求:删除临时模型文件
`gnn_attack/` 根目录不得包含 `.pth` 临时文件。

#### 场景:根目录无 pth 文件
- **当** 列出 `gnn_attack/` 根目录所有 `.pth` 文件
- **那么** 不存在任何 `.pth` 文件

### 需求:旧测试文件与新版函数对齐
`tests/test_graph_build.py` 必须使用 `build_cooc_message_graph` 而非已删除的 deprecated 函数进行测试。

#### 场景:test_graph_build 使用新函数
- **当** 搜索 `tests/test_graph_build.py` 中 import 的函数
- **那么** 必须从 `gnn_model` import `build_cooc_message_graph`
- **那么** 不得 import `build_message_passing_graph` 或 `build_message_passing_graph_from_features`
