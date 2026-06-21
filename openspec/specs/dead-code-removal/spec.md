### 需求: 删除所有不被新架构引用的源代码文件

`gnn_attack/` 目录下不得存在被三阶段架构取代的旧文件：`train_gnn_v2.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py`。

#### 场景: 旧文件不存在

- **当** 检查 `gnn_attack/` 目录
- **那么** 不存在 `gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py`

### 需求: 删除 temporary 文件和 deprecated 函数

根目录不得存在临时 `.pth` 文件。`gnn_model.py` 不得包含 `build_message_passing_graph` 等 deprecated 函数。

#### 场景: 根目录无 pth 文件

- **当** 列出 `gnn_attack/` 根目录
- **那么** 不存在任何 `.pth` 文件

#### 场景: deprecated 函数已移除

- **当** 搜索 `gnn_model.py`
- **那么** 不存在 `build_message_passing_graph(` 或 `build_message_passing_graph_from_features(` 函数定义
