## 新增需求

### 需求: 特征提取测试

`tests/test_extract_features.py` 必须验证 `extract_node_features` 和 `extract_edge_features` 的输入输出正确性。

#### 场景: 节点特征 shape 正确

- **当** 输入 N×N 共现矩阵
- **那么** 输出 shape 必须为 `[N, 16]`，dtype 为 float32

#### 场景: N<4 边界情况

- **当** N < 4（如 N=3）
- **那么** 输出必须为 `[N, 16]` 全零矩阵，不抛出异常

#### 场景: 边特征维度正确

- **当** 输入共现矩阵、响应集、edge_index `[2, E]`
- **那么** 输出 shape 必须为 `[E, 4]`

### 需求: 图构建测试

`tests/test_graph_build.py` 必须验证消息传递图的稀疏格式和连通性。

#### 场景: 稀疏格式 edge_index

- **当** 输入 N×N 共现矩阵（N=10）
- **那么** 输出 edge_index 的 shape 必须为 `[2, E]`，其中 E ≥ k·N（对称化后）
- **而且** 不存在自环（edge_index[0, e] ≠ edge_index[1, e]）

### 需求: 边预测测试

`tests/test_edge_prediction.py` 必须验证 EdgePredictionGNN 的 forward 和 predict_all_pairs。

#### 场景: forward 输出 shape

- **当** 输入 node_features `[N, 16]`、edge_index `[2, E]`
- **那么** `node_emb` shape 为 `[N, 32]`，`edge_logits` shape 为 `[E]`

#### 场景: predict_all_pairs 输出

- **当** 输入 node_emb `[N, 32]`
- **那么** 输出 `[N, N]` 边概率矩阵，且对称（`probs[i,j] == probs[j,i]`）

### 需求: 精度计算测试

`tests/test_accuracy.py` 必须验证 `_get_correct_edges_at_scale` 不生成自环且正确识别邻接边。

#### 场景: 无自环

- **当** 坐标 map 包含两个相邻点（曼哈顿距离 = 1）
- **那么** 返回的边集中每条边的两个端点坐标必须不同
- **而且** 不存在 `(coord, coord)` 形式的自环

#### 场景: 正确识别邻接边

- **当** 坐标 map 包含相邻点 (0,0) 和 (0,1)
- **那么** 返回的边集必须包含 `((0,0), (0,1))` 和 `((0,1), (0,0))`
