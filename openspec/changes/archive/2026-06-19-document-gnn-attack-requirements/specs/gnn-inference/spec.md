## 新增需求

### 需求: 共现矩阵计算

`gnn_range_attack.compute_cooccurrence_matrix()` 必须从采样响应集合计算归一化共现矩阵。

#### 场景: 收集所有点并建立索引

- **当** 输入响应列表（每个元素是点 ID 集合）
- **那么** 系统收集所有出现的点 ID，排序后建立 `token_to_idx` 映射
- **而且** 返回共现矩阵、点列表、索引映射三个值

#### 场景: 计算共现计数

- **当** 点索引建立完毕
- **那么** 系统对每条响应中的所有点对 `(i,j)`（i ≠ j）累加共现计数
- **而且** 矩阵对称：`cooc[i,j] += 1` 同时 `cooc[j,i] += 1`

#### 场景: Jaccard 风格归一化

- **当** 共现计数矩阵计算完毕
- **那么** 系统将每行除以该行的总和（+ 1e-8），得到归一化共现矩阵

### 需求: GNN 边预测推理

`gnn_range_attack.gnn_range_attack()` 必须加载训练好的 GNN 模型，从共现矩阵预测点对间的空间邻接关系。

#### 场景: 加载模型

- **当** `model_or_path` 是文件路径字符串
- **那么** 系统从 checkpoint 加载 `model_state_dict`、`input_dim`、`hidden_dim`、`emb_dim`
- **而且** 使用 checkpoint 中存储的 `input_dim` 重建模型架构（当 checkpoint 未存储时使用传入的 `input_dim` 作为回退）

#### 场景: 构建 k-NN 消息传递图

- **当** 共现矩阵为 N×N
- **那么** 系统基于共现行向量的余弦相似度构建 k=min(10, N-1) 的最近邻图
- **而且** 邻接矩阵进行 GCN 风格行归一化

#### 场景: GNN 前向推理

- **当** 共现矩阵和 k-NN 邻接矩阵就绪
- **那么** 系统以 eval 模式运行 `EdgePredictionGNN.forward()`，得到边对数几率和 sigmoid 概率

#### 场景: 边提取

- **当** 边概率矩阵 `edge_probs[N, N]` 计算完毕
- **那么** 系统提取所有 `prob >= threshold` 的点对 `(i,j)`（i < j）作为预测边

#### 场景: 图构建

- **当** 预测边列表就绪
- **那么** 系统构建 NetworkX 图，所有点作为孤立节点加入（保留孤立点信息）
- **而且** 返回 Graph 对象、边数量和 `go_back` 映射（节点索引 → 原始点 ID）

### 需求: 模型加载兼容性

`_load_model()` 必须支持传入模型对象（直接使用）和文件路径（从磁盘加载）两种模式。

#### 场景: 传入模型对象

- **当** `model_or_path` 是 `EdgePredictionGNN` 实例
- **那么** 系统直接返回该实例

#### 场景: 传入文件路径

- **当** `model_or_path` 是字符串
- **那么** 系统检查文件是否存在，使用 `torch.load` 加载 checkpoint
- **而且** checkpoint 必须包含 `model_state_dict` 键

#### 场景: 模型文件不存在

- **当** `model_or_path` 指向不存在的文件
- **那么** 系统抛出 `FileNotFoundError`
