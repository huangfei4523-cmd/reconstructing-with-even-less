### 需求: 节点特征提取

`extract_node_features()` 必须从共现矩阵中提取固定维度 F=16 的互不冗余统计特征向量，每个特征与数据集点数 N 无关。

#### 场景: 提取完整特征集

- **当** 输入为 N×N 共现矩阵
- **那么** 系统对每个节点计算 16 维特征向量：F1 平均共现度、F2 最大共现度、F3 共现稀疏度 `nonzero/N`、F4 标准差、F5 共现偏度 `skew(vec)`、F6 总强度归一化、F7 共现峰度 `kurtosis(vec)`、F8 聚类系数（保留）、F9-F12 中心性和集中度、F13-F16 保留扩展位

#### 场景: 特征维度互不冗余

- **当** 输入 N×N 共现矩阵（N≥4）
- **那么** F3 ≠ F5 ≠ F7（当 N>4 且共现向量有差异时）

#### 场景: N 极小的情况

- **当** N < 2
- **那么** 系统返回全零特征矩阵 `[N, 16]`

### 需求: 边特征提取

`extract_edge_features()` 必须为每条候选边计算 pairwise 统计特征。

#### 场景: 计算边特征

- **当** 输入为共现矩阵、响应集合和边索引 `[2, E]`
- **那么** 系统对每条边 `(i,j)` 计算 4 维特征：归一化共现计数、Jaccard 系数、余弦相似度、Adamic-Adar 指数

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

`gnn_range_attack.gnn_range_attack()` 必须加载训练好的 GNN 模型，从共现矩阵提取节点特征和边特征后进行 GraphSAGE 消息传递和边预测。

#### 场景: 特征提取与消息传递图构建

- **当** 共现矩阵计算完毕
- **那么** 系统先调用 `extract_node_features()` 生成 `[N, F]` 节点特征，再调用 `build_message_passing_graph()` 基于共现余弦相似度构建 k=min(10, N-1) 的稀疏最近邻图 `edge_index[2, E]`
- **而且** 调用 `extract_edge_features()` 为消息传递图的每条边计算 4 维边特征

#### 场景: 加载模型

- **当** `model_or_path` 是文件路径字符串
- **那么** 系统从 checkpoint 加载 `model_state_dict`、`feature_dim`、`hidden_dim`、`emb_dim`

#### 场景: GNN 前向推理

- **当** 节点特征和消息传递图就绪
- **那么** 系统以 eval 模式运行 `EdgePredictionGNN.forward()`，得到节点嵌入和所有点对的边概率

#### 场景: 边提取

- **当** 边概率矩阵计算完毕
- **那么** 系统提取所有 `prob >= threshold` 的点对 `(i,j)`（i < j）作为预测边

#### 场景: 图构建

- **当** 预测边列表就绪
- **那么** 系统构建 NetworkX 图，所有点作为孤立节点加入
- **而且** 返回 Graph 对象、边数量和 `go_back` 映射

### 需求: 模型加载兼容性

`_load_model()` 必须支持传入模型对象和文件路径两种模式。Checkpoint 格式使用 `feature_dim` 字段。

#### 场景: 传入模型对象

- **当** `model_or_path` 是 `EdgePredictionGNN` 实例
- **那么** 系统直接返回该实例

#### 场景: 传入文件路径（新 checkpoint 格式）

- **当** `model_or_path` 是字符串
- **那么** 系统检查文件是否存在，使用 `torch.load` 加载 checkpoint
- **而且** checkpoint 必须包含 `model_state_dict`、`feature_dim` 键

#### 场景: 传入旧 checkpoint 格式

- **当** checkpoint 包含 `input_dim` 但无 `feature_dim`
- **那么** 系统抛出 `ValueError` 并提示用户使用新模型重新训练

#### 场景: 模型文件不存在

- **当** `model_or_path` 指向不存在的文件
- **那么** 系统抛出 `FileNotFoundError`
