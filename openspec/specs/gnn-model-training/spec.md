### 需求: 合成训练数据生成

`gnn_model.generate_training_data()` 必须在合成 2D 网格上模拟攻击场景，生成共现矩阵（特征）和邻接矩阵（标签）对。

#### 场景: 生成随机点集

- **当** 指定 `grid_size=(N0, N1)` 和 `max_points_per_cell=M`
- **那么** 系统在网格每个格子 `(i,j)` 中随机生成 1 到 M 个点，每个点分配唯一随机 token
- **而且** 点数 N 小于 4 的样本被跳过

#### 场景: 生成全部范围查询响应

- **当** 点集生成完毕
- **那么** 系统对 `min0..max0`、`min1..max1` 四重循环生成全部范围查询的响应集合
- **而且** 只保留至少有 2 个点的非空响应

#### 场景: 模拟 "Even Less" 采样

- **当** 全部响应生成完毕
- **那么** 系统按 `response_sampling_ratio`（默认 5%）随机采样响应子集

#### 场景: 计算共现矩阵

- **当** 采样响应就绪
- **那么** 系统计算 N×N 共现矩阵，其中 `matrix[i,j]` = 点 i 和点 j 在采样响应中同时出现的次数（i ≠ j）
- **而且** 矩阵为 float32 类型

#### 场景: 计算 Ground Truth 邻接矩阵

- **当** 点集坐标已知
- **那么** 系统计算 N×N 二值邻接矩阵，其中 `matrix[i,j] = 1.0` 当且仅当点 i 和点 j 的曼哈顿距离 ≤ 1

### 需求: EdgePredictionGNN 模型架构

`EdgePredictionGNN` 模型必须实现双消息传递层 + 边缘预测头的 GNN 架构。

#### 场景: 输入处理

- **当** 输入为 `[N, input_dim]` 共现矩阵和 `[N, N]` 归一化 k-NN 邻接矩阵
- **那么** 节点编码器通过两个 Linear+ReLU 层将每个点的共现特征向量编码为 `emb_dim` 维嵌入

#### 场景: 消息传递

- **当** 节点编码完成
- **那么** 系统通过两次 `Linear(emb_dim→emb_dim) @ (k-NN_adj @ h)` 操作进行 GCN 风格消息传递
- **而且** 每层后跟 LayerNorm 和 ReLU

#### 场景: 边预测

- **当** 消息传递完成得到 `[N, emb_dim]` 节点嵌入
- **那么** 系统对所有 `(i,j)` 对拼接 `[emb_i, emb_j]` 输入三层 MLP 预测对数几率
- **而且** 输出矩阵强制对称化：`(logits + logits.T) / 2`

### 需求: 训练管线

`train_gnn.py` 必须提供完整的模型训练脚本，包括数据生成、训练/验证划分、模型保存。

#### 场景: 训练数据划分

- **当** `samples` 个样本生成完毕
- **那么** 系统按 `val_split` 比例（默认 15%）随机划分训练集和验证集
- **而且** 每个样本以 batch_size=1 加载（因为不同样本的点数可能不同）

#### 场景: Focal Loss 训练

- **当** 训练迭代
- **那么** 系统使用 Binary Cross Entropy + 正样本加权（负样本数/正样本数）+ Focal Weight `(1-pt)²`
- **而且** 使用 Adam 优化器 + CosineAnnealingLR 学习率调度 + 梯度裁剪 1.0

#### 场景: 模型保存

- **当** 训练完成
- **那么** 系统保存 checkpoint 包含 `model_state_dict`、`input_dim`、`hidden_dim`、`emb_dim`、`train_loss`、`val_loss`
