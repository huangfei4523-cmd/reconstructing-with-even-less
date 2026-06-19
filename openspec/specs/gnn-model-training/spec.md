### 需求: 合成训练数据生成

`gnn_model.generate_training_data_v2()` 必须在合成 2D/3D 网格、不规则点云等多种场景上模拟攻击，生成节点特征、边标签、采样响应的三元组。

#### 场景: 多场景配置混合

- **当** 调用 `generate_training_data_v2(num_samples, configs)`
- **那么** 系统按 `configs` 指定的场景类型和占比混合生成样本
- **而且** 支持的场景类型必须包含：`grid_2d`（规则网格）、`random_2d`（随机点云）、`random_3d`（3D 随机点云）

#### 场景: 生成随机点集

- **当** 指定场景类型和参数
- **那么** 系统在对应坐标空间生成点集，每个点分配唯一随机 token
- **而且** 点数 N 小于 4 的样本被跳过

#### 场景: 生成全部范围查询响应

- **当** 点集生成完毕
- **那么** 系统对坐标范围生成全部范围查询的响应集合
- **而且** 只保留至少有 2 个点的非空响应

#### 场景: 模拟 "Even Less" 采样

- **当** 全部响应生成完毕
- **那么** 系统按 `response_sampling_ratio`（默认 5%）随机采样响应子集

#### 场景: 计算共现矩阵与提取特征

- **当** 采样响应就绪
- **那么** 系统先计算 N×N 共现矩阵，再调用 `extract_node_features()` 提取 `[N, F]` 固定维度节点特征

#### 场景: 计算 Ground Truth 邻接标签

- **当** 点集坐标已知
- **那么** 系统计算 N×N 二值邻接矩阵，其中 `matrix[i,j] = 1.0` 当且仅当点 i 和点 j 的曼哈顿距离 ≤ 1
- **而且** 对 3D 点使用三维曼哈顿距离

#### 场景: 训练数据包含响应集

- **当** 调用 `generate_training_data_v2(num_samples, configs)`
- **那么** 返回的三元组必须包含 `responses` 字段（每个样本对应的采样响应列表）
- **而且** `CooccurrenceDataset` 存储 responses 供训练循环使用

#### 场景: 3D 查询采样优化

- **当** 3D 场景的全部查询组合数超过 5000
- **那么** 系统随机采样至多 5000 条查询替代全枚举

### 需求: EdgePredictionGNN 模型架构

`EdgePredictionGNN` 必须实现基于固定维度节点特征输入的 GraphSAGE 风格消息传递 + 边预测头架构，所有 `nn.Linear` 层输入维度与数据集点数 N 无关。

#### 场景: 固定维度节点编码

- **当** 输入为 `node_features[N, F]`（F=16 固定）和 `edge_index[2, E]` 稀疏消息传递图
- **那么** 节点编码器 `Linear(F→64)→BatchNorm→ReLU→Linear(64→32)→BatchNorm→ReLU` 将每个节点编码为 32 维嵌入

#### 场景: GraphSAGE 消息传递

- **当** 节点编码完成
- **那么** 系统执行两轮 GraphSAGE 风格消息传递：对每个节点 i，计算邻居消息的均值后与自身拼接，通过 Linear 层更新
- **而且** 每层后跟 LayerNorm 和 ReLU

#### 场景: 边预测（含边特征增强）

- **当** 消息传递完成得到 `[N, 32]` 节点嵌入
- **那么** 系统对 `edge_index` 中的每条边拼接 `[emb_i, emb_j, edge_feat_ij]`（维度 32+32+4=68），输入四层 MLP `68→64→32→16→1`

#### 场景: 推断所有点对的边概率

- **当** 需要评估所有 N×N 点对的边概率（全连接推理）
- **那么** 系统通过 `node_emb` 张量广播 + MLP 计算所有边缘对数几率
- **而且** 对 N > 500 的情况分批次计算以避免 OOM

### 需求: 训练管线

`train_gnn.py` 必须提供完整的模型训练脚本，适配新的图数据和模型架构。

#### 场景: 训练数据划分

- **当** `samples` 个样本生成完毕
- **那么** 系统按 `val_split` 比例（默认 15%）随机划分训练集和验证集
- **而且** 每个样本以 batch_size=1 加载

#### 场景: Focal Loss 训练

- **当** 训练迭代
- **那么** 系统使用 Binary Cross Entropy + 正样本加权 + Focal Weight `(1-pt)²`
- **而且** 使用 Adam 优化器 + CosineAnnealingLR 学习率调度 + 梯度裁剪 1.0

#### 场景: 训练时计算边特征

- **当** 训练循环前向传播
- **那么** 系统必须从 responses 调用 `extract_edge_features()` 计算真实边特征
- **而且** 禁止传入 `None` 用零填充替代

#### 场景: 学习率调度无条件执行

- **当** `val_loader=None` 且训练进行到 epoch 结束
- **那么** 系统必须调用 `scheduler.step()` 更新学习率

#### 场景: 模型保存

- **当** 训练完成
- **那么** 系统保存 checkpoint 包含 `model_state_dict`、`feature_dim`、`hidden_dim`、`emb_dim`、`num_message_layers`、`train_loss`、`val_loss`
