## 修改需求

### 需求: 训练管线

`train_gnn_model()` 必须每个 epoch 无条件调用 `scheduler.step()`，不受 `val_loader` 存在性影响。

#### 场景: 无验证集训练

- **当** `val_loader=None` 且训练进行到 epoch 结束
- **那么** 系统必须调用 `scheduler.step()` 更新学习率
- **而且** 学习率按 CosineAnnealingLR 计划衰减

### 需求: 合成训练数据生成

`generate_training_data_v2()` 必须在生成训练样本时同时产出响应集合，供训练阶段的边特征提取使用。

#### 场景: 训练数据包含响应集

- **当** 调用 `generate_training_data_v2(num_samples, configs)`
- **那么** 返回的三元组必须包含 `responses` 字段（每个样本对应的采样响应列表）
- **而且** `CooccurrenceDataset` 存储 responses 供训练循环使用

#### 场景: 训练时计算边特征

- **当** 训练循环前向传播
- **那么** 系统必须从 responses 调用 `extract_edge_features()` 计算真实边特征
- **而且** 禁止传入 `None` 用零填充替代

#### 场景: 3D 查询采样优化

- **当** 3D 场景的全部查询组合数超过 5000
- **那么** 系统随机采样至多 5000 条查询替代全枚举
- **而且** 采样后的响应仍能生成有意义的共现矩阵
