## 新增需求

### 需求: 共现矩阵完整传递链路

ID: `req-cooc-passthrough`

真正的共现矩阵必须从数据生成函数完整传递到训练和验证循环，不得在任何环节丢失。

#### 场景: _make_sample_from_points 返回共现矩阵

- **当** `_make_sample_from_points()` 完成一个训练样本的生成
- **那么** 返回值必须包含第 4 个元素 `cooc`（[N,N] float32 共现计数矩阵）
- **而且** `cooc` 是未归一化的、从采样响应中累加的点对共现计数

#### 场景: generate_training_data_v2 收集共现矩阵

- **当** `generate_training_data_v2()` 收集所有样本
- **那么** 返回值必须包含第 4 个列表 `cooc_list`
- **而且** `cooc_list` 长度等于 `num_samples`

#### 场景: CooccurrenceDataset 存储并返回共现矩阵

- **当** `CooccurrenceDataset.__getitem__(idx)` 被调用
- **那么** 返回值必须为四元组 `(node_feat, adj_gt, resp, cooc)`
- **而且** `cooc` 不被转换为 torch.Tensor（保持 numpy 数组以节省内存）

#### 场景: build_cooc_message_graph 接收真正的共现矩阵

- **当** 训练或验证循环构建消息传递图
- **那么** `build_cooc_message_graph()` 的输入必须是真正的 `cooc` 矩阵（从 `CooccurrenceDataset` 获取）
- **而且** 禁止使用 `node_feat[:,:3] @ node_feat[:,:3].T` 作为替代输入
