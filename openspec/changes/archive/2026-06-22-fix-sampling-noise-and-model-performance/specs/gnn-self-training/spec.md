# gnn-self-training（增量修改）

## 修改需求

### 需求：Phase 2 诊断日志

ID: `req-phase2-diag`

SelfTrainingLoop 每轮迭代必须输出诊断信息，帮助判断模型是否在有效学习。

#### 场景：输出伪标签置信度分布

- **当** 每轮筛选伪标签后
- **那么** 打印 Top-K 正样本的平均预测概率、负样本的平均预测概率
- **且** 打印全部 N(N-1)/2 个点对的概率分布熵

#### 场景：输出特征统计对比

- **当** Phase 2 开始
- **那么** 打印 C_target 的关键统计：每行总强度均值/std、非零邻居数均值/std
- **且** 打印 Phase 1 预训练时合成数据的对应统计（从 checkpoint 读取或标注期望范围）

### 需求：Phase 1 验证集 P/R 总结

ID: `req-phase1-pr-summary`

Phase 1 训练结束时输出最佳阈值下的 Precision/Recall。

#### 场景：打印最佳验证 P/R

- **当** Phase 1 训练完成，多阈值评估结束后
- **那么** 打印一行 "Best: threshold=X.X P=Y.YYY R=Z.ZZZ"
- **且** 这个值被写入 Phase 1 checkpoint 的元数据中
