## 上下文

`SelfTrainingLoop` 在迭代微调阶段调用 `model.forward()` 两次（正伪标签边、负伪标签边），但 `forward()` 同时用 `edge_index` 做消息传递和边预测。当前只传入伪标签边 → 消息传递图只覆盖少数节点 → 节点嵌入质量下降。

## 目标 / 非目标

**目标：**
- 微调时用完整共现图做消息传递，在伪标签边上单独预测
- 一致性正则化使用相同消息传递图结构

**非目标：**
- 不修改 `EdgePredictionGNN.forward()` 的接口
- 不修改 `SelfTrainingLoop` 的整体收敛/发散逻辑

## 决策

### D1: 微调时用完整图做消息传递 + 伪标签边单独预测

`forward()` 返回 `node_emb`（节点嵌入）和 `edge_logits`。在微调时：先调用 `forward()` 传入完整共现图获得 `node_emb`；然后用 `node_emb` 对伪标签边计算 logits（拼接 `node_emb[src]` + `node_emb[dst]` → `edge_predictor`）。

替代方案：新增 `model.encode_only()` 方法 — 拒绝，增加接口复杂度且只在微调时使用。

### D2: 一致性正则化使用相同 edge_index

将第 155-161 行的两次 `forward()` 调用改为使用相同的 `edge_idx_t`，只替换 `node_feat`。

## 风险 / 权衡

- [风险] 微调时额外调用一次完整图 forward → 计算量增加 1 倍 — 可接受，微调样本数少
