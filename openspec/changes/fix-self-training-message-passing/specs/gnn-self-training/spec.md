## 修改需求

### 需求: Phase 2 自训练模块

`gnn_self_training.py` 的微调阶段必须使用完整共现图进行消息传递，伪标签边单独计算边预测。

#### 场景: 微调消息传递使用完整图

- **当** `SelfTrainingLoop` 的每次微调迭代执行 `model.forward()`
- **那么** 消息传递的 `edge_index` 必须是完整的共现图（`edge_idx_t`，通过 `build_cooc_message_graph` 构建）
- **而且** 边预测 logits 只对伪标签边计算：`edge_predictor(cat(node_emb[src], node_emb[dst], edge_feat))`
- **而且** 不得只用伪标签边（正或负）作为消息传递图

#### 场景: 一致性正则化 logits 长度一致

- **当** 计算 `C_pert` 扰动后的 consistency loss
- **那么** `logits_orig` 和 `logits_p` 必须使用相同的 `edge_idx_t` 做消息传递
- **那么** 只需要替换 `node_feat`（扰动前后），不替换 `edge_index`
- **那么** `logits_orig` 和 `logits_p` 长度必然相同，无需 `min_len` 截断
