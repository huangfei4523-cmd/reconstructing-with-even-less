## 为什么

全量代码审计发现 `gnn_self_training.py` 中存在两个缺陷：

1. **微调时消息传递图不完整**：`model.forward()` 将 `edge_index` 同时用于消息传递和边预测。微调时只传入伪标签边（正或负），导致消息只在少量边（~12N 条）上进行，绝大多数节点无消息流入，节点嵌入质量下降为空向量聚合。

2. **一致性正则化 logits 长度不匹配**：对共现矩阵做扰动后 `build_cooc_message_graph` 可能产生不同数量的边，导致 `logits_orig` 和 `logits_p` 长度不一致，当前用 `min_len` 截断只是临时补救。

## 变更内容

- 重写 `SelfTrainingLoop` 中的微调步骤：用完整共现图做消息传递获得节点嵌入，然后单独对伪标签边计算 logits 和 loss
- 修复一致性正则化：使用相同的 `edge_idx_t` 做消息传递，只替换 `node_feat`，确保 logits 长度一致
- 验证修复后 `pytest tests/test_phase2_pseudo.py tests/test_integration.py` 全部 PASS

## 功能 (Capabilities)

### 修改功能
- `gnn-self-training`: Phase 2 微调阶段的消息传递必须使用完整共现图，在伪标签边上单独计算边预测 logits；一致性正则化必须使用相同图结构的 logits

## 影响

- 修改 `gnn_self_training.py` 中 `SelfTrainingLoop` 的微调循环（约 20 行）
