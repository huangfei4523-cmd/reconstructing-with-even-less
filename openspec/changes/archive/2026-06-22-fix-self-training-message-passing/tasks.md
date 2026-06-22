## 1. 修复微调消息传递

- [x] 1.1 在 `SelfTrainingLoop` 微调循环中，先调用 `model(node_feat, edge_idx_t, None, edge_w_t)` 用完整图获得 `node_emb`
- [x] 1.2 对正伪标签边单独拼接 `node_emb[pseudo_pos_src]`, `node_emb[pseudo_pos_dst]` → `edge_predictor` → pos_logits
- [x] 1.3 对负伪标签边单独拼接 → `edge_predictor` → neg_logits
- [x] 1.4 移除 `pos_tensor.t().contiguous()` 和 `neg_tensor.t().contiguous()` 的直接传入

## 2. 修复一致性正则化

- [x] 2.1 将第 155-161 行的两次 `forward()` 改为使用相同的 `edge_idx_t`
- [x] 2.2 移除 `min_len` 截断 hack

## 3. 验证

- [x] 3.1 运行 `pytest tests/test_phase2_pseudo.py -v` — 全部 PASS
- [x] 3.2 运行 `pytest tests/test_integration.py -v` — 全部 PASS

> **最终验证门禁：** 测试全 PASS，Phase 2 收敛轮数不劣于修复前
