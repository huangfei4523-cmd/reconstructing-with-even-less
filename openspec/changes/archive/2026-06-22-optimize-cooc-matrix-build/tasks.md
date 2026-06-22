## 1. 实现闭式公式

- [x] 1.1 在 `attack_gnn.py` 中用闭式公式替换暴力流式循环构建 C_target（约 20 行）
- [x] 1.2 采样率 p% 通过乘系数实现，跳过 `get_responses_no_vals` 和 `sample_uniform` 调用
- [x] 1.3 分离采样率参数到 `args.output/results_*.json` 中，确保结果可复现

## 2. 等价性验证

- [x] 2.1 对 cali_self（5 点）用闭式公式和暴力枚举分别构建 C_target，确认逐元素误差 < 1e-5
- [x] 2.2 确认采样率 p=10/p=50/p=100 的归一化 C_target 一致

**最终门禁：** cali_self 上闭式公式与暴力枚举的 C_target 逐元素相等（相对误差 < 1e-5），且攻击全流程无报错
