## 为什么

`attack_gnn.py` 流式构建共现矩阵 C_target 在 cali_50（N≈1000）上实测耗时超过 24 小时。根本原因：遍历 1.6M 条采样矩形查询，每条查询内嵌 `O(|r|²)` 共现累加。实际存在闭式公式可在 `O(N²)` 时间内计算出完全等价的结果。

## 变更内容

- 用闭式公式替换暴力遍历构建 C_target：C[i,j] = min(xᵢ,xⱼ) × min(yᵢ,yⱼ) × (N0-max(xᵢ,xⱼ)) × (N1-max(yᵢ,yⱼ))
- 采样率 p% 通过直接乘系数 p/100 实现，跳过 `get_responses_no_vals` + `sample_uniform`
- 确保闭式公式的共现矩阵与暴力遍历的统计结果**完全一致**

## 功能 (Capabilities)

### 修改功能
- `gnn-attack-pipeline`: 共现矩阵的构建方式从暴力流式循环改为闭式公式，要求输出结果与旧方法完全等价

## 影响

- 修改 `attack_gnn.py` 约 20 行（共现矩阵构建段）
- 不再依赖 `get_responses_no_vals` 和 `sample_uniform` 计算 C_target
- 性能：cali_50 从 ~24h 降至 <1s
