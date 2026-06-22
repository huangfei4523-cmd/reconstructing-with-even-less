# 设计文档

## D1：采样噪声实现策略

### 当前（空操作）

```
C_target  = 闭式公式(N0,N1,所有点坐标)   -> 全量共现计数值
C_target *= p/100                         -> 均匀缩放
C_target  = normalize(C_target)           -> p 因子消失！
```

### 修复后（真实采样模拟）

对每对 (i,j)，共现计数 ~ Binomial(n=C_full[i,j], p=p/100)。

但逐个点对加二项噪声代价高。等效方案：对整个矩阵加相对噪声：

```
C_target  = 闭式公式(N0,N1,所有点坐标)
σ_rel     = sqrt((100-p)/(p * C_target + eps))   # 二项分布相对标准差
noise     = N(0, σ_rel) × C_target                # 逐元素噪声
C_target += noise
C_target  = clip(C_target, 0, ∞)
C_target  = normalize(C_target)
```

当 p=100：σ_rel=0，无噪声 → 等同当前
当 p=10：σ_rel≈0.3，噪声显著 → 模拟低采样不确定性
当 p=1：σ_rel≈1.0，高噪声 → 极度低采样场景

### 合理性

- p=100 → 零噪声：攻击者看到所有查询，共现矩阵精确
- p=10 → σ_rel=30%：每对 (i,j) 的共现计数波动约 30%
- p=1 → σ_rel=100%：极度低信噪比

## D2：Phase 2 诊断日志

当前 Phase 2 只有 iter 级别的边数和 overlap，不知道模型实际上在学什么。

增加：
- **伪标签置信度分布**：Top-K 正样本的平均概率 vs 随机猜测概率
- **预测熵**：所有 pair 的概率分布熵（高熵=不确定，低熵=确定但不一定对）
- **特征统计**：C_target 的度数分布均值/方差，与合成数据的对比

## D3：Phase 1 验证集 P/R 检查

如果 Phase 1 在合成验证集上 P/R 已经很低，那 Phase 2 不可能变好。

直接在 `train_phase1()` 已有的 threshold 评估基础上，增加最佳 threshold 的 P/R 总结行。
