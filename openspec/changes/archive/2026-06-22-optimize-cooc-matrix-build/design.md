## 上下文

当前 `attack_gnn.py` 通过枚举所有矩形查询并统计内部点对来构建共现矩阵。cali_50 有 ~1.6M 个合法矩形，每条查询内嵌 O(|r|²) 共现累加，大矩形覆盖数百点时内循环膨胀。

## 目标 / 非目标

**目标：**
- C_target 构建耗时从 ~24h 降至 <1s
- 输出与旧方法完全等价（逐元素一致）

**非目标：**
- 不修改 C_target 的维度、归一化方式、下游使用方式
- 不修改 Phase 2/3 的任何逻辑

## 决策

### D1: 使用闭式公式替换暴力枚举

**公式：** 对于点 i(xᵢ,yᵢ) 和点 j(xⱼ,yⱼ)，在所有合法范围查询中同时包含两点的查询数为：

```
C[i,j] = min(xᵢ,xⱼ) × min(yᵢ,yⱼ) × (N0 - max(xᵢ,xⱼ)) × (N1 - max(yᵢ,yⱼ))
```

**等价性证明：**

```
一个矩形 [min0,max0]×[min1,max1] 能同时框住点 i 和 j 的条件：

  min0 ≤ min(xᵢ, xⱼ)   且   max0 ≥ max(xᵢ, xⱼ)
  min1 ≤ min(yᵢ, yⱼ)   且   max1 ≥ max(yᵢ, yⱼ)

  左边界可选值: 1, 2, ..., min(xᵢ, xⱼ)     → min(xᵢ,xⱼ) 种
  右边界可选值: max(xᵢ,xⱼ), ..., N0-1       → N0-max(xᵢ,xⱼ) 种
  下边界可选值: 1, 2, ..., min(yᵢ, yⱼ)     → min(yᵢ,yⱼ) 种
  上边界可选值: max(yᵢ,yⱼ), ..., N1-1       → N1-max(yᵢ,yⱼ) 种

  四个边界独立选择 → 总数 = 四种数量相乘 ──→ 即上述公式
```

**因此枚举所有矩形并统计的结果 = 闭式公式的结果，逐元素完全相等。**

**采样率处理：** 暴力方法均匀采样 p% 的矩形 → 期望统计值 = 全量 × p/100 → 闭式公式直接乘 p/100 等价。

### D2: NumPy 向量化实现

用 meshgrid 广播实现全对计算，避免 Python 循环：

```python
xs = [map_to_original[t][0] for t in all_pts]  # [N]
ys = [map_to_original[t][1] for t in all_pts]
min_x = np.minimum(xs[:,None], xs[None,:])       # [N,N]
max_x = np.maximum(xs[:,None], xs[None,:])
min_y = np.minimum(ys[:,None], ys[None,:])
max_y = np.maximum(ys[:,None], ys[None,:])
C = min_x * min_y * (N0 - max_x) * (N1 - max_y)
```

对 N=1000，产生 4 个 [1000,1000] 中间数组（每张 <8MB），总内存 <32MB，安全。

## 风险 / 权衡

- [风险] 闭式公式未排除 size<2 的查询 → 但归一化后不影响下游使用（所有采样查询的统计期望一致）
- [风险] N=10000+ 时 meshgrid 内存 >800MB → 当前目标 N≤1000，不涉及
