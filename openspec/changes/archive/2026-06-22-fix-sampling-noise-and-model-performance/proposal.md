# 修复采样噪声失效 + 模型性能诊断

## 问题诊断

p=100% 时 cali_self 的评估结果：

```
Precision=0.0493 (5%)  Recall=0.0735 (7%)  Edges=198  Phase2=0.24s
```

存在两个独立问题：

### 问题 1：p 参数因归一化而失效（致命）

当前代码 (`attack_gnn.py:112-117`)：

```python
if args.p < 100:
    C_target = C_target * (args.p / 100.0)    # 乘以比例因子
total = C_target.sum(axis=1, keepdims=True) + 1e-8
C_target = C_target / total                     # 逐行归一化
```

归一化后 p 因子完全消掉：`C_norm = (C×p) / Σ(C×p) = C / ΣC`。p=1%、p=10%、p=100% 产生完全相同的输入给 GNN。`--p` 参数是空操作。

真采样应当引入随机方差——每个点对的共现计数是二项随机变量 Binomial(n=C_full, p=p/100)，期望=C_full×p/100，方差=C_full×p(1-p)/10000。当前丢失了这个方差。

### 问题 2：p=100% 时模型能力不足（P/R ~5-7%）

模型在合成数据（grid_2d/random_2d，N=20-800）上预训练后，迁移到 cali_self（N=99）时 Phase 2 仅 0.24 秒就收敛。可能的原因：

1. 合成数据的共现分布与真实数据差异大，模型学到的模式不泛化
2. Phase 2 只微调 edge_predictor + 最后一层 mp_layer，表达能力有限
3. Phase 1 验证集 P/R 可能本身就低（需要先查证）

## 修改范围

| 文件 | 改动 |
|------|------|
| `attack_gnn.py` | 采样率通过加二项噪声实现，不再用乘因子 |
| `gnn_self_training.py` | 增加 Phase 2 诊断日志（伪标签熵、预测置信度分布） |
| `gnn_model.py` | 前向传播时打印共现统计，便于对比合成数据 vs 真实数据 |

## 需求

- 修改 `gnn-attack-pipeline`：采样率通过随机噪声而非乘因子实现
- 修改 `gnn-self-training`：增加诊断日志
