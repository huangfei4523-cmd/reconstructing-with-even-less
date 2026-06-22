## 1. 修复采样噪声

- [x] 1.1 在 `attack_gnn.py` 的闭式公式后，p<100 时加二项相对噪声：σ_rel = sqrt((100-p)/(p*c + eps))，noise = randn * σ_rel * c，clip(c+noise, 0)
- [x] 1.2 确认 p=100 时无噪声分支不做任何修改
- [x] 1.3 验证：对 cali_self 分别跑 p=100 和 p=10，确认两次 C_target 的归一化值不同（p=10 有波动）

## 2. Phase 2 诊断日志

- [x] 2.1 在 `SelfTrainingLoop` iter 日志中加 Top-10% 伪正标签的平均概率
- [x] 2.2 在 `SelfTrainingLoop` 启动时打印 C_target 统计（共现强度均值/std、非零邻居数均值/std）
- [x] 2.3 在 `attack_gnn.py` Phase 2 结束后打印预测概率分布摘要（min/p25/median/p75/max）

## 3. Phase 1 验证总结

- [x] 3.1 在 `train_phase1()` 多阈值循环后加最佳 P/R 行打印
- [x] 3.2 将最佳 threshold+P/R 写入 checkpoint 的 `phase1_val_best` 字段

## 4. 验证

- [x] 4.1 运行 `python attack_gnn.py --train --target-data cali_self --p 100 --train-samples 10 --train-epochs 2` 确认无报错
- [x] 4.2 运行 `python attack_gnn.py --train --target-data cali_self --p 10 --train-samples 10 --train-epochs 2` 确认 C_target 与 p=100 不同
- [x] 4.3 运行 `pytest tests/ -v` 全测试 PASS
