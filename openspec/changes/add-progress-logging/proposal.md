## 为什么

Phase 2 数据准备阶段和 SelfTrainingLoop 迭代阶段无任何进度反馈，用户看到"数据准备完成"后长时间无输出，无法判断是卡死还是在正常计算。

## 变更内容

- `attack_gnn.py` 两遍流式扫描加 tqdm 进度条
- `gnn_self_training.py` SelfTrainingLoop 每轮打印开始/结束和耗时
- 第一遍扫描优化：直接用 `set(points)` 替代流式收集 `all_pts`

## 影响

- 修改 `attack_gnn.py` (~5 行)
- 修改 `gnn_self_training.py` (~3 行)
