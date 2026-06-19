## 1. 代码审查与分析

- [x] 1.1 完成 `gnn_attack/` 目录下全部 11 个源文件的逐文件阅读
- [x] 1.2 梳理模块间依赖关系和调用链（train_gnn.py → gnn_model.py → gnn_range_attack.py → gnn_attack.py）
- [x] 1.3 绘制架构图和数据流图

## 2. 核心数据结构和算法复原

- [x] 2.1 复原共现矩阵的计算逻辑（O(N²·|R|) 复杂度、Jaccard 归一化）
- [x] 2.2 复原 k-NN 消息传递图的构建逻辑（余弦相似度 + 行归一化）
- [x] 2.3 复原 EdgePredictionGNN 的前向传播流程（编码→双消息传递→边预测）
- [x] 2.4 复原 Focal Loss + 正样本加权的训练损失设计
- [x] 2.5 复原训练数据生成策略（合成网格 + 5% 响应采样）

## 3. 缺陷识别

- [x] 3.1 识别 input_dim 训练/推理维度不匹配问题（模型架构层面）
- [x] 3.2 识别 `data_loader.py` 中路径解析错误（`../reconstructing-with-even-less/attack.py` 不存在）
- [x] 3.3 识别 `_get_correct_edges_at_scale` 中自环 Bug（`tuple(coord)` 应为 `tuple(dictionarry[nt])`）
- [x] 3.4 识别 `sympy` 依赖在 `requirements.txt` 中缺失
- [x] 3.5 识别大 N 时的内存风险（`[N, N, 2*emb_dim]` 张量）
- [x] 3.6 识别训练数据局限性（仅 2D 规则网格，泛化能力未验证）

## 4. 文档产出

- [x] 4.1 撰写 proposal.md（变更动机、范围、非目标）
- [x] 4.2 撰写 design.md（架构决策、关键接口、风险表）
- [x] 4.3 撰写 specs/gnn-attack-pipeline/spec.md（主入口和数据集加载需求）
- [x] 4.4 撰写 specs/gnn-model-training/spec.md（模型架构和训练需求）
- [x] 4.5 撰写 specs/gnn-inference/spec.md（推理引擎需求）
- [x] 4.6 撰写 specs/baseline-comparison/spec.md（对比评估需求）
- [x] 4.7 撰写 tasks.md（本文档）
