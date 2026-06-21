## 1. Phase 1 核心模型重构

- [x] 1.1 重写 `EdgePredictionGNN.forward` — 移除 k-NN 图，改为接受 edge_index + edge_weights，消息聚合按 edge_weights 加权求和
- [x] 1.2 新增 `build_cooc_message_graph(cooc_matrix)` — 返回所有 C[i,j]>0 的边索引和归一化权重 α=softmax(1/log(1+C[i,j]))
- [x] 1.3 重写 `generate_training_data_v2` 的默认 configs — 按详细设计 §1.2 Step 1 定义的参数空间网格（N×形状×p, 210 配置，每种 5-10 样本）
- [x] 1.4 新增 `train_gnn_v2.py` — Phase 1 训练入口，输出 checkpoint + 每 epoch 的 val P/R 日志
- [ ] 1.5 实现 Phase 1 反馈闭环 — 验证集 AUC<0.7 的配置自动增补样本并重新训练

> ⚠ 1.5 反馈闭环需要多轮训练（耗时），标记为后续优化项

> **验证门禁：** `python train_gnn_v2.py --samples 200 --epochs 10 --save test.pth` 训练 val_loss < 0.15, 输出多阈值 val P/R

## 2. Phase 2 自训练模块

- [x] 2.1 新增 `gnn_self_training.py` — `SelectPseudoLabels(prob_matrix, N)` 伪标签筛选函数（Top-2N 正、Bottom-10N 负、中间丢弃、覆盖度检查）
- [x] 2.2 新增 `SelfTrainingLoop(model, C_target, max_iter, device)` — 迭代微调主循环（推理→伪标签→微调→收敛检查）
- [x] 2.3 实现一致性正则化 — `PerturbCooc(C, drop_pct=0.05)` + L2 consistency loss
- [x] 2.4 实现收敛和发散判定 — 边重合率≥90%→收敛, 连续3轮下降→发散

> **验证门禁：** `python -m pytest tests/test_phase2_pseudo.py -v` 全部 PASS

## 3. Phase 3 形状重建模块

- [x] 3.1 新增 `gnn_reconstruction.py` — `ForceDirectedLayout(E_hat, N, anchors=None)` 力导向布局函数
- [x] 3.2 实现弹簧-斥力能量函数的迭代优化（至收敛或最大迭代）
- [x] 3.3 实现 `ProcrustesAlign(pred_pos, true_anchors)` 对齐函数
- [x] 3.4 实现 `CheckReconstructionFailure(pos, E_hat)` 失败检测（边数/方差/连通分量）

> **验证门禁：** `python -m pytest tests/test_phase3_layout.py -v` 全部 PASS

## 4. 攻击入口脚本

- [x] 4.1 新增 `attack_gnn_v2.py` — 三阶段串联脚本，参数 `--phase1-model` `--target-data` `--output`
- [x] 4.2 实现 Phase 1→Phase 2 的数据传递（checkpoint 加载 + 共现矩阵计算）
- [x] 4.3 实现 Phase 2→Phase 3 的数据传递（E_hat → 力导向布局）
- [x] 4.4 输出重建形状的可视化和 JSON 结果

> **验证门禁：** `python attack_gnn_v2.py --phase1-model test.pth --target-data cali_self` 无报错完成

## 5. 设计-代码比对 Review

- [x] 5.1 创建 `docs/design-code-review.md` — 建立追溯表框架
- [x] 5.2 逐接口比对 11 个接口点
- [x] 5.3 标记对齐状态：全部 ✓
- [x] 5.4 无 ✗/⚠ 项

> **完成标准：** `docs/design-code-review.md` 中 11 个接口点全部有记录，状态清晰

## 6. 测试套件

- [x] 6.1 创建 `tests/test_phase1_data.py`
- [x] 6.2 创建 `tests/test_phase2_pseudo.py`
- [x] 6.3 创建 `tests/test_phase3_layout.py`
- [x] 6.4 创建 `tests/test_integration.py`

> **验证门禁：** `python -m pytest tests/ -v` 全部 PASS（≥16 个测试用例）
