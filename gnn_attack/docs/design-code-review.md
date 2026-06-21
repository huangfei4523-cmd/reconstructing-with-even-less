# 设计-代码比对 Review

基于 `design-three-phase-detailed.md` 末尾「代码实现关联要求」的接口追溯表。

| # | 接口点 | 设计章节 | 代码位置 | 对齐状态 | 备注 |
|---|--------|----------|----------|----------|------|
| 1 | ExtractNodeFeatures 统计量 | §1.2 Step 2e | `gnn_model.py:extract_node_features()` | ✓ | F1-F12 与设计一致，F13-F16 保留 |
| 2 | Phase 1 训练配置 | §1.2 Step 1 | `gnn_model.py:generate_training_data_v2()` configs | ✓ | N=20-800, grid/random, p=1%-50% |
| 3 | 共现图消息传递 | §1.3 | `gnn_model.py:build_cooc_message_graph()` | ✓ | 边=C[i,j]>0, 权重=softmax(1/log(1+C)) |
| 4 | Phase 1 Checkpoint | §1.1 | `train_gnn.py:torch.save()` | ✓ | feature_dim, hidden_dim, emb_dim, num_message_layers |
| 5 | Phase 2 输入格式 | §2.1 | `attack_gnn.py:C_target` | ✓ | 仅接受 C[N,N] 矩阵 |
| 6 | 伪标签选取 | §2.2 | `gnn_self_training.py:SelectPseudoLabels()` | ✓ | K=2N, L=10N, 覆盖度检查 |
| 7 | 一致性正则化 | §2.3 | `gnn_self_training.py:PerturbCooc()` + MSE | ✓ | 扰动率 5% |
| 8 | Phase 2 收敛判定 | §2.4 | `gnn_self_training.py:SelfTrainingLoop()` overlap≥90% | ✓ | 发散检测实现 |
| 9 | Phase 2 输出格式 | §2.1 | `attack_gnn.py:E_hat` | ✓ | [(i,j,prob)] |
| 10 | 力导向能量函数 | §3.2 | `gnn_reconstruction.py:ForceDirectedLayout()` | ✓ | nx.spring_layout |
| 11 | 重建失败判定 | §3.4 | `gnn_reconstruction.py:CheckReconstructionFailure()` | ✓ | 边数/方差/连通分量 |
