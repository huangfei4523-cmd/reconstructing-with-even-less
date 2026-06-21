## 新增需求

### 需求:attack_gnn_v2.py 支持 --train 一键训练参数
`attack_gnn_v2.py` 必须支持 `--train` 参数，当指定时自动执行 Phase 1 训练（生成合成数据并训练 GNN），完成后继续执行 Phase 2 自训练和 Phase 3 形状重建。

#### 场景:指定 --train 时自动训练
- **当** 运行 `python attack_gnn_v2.py --train --target-data cali_self`
- **那么** 首先进入 Phase 1 训练（打印训练 loss 日志）
- **然后** 训练完成后自动进入 Phase 2 自训练
- **然后** Phase 2 完成后自动进入 Phase 3 形状重建
- **然后** 模型保存到 `results/phase1_model.pth`

#### 场景:指定 --train --train-epochs --train-samples 控制训练
- **当** 运行 `python attack_gnn_v2.py --train --train-epochs 10 --train-samples 50 --target-data cali_self`
- **那么** 训练使用 10 个 epoch 和 50 个合成样本
- **那么** 不使用默认的 epoch 和 samples 值

#### 场景:不指定 --train 时仅做推理
- **当** 运行 `python attack_gnn_v2.py --phase1-model results/model.pth --target-data cali_self`
- **那么** 跳过 Phase 1 训练
- **那么** 直接从 Phase 2 自训练开始

### 需求:train_gnn_v2.py 的训练逻辑可被 attack_gnn_v2.py 复用
`train_gnn_v2.py` 必须将训练逻辑提取为可导入函数 `train_phase1(epochs, samples, save_path, device) -> model_path`，供 `attack_gnn_v2.py` 调用。

#### 场景:train_phase1 可被外部导入
- **当** 从 `train_gnn_v2` import `train_phase1`
- **那么** 导入成功，不抛出异常

#### 场景:train_phase1 返回模型路径
- **当** 调用 `train_phase1(epochs=2, samples=10, save_path="results/test.pth", device="cpu")`
- **那么** 返回模型保存路径（字符串）
- **那么** `results/test.pth` 文件存在且可被 `torch.load` 加载

### 需求:复合入口必须输出 Precision/Recall 评估
当 `attack_gnn_v2.py` 运行完成时，如果目标数据集的真实坐标已知（所有内置数据集均满足），必须计算并输出 Precision 和 Recall。

#### 场景:复合入口输出 P/R
- **当** 运行 `python attack_gnn_v2.py --train --target-data cali_self`
- **那么** 终端输出包含 `Precision:` 和 `Recall:` 两行
- **那么** 结果 JSON 包含 `precision` 和 `recall` 字段
