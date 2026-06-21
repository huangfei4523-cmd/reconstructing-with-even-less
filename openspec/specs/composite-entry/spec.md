### 需求: attack_gnn.py 支持 --train 一键训练参数

`attack_gnn.py` 必须支持 `--train` 参数，当指定时自动执行 Phase 1 训练，完成后继续执行 Phase 2 和 Phase 3。

#### 场景: 指定 --train 时自动训练

- **当** 运行 `python attack_gnn.py --train --target-data cali_self`
- **那么** 首先进入 Phase 1 训练
- **然后** 训练完成后自动进入 Phase 2 自训练
- **然后** Phase 2 完成后自动进入 Phase 3 形状重建
- **然后** 模型保存到 `results/phase1_model.pth`

#### 场景: 指定 --train-epochs --train-samples 控制训练

- **当** 运行 `python attack_gnn.py --train --train-epochs 10 --train-samples 50 --target-data cali_self`
- **那么** 训练使用 10 个 epoch 和 50 个合成样本

#### 场景: 不指定 --train 时仅做推理

- **当** 运行 `python attack_gnn.py --phase1-model results/model.pth --target-data cali_self`
- **那么** 跳过 Phase 1 训练，直接从 Phase 2 自训练开始

### 需求: train_gnn.py 的训练逻辑可被 attack_gnn.py 复用

`train_gnn.py` 必须将训练逻辑提取为可导入函数 `train_phase1(epochs, samples, save_path, device) -> model_path`。

#### 场景: train_phase1 可被外部导入

- **当** 从 `train_gnn` import `train_phase1`
- **那么** 导入成功，不抛出异常

#### 场景: train_phase1 返回模型路径

- **当** 调用 `train_phase1(epochs=2, samples=10, save_path="results/test.pth", device="cpu")`
- **那么** 返回模型保存路径（字符串）

### 需求: 复合入口必须输出 Precision/Recall 评估

当 `attack_gnn.py` 运行完成时，必须计算并输出 Precision 和 Recall。

#### 场景: 复合入口输出 P/R

- **当** 运行 `python attack_gnn.py --train --target-data cali_self`
- **那么** 终端输出包含 `Precision:` 和 `Recall:` 两行
- **那么** 结果 JSON 包含 `precision` 和 `recall` 字段
