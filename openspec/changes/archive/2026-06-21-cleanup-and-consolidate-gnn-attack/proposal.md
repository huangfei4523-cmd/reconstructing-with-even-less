## 为什么

`gnn_attack/` 目录经过三轮架构重构（v0 矩阵输入 → v1 固定特征 → v2 三阶段），积压了大量已不再使用的旧文件。`implement-three-phase-gnn-attack` 完成后新老代码并存，依赖关系混乱，新增的 `train_gnn_v2.py`、`attack_gnn_v2.py`、`gnn_self_training.py`、`gnn_reconstruction.py` 构成了独立的新管线，而旧文件 `train_gnn.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py` 仅互相引用、无任何新代码依赖。README.md 仍在描述旧管线，且模型文件和结果散落在根目录和 `results/` 之间无统一约定。需要一次彻底的清理和整合。

## 变更内容

- 删除 4 个旧文件（`train_gnn.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py`）
- 从 `gnn_attack.py` 提取 `_get_correct_edges_at_scale` 和 `check_accuracy_with_edges` 到 `gnn_model.py`
- 从 `gnn_model.py` 移除两个 deprecated 函数（`build_message_passing_graph`、`build_message_passing_graph_from_features`）
- 删除临时 `.pth` 文件（`test.pth`、根目录 `gnn_model.pth`）
- 统一所有模型文件和输出到 `results/` 目录
- 为 `attack_gnn_v2.py` 增加 `--train` 参数，实现一键训练+攻击复合入口
- 重写 README.md，反映新架构和新的文件约定

## 功能 (Capabilities)

### 新增功能
- `dead-code-removal`: 清理所有不再被新架构引用的旧代码文件、临时文件、deprecated 函数
- `composite-entry`: 提供 `attack_gnn_v2.py --train` 复合入口，可在一次命令中完成训练→自训练→形状重建全流程

### 修改功能
- `gnn-attack-pipeline`: README.md 和入口脚本中的路径约定统一为 `results/` 目录

## 影响

- 移除 `train_gnn.py`（旧训练脚本，被 `train_gnn_v2.py` 替代）
- 移除 `gnn_range_attack.py`（旧推理引擎，被 `gnn_self_training.py` 替代）
- 移除 `range_attack.py`（原始方法实现，仅旧入口引用）
- 移除 `gnn_attack.py`（旧入口脚本，评估函数提取后删除）
- 移除 `gnn_model.py` 中两个 deprecated 函数
- 移除 `test.pth`、根目录 `gnn_model.pth`（临时文件）
- 修改 `gnn_model.py`（新增评估工具函数）
- 修改 `attack_gnn_v2.py`（新增 `--train` 参数 + P/R 评估输出）
- 重写 `README.md`
