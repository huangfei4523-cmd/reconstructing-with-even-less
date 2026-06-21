## 上下文

旧 v0/v1 文件已全部删除，`gnn_attack/` 中仅存 4 个三阶段核心脚本：`train_gnn_v2.py`、`attack_gnn_v2.py`、`gnn_self_training.py`、`gnn_reconstruction.py`。其中前两个的 `_v2` 后缀是历史遗留的区分标记，已无必要。

## 目标 / 非目标

**目标：**
- 将 `train_gnn_v2.py` → `train_gnn.py`，`attack_gnn_v2.py` → `attack_gnn.py`
- 更新所有引用这些文件名的代码导入、文档、设计文档

**非目标：**
- 不修改 `gnn_self_training.py` 和 `gnn_reconstruction.py`（它们始终没有 `_v2` 后缀）
- 不修改测试文件（测试文件 import 的是模块内函数，不 import 文件名）

## 决策

### D1: attack_gnn_v2.py → attack_gnn.py

不还原为 `gnn_attack.py`（旧文件名），因为目录名也是 `gnn_attack`，同名会导致 Python import 冲突（`import gnn_attack` 会优先匹配目录的 `__init__.py` 而非文件）。`attack_gnn.py` 语义清晰且无冲突。

替代方案：`gnn_attack.py` — 拒绝，与目录同名导致 import 问题。

### D2: 文件内模块引用更新

`attack_gnn.py` 中 `from train_gnn_v2 import train_phase1` 改为 `from train_gnn import train_phase1`。两文件自身的 docstring 和注释中的文件名引用同步更新。

## 风险 / 权衡

- [风险] 任何外部脚本直接 `python train_gnn_v2.py` 的调用会失效 → 影响范围：README 命令示例和 `attack_gnn.py` 内部 import，均在本变更范围内更新
