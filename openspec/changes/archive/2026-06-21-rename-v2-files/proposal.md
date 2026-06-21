## 为什么

`cleanup-and-consolidate-gnn-attack` 已删除所有 v0/v1 旧文件（`train_gnn.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py`），现 `gnn_attack/` 目录下仅保留三阶段架构的 4 个核心脚本。文件名中的 `_v2` 后缀不再有区分意义，应去除以简化命名。

## 变更内容

- 重命名 `train_gnn_v2.py` → `train_gnn.py`
- 重命名 `attack_gnn_v2.py` → `attack_gnn.py`
- 更新 `attack_gnn.py` 中 `from train_gnn_v2 import` → `from train_gnn import`
- 更新 `README.md` 中所有 `_v2` 文件名引用
- 更新 `docs/design-code-review.md` 中的文件路径引用

## 功能 (Capabilities)

### 新增功能
- `rename-v2-files`: 去除核心脚本的 `_v2` 后缀，使文件命名与当前唯一架构保持一致

## 影响

- 重命名 `gnn_attack/train_gnn_v2.py` → `gnn_attack/train_gnn.py`
- 重命名 `gnn_attack/attack_gnn_v2.py` → `gnn_attack/attack_gnn.py`
- 修改 `attack_gnn.py` 中的 import 语句
- 修改 `README.md` 中的命令示例和参数表
- 修改 `docs/design-code-review.md` 中的文件路径引用
