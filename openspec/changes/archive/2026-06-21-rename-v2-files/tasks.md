## 1. 文件重命名

- [x] 1.1 重命名 `train_gnn_v2.py` → `train_gnn.py`，更新文件内 docstring 和注释中的名称引用
- [x] 1.2 重命名 `attack_gnn_v2.py` → `attack_gnn.py`，更新 `from train_gnn_v2 import` → `from train_gnn import`，更新 docstring

## 2. 文档更新

- [x] 2.1 更新 `README.md` — 项目结构、命令示例、参数表中所有 `_v2` 引用 → 去掉 `_v2`
- [x] 2.2 更新 `docs/design-code-review.md` — 文件路径 `train_gnn_v2.py`/`attack_gnn_v2.py` → `train_gnn.py`/`attack_gnn.py`

## 3. 验证

- [x] 3.1 确认 `gnn_attack/` 目录中不存在 `_v2` 后缀的 `.py` 文件
- [x] 3.2 运行 `python -m pytest tests/ -v` 确保全测试 PASS

> **最终验证门禁：** 目录无 `_v2` 文件，全测试 PASS
