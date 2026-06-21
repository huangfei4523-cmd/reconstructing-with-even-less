## 1. 评估函数迁移 + deprecated 清理

- [x] 1.1 从 `gnn_attack.py` 提取 `_get_correct_edges_at_scale` 到 `gnn_model.py` §8 评估工具区
- [x] 1.2 从 `gnn_attack.py` 提取 `check_accuracy_with_edges` 到 `gnn_model.py` §8 评估工具区
- [x] 1.3 从 `gnn_model.py` 删除 `build_message_passing_graph` 和 `build_message_passing_graph_from_features`
- [x] 1.4 更新 `tests/test_graph_build.py` 改为 import `build_cooc_message_graph` 并修改用例适配新签名
- [x] 1.5 更新 `tests/test_accuracy.py` 改为从 `gnn_model` import `_get_correct_edges_at_scale`

> **验证门禁：** `pytest tests/test_graph_build.py tests/test_accuracy.py` 全部 PASS

## 2. 删除旧文件

- [x] 2.1 删除 `gnn_attack/train_gnn.py`
- [x] 2.2 删除 `gnn_attack/gnn_range_attack.py`
- [x] 2.3 删除 `gnn_attack/range_attack.py`
- [x] 2.4 删除 `gnn_attack/gnn_attack.py`（评估函数已在任务 1.1-1.2 提取）
- [x] 2.5 删除 `gnn_attack/test.pth` 和 `gnn_attack/gnn_model.pth`（临时模型文件）

> **验证门禁：** `gnn_attack/` 目录中不包含以上 7 个文件；`pytest tests/` 全部 PASS

## 3. 训练逻辑提取 + 复合入口

- [x] 3.1 在 `train_gnn_v2.py` 中将训练逻辑提取为 `def train_phase1(epochs, samples, save_path, device) -> str`，保留原有 `if __name__ == "__main__"` 调用它
- [x] 3.2 在 `attack_gnn_v2.py` 中新增 `--train`、`--train-epochs`（默认 30）、`--train-samples`（默认 200）参数
- [x] 3.3 当指定 `--train` 时，内部调用 `train_gnn_v2.train_phase1(...)` 执行训练，然后将模型用于 Phase 2
- [x] 3.4 在 Phase 3 完成后，调用 `_get_correct_edges_at_scale` + `check_accuracy_with_edges` 输出 Precision/Recall
- [x] 3.5 结果 JSON 新增 `precision`、`recall` 字段

> **验证门禁：** `python attack_gnn_v2.py --train --train-epochs 2 --train-samples 10 --target-data cali_self` 全流程无报错，终端输出含 Precision/Recall 两行

## 4. README.md 重写

- [x] 4.1 重写「项目结构」章节，列出当前实际存在的 9 个核心文件（不含 datasets/tests/docs）
- [x] 4.2 新增「三阶段架构」章节，引用设计文档中的 Phase 1/2/3 示意图
- [x] 4.3 更新「使用方式」章节，用 `train_gnn_v2.py` 和 `attack_gnn_v2.py` 替换旧命令，所有 `--save`/`--output` 指向 `results/`
- [x] 4.4 更新「脚本参数详解」章节，移除 `train_gnn.py` 和 `gnn_attack.py` 参数表，新增 `train_gnn_v2.py` 和 `attack_gnn_v2.py` 参数表
- [x] 4.5 删除「预期提升」章节（不再适用），删除「与原版的关系」章节（旧文件已清理）

> **验证门禁：** README.md 中不出现 `train_gnn.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py` 文件名

## 5. 全量回归测试

- [x] 5.1 运行 `pytest tests/ -v` 确保所有测试全部 PASS
- [x] 5.2 运行 `python attack_gnn_v2.py --train --train-epochs 2 --train-samples 10 --target-data cali_self` 全流程通过

> **验证门禁：** 所有测试 PASS，全流程无报错，`results/` 中包含 `.pth`、`.json`、`.png` 各至少一个文件
