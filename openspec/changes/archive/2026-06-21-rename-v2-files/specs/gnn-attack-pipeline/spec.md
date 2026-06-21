## 修改需求

### 需求:README.md 反映当前文件名
README.md 中的项目结构、命令示例、参数表必须使用 `train_gnn.py` 和 `attack_gnn.py` 而非带 `_v2` 后缀的文件名。

#### 场景:README 命令使用新文件名
- **当** 阅读 README.md 的命令示例
- **那么** 所有命令使用 `train_gnn.py` 而非 `train_gnn_v2.py`
- **那么** 所有命令使用 `attack_gnn.py` 而非 `attack_gnn_v2.py`
