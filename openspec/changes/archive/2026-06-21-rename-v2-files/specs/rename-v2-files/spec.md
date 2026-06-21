## 新增需求

### 需求:核心脚本文件名不含 _v2 后缀
`gnn_attack/` 目录下的核心脚本不得携带 `_v2` 历史版本后缀。

#### 场景:训练脚本无 v2 后缀
- **当** 检查 `gnn_attack/` 目录
- **那么** 存在 `train_gnn.py`
- **那么** 不存在 `train_gnn_v2.py`

#### 场景:攻击入口无 v2 后缀
- **当** 检查 `gnn_attack/` 目录
- **那么** 存在 `attack_gnn.py`
- **那么** 不存在 `attack_gnn_v2.py`

### 需求:attack_gnn.py 正确导入 train_gnn
`attack_gnn.py` 的 import 语句必须引用新文件名。

#### 场景:import 语句更新
- **当** 搜索 `attack_gnn.py` 中的 import 语句
- **那么** 存在 `from train_gnn import train_phase1`
- **那么** 不存在 `from train_gnn_v2 import`

### 需求:所有文档引用使用新文件名
`README.md` 和 `docs/design-code-review.md` 中的所有文件名引用必须不含 `_v2`。

#### 场景:README 无 v2 引用
- **当** 搜索 README.md 中的 `_v2`
- **那么** 不存在 `train_gnn_v2` 或 `attack_gnn_v2`

#### 场景:design-code-review 无旧路径
- **当** 搜索 `docs/design-code-review.md` 中的 `train_gnn_v2` 或 `attack_gnn_v2`
- **那么** 不存在匹配项
