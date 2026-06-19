## 修改需求

### 需求: 数据集加载统一化

`data_loader.py` 必须提供统一的数据集加载接口 `load_dataset(name, N0, N1)`。硬编码数据集（cali_50、nh）的加载逻辑必须从 `exec()` 解析源码改为包内 Python import 方式。

#### 场景: 加载硬编码数据集（包内导入）

- **当** 数据集名称为 `cali_50` 或 `nh`
- **那么** 系统通过 `sys.path` 添加项目根目录后 `from dataset import cali_all, nh` 导入对应变量
- **而且** 通过 `process_database.scale_points` 和 `make_database_from_points` 转换为标准格式

#### 场景: 加载 pickle 数据集

- **当** 数据集名称为 `dg`、`crg` 或 `boat`
- **那么** 系统从 `datasets/` 目录加载对应的 pickle 文件
- **而且** 通过 `process_database.make_database_from_points` 或 `make_database_from_points_3D` 转换为标准格式

#### 场景: 生成合成数据集

- **当** 数据集名称为 `grid`
- **那么** 系统通过 `process_database.get_random_database(N0, N1, 1)` 动态生成网格数据

### 需求: 防止评估除零（修复自环 Bug）

攻击评估的 `_get_correct_edges_at_scale()` 必须正确使用邻居点的坐标构建 Ground Truth 边集。

#### 场景: 构建正确边集

- **当** 遍历每个点 i 及其邻居点 n ∈ N(i)
- **那么** 系统使用 `tuple(dictionarry[nt])`（邻居坐标）而非 `tuple(coord)`（自身坐标）构建边对
- **而且** 正确边集不包含任何自环 `(coord, coord)`

#### 场景: Recall 安全除法

- **当** `max_correct` 长度为 0
- **那么** Recall 计算使用 `len(correct_edges_set) / (len(max_correct) + 1e-10)` 防止 ZeroDivisionError

#### 场景: Precision 安全除法

- **当** 图 `G` 没有任何边
- **那么** Precision 计算使用 `len(correct_edges_set) / (len(correct_edges_set) + incorrect + 1e-10)` 防止 ZeroDivisionError
