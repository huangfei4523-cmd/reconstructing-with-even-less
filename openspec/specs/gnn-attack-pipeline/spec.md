### 需求: 主入口统一攻击流程

`gnn_attack.py` 作为 GNN 增强攻击的主入口，必须串联数据集加载、响应生成、GNN 攻击评估、原始方法对比的完整流程。

#### 场景: 加载数据集并生成响应

- **当** 用户通过命令行指定 `-points=<name>`、`-dist=<dist>`、`-p=<percent>` 参数
- **那么** 系统通过 `data_loader.load_dataset()` 加载指定数据集，返回包含 `points`、`map_to_original`、`N0`、`N1` 的配置字典
- **而且** 系统通过 `process_database` 生成全部范围查询响应、按指定分布采样、翻译为实际响应集合

#### 场景: GNN 攻击评估

- **当** 采样响应就绪且模型文件存在
- **那么** 系统调用 `gnn_range_attack.gnn_range_attack()` 执行 GNN 增强攻击，返回重建图 `G_gnn` 和 `go_back` 映射
- **而且** 系统计算 Precision 和 Recall 指标，确保分母使用 `1e-10` 防止除零

#### 场景: 原始方法对比

- **当** 用户指定 `--baseline` 开关
- **那么** 系统在同一数据集上调用 `range_attack.general()` 运行原始攻击，输出精度、召回率、耗时对比

#### 场景: 结果持久化

- **当** 攻击评估完成
- **那么** 系统将 GNN 和 Baseline 的 Precision、Recall、Time、EdgesUsed 写入 `results/` 目录下的 JSON 文件
- **而且** 系统生成 Kamada-Kawai 布局的可视化图片（除非指定 `--compare-only`）

### 需求: 数据集加载统一化

`data_loader.py` 必须提供统一的数据集加载接口 `load_dataset(name, N0, N1)`。硬编码数据集（cali_50、nh）的加载逻辑必须使用包内 Python import 方式。

#### 场景: 加载硬编码数据集（包内导入）

- **当** 数据集名称为 `cali_50`、`cali_self` 或 `nh`
- **那么** 系统通过 `sys.path` 添加项目根目录后 `from dataset import cali_all, cali_self, nh` 导入对应变量
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
