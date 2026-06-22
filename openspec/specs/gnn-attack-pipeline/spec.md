### 需求: 主入口统一攻击流程

`attack_gnn.py` 作为三阶段 GNN 攻击的主入口，必须串联 Phase 1 训练（可选）、Phase 2 自训练、Phase 3 形状重建的完整流程。

#### 场景: 构建共现矩阵（闭式公式 + 二项噪声）

- **当** 数据集加载完成
- **那么** 系统使用闭式公式直接计算 C_target：C[i,j] = min(xᵢ,xⱼ) × min(yᵢ,yⱼ) × (N0-max(xᵢ,xⱼ)) × (N1-max(yᵢ,yⱼ))
- **而且** 构建结果与暴力枚举完全等价（逐元素误差 < 1e-5）
- **而且** cali_50（N≈1000）构建耗时 < 5 秒

#### 场景: p=100 时无采样噪声

- **当** 用户指定 `--p 100`
- **那么** C_target 不加任何噪声，直接逐行归一化

#### 场景: p<100 时引入二项采样噪声

- **当** 用户指定 `--p N`（N<100）
- **那么** C_target 的每个元素 c_ij 先乘以 p/100，再加相对噪声 σ_rel = sqrt((100-p)/(p × c_ij + ε))
- **而且** 噪声服从正态分布 N(0, σ_rel × c_ij)
- **而且** 加噪后 clip 到 [0, ∞)，再逐行归一化
- **而且** 低 p 产生更大方差（p=10 时的波动 > p=50）

#### 场景: 一键全流程（--train）

- **当** 用户指定 `--train` 参数
- **那么** 系统首先执行 Phase 1 训练（通过 `train_gnn.train_phase1()` 生成合成数据并训练 GNN）
- **而且** 训练完成后自动执行 Phase 2 自训练（`SelfTrainingLoop`）
- **而且** Phase 2 完成后自动执行 Phase 3 形状重建（`ForceDirectedLayout`）
- **而且** 最终输出 Precision/Recall 评估和重建可视化

#### 场景: 仅推理（--phase1-model）

- **当** 用户指定 `--phase1-model=<path>` 而不指定 `--train`
- **那么** 系统跳过 Phase 1，直接从指定路径加载预训练模型
- **而且** 依次执行 Phase 2 和 Phase 3

#### 场景: 结果持久化

- **当** 攻击流程完成
- **那么** 系统将 Precision、Recall、Edges、Phase 时间写入 `results/result_*.json` 文件
- **而且** 系统生成力导向布局的重建可视化图片到 `results/recon_*.png`

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

### 需求: 防止评估除零

攻击评估的 `_get_correct_edges_at_scale()` 和 `check_accuracy_with_edges()` 必须位于 `gnn_model.py` 的 §8 评估工具区。计算时必须使用安全除法防止 ZeroDivisionError。

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

### 需求: 所有模型文件和输出统一在 results/ 目录下

所有脚本生成的模型文件（`.pth`）、结果文件（`.json`）、可视化图片（`.png`）必须默认输出到 `gnn_attack/results/` 目录。禁止在 `gnn_attack/` 根目录生成上述文件。

#### 场景: 训练模型保存到 results/

- **当** 运行 `python train_gnn.py --epochs 2 --samples 10`
- **那么** 默认将模型保存到 `results/phase1_model.pth`
- **那么** `gnn_attack/` 根目录不生成 `.pth` 文件

#### 场景: 攻击结果保存到 results/

- **当** 运行 `python attack_gnn.py --train --target-data cali_self`
- **那么** 结果 JSON 保存在 `results/result_cali_self_p100.json`
- **那么** 可视化图片保存在 `results/recon_cali_self_p100.png`

### 需求: 核心脚本使用无后缀文件名

`gnn_attack/` 目录下的核心脚本必须使用 `train_gnn.py` 和 `attack_gnn.py`，不得携带 `_v2` 历史版本后缀。

#### 场景: 脚本无 v2 后缀

- **当** 检查 `gnn_attack/` 目录
- **那么** 存在 `train_gnn.py` 和 `attack_gnn.py`
- **那么** 不存在 `train_gnn_v2.py` 或 `attack_gnn_v2.py`

#### 场景: README 使用新文件名

- **当** 阅读 README.md 的命令示例
- **那么** 所有命令使用 `train_gnn.py` 和 `attack_gnn.py`
