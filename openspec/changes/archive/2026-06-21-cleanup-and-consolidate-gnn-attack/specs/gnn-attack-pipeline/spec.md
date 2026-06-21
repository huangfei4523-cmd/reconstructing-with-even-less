## 修改需求

### 需求:所有模型文件和输出必须统一在 results/ 目录下
所有脚本生成的模型文件（`.pth`）、结果文件（`.json`）、可视化图片（`.png`）必须默认输出到 `gnn_attack/results/` 目录。禁止在 `gnn_attack/` 根目录生成上述文件。

#### 场景:训练模型保存到 results/
- **当** 运行 `python train_gnn_v2.py --epochs 2 --samples 10`
- **那么** 默认将模型保存到 `results/phase1_model.pth`
- **那么** `gnn_attack/` 根目录不生成 `.pth` 文件

#### 场景:攻击结果保存到 results/
- **当** 运行 `python attack_gnn_v2.py --train --target-data cali_self`
- **那么** 结果 JSON 保存在 `results/result_cali_self_p100.json`
- **那么** 可视化图片保存在 `results/recon_cali_self_p100.png`

### 需求:README.md 反映新架构和路径约定
README.md 中的项目结构、命令示例、参数表必须反映三阶段架构和 `results/` 统一路径约定。不得包含已删除文件的引用。

#### 场景:README 项目结构仅列出当前文件
- **当** 阅读 README.md 的「项目结构」章节
- **那么** 不包含 `train_gnn.py`、`gnn_range_attack.py`、`gnn_attack.py`、`range_attack.py`
- **那么** 包含 `train_gnn_v2.py`、`attack_gnn_v2.py`、`gnn_self_training.py`、`gnn_reconstruction.py`

#### 场景:README 命令示例使用统一路径
- **当** 阅读 README.md 的示例命令
- **那么** 所有 `--save`、`--model`、`--output` 参数指向 `results/` 目录
- **那么** 不存在指向根目录的 `.pth` 路径
