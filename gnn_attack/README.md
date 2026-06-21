# GNN 增强的 "Reconstructing with Even Less" 攻击

基于论文 *Reconstructing with Even Less: Leakage Amplification and Graph Drawing*，使用 **图神经网络 (GNN)** 替代原始的集合交集方法来进行泄漏放大和边预测。

## 三阶段架构

```
Phase 1: 预训练（攻击者自生成）         Phase 2: 自训练（目标加密数据）      Phase 3: 形状重建
───────────────────────                 ─────────────────────────        ──────────────
多种形状 × 多种采样率 × 多种N           目标加密查询响应                    推断的边集合
       │                                      │                              │
       ▼                                      ▼                              ▼
  参数空间网格 → 训练 GNN              共现矩阵 C[N,N] →                 力导向布局 →
  学「空间近≈共现多」                  伪标签迭代微调                     2D 坐标 + 可视化
       │                                      │
       └──── GNN_pre ────→ 自训练 ────→ GNN_fine + E_hat
```

详细设计文档：`docs/design-three-phase-summary.md` / `docs/design-three-phase-detailed.md`

## 项目结构

```
gnn_attack/
├── train_gnn.py              # Phase 1 训练脚本（多场景参数空间网格训练）
├── attack_gnn.py             # 复合入口：一键训练+攻击/仅推理
├── gnn_self_training.py      # Phase 2 自训练引擎（伪标签+迭代微调+一致性正则化）
├── gnn_reconstruction.py     # Phase 3 力导向布局 + 形状重建
├── gnn_model.py              # 核心模型（共现图消息传递 + 评估工具）
├── data_loader.py            # 数据集加载工具
├── process_database.py       # 范围查询响应生成 + 采样
├── datasets/                 # 数据集文件
├── docs/                     # 设计文档
├── tests/                    # 测试套件
├── requirements.txt          # 依赖
└── README.md                 # 本文件
```

所有模型 checkpoint (`.pth`) 和输出结果 (`.json`, `.png`) 默认存放在 `results/` 目录。

## 安装依赖

```bash
cd gnn_attack
pip install -r requirements.txt
```

## 使用方式

### 1. 一键全流程（推荐）

**输入：** 目标数据集名  
**输出：** `results/phase1_model.pth` + `results/result_*.json` + `results/recon_*.png`

```bash
# 快速验证（约 30 秒）
python attack_gnn.py --train --train-epochs 2 --train-samples 10 --target-data cali_self

# 正式运行
python attack_gnn.py --train --train-epochs 30 --train-samples 200 --target-data cali_self
```

### 2. 分步执行

**Phase 1: 预训练**

```bash
# 快速测试
python train_gnn.py --epochs 10 --samples 20 --save results/phase1_model.pth

# 正式训练
python train_gnn.py --epochs 30 --samples 500 --save results/phase1_model.pth
```

**Phase 2+3: 攻击**

```bash
python attack_gnn.py --phase1-model results/phase1_model.pth --target-data cali_self

# 多采样率对比
for p in 1 5 10 50 100; do
    python attack_gnn.py --phase1-model results/phase1_model.pth \
        --target-data cali_self --p $p
done
```

## 脚本参数详解

### `train_gnn.py` — Phase 1 预训练

**输入：** 无（自动生成合成训练数据）  
**输出：** `results/phase1_model.pth`

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--epochs` | int | 训练轮数 | 30 |
| `--lr` | float | 学习率 (Adam) | 0.001 |
| `--hidden` | int | 隐藏层维度 | 64 |
| `--emb` | int | 节点嵌入维度 | 32 |
| `--samples` | int | 合成训练样本数（参数空间网格） | 500 |
| `--val-split` | float | 验证集划分比例 | 0.15 |
| `--save` | str | 模型保存路径 | results/phase1_model.pth |

### `attack_gnn.py` — 三阶段攻击复合入口

**输入：** 目标数据集名 + (model 或 --train)  
**输出：** `results/result_*.json`（含 Precision/Recall）+ `results/recon_*.png`

#### 模型参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--train` | flag | 自动执行 Phase 1 训练 | False |
| `--train-epochs` | int | Phase 1 训练轮数（--train 时生效） | 30 |
| `--train-samples` | int | Phase 1 合成样本数（--train 时生效） | 200 |
| `--phase1-model` | str | 已有模型路径（不指定 --train 时必填） | None |

#### 数据参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--target-data` | str | 目标数据集 | cali_self |
| `--p` | float | 响应采样百分比 (0~100) | 100 |

#### 输出控制

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--output` | str | 结果输出目录 | results |
| `--device` | str | 推理设备 `cpu` 或 `cuda` | cpu |

## 已知设计限制

- Phase 2 自训练依赖 Phase 1 预训练质量（验证集 AUC > 0.7 时方可靠）
- Phase 3 力导向布局正确性受推断边 Recall 影响（Recall < 0.5 时形状可能明显变形）
- 当前仅支持 2D 数据集，3D 数据集（nh, crg）的 Phase 3 布局需 3D 扩展
