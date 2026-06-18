# GNN 增强的 "Reconstructing with Even Less" 攻击

基于论文 *Reconstructing with Even Less: Leakage Amplification and Graph Drawing*，使用 **图神经网络 (GNN)** 替代原始的集合交集方法来进行泄漏放大和边预测。

## 设计思路

### 原始方法的问题

```
响应集合
  ↓ 两两取交集（硬约束）
  ↓ 找大小为2的响应
  ↓ 建图
```

原始方法仅利用 **大小为 2 的直接交集**来推断邻接关系，信息利用效率受限。

### GNN 增强方法

```
响应集合
  ↓ 计算共现矩阵（所有响应大小）
  ↓ GNN 消息传递（学习空间关系模式）
  ↓ 软阈值边预测
  ↓ 建图
```

GNN 可以从**所有响应大小**的共现模式中学习，利用更多的统计信息，在信息极度稀疏时（如 p=1%）可能更鲁棒。

## 项目结构

```
gnn_attack/
├── __init__.py
├── gnn_model.py              # GNN 模型定义 + 训练数据生成
├── train_gnn.py              # 训练脚本
├── gnn_range_attack.py       # GNN 增强的攻击引擎
├── gnn_attack.py             # 主入口（攻击 + 评估 + 可视化）
├── range_attack.py           # 原始方法副本（用于 baseline 对比）
├── process_database.py       # 数据库处理（从原版复制）
├── data_loader.py            # 数据集加载工具
├── datasets/                 # 数据集文件（从原版复制）
├── requirements.txt          # 依赖
└── README.md                 # 本文件
```

## 安装依赖

```bash
cd gnn_attack
conda create -n gnn-attack python=3.9
conda activate gnn-attack

# 安装 PyTorch (CPU 版)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt

# （可选）如果需要 Cuda 版 PyTorch
# pip install torch
```

## 使用方式

### 1. 训练 GNN 模型

```bash
# 在合成网格数据上训练
python train_gnn.py --epochs 50 --samples 500 --save gnn_model.pth

# 自定义参数
python train_gnn.py --epochs 100 --lr 0.001 --hidden 64 --emb 32 \
    --samples 1000 --grid 15 15 --ratio 0.03 --save gnn_model.pth
```

### 2. 运行 GNN 增强攻击

```bash
# 基本用法
python gnn_attack.py -points=cali_50 -dist=uniform -p=10 --model gnn_model.pth

# 极稀疏场景 (1% 响应)
python gnn_attack.py -points=grid -dist=uniform -p=1 --model gnn_model.pth

# 同时运行原始方法做对比
python gnn_attack.py -points=cali_50 -dist=uniform -p=5 --model gnn_model.pth --baseline
```

### 3. 对比实验

```bash
# 多点采样率对比
for p in 1 2 5 10 20 50 100; do
    python gnn_attack.py -points=grid -dist=uniform -p=$p \
        --model gnn_model.pth --baseline --compare-only
done
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-points` | 数据集 (cali_50, grid, dg, crg, nh, boat) | small_grid |
| `-p` | 响应采样百分比 | 100 |
| `-dist` | 采样分布 (uniform, beta, gaussian) | uniform |
| `--model` | GNN 模型权重路径 | gnn_model.pth |
| `--threshold` | 边预测阈值 | 0.5 |
| `--baseline` | 同时运行原始方法对比 | False |
| `--device` | 推理设备 (cpu/cuda) | cpu |

## 预期提升

GNN 在以下场景中可能优于原始方法：

1. **极低采样率 (p=1%~5%)**：GNN 能利用全共现矩阵的统计模式
2. **噪声响应**：GNN 的软阈值比硬交集更鲁棒
3. **不规则点分布**：GNN 可以学习特定数据分布的邻接模式
4. **跨数据集泛化**：在非网格数据（如 cali_50）上，GNN 可能学到更通用的空间关系

## 与原版的关系

- 原版 `reconstructing-with-even-less/` 代码**完全不变**
- `gnn_attack/` 是独立的改进版本，共享 `process_database.py` 和 `datasets/`
- `range_attack.py` 的副本用于 `--baseline` 对比
