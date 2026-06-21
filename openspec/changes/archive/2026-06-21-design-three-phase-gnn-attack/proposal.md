## 为什么

当前 GNN 攻击实现是单阶段的：用合成数据训练一个 GNN，然后直接推理目标加密数据。代码审查和实验结果已证明该方案在真实数据集（cali_50，13 小时推理输出 0 条边）上完全无效。根因不是「GNN 不行」，而是架构缺少两个关键阶段：让模型适应目标数据分布的自训练阶段，以及从推断的边重建出几何形状的布局阶段。

本提案定义一套完整的**三阶段可搜索加密攻击方案**——预训练（学通用规律）→ 自训练（贴合目标分布）→ 布局重建（复现形状）——产出概要设计文档和详细设计文档两份交付物。**这些设计文档将成为后续 gnn-attack 代码实现的强制依据——所有代码实现必须与设计文档中定义的接口契约、数据流和验证标准保持一致。**

## 变更内容

本变更是纯设计文档变更。交付两份设计文档：

| 交付物 | 受众 | 内容 |
|--------|------|------|
| `docs/design-three-phase-summary.md` | 概要设计 | 攻击场景全景、三阶段数据流图、各阶段输入输出总览、与原始方法/当前方案的对比 |
| `docs/design-three-phase-detailed.md` | 详细设计 | 每个阶段的算法原理、伪代码、接口契约、验证标准、边界条件处理 |

### 移除

- 移除 `docs/gnn-attack-design.md` — 旧的 v0→v1 架构演进文档，已被三阶段设计取代
- 移除 `docs/analysis-current-issues.md` — 问题分析已整合到三阶段设计的上下文说明中
- 移除 `docs/design-incremental-improvement.md` — 增量改进思路已被完整的三阶段方案吸收
- 移除 `docs/design-full-improvement.md` — 全量改进思路已被完整的三阶段方案吸收

不修改任何源代码文件。

## 功能 (Capabilities)

### 新增功能

- `attack-scenario-model`: 攻击者能力边界定义——明确每个阶段攻击者能获取什么数据、数据从哪里来、不能获取什么数据
- `gnn-pretraining`: Phase 1 预训练——攻击者用自生成合成数据训练 GNN，学「空间相邻≈高频共现」
- `gnn-self-training`: Phase 2 自训练——在目标加密数据的共现矩阵上，用伪标签迭代微调 GNN
- `shape-reconstruction`: Phase 3 形状重建——从推断的边通过力导向布局恢复 2D 点分布

## 影响

- 新增：`docs/design-three-phase-summary.md`（概要设计文档）
- 新增：`docs/design-three-phase-detailed.md`（详细设计文档）
- 移除：`docs/gnn-attack-design.md`、`docs/analysis-current-issues.md`、`docs/design-incremental-improvement.md`、`docs/design-full-improvement.md`（旧设计文档，已被三阶段方案取代）
- 主规范新增：`attack-scenario-model`、`gnn-pretraining`、`gnn-self-training`、`shape-reconstruction`
- 不修改任何源代码
- **后续 gnn-attack 模块的所有代码实现必须以这两份设计文档为权威参考——函数接口、数据格式、验证指标必须与设计文档保持一致**
