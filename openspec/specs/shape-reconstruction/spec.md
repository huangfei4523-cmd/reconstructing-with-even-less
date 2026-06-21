### 需求: Phase 3 力导向布局形状重建

`gnn_reconstruction.py` 必须实现从推断边集合到 2D 坐标的力导向布局重建。

#### 场景: 无锚点布局

- **当** 调用 `ForceDirectedLayout(E_hat, N)`
- **那么** 构建 NetworkX 图，边权重 = 推断概率
- **而且** 运行 spring_layout 弹簧-斥力迭代至收敛或达到最大迭代次数
- **而且** 返回 pos[N,2] 坐标数组

#### 场景: Procrustes 对齐

- **当** 已知部分锚点坐标
- **那么** 系统对布局结果运行 Procrustes 分析对齐到锚点坐标系

#### 场景: 重建失败检测

- **当** 调用 `CheckReconstructionFailure(pos, E_hat)`
- **那么** 若推断边数 < N/2 → 标记失败
- **而且** 若坐标方差 < 1e-4 → 标记失败
- **而且** 若图连通分量 > N/5 → 标记警告
