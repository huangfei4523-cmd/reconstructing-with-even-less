### 需求: 原始攻击方法对比执行

`gnn_attack.py` 在 `--baseline` 模式下必须同时运行原始集合交集攻击方法，与 GNN 方法输出同一评估标准下的对比结果。

#### 场景: 触发基线对比

- **当** 用户指定 `--baseline` 命令行开关
- **那么** 系统在 GNN 攻击完成后调用 `range_attack.general(new_responses)` 执行原始方法
- **而且** 使用相同的 `check_accuracy_with_edges` 函数评估精度和召回率

#### 场景: 输出对比结果

- **当** GNN 和 Baseline 评估均完成
- **那么** 结果 JSON 包含 `gnn` 和 `baseline` 两个键，分别记录 precision、recall、time、edges_used
- **而且** 终端打印对比摘要（GNN vs Original 的 P/R/T）

#### 场景: 对比可视化

- **当** `--compare-only` 未指定
- **那么** 系统分别为 GNN 和原始方法生成 Kamada-Kawai 布局的可视化图片，保存到 `results/` 目录

### 需求: 原始 attack.py 方法保留

`range_attack.py` 必须是原版 `attack.py` 中攻击核心逻辑的完整副本，以确保基线对比的公平性。

#### 场景: 泄漏放大流程

- **当** 调用 `leakage_augment(responses)`
- **那么** 系统依次执行 `fast_augment_responses`（两两取交集）、`reduce_to_domain_points`（点域归约）、`translate_responses_domain`（域值替换）
- **而且** 返回处理后的响应和 `go_back` 映射

#### 场景: 素数响应提取

- **当** 调用 `find_prime_responses(new_responses, 2)`
- **那么** 系统筛选所有大小为 2 且大小为素数的响应集合

#### 场景: 图构建

- **当** 素数响应就绪
- **那么** `make_simple_graph` 通过 `go_back` 反向映射构建 NetworkX 图

### 需求: 依赖声明完整性

`requirements.txt` 必须包含所有运行时 import 的第三方依赖。

#### 场景: sympy 依赖

- **当** 运行 `--baseline` 模式
- **那么** `import sympy` 必须成功，即 `requirements.txt` 必须包含 `sympy>=1.9` 依赖项

#### 场景: 所有导入可解析

- **当** 执行 `python -c "import range_attack"` 从 gnn_attack 目录
- **那么** 所有 `import` 语句不得抛出 `ModuleNotFoundError`
