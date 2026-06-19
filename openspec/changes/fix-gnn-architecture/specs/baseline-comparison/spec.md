## 修改需求

### 需求: 依赖声明完整性

`requirements.txt` 必须包含所有运行时 import 的第三方依赖。

#### 场景: sympy 依赖

- **当** 运行 `--baseline` 模式
- **那么** `import sympy` 必须成功，即 `requirements.txt` 必须包含 `sympy>=1.9` 依赖项

#### 场景: 所有导入可解析

- **当** 执行 `python -c "import range_attack"` 从 gnn_attack 目录
- **那么** 所有 `import` 语句不得抛出 `ModuleNotFoundError`
