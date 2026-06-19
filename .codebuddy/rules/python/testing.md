---
paths:
  - "**/*.py"
  - "**/*.pyi"
---
# Python Testing

> This file extends [common/testing.md](../common/testing.md) with Python specific content.

## 共享库测试覆盖规范

**所有 `src/sd3_backdoor/` 下的共享库模块必须有对应的 `tests/test_<module>.py` 基础测试文件。**

- 测试文件命名: `tests/test_<package>_<module>.py`（如 `src/sd3_backdoor/core/model.py` → `tests/test_core_model.py`）
- 基础测试包含：导入验证、函数签名检查、参数有效性、边界情况
- GPU 依赖的测试使用 `@pytest.mark.skip(reason='需要 GPU')` 跳过
- 运行: `pytest tests/ -v --tb=short`

## Framework

Use **pytest** as the testing framework.

## Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

## Test Organization

Use `pytest.mark` for test categorization:

```python
import pytest

@pytest.mark.unit
def test_calculate_total():
    ...

@pytest.mark.integration
def test_database_connection():
    ...
```

## Reference

See skill: `python-testing` for detailed pytest patterns and fixtures.
