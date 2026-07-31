# 迁移指南

本文件用于记录 RECS 显式 API、语法糖、默认行为或数据语义发生变化时，用户从旧版本迁移到新版本需要执行的操作。

使用边界：

- `docs/CHANGELOG.md` 记录“发生了什么”。
- 本文件记录“用户代码应该怎么改”。
- `docs/API_REFERENCE.md` 和 `docs/USER_MANUAL.md` 只保留当前推荐写法。

## 当前状态

当前暂无需要迁移的已发布破坏性变更。

## 迁移条目模板

复制本节作为新版本迁移条目。

````markdown
## 从 vX.Y 迁移到 vA.B

### 1. 变更名称

变更类型：

- Breaking change / Deprecated / Behavior change

影响范围：

- 影响的类、函数、参数或返回值。
- 哪些用户代码会受影响。

旧写法：

```python
# old code
```

新写法：

```python
# new code
```

自动检查建议：

```bash
rg "old_api_or_pattern" .
```

兼容期：

- 从 `vX.Y` 开始发出 `DeprecationWarning`。
- 计划在 `vA.B` 移除旧写法。

注意事项：

- 需要用户人工判断的边界情况。
````

## 维护规则

- 每个破坏性变更必须有迁移条目。
- 每个废弃 API 必须写明废弃版本、替代写法和计划移除版本。
- 如果旧写法仍兼容，应在运行时发出 `DeprecationWarning`。
- 如果无法自动兼容，应在 release notes 中把该条目标为 `Breaking Changes`。
