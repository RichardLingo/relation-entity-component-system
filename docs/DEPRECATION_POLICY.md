# API 弃用策略

本文件定义 RECS 上游库在 API、语法糖、默认行为发生变化时的弃用和移除规则。

## 目标

- 让用户在升级前能知道哪些写法即将变化。
- 让用户在运行旧代码时能看到明确提示。
- 避免无提示地删除用户正在使用的 API。

## 版本规则

RECS 采用语义化版本思路：

- Patch：修复 bug，不应引入破坏性变更。
- Minor：新增兼容功能；在 `0.x` 阶段如有必要可引入小范围破坏性变更，但必须写明迁移方法。
- Major：允许破坏性变更，必须提供迁移指南。

## 弃用流程

推荐流程：

1. 在新版本中保留旧 API。
2. 旧 API 运行时发出 `DeprecationWarning`。
3. `docs/CHANGELOG.md` 记录 `Deprecated`。
4. `docs/MIGRATION_GUIDE.md` 提供旧写法和新写法对照。
5. 至少经过一个发布周期后再移除旧 API。
6. 移除时在 `CHANGELOG.md` 记录 `Removed` 和迁移入口。

## Warning 文案模板

```python
import warnings

warnings.warn(
    "`old_api(...)` is deprecated since RECS vX.Y and will be removed in vA.B. "
    "Use `new_api(...)` instead. See docs/MIGRATION_GUIDE.md.",
    DeprecationWarning,
    stacklevel=2,
)
```

要求：

- 必须说明废弃开始版本。
- 必须说明计划移除版本，若尚未确定则写 `a future release`。
- 必须给出替代 API。
- 必须使用 `stacklevel=2` 或更合适的值，让 warning 指向用户代码。

## 何时允许直接移除

只有以下情况可以不经过兼容期：

- API 从未发布到正式版本。
- API 明确标记为内部实现细节。
- 原行为存在严重 bug、安全风险或数据破坏风险。
- 继续兼容会导致实现不可维护，并且已在 release notes 中显著说明。

## 文档同步要求

发生弃用或破坏性变更时，至少同步更新：

- `docs/CHANGELOG.md`
- `docs/MIGRATION_GUIDE.md`
- `docs/API_REFERENCE.md`
- `docs/USER_MANUAL.md`
- 相关 `demos/`
- 发布说明或 GitHub Release Notes

